# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.glm4_moe_lite.tt.layer0_tt import _alloc_contiguous_page_table, _round_up
from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _sampling_result_to_token(result, mesh_device) -> int:
    """Extract token ID from trace-sampling result, handling mesh-distributed tensors."""
    if isinstance(result, tuple):
        values_tt, indices_tt = result
    else:
        indices_tt = result
        values_tt = None

    is_mesh = mesh_device.__class__.__name__ == "MeshDevice"
    if is_mesh:
        if values_tt is not None:
            val_shards = ttnn.get_device_tensors(values_tt)
            idx_shards = ttnn.get_device_tensors(indices_tt)
            best_val, best_idx = float("-inf"), 0
            for v_tt, i_tt in zip(val_shards, idx_shards):
                v = float(ttnn.to_torch(v_tt.cpu()).flatten()[0].item())
                if v > best_val:
                    best_val = v
                    best_idx = int(ttnn.to_torch(i_tt.cpu()).flatten()[0].item())
            return best_idx
        device_tensors = ttnn.get_device_tensors(indices_tt)
        return int(ttnn.to_torch(device_tensors[0].cpu()).flatten()[0].item())
    return int(ttnn.to_torch(indices_tt.cpu()).flatten()[0].item())


def _set_default_fabric_config(num_devices: int) -> None:
    """Match vLLM/tt-metal conftest behavior: set fabric before opening a mesh.

    Some systems can segfault during mesh discovery/open if fabric isn't set.
    """
    if int(num_devices) <= 1:
        return
    is_6u = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_6u else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)


def _parse_tt_dtype(raw: str) -> ttnn.DataType:
    raw = (raw or "").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if raw in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported --kv-cache-dtype={raw!r} (expected bf16 or bf8)")


def _alloc_paged_kvpe_cache_from_cpu(
    *,
    device: object,
    max_num_blocks: int,
    block_size: int,
    kvpe_dim: int,
    tt_dtype: ttnn.DataType,
) -> ttnn.Tensor:
    """Allocate a paged KVPE cache without `ttnn.zeros`.

    `ttnn.zeros(..., device=MeshDevice)` has been observed to hang in some
    environments; for bring-up we prefer staging a CPU zero tensor and uploading
    it via `ttnn.as_tensor`.
    """
    torch_dtype = torch.bfloat16
    host = torch.zeros((int(max_num_blocks), 1, int(block_size), int(kvpe_dim)), dtype=torch_dtype, device="cpu")
    is_mesh_device = device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh_device else None
    return ttnn.as_tensor(
        host,
        device=device,
        mesh_mapper=mesh_mapper,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=tt_dtype,
        cache_file_name=None,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Run GLM-4.7-Flash end-to-end on TT without vLLM (debug helper).")
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument(
        "--prompt", default="Say hello in exactly 3 words.", help="Prompt text (ignored if --prompt-file is set)."
    )
    ap.add_argument(
        "--prompt-file",
        default=None,
        metavar="PATH",
        help="Read prompt from file (e.g. long document to summarize). Overrides --prompt.",
    )
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--cache-dir",
        default="~/.cache/ttnn/models/glm4_moe_lite/vllm",
        help="TT weight cache dir (default reuses the vLLM cache to avoid reconverting all layers).",
    )
    ap.add_argument("--mesh-rows", type=int, default=1, help="Number of rows in mesh shape.")
    ap.add_argument("--mesh-cols", type=int, default=1, help="Number of columns in mesh shape.")
    ap.add_argument(
        "--device-ids",
        default="auto",
        help="Comma-separated physical device ids (length=mesh-cols) or 'auto' to let TTNN choose.",
    )
    ap.add_argument("--kv-cache-dtype", default="bf16", help="bf16 (correctness) or bf8 (memory/perf).")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--min-cache-tokens", type=int, default=128, help="Allocate at least this many tokens in KV cache.")
    ap.add_argument("--batch-size", type=int, default=1, help="Batch size for decode (replicates the prompt B times).")
    ap.add_argument(
        "--phase",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Which phase to profile: prefill (real flash_mla_prefill), decode, or both.",
    )
    ap.add_argument(
        "--enable-trace",
        action="store_true",
        default=False,
        help="Enable traced decode execution (captures device command trace on first call, replays on subsequent calls).",
    )
    ap.add_argument(
        "--trace-mode",
        choices=["logits", "sampling"],
        default="logits",
        help="Trace mode: 'logits' returns logits to host (same as eager), 'sampling' does on-device greedy top-1.",
    )
    ap.add_argument(
        "--simulate-context-len",
        type=int,
        default=0,
        help="Repeat prompt tokens to reach this context length (0=disabled). "
        "Useful for testing prefill/decode at long sequence lengths without a real long prompt.",
    )
    ap.add_argument(
        "--fake-context-len",
        type=int,
        default=0,
        help="Skip prefill and treat KV cache as already filled to this length (0=disabled). "
        "Cache is zero-filled; decode runs at this context length for fast decode-only benchmarking. "
        "Incompatible with phase=prefill.",
    )
    ap.add_argument(
        "--input-tokens-json",
        default=None,
        metavar="PATH",
        help="Load pre-tokenized input_ids from a JSON file (skips tokenizer entirely). "
        "JSON must have an 'input_ids' key with a list of integer token IDs. "
        "Overrides --prompt, --prompt-file, and --simulate-context-len.",
    )
    ap.add_argument(
        "--warmup",
        action="store_true",
        default=False,
        help="Run a warmup prefill+decode before the timed run to exclude compilation "
        "overhead from measurements (matches Llama 70B methodology).",
    )
    args = ap.parse_args()

    model_id = str(args.model_id)
    snap = Path(resolve_best_effort_snapshot_dir(model_id))
    missing = find_missing_shards(snap)
    if missing:
        raise SystemExit(
            f"Snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first (example: {missing[0]})"
        )

    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    tok = None
    if getattr(args, "input_tokens_json", None):
        json_path = Path(os.path.expanduser(str(args.input_tokens_json)))
        if not json_path.is_file():
            raise SystemExit(f"--input-tokens-json not found: {json_path}")
        with open(json_path) as f:
            data = json.load(f)
        prompt_ids_single = torch.tensor([data["input_ids"]], dtype=torch.int32)  # [1, S]
        prompt_len = int(prompt_ids_single.shape[1])
        print(f"[DEBUG] Loaded {prompt_len} pre-tokenized tokens from {json_path}", flush=True)
    else:
        prompt_text = str(args.prompt)
        if getattr(args, "prompt_file", None):
            path = Path(os.path.expanduser(str(args.prompt_file)))
            if not path.is_file():
                raise SystemExit(f"--prompt-file not found: {path}")
            prompt_text = path.read_text(encoding="utf-8", errors="replace").strip()
            print(f"[DEBUG] Loaded prompt from {path} ({len(prompt_text)} chars)", flush=True)

        tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
        enc = tok(prompt_text, return_tensors="pt", add_special_tokens=True)
        prompt_ids_single = enc["input_ids"].to(dtype=torch.int32)  # [1, S]
        if prompt_ids_single.ndim != 2 or int(prompt_ids_single.shape[0]) != 1:
            raise ValueError(f"expected prompt input_ids [1,S], got {tuple(prompt_ids_single.shape)}")
        prompt_len = int(prompt_ids_single.shape[1])

    fake_context_len = int(args.fake_context_len)
    if fake_context_len > 0:
        if args.phase == "prefill":
            raise SystemExit("--fake-context-len requires phase=decode or phase=both (prefill is skipped).")
        # Use short prompt (one token) for bootstrap decode; context length is fake.
        prompt_len = fake_context_len
        prompt_ids_single = prompt_ids_single[:, :1].contiguous()  # [1, 1]
        prompt_lens = [1] * batch_size
        print(
            f"[DEBUG] Fake context: skip prefill, KV cache zero-filled, decode at context_len={fake_context_len}",
            flush=True,
        )
    else:
        sim_ctx = int(args.simulate_context_len)
        if sim_ctx > 0 and sim_ctx > prompt_len:
            repeats = (sim_ctx + prompt_len - 1) // prompt_len
            prompt_ids_single = prompt_ids_single.repeat(1, repeats)[:, :sim_ctx]  # [1, sim_ctx]
            prompt_len = sim_ctx
            print(f"[DEBUG] Expanded prompt to {prompt_len} tokens via repetition", flush=True)
        prompt_lens = [prompt_len] * batch_size

    prompt_ids = prompt_ids_single.repeat(batch_size, 1)  # [B, S]

    mesh_rows = int(args.mesh_rows)
    mesh_cols = int(args.mesh_cols)
    if mesh_rows <= 0 or mesh_cols <= 0:
        raise ValueError("--mesh-rows and --mesh-cols must be > 0")
    n_devices = mesh_rows * mesh_cols
    device_ids_raw = str(args.device_ids).strip().lower()
    device_ids: list[int] | None
    if device_ids_raw in {"auto", ""}:
        device_ids = None
    else:
        device_ids = [int(x) for x in str(args.device_ids).split(",") if x.strip() != ""]
        if len(device_ids) != n_devices:
            raise ValueError(
                f"--device-ids must have exactly mesh-rows*mesh-cols entries (got {len(device_ids)} vs {n_devices})"
            )

    # Ensure MoE is enabled (matches production intent).
    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"

    kv_cache_dtype = _parse_tt_dtype(str(args.kv_cache_dtype))
    block_size = int(args.block_size)
    max_new_tokens = int(args.max_new_tokens)

    total_len = prompt_len + max_new_tokens
    total_len = max(total_len, int(args.min_cache_tokens))
    blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)

    # Open a mesh device for consistency with vLLM (even if mesh-cols=1).
    _set_default_fabric_config(n_devices)
    open_kwargs = {
        "mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols),
        "dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    }
    if device_ids is not None:
        open_kwargs["physical_device_ids"] = device_ids
    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        print("[DEBUG] Creating model...", flush=True)
        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=Path(os.path.expanduser(str(args.cache_dir))),
            max_seq_len=int(blocks_per_seq * block_size),
        )

        print(f"[DEBUG] Model created. num_layers_to_run={runner.num_layers_to_run}", flush=True)
        kvpe_dim = int(runner.hparams.kv_lora_rank + runner.hparams.qk_rope_head_dim)
        kv_cache = [
            _alloc_paged_kvpe_cache_from_cpu(
                device=mesh_device,
                max_num_blocks=int(batch_size * blocks_per_seq),
                block_size=block_size,
                kvpe_dim=kvpe_dim,
                tt_dtype=kv_cache_dtype,
            )
            for _ in range(int(runner.num_layers_to_run))
        ]
        page_table = _alloc_contiguous_page_table(batch=batch_size, blocks_per_seq=blocks_per_seq)

        # Pre-load all layer weights before the decode loop.  Lazy loading
        # inside decode() can deadlock because ttnn.as_tensor (host->device DMA)
        # contends with in-flight device compute ops dispatched earlier in the
        # same decode call (embedding, reshape, etc.).
        t_preload = time.perf_counter()
        for li in range(runner.num_layers_to_run):
            runner._ensure_layer_weights(li)
        print(
            f"Pre-loaded {runner.num_layers_to_run} layer(s) in {time.perf_counter() - t_preload:.1f}s",
            flush=True,
        )

        phase = str(args.phase)
        use_fake_context = fake_context_len > 0

        # -- Prefill phase (skipped when --fake-context-len) --
        if use_fake_context:
            prefill_s = 0.0
            logits = None
            last_sampling_result = None
            use_trace = args.enable_trace
            use_sampling = use_trace and args.trace_mode == "sampling"
            # Bootstrap decode at start_pos=fake_context_len (cache is zero-filled).
            bootstrap_token = prompt_ids[:, 0:1].contiguous()  # [B, 1]
            pos_bootstrap = torch.tensor([fake_context_len] * batch_size, dtype=torch.int32)
            print(f"[DEBUG] Bootstrap decode at start_pos={fake_context_len} (no prefill)...", flush=True)
            t0 = time.perf_counter()
            if use_sampling:
                last_sampling_result = runner.decode(
                    tokens=bootstrap_token,
                    start_pos=pos_bootstrap,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    sampling_params=True,
                )
            else:
                logits = runner.decode(
                    tokens=bootstrap_token,
                    start_pos=pos_bootstrap,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=use_trace,
                )
            prefill_s = time.perf_counter() - t0
        elif phase in ("prefill", "both"):
            if args.warmup:
                print(f"\n=== Warmup prefill (compile) ===", flush=True)
                t_wu = time.perf_counter()
                _ = runner.prefill(
                    tokens=prompt_ids,
                    prompt_lens=prompt_lens,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    seq_pad_multiple=block_size,
                )
                warmup_prefill_s = time.perf_counter() - t_wu
                print(f"warmup_prefill_s={warmup_prefill_s:.3f}", flush=True)

            t0 = time.perf_counter()
            logits = runner.prefill(
                tokens=prompt_ids,
                prompt_lens=prompt_lens,
                page_table=page_table,
                kv_cache=kv_cache,
                seq_pad_multiple=block_size,
            )
            prefill_s = time.perf_counter() - t0
            print(f"\n=== Prefill (real flash_mla_prefill) ===", flush=True)
            print(f"batch_size={batch_size} prompt_len={prompt_len} prefill_s={prefill_s:.3f}", flush=True)
            if phase == "prefill":
                for t in kv_cache:
                    ttnn.deallocate(t)
                ttnn.close_mesh_device(mesh_device)
                return 0
        else:
            print(f"[DEBUG] Starting iterative warm-up decode for {prompt_len} prompt tokens...", flush=True)
            t0 = time.perf_counter()
            logits = None
            use_trace = args.enable_trace
            use_sampling = use_trace and args.trace_mode == "sampling"
            for t in range(prompt_len):
                print(f"[DEBUG] Warm-up decode token {t}/{prompt_len}...", flush=True)
                tok_in = prompt_ids[:, t : t + 1].contiguous()  # [B, 1]
                pos = torch.tensor([t] * batch_size, dtype=torch.int32)  # [B]
                if use_sampling:
                    result = runner.decode(
                        tokens=tok_in,
                        start_pos=pos,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        enable_trace=True,
                        sampling_params=True,
                    )
                    logits = None
                    last_sampling_result = result
                else:
                    logits = runner.decode(
                        tokens=tok_in,
                        start_pos=pos,
                        page_table=page_table,
                        kv_cache=kv_cache,
                        enable_trace=use_trace,
                    )
                print(f"[DEBUG] Warm-up decode token {t} complete", flush=True)
            if use_sampling:
                assert last_sampling_result is not None
            else:
                assert logits is not None
            prefill_s = time.perf_counter() - t0

        # -- Decode phase --
        use_trace = args.enable_trace
        use_sampling = use_trace and args.trace_mode == "sampling"
        mode_label = "eager"
        if use_trace:
            mode_label = f"trace-{args.trace_mode}"

        # For batch>1, track per-sequence generated tokens; use sequence 0 for display.
        generated: list[list[int]] = [[] for _ in range(batch_size)]
        if use_sampling and (phase == "decode" or use_fake_context):
            token_in = _sampling_result_to_token(last_sampling_result, mesh_device)
            tokens_in = [token_in] * batch_size
        else:
            # logits: [B, 1, vocab] or [B, vocab]
            logits_flat = logits.reshape(batch_size, -1)
            tokens_in = [int(torch.argmax(logits_flat[b]).item()) for b in range(batch_size)]
        for b in range(batch_size):
            generated[b].append(tokens_in[b])

        # Warmup decode: run one decode step to compile/capture trace before timing.
        if args.warmup and use_trace:
            print(f"\n=== Warmup decode (trace capture) ===", flush=True)
            t_wu_dec = time.perf_counter()
            wu_pos = torch.tensor([prompt_len] * batch_size, dtype=torch.int32)
            wu_tok = torch.tensor([[t] for t in tokens_in], dtype=torch.int32)
            if use_sampling:
                _ = runner.decode(
                    tokens=wu_tok,
                    start_pos=wu_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    sampling_params=True,
                )
            else:
                _ = runner.decode(
                    tokens=wu_tok,
                    start_pos=wu_pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=use_trace,
                )
            warmup_decode_s = time.perf_counter() - t_wu_dec
            print(f"warmup_decode_s={warmup_decode_s:.3f}", flush=True)

        step_times: list[float] = []
        t_decode0 = time.perf_counter()
        # When using fake context, bootstrap already wrote at prompt_len; start loop at prompt_len+1.
        pos_offset = 1 if use_fake_context else 0
        for step in range(max_new_tokens - 1):
            pos = torch.tensor([prompt_len + pos_offset + step] * batch_size, dtype=torch.int32)  # [B]
            tok_in = torch.tensor([[t] for t in tokens_in], dtype=torch.int32)  # [B, 1]
            t_step = time.perf_counter()

            if use_sampling:
                result = runner.decode(
                    tokens=tok_in,
                    start_pos=pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=True,
                    sampling_params=True,
                )
                token_in = _sampling_result_to_token(result, mesh_device)
                tokens_in = [token_in] * batch_size
            else:
                logits = runner.decode(
                    tokens=tok_in,
                    start_pos=pos,
                    page_table=page_table,
                    kv_cache=kv_cache,
                    enable_trace=use_trace,
                )
                logits_flat = logits.reshape(batch_size, -1)
                tokens_in = [int(torch.argmax(logits_flat[b]).item()) for b in range(batch_size)]

            step_ms = (time.perf_counter() - t_step) * 1000
            step_times.append(step_ms)
            for b in range(batch_size):
                generated[b].append(tokens_in[b])
        decode_s = time.perf_counter() - t_decode0

        full_ids = torch.cat([prompt_ids[0:1], torch.tensor([generated[0]], dtype=torch.int32)], dim=1)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
        text = tok.decode(full_ids[0].tolist(), skip_special_tokens=True)
        print("", flush=True)
        print(f"=== TT greedy decode ({mode_label}) ===", flush=True)
        print(
            f"mesh_shape=({mesh_rows},{mesh_cols}) device_ids={device_ids if device_ids is not None else 'auto'} "
            f"kv_cache_dtype={args.kv_cache_dtype} dispatch=ETH phase={phase} batch_size={batch_size}",
            flush=True,
        )
        print(
            f"prompt_len={prompt_len} new_tokens={len(generated[0])} blocks_per_seq={blocks_per_seq}"
            + (" (fake context, no prefill)" if use_fake_context else ""),
            flush=True,
        )
        num_gen = max(1, len(generated[0]) - 1)
        if decode_s > 0:
            print(
                f"prefill_s={prefill_s:.3f} decode_tok_s={decode_s / num_gen:.4f} tok_s={num_gen / decode_s:.2f}",
                flush=True,
            )
        if step_times:
            print(f"\n--- Per-token decode latency (ms) ---", flush=True)
            if len(step_times) >= 2:
                first_ms = step_times[0]
                rest = step_times[1:]
                print(
                    f"  first token:  {first_ms:>10.1f} ms  {'(includes trace capture)' if use_trace else ''}",
                    flush=True,
                )
                print(
                    f"  subsequent:   mean={statistics.mean(rest):>8.1f}  min={min(rest):>8.1f}  max={max(rest):>8.1f}",
                    flush=True,
                )
            else:
                print(f"  single token: {step_times[0]:>10.1f} ms", flush=True)
        print("", flush=True)
        print(text, flush=True)

        # Cleanup.
        for t in kv_cache:
            ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
