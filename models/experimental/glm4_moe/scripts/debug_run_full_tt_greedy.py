# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Debug: run GLM-4.7 MoE REAP (glm4_moe / Glm4MoeTT) on TT without vLLM.

Similar role to ``glm4_moe_lite/scripts/debug_run_full_tt_greedy.py``, but for the
full REAP model (GQA + MoE, paged K/V per layer).

Prerequisites:
  - Weights: HuggingFace snapshot with ``model.safetensors.index.json`` + shards.
    Use ``HF_MODEL`` / ``--model-id`` or optional ``GLM4_MOE_SNAPSHOT_DIR`` (same
    semantics as ``generator_vllm``).
  - Mesh: e.g. Galaxy TG ``--mesh-rows 8 --mesh-cols 4`` (32 devices).
  - For traced decode (``--enable-trace``), use on-device reduces so host reads are
    not used during trace capture::

        export GLM4_MOE_REDUCE_IMPL=native
        export GLM4_MOE_EP_REDUCE_DEVICE=1

  - Optional partial model (fast bring-up)::

        export GLM4_MOE_DEBUG_ALLOW_PARTIAL_LAYERS=1
        export GLM4_MOE_NUM_LAYERS=4

  - Optional CCL fabric links / topology (attention async rs/ag, MoE dispatch)::

        export GLM4_MOE_CCL_NUM_LINKS=4
        export GLM4_MOE_CCL_TOPOLOGY=ring   # or linear (default)
        # Per mesh axis: GLM4_MOE_CCL_NUM_LINKS_AXIS0 / GLM4_MOE_CCL_NUM_LINKS_AXIS1

Run from tt-metal root with ``PYTHONPATH`` including the repo root, e.g.::

    export TT_METAL_HOME=/path/to/tt-metal   # optional; also accepts TT_METAL
    cd "$TT_METAL_HOME"
    export PYTHONPATH=$(pwd)
    export GLM4_MOE_REDUCE_IMPL=native
    export GLM4_MOE_EP_REDUCE_DEVICE=1
    python models/experimental/glm4_moe/scripts/debug_run_full_tt_greedy.py \\
        --mesh-rows 8 --mesh-cols 4 --model-id cerebras/GLM-4.7-REAP-218B-A32B
"""

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
from models.experimental.glm4_moe.tt.layer_weights import _tp_axis_and_size
from models.experimental.glm4_moe.tt.model_tt import Glm4MoeTT
from models.experimental.glm4_moe.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


def _round_up(x: int, multiple: int) -> int:
    return ((int(x) + int(multiple) - 1) // int(multiple)) * int(multiple)


def _alloc_contiguous_page_table(*, batch: int, blocks_per_seq: int) -> torch.Tensor:
    """User i uses blocks [i*BPS : i*BPS + BPS)."""
    max_num_blocks = int(batch) * int(blocks_per_seq)
    return torch.arange(max_num_blocks, dtype=torch.int32).reshape(int(batch), int(blocks_per_seq))


def _parse_tt_dtype(raw: str) -> ttnn.DataType:
    raw = (raw or "").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if raw in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    if raw in {"bf4", "bfloat4_b"}:
        return ttnn.bfloat4_b
    raise ValueError(f"Unsupported --kv-cache-dtype={raw!r}")


def _tp_size_for_kv(device: object) -> int:
    if device.__class__.__name__ != "MeshDevice":
        return 1
    _, tp = _tp_axis_and_size(device)
    return max(1, int(tp))


def _alloc_gqa_kv_cache(
    *,
    device: object,
    num_key_value_heads: int,
    head_dim: int,
    num_layers: int,
    num_blocks: int,
    block_size: int,
    tt_dtype: ttnn.DataType,
) -> list[list]:
    tp = _tp_size_for_kv(device)
    n_local_kv = int(num_key_value_heads) // max(1, tp)
    host = torch.zeros(
        (int(num_blocks), n_local_kv, int(block_size), int(head_dim)),
        dtype=torch.bfloat16,
        device="cpu",
    )
    is_mesh = device.__class__.__name__ == "MeshDevice"
    mesh_mapper = ttnn.ReplicateTensorToMesh(device) if is_mesh else None
    out: list[list] = []
    for _ in range(int(num_layers)):
        tt_k = ttnn.as_tensor(
            host,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=tt_dtype,
            cache_file_name=None,
        )
        tt_v = ttnn.as_tensor(
            host,
            device=device,
            mesh_mapper=mesh_mapper,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=tt_dtype,
            cache_file_name=None,
        )
        out.append([tt_k, tt_v])
    return out


def _set_default_fabric_config(num_devices: int) -> None:
    if int(num_devices) <= 1:
        return
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(
        fabric,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )


def _resolve_snapshot(model_id: str) -> Path:
    hint = os.environ.get("GLM4_MOE_SNAPSHOT_DIR", "").strip()
    hint_path = Path(hint) if hint else None
    return Path(resolve_best_effort_snapshot_dir(model_id, hint_dir=hint_path))


def main() -> int:
    ap = argparse.ArgumentParser(description="Run GLM-4.7 REAP MoE (glm4_moe) on TT without vLLM (debug helper).")
    ap.add_argument(
        "--model-id",
        default=os.environ.get("HF_MODEL") or os.environ.get("GLM4_MOE_HF_MODEL") or "cerebras/GLM-4.7-REAP-218B-A32B",
    )
    ap.add_argument("--prompt", default="Say hello in exactly 3 words.")
    ap.add_argument("--prompt-file", default=None, metavar="PATH")
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument(
        "--cache-dir",
        default="~/.cache/ttnn/models/glm4_moe/debug_greedy",
        help="TT tensor cache (weight conversion artifacts).",
    )
    ap.add_argument("--mesh-rows", type=int, default=8)
    ap.add_argument("--mesh-cols", type=int, default=4)
    ap.add_argument("--device-ids", default="auto", help="'auto' or comma-separated ids (len = rows*cols).")
    ap.add_argument("--kv-cache-dtype", default="bf8")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--min-cache-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-batch-size", type=int, default=32, help="Passed to Glm4MoeTT.create (RoPE / config).")
    ap.add_argument(
        "--enable-trace",
        action="store_true",
        help="Use traced decode (requires on-device reduces; see module docstring).",
    )
    ap.add_argument(
        "--trace-mode",
        choices=["logits", "sampling"],
        default="logits",
        help="With --enable-trace: return logits to host (logits) or on-device greedy top-1 (sampling).",
    )
    ap.add_argument(
        "--warmup-decode-trace", action="store_true", help="Extra decode step to capture trace before timing."
    )
    ap.add_argument(
        "--simulate-context-len",
        type=int,
        default=0,
        help="Repeat prompt tokens to reach this length (0=off). Same idea as glm4_moe_lite debug script.",
    )
    ap.add_argument(
        "--input-tokens-json",
        default=None,
        metavar="PATH",
        help="JSON with 'input_ids' list; skips tokenizer.",
    )
    args = ap.parse_args()

    model_id = str(args.model_id)
    os.environ.setdefault("HF_MODEL", model_id)
    os.environ.setdefault("GLM4_MOE_HF_MODEL", model_id)

    snap = _resolve_snapshot(model_id)
    missing = find_missing_shards(snap)
    if missing:
        raise SystemExit(f"Snapshot missing {len(missing)} shards (e.g. {missing[0]}). Download weights first.")

    batch_size = int(args.batch_size)
    if batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    if args.input_tokens_json:
        p = Path(os.path.expanduser(args.input_tokens_json))
        data = json.loads(p.read_text())
        prompt_ids_single = torch.tensor([data["input_ids"]], dtype=torch.int32)
        tok = None
    else:
        text = str(args.prompt)
        if args.prompt_file:
            pf = Path(os.path.expanduser(args.prompt_file))
            text = pf.read_text(encoding="utf-8", errors="replace").strip()
        tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, use_fast=True)
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        prompt_ids_single = enc["input_ids"].to(dtype=torch.int32)

    if prompt_ids_single.ndim != 2 or prompt_ids_single.shape[0] != 1:
        raise SystemExit(f"expected prompt as [1,S], got {tuple(prompt_ids_single.shape)}")
    prompt_len = int(prompt_ids_single.shape[1])
    sim_ctx = int(args.simulate_context_len)
    if sim_ctx > 0 and sim_ctx > prompt_len:
        repeats = (sim_ctx + prompt_len - 1) // prompt_len
        prompt_ids_single = prompt_ids_single.repeat(1, repeats)[:, :sim_ctx]
        prompt_len = sim_ctx
        print(f"[debug glm4_moe] simulate-context-len expanded prompt to {prompt_len} tokens", flush=True)

    prompt_ids = prompt_ids_single.repeat(batch_size, 1)
    prompt_lens = [prompt_len] * batch_size

    mesh_rows = int(args.mesh_rows)
    mesh_cols = int(args.mesh_cols)
    n_devices = mesh_rows * mesh_cols
    raw_ids = str(args.device_ids).strip().lower()
    if raw_ids in {"auto", ""}:
        device_ids = None
    else:
        device_ids = [int(x) for x in str(args.device_ids).split(",") if x.strip()]
        if len(device_ids) != n_devices:
            raise SystemExit(f"--device-ids must have length {n_devices}, got {len(device_ids)}")

    block_size = int(args.block_size)
    max_new = int(args.max_new_tokens)
    total_tokens = prompt_len + max_new
    total_tokens = max(total_tokens, int(args.min_cache_tokens))
    # attention_tt.forward_prefill asserts seq_len % 128 == 0; model_tt prefill caps padded
    # length at max_seq_len, so capacity must be at least round_up(prompt_len, 128).
    _attn_prefill_tile = 128
    total_tokens = max(total_tokens, _round_up(prompt_len, _attn_prefill_tile))
    blocks_per_seq = max(1, _round_up(total_tokens, block_size) // block_size)
    max_seq_len = int(blocks_per_seq * block_size)

    kv_dtype = _parse_tt_dtype(str(args.kv_cache_dtype))
    cache_dir = Path(os.path.expanduser(str(args.cache_dir)))
    cache_dir.mkdir(parents=True, exist_ok=True)

    _set_default_fabric_config(n_devices)
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    dispatch_cfg = (
        ttnn.DispatchCoreConfig(axis=ttnn.DispatchCoreAxis.ROW)
        if is_galaxy
        else ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH)
    )
    open_kw: dict = {"mesh_shape": ttnn.MeshShape(mesh_rows, mesh_cols), "dispatch_core_config": dispatch_cfg}
    if device_ids is not None:
        open_kw["physical_device_ids"] = device_ids

    mesh_device = ttnn.open_mesh_device(**open_kw)
    try:
        print("[debug glm4_moe] Glm4MoeTT.create ...", flush=True)
        t0 = time.perf_counter()
        runner = Glm4MoeTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=cache_dir,
            max_seq_len=max_seq_len,
            max_batch_size=int(args.max_batch_size),
        )
        print(
            f"[debug glm4_moe] create done in {time.perf_counter() - t0:.1f}s "
            f"layers={runner.num_layers_to_run} max_seq_len={max_seq_len}",
            flush=True,
        )

        num_blocks = batch_size * blocks_per_seq
        kv_cache = _alloc_gqa_kv_cache(
            device=mesh_device,
            num_key_value_heads=int(runner.hparams.num_key_value_heads),
            head_dim=int(runner.hparams.head_dim),
            num_layers=int(runner.num_layers_to_run),
            num_blocks=num_blocks,
            block_size=block_size,
            tt_dtype=kv_dtype,
        )
        page_table = _alloc_contiguous_page_table(batch=batch_size, blocks_per_seq=blocks_per_seq)

        use_trace = bool(args.enable_trace)
        use_sampling = use_trace and str(args.trace_mode) == "sampling"
        if use_trace:
            print(
                "[debug glm4_moe] enable_trace=True — ensure GLM4_MOE_REDUCE_IMPL=native "
                "and GLM4_MOE_EP_REDUCE_DEVICE=1 to avoid host reads during trace.",
                flush=True,
            )

        print("[debug glm4_moe] prefill ...", flush=True)
        t_pf = time.perf_counter()
        logits = runner.prefill(
            tokens=prompt_ids,
            prompt_lens=prompt_lens,
            page_table=page_table,
            kv_cache=kv_cache,
            seq_pad_multiple=block_size,
        )
        prefill_s = time.perf_counter() - t_pf
        print(f"[debug glm4_moe] prefill_s={prefill_s:.3f}", flush=True)

        logits_flat = logits.reshape(batch_size, -1)
        next_ids = [int(torch.argmax(logits_flat[b]).item()) for b in range(batch_size)]
        generated: list[list[int]] = [[t] for t in next_ids]

        if args.warmup_decode_trace and use_trace:
            wpos = torch.tensor([prompt_len] * batch_size, dtype=torch.int32)
            wtok = torch.tensor([[t] for t in next_ids], dtype=torch.int32)
            sp = True if use_sampling else None
            _ = runner.decode(
                tokens=wtok,
                start_pos=wpos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=True,
                sampling_params=sp,
            )

        step_times: list[float] = []
        t_dec0 = time.perf_counter()
        cur = next_ids
        for step in range(max_new - 1):
            pos = torch.tensor([prompt_len + step] * batch_size, dtype=torch.int32)
            tok_in = torch.tensor([[t] for t in cur], dtype=torch.int32)
            t_step = time.perf_counter()
            out = runner.decode(
                tokens=tok_in,
                start_pos=pos,
                page_table=page_table,
                kv_cache=kv_cache,
                enable_trace=use_trace,
                sampling_params=True if use_sampling else None,
            )
            if use_sampling:
                ids_t = out.reshape(-1).to(torch.int32).cpu()
                cur = [int(ids_t[b].item()) for b in range(batch_size)]
            else:
                logits_flat = out.reshape(batch_size, -1)
                cur = [int(torch.argmax(logits_flat[b]).item()) for b in range(batch_size)]
            step_times.append((time.perf_counter() - t_step) * 1000)
            for b in range(batch_size):
                generated[b].append(cur[b])
        decode_s = time.perf_counter() - t_dec0

        if tok is None:
            tok = AutoTokenizer.from_pretrained(str(snap), local_files_only=True, use_fast=True)
        full = torch.cat([prompt_ids[0:1], torch.tensor([generated[0]], dtype=torch.int32)], dim=1)
        text = tok.decode(full[0].tolist(), skip_special_tokens=True)

        print("", flush=True)
        print("=== glm4_moe TT greedy (no vLLM) ===", flush=True)
        mode = "eager"
        if use_trace:
            mode = f"trace-{args.trace_mode}"
        print(
            f"model_id={model_id} mesh=({mesh_rows},{mesh_cols}) mode={mode} "
            f"kv_dtype={args.kv_cache_dtype} prompt_len={prompt_len} new_toks={len(generated[0])}",
            flush=True,
        )
        n_gen = max(1, len(generated[0]) - 1)
        print(f"prefill_s={prefill_s:.3f} decode_s={decode_s:.3f} tok/s={n_gen / max(decode_s, 1e-9):.2f}", flush=True)
        # Line parsed by run_sweep_isl_batch.py (matches glm4_moe_lite format).
        if decode_s > 0:
            print(
                f"prefill_s={prefill_s:.3f} decode_tok_s={decode_s / n_gen:.4f} tok_s={n_gen / decode_s:.2f}",
                flush=True,
            )
        else:
            print(f"prefill_s={prefill_s:.3f} decode_tok_s=0.0000 tok_s=0.00", flush=True)
        if step_times:
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
                print(f"  single decode step: {step_times[0]:>10.1f} ms", flush=True)
            if prefill_s is not None:
                ttft_ms = prefill_s * 1000.0 + step_times[0]
                print(f"ttft_ms={ttft_ms:.1f}", flush=True)
        elif max_new <= 1 and prefill_s is not None:
            print(f"ttft_ms={prefill_s * 1000.0:.1f}", flush=True)
        print("", flush=True)
        print(text, flush=True)

        for layer_pair in kv_cache:
            for t in layer_pair:
                ttnn.deallocate(t, force=False)
    finally:
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
