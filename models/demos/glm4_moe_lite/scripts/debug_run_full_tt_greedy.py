# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.glm4_moe_lite.tt.layer0_tt import _alloc_contiguous_page_table, _round_up
from models.demos.glm4_moe_lite.tt.model_tt import Glm4MoeLiteDenseOnlyTT
from models.demos.glm4_moe_lite.tt.weights import find_missing_shards, resolve_best_effort_snapshot_dir


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
    ap.add_argument("--prompt", default="Say hello in exactly 3 words.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument(
        "--cache-dir",
        default="~/.cache/ttnn/models/glm4_moe_lite/vllm",
        help="TT weight cache dir (default reuses the vLLM cache to avoid reconverting all layers).",
    )
    ap.add_argument("--mesh-cols", type=int, default=1, help="Use mesh shape (1,mesh_cols).")
    ap.add_argument(
        "--device-ids",
        default="auto",
        help="Comma-separated physical device ids (length=mesh-cols) or 'auto' to let TTNN choose.",
    )
    ap.add_argument("--kv-cache-dtype", default="bf16", help="bf16 (correctness) or bf8 (memory/perf).")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--min-cache-tokens", type=int, default=128, help="Allocate at least this many tokens in KV cache.")
    ap.add_argument(
        "--phase",
        choices=["prefill", "decode", "both"],
        default="both",
        help="Which phase to profile: prefill (real flash_mla_prefill), decode, or both.",
    )
    args = ap.parse_args()

    model_id = str(args.model_id)
    snap = Path(resolve_best_effort_snapshot_dir(model_id))
    missing = find_missing_shards(snap)
    if missing:
        raise SystemExit(
            f"Snapshot missing {len(missing)} shards; run ensure_glm47_weights.sh first (example: {missing[0]})"
        )

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
    enc = tok(str(args.prompt), return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(dtype=torch.int32)
    if prompt_ids.ndim != 2 or int(prompt_ids.shape[0]) != 1:
        raise ValueError(f"expected prompt input_ids [1,S], got {tuple(prompt_ids.shape)}")
    prompt_len = int(prompt_ids.shape[1])

    mesh_cols = int(args.mesh_cols)
    if mesh_cols <= 0:
        raise ValueError("--mesh-cols must be > 0")
    device_ids_raw = str(args.device_ids).strip().lower()
    device_ids: list[int] | None
    if device_ids_raw in {"auto", ""}:
        device_ids = None
    else:
        device_ids = [int(x) for x in str(args.device_ids).split(",") if x.strip() != ""]
        if len(device_ids) != mesh_cols:
            raise ValueError(f"--device-ids must have exactly mesh-cols entries (got {len(device_ids)} vs {mesh_cols})")

    # Ensure MoE is enabled (matches production intent).
    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"

    kv_cache_dtype = _parse_tt_dtype(str(args.kv_cache_dtype))
    block_size = int(args.block_size)
    max_new_tokens = int(args.max_new_tokens)

    total_len = prompt_len + max_new_tokens
    total_len = max(total_len, int(args.min_cache_tokens))
    blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)

    # Open a mesh device for consistency with vLLM (even if mesh-cols=1).
    _set_default_fabric_config(mesh_cols)
    open_kwargs = {
        "mesh_shape": ttnn.MeshShape(1, mesh_cols),
        "dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    }
    if device_ids is not None:
        open_kwargs["physical_device_ids"] = device_ids
    mesh_device = ttnn.open_mesh_device(**open_kwargs)
    try:
        runner = Glm4MoeLiteDenseOnlyTT.create(
            device=mesh_device,
            snapshot_dir=snap,
            cache_dir=Path(os.path.expanduser(str(args.cache_dir))),
            max_seq_len=int(blocks_per_seq * block_size),
        )

        kvpe_dim = int(runner.hparams.kv_lora_rank + runner.hparams.qk_rope_head_dim)
        kv_cache = [
            _alloc_paged_kvpe_cache_from_cpu(
                device=mesh_device,
                max_num_blocks=int(1 * blocks_per_seq),
                block_size=block_size,
                kvpe_dim=kvpe_dim,
                tt_dtype=kv_cache_dtype,
            )
            for _ in range(int(runner.num_layers_to_run))
        ]
        page_table = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)

        # Pre-load all layer weights before the decode loop.  Lazy loading
        # inside decode() can deadlock because ttnn.as_tensor (host→device DMA)
        # contends with in-flight device compute ops dispatched earlier in the
        # same decode call (embedding, reshape, etc.).
        t_preload = time.perf_counter()
        for li in range(runner.num_layers_to_run):
            runner._ensure_layer_weights(li)
        print(
            f"Pre-loaded {runner.num_layers_to_run} layer(s) in {time.perf_counter()-t_preload:.1f}s",
            flush=True,
        )

        phase = str(args.phase)

        # -- Prefill phase --
        if phase in ("prefill", "both"):
            t0 = time.perf_counter()
            logits = runner.prefill(
                tokens=prompt_ids,
                prompt_lens=[prompt_len],
                page_table=page_table,
                kv_cache=kv_cache,
                seq_pad_multiple=block_size,
            )
            prefill_s = time.perf_counter() - t0
            print(f"\n=== Prefill (real flash_mla_prefill) ===", flush=True)
            print(f"prompt_len={prompt_len} prefill_s={prefill_s:.3f}", flush=True)
            if phase == "prefill":
                for t in kv_cache:
                    ttnn.deallocate(t)
                ttnn.close_mesh_device(mesh_device)
                return 0
        else:
            t0 = time.perf_counter()
            logits = None
            for t in range(prompt_len):
                tok_in = prompt_ids[:, t : t + 1].contiguous()
                pos = torch.tensor([t], dtype=torch.int32)
                logits = runner.decode(tokens=tok_in, start_pos=pos, page_table=page_table, kv_cache=kv_cache)
            assert logits is not None
            prefill_s = time.perf_counter() - t0

        # -- Decode phase --
        generated: list[int] = []
        token_in = int(torch.argmax(logits.reshape(-1)).item())
        generated.append(token_in)

        t_decode0 = time.perf_counter()
        for step in range(max_new_tokens - 1):
            pos = torch.tensor([prompt_len + step], dtype=torch.int32)
            tok_in = torch.tensor([[token_in]], dtype=torch.int32)
            logits = runner.decode(tokens=tok_in, start_pos=pos, page_table=page_table, kv_cache=kv_cache)
            token_in = int(torch.argmax(logits.reshape(-1)).item())
            generated.append(token_in)
        decode_s = time.perf_counter() - t_decode0

        full_ids = torch.cat([prompt_ids, torch.tensor([generated], dtype=torch.int32)], dim=1)
        text = tok.decode(full_ids[0].tolist(), skip_special_tokens=True)
        print("")
        print("=== TT greedy decode (direct) ===", flush=True)
        print(
            f"mesh_shape=(1,{mesh_cols}) device_ids={device_ids if device_ids is not None else 'auto'} "
            f"kv_cache_dtype={args.kv_cache_dtype} dispatch=ETH phase={phase}",
            flush=True,
        )
        print(f"prompt_len={prompt_len} new_tokens={len(generated)} blocks_per_seq={blocks_per_seq}", flush=True)
        if decode_s > 0:
            print(
                f"prefill_s={prefill_s:.3f} decode_tok_s={decode_s/ max(1,len(generated)-1):.4f} tok_s={(len(generated)-1)/decode_s:.2f}",
                flush=True,
            )
        print("")
        print(text, flush=True)

        # Cleanup.
        for t in kv_cache:
            ttnn.deallocate(t)
    finally:
        ttnn.close_mesh_device(mesh_device)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
