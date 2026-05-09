# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""End-to-end GLM-4.7-Flash generation via the hybrid implementation.

This is the hybrid's counterpart to the agentic's debug_run_full_tt_greedy.py.
It uses the hybrid's HybridGlm4Runner with the same agentic backend, but goes
through the hybrid's module framework:
  1. Load model config via HybridGlm4Runner
  2. Allocate compressed KVPE cache (BF8 by default)
  3. Initialize MoE runtime
  4. Load weights via agentic's convert_decoder_layer_weights
  5. Run prefill + decode greedy generation
  6. Report latency and throughput KPIs

Usage:
  # Single chip (N150)
  python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
    --mesh-cols 1 --prompt "Hello world" --max-new-tokens 32

  # T3K (8 chips)
  python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
    --mesh-cols 8 --prompt "Hello world" --max-new-tokens 64

  # T3K with all optimizations
  GLM4_MOE_LITE_DRAM_SHARDED_WEIGHTS=1 GLM4_MOE_LITE_FUSED_KV_BRANCH=1 \\
  GLM4_MOE_LITE_FUSED_MOE=1 GLM4_MOE_LITE_TP=1 \\
  python3 models/demos/glm4_moe_lite_hybrid/scripts/run_hybrid_e2e.py \
    --mesh-cols 8 --prompt "Hello world" --max-new-tokens 64 --kv-cache-dtype bf8
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

import ttnn
from models.demos.glm4_moe_lite.tt.decoder_layer_tt import (
    prepare_decode_rope_and_positions_tt,
    run_decoder_layer_decode_one_step_update_cache_tt,
)
from models.demos.glm4_moe_lite.tt.layer0_tt import _alloc_contiguous_page_table, _round_up, make_rope_tensors
from models.demos.glm4_moe_lite.tt.layer_weights import convert_decoder_layer_weights
from models.demos.glm4_moe_lite.tt.tt_embedding import convert_embedding_weight_to_tt, run_tt_embedding
from models.demos.glm4_moe_lite.tt.weights import (
    find_missing_shards,
    load_glm_lazy_state_dict,
    resolve_best_effort_snapshot_dir,
)
from models.demos.glm4_moe_lite_hybrid.core.config import Glm4MoeLiteHParams
from models.demos.glm4_moe_lite_hybrid.core.runtime_config import Glm4RuntimeConfig
from models.demos.glm4_moe_lite_hybrid.modules.kvpe_cache import CompressedKVPECache, CompressedKVPECacheConfig
from models.demos.glm4_moe_lite_hybrid.modules.moe import HybridGlm4MoERuntimeManager


def _set_default_fabric_config(num_devices: int) -> None:
    if num_devices <= 1:
        return
    is_galaxy = ttnn.cluster.get_cluster_type() == ttnn.cluster.ClusterType.GALAXY
    fabric = ttnn.FabricConfig.FABRIC_1D_RING if is_galaxy else ttnn.FabricConfig.FABRIC_1D
    ttnn.set_fabric_config(fabric, ttnn.FabricReliabilityMode.STRICT_INIT)


def _parse_tt_dtype(raw: str) -> ttnn.DataType:
    raw = (raw or "").strip().lower()
    if raw in {"bf16", "bfloat16"}:
        return ttnn.bfloat16
    if raw in {"bf8", "bfloat8_b"}:
        return ttnn.bfloat8_b
    raise ValueError(f"Unsupported dtype: {raw!r}")


def _tt_to_torch_device0(t: ttnn.Tensor, device) -> torch.Tensor:
    if device.__class__.__name__ == "MeshDevice":
        return ttnn.to_torch(ttnn.get_device_tensors(t)[0])
    return ttnn.to_torch(t)


def main() -> int:
    ap = argparse.ArgumentParser(description="GLM-4.7-Flash end-to-end via hybrid implementation.")
    ap.add_argument("--model-id", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--prompt", default="Explain quantum computing in simple terms.")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--mesh-cols", type=int, default=1)
    ap.add_argument("--kv-cache-dtype", default="bf8")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--device-ids", default="auto")
    args = ap.parse_args()

    mesh_cols = int(args.mesh_cols)
    block_size = int(args.block_size)
    max_new_tokens = int(args.max_new_tokens)
    kv_dtype = _parse_tt_dtype(args.kv_cache_dtype)

    # ── Step 1: Resolve snapshot and tokenize ──
    print("=" * 72)
    print("  GLM-4.7-Flash End-to-End: Hybrid Implementation")
    print("=" * 72)

    snap = Path(resolve_best_effort_snapshot_dir(args.model_id))
    missing = find_missing_shards(snap)
    if missing:
        raise SystemExit(f"Snapshot missing {len(missing)} shards")

    tok = AutoTokenizer.from_pretrained(snap, local_files_only=True, use_fast=True)
    enc = tok(args.prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc["input_ids"].to(dtype=torch.int32)
    prompt_len = int(prompt_ids.shape[1])

    print(f"  Prompt: {args.prompt!r}")
    print(f"  Prompt tokens: {prompt_len}")
    print(f"  Max new tokens: {max_new_tokens}")
    print(f"  Mesh: (1, {mesh_cols})")
    print(f"  KV cache dtype: {args.kv_cache_dtype}")

    # ── Step 2: Load config via hybrid ──
    hf_cfg = json.loads((snap / "config.json").read_text())
    hparams = Glm4MoeLiteHParams.from_hf_config(SimpleNamespace(**hf_cfg))
    hparams.validate()

    hidden = int(hparams.hidden_size)
    num_layers = int(hparams.num_hidden_layers)
    kvpe_dim = int(hparams.kv_lora_rank) + int(hparams.qk_rope_head_dim)

    print(f"  Hidden: {hidden}, Layers: {num_layers}, KVPE dim: {kvpe_dim}")
    print(f"  Experts: {hparams.n_routed_experts}, per token: {hparams.num_experts_per_tok}")

    # ── Step 3: Open device ──
    os.environ["GLM4_MOE_LITE_ENABLE_MOE"] = "1"
    if mesh_cols <= 1:
        os.environ.setdefault("GLM4_MOE_LITE_EVICT_WEIGHTS", "1")

    _set_default_fabric_config(mesh_cols)

    device_ids_raw = args.device_ids.strip().lower()
    open_kwargs = {
        "mesh_shape": ttnn.MeshShape(1, mesh_cols),
        "dispatch_core_config": ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.ETH),
    }
    if device_ids_raw not in {"auto", ""}:
        open_kwargs["physical_device_ids"] = [int(x) for x in device_ids_raw.split(",")]

    device = ttnn.open_mesh_device(**open_kwargs)
    cfg = Glm4RuntimeConfig.from_env(device=device)

    grid = device.compute_with_storage_grid_size()
    print(f"  Compute grid: {grid.x}x{grid.y}")

    try:
        # ── Step 4: Allocate KV cache via hybrid CompressedKVPECache ──
        total_len = max(prompt_len + max_new_tokens, 128)
        blocks_per_seq = max(1, _round_up(total_len, block_size) // block_size)
        max_num_blocks = 1 * blocks_per_seq

        print(f"\n[1/5] Allocating compressed KVPE cache...", flush=True)
        t0 = time.perf_counter()
        cache_config = CompressedKVPECacheConfig(
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            num_layers=num_layers,
            dtype=kv_dtype,
        )
        kvpe_cache = CompressedKVPECache(hparams, cache_config)
        kvpe_cache.to_device(device, batch_size=1)
        cache_ms = (time.perf_counter() - t0) * 1000
        cache_mb = max_num_blocks * block_size * kvpe_dim * (1 if kv_dtype == ttnn.bfloat8_b else 2) / 1024 / 1024
        print(f"  Cache: {num_layers} layers x [{max_num_blocks},{block_size},{kvpe_dim}] @ {args.kv_cache_dtype}")
        print(f"  Total cache: {cache_mb * num_layers:.1f} MB, allocated in {cache_ms:.0f} ms")

        # Build page table
        page_table_host = _alloc_contiguous_page_table(batch=1, blocks_per_seq=blocks_per_seq)

        # ── Step 5: Load weights ──
        print(f"\n[2/5] Loading model weights...", flush=True)
        state = load_glm_lazy_state_dict(str(snap))

        # MoE runtime via hybrid manager
        moe_mgr = HybridGlm4MoERuntimeManager()
        moe_runtime = moe_mgr.get_or_create(device, hparams)

        # Embedding
        embed_w = convert_embedding_weight_to_tt(
            device=device,
            embed_weight=state["model.embed_tokens.weight"],
        )

        # LM head
        lm_head_torch = state["lm_head.weight"]
        lm_head_w = ttnn.as_tensor(
            lm_head_torch.transpose(-2, -1).unsqueeze(0).unsqueeze(0).contiguous(),
            device=device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(device) if device.__class__.__name__ == "MeshDevice" else None,
        )

        # Final norm
        from models.common.rmsnorm import RMSNorm

        final_norm = RMSNorm(
            device=device,
            dim=hidden,
            eps=hparams.rms_norm_eps,
            state_dict=state,
            state_dict_prefix="model.",
            weight_key="norm",
            weight_cache_path=None,
            weight_dtype=ttnn.bfloat16,
            is_distributed=False,
        )

        # RoPE
        rope_dim = int(hparams.qk_rope_head_dim)
        rope = make_rope_tensors(
            device=device,
            seq_len=total_len + 128,
            rope_dim=rope_dim,
            rope_theta=float(hparams.rope_theta),
        )

        # Layer weights (lazy: load one at a time for single device)
        t_preload = time.perf_counter()
        layer_weights = [None] * num_layers
        evict = os.environ.get("GLM4_MOE_LITE_EVICT_WEIGHTS", "") == "1"
        for li in range(num_layers):
            is_dense = li < int(hparams.first_k_dense_replace)
            layer_weights[li] = convert_decoder_layer_weights(
                device=device,
                state=state,
                layer_idx=li,
                hparams=hparams,
                enable_moe=not is_dense,
                skip_fused_kv_branch=True,
            )
        preload_s = time.perf_counter() - t_preload
        print(f"  Loaded {num_layers} layers in {preload_s:.1f}s")

        # ── Step 6: Prefill ──
        print(f"\n[3/5] Prefill ({prompt_len} tokens)...", flush=True)
        t0 = time.perf_counter()

        # Iterative decode-style prefill (one token at a time)
        logits = None
        for t_idx in range(prompt_len):
            tok_in = prompt_ids[:, t_idx : t_idx + 1].contiguous()
            pos = torch.tensor([t_idx], dtype=torch.int32)

            x = run_tt_embedding(device=device, token_ids=tok_in, tt_weight=embed_w)
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, (1, 1, 1, hidden))

            tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
                device=device,
                rope=rope,
                positions=pos,
            )

            for li in range(num_layers):
                w = layer_weights[li]
                x = run_decoder_layer_decode_one_step_update_cache_tt(
                    device=device,
                    x_embed_tok=x,
                    tt_positions=tt_positions,
                    page_table_tt=kvpe_cache.page_table,
                    kvpe_cache=kvpe_cache.get_cache(li),
                    cos_batch=cos_batch,
                    sin_batch=sin_batch,
                    trans_matrix=rope["trans_matrix"],
                    cos_decode=None,
                    sin_decode=None,
                    trans_decode=None,
                    rope_sharded_cfg=None,
                    w=w,
                    hparams=hparams,
                    moe_runtime=moe_runtime if w.moe is not None else None,
                    profile=None,
                    use_decode_rope=False,
                )

            x = final_norm(x, mode="decode")
            logits = ttnn.linear(x, lm_head_w)
            ttnn.deallocate(x, force=False)

        prefill_s = time.perf_counter() - t0
        print(f"  Prefill: {prefill_s:.3f}s ({prompt_len / prefill_s:.1f} tok/s)")

        # ── Step 7: Decode ──
        print(f"\n[4/5] Decode ({max_new_tokens} tokens)...", flush=True)

        vocab = int(hparams.vocab_size)
        logits_torch = _tt_to_torch_device0(logits, device).float().reshape(-1)[:vocab]
        token_in = int(torch.argmax(logits_torch).item())
        generated = [token_in]

        step_times = []
        t_decode0 = time.perf_counter()

        for step in range(max_new_tokens - 1):
            pos = torch.tensor([prompt_len + step], dtype=torch.int32)
            tok_in = torch.tensor([[token_in]], dtype=torch.int32)

            t_step = time.perf_counter()

            x = run_tt_embedding(device=device, token_ids=tok_in, tt_weight=embed_w)
            if x.layout != ttnn.TILE_LAYOUT:
                x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
            x = ttnn.reshape(x, (1, 1, 1, hidden))

            tt_positions, cos_batch, sin_batch = prepare_decode_rope_and_positions_tt(
                device=device,
                rope=rope,
                positions=pos,
            )

            for li in range(num_layers):
                w = layer_weights[li]
                x = run_decoder_layer_decode_one_step_update_cache_tt(
                    device=device,
                    x_embed_tok=x,
                    tt_positions=tt_positions,
                    page_table_tt=kvpe_cache.page_table,
                    kvpe_cache=kvpe_cache.get_cache(li),
                    cos_batch=cos_batch,
                    sin_batch=sin_batch,
                    trans_matrix=rope["trans_matrix"],
                    cos_decode=None,
                    sin_decode=None,
                    trans_decode=None,
                    rope_sharded_cfg=None,
                    w=w,
                    hparams=hparams,
                    moe_runtime=moe_runtime if w.moe is not None else None,
                    profile=None,
                    use_decode_rope=False,
                )

            x = final_norm(x, mode="decode")
            logits = ttnn.linear(x, lm_head_w)
            ttnn.deallocate(x, force=False)

            logits_torch = _tt_to_torch_device0(logits, device).float().reshape(-1)[:vocab]
            token_in = int(torch.argmax(logits_torch).item())
            generated.append(token_in)

            step_ms = (time.perf_counter() - t_step) * 1000
            step_times.append(step_ms)

        decode_s = time.perf_counter() - t_decode0

        # ── Step 8: Report ──
        full_ids = torch.cat([prompt_ids, torch.tensor([generated], dtype=torch.int32)], dim=1)
        text = tok.decode(full_ids[0].tolist(), skip_special_tokens=True)

        num_gen = max(1, len(generated) - 1)

        print(f"\n[5/5] Results")
        print(f"\n{'='*72}")
        print(f"  GLM-4.7-Flash Hybrid End-to-End Results")
        print(f"{'='*72}")
        print(f"  Implementation:  hybrid (TTNNModule framework + agentic backend)")
        print(f"  Device:          (1, {mesh_cols}) mesh")
        print(f"  KV cache:        compressed KVPE @ {args.kv_cache_dtype} ({cache_mb * num_layers:.0f} MB)")
        print(f"  MoE runtime:     {hparams.n_routed_experts} experts, {hparams.num_experts_per_tok}/tok, sparse")
        print(f"  Weight eviction: {'on' if evict else 'off'}")
        print()
        print(f"  prompt_len={prompt_len}  new_tokens={len(generated)}  blocks_per_seq={blocks_per_seq}")
        if decode_s > 0:
            print(f"  prefill_s={prefill_s:.3f}  decode_tok_s={decode_s / num_gen:.4f}  tok_s={num_gen / decode_s:.2f}")

        if step_times:
            print(f"\n  --- Per-token decode latency (ms) ---")
            if len(step_times) >= 2:
                first_ms = step_times[0]
                rest = step_times[1:]
                print(f"    first token:  {first_ms:>10.1f} ms")
                print(
                    f"    subsequent:   mean={statistics.mean(rest):>8.1f}  min={min(rest):>8.1f}  max={max(rest):>8.1f}"
                )
            else:
                print(f"    single token: {step_times[0]:>10.1f} ms")

        print(f"\n  --- Generated text ---")
        print(f"  {text}")
        print()

    finally:
        ttnn.close_mesh_device(device)
        print("Device closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
