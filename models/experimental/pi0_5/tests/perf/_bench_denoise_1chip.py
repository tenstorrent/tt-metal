# SPDX-License-Identifier: Apache-2.0
"""Single-chip denoise per-layer latency harness (production config).

Production env flags are applied by default — no need to source pi05_production.env.
Default config is L1 weights + matmul_decode MLP (fastest). Use --config to switch:

  --config l1_md   L1 weights + matmul_decode MLP  (default)
  --config l1      L1 weights, standard matmul
  --config dram    DRAM weights, standard matmul

Run:
  python _bench_denoise_1chip.py --device-id 23
  python _bench_denoise_1chip.py --device-id 23 --config dram
  python -m tracy -p -r -n denoise_l1_md _bench_denoise_1chip.py --device-id 23
"""
import argparse
import os
import statistics
import time

# Parse --config early so PI0_MD_DENOISE is set before model imports.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", choices=["dram", "l1", "l1_md"], default="l1_md")
_pre_args, _ = _pre.parse_known_args()

# Apply production defaults before any model import (env vars must be set early).
# These mirror _bench_runs/pi05_production.env; individual vars can still be
# overridden from the shell before running the script.
_PROD_DEFAULTS = {
    "PI0_EXPERT_MM_LOFI": "1",
    "PI0_ROPE_TABLES_L1": "1",
    "PI0_MM_SWEEP_V2": "1",
    "PI0_DENOISE_MM_TUNE": "1",
    "PI0_UPSTREAM_MASKS": "1",
    "QWEN_NLP_CONCAT_HEADS_HEAD_SPLIT": "1",
    "QWEN_NLP_CREATE_HEADS_HEAD_SPLIT": "1",
    "PI0_MQA_HEAD_SPLIT": "1",
    "PI0_SDPA_DENOISE_K_FORCE": "96",
    "PI0_MD_DENOISE": "1" if _pre_args.config == "l1_md" else "0",
}
for _k, _v in _PROD_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import ExpertChunkSlice

N_WARMUP = 3
N_ITER = 10


def _synthetic_weights(config):
    W, M = config.width, config.mlp_dim
    H = config.num_heads * config.head_dim
    KVH = config.num_kv_heads * config.head_dim
    w = {}
    for i in range(config.depth):
        p = f"model.layers.{i}."
        w[f"{p}self_attn.q_proj.weight"] = torch.randn(H, W) * 0.02
        w[f"{p}self_attn.k_proj.weight"] = torch.randn(KVH, W) * 0.02
        w[f"{p}self_attn.v_proj.weight"] = torch.randn(KVH, W) * 0.02
        w[f"{p}self_attn.o_proj.weight"] = torch.randn(W, H) * 0.02
        w[f"{p}mlp.gate_proj.weight"] = torch.randn(M, W) * 0.02
        w[f"{p}mlp.up_proj.weight"] = torch.randn(M, W) * 0.02
        w[f"{p}mlp.down_proj.weight"] = torch.randn(W, M) * 0.02
        w[f"{p}input_layernorm.weight"] = torch.ones(W)
        w[f"{p}post_attention_layernorm.weight"] = torch.ones(W)
        w[f"{p}input_layernorm.dense.weight"] = torch.randn(3 * W, W) * 0.02
        w[f"{p}post_attention_layernorm.dense.weight"] = torch.randn(3 * W, W) * 0.02
    return w


def _move_weights_to_l1(chunk):
    """Move EVERY persistent denoise weight to L1-interleaved at init: projection
    weights (wqkv, o_proj, gate/up/down), modulation Dense weight+bias, RMSNorm
    gammas, and RoPE cos/sin tables. Scans all ttnn.Tensor attributes on the chunk,
    each block, its attention, and its mlp (init-time => only weights/tables exist)."""
    L1 = ttnn.L1_MEMORY_CONFIG
    moved = []

    def move_obj(obj, label, skip=()):
        if obj is None:
            return
        for k, v in vars(obj).items():
            if k in skip:
                continue
            if (
                isinstance(v, ttnn.Tensor)
                and v.storage_type() == ttnn.StorageType.DEVICE
                and v.memory_config().buffer_type != ttnn.BufferType.L1
            ):
                setattr(obj, k, ttnn.to_memory_config(v, L1))
                moved.append(f"{label}.{k}")

    move_obj(chunk, "chunk")  # cos_meta / sin_meta (RoPE tables)
    for i, blk in enumerate(chunk.blocks):
        move_obj(blk, f"blk{i}")  # mod_weight, mod_bias, norm gammas
        move_obj(blk.attention, f"blk{i}.attn")  # wqkv, o_proj, cos/sin
        # If matmul_decode is active, gate/up/down are already L1-sharded (_gate_md
        # etc.); skip the unused DRAM originals so we don't double-allocate L1.
        mlp_skip = ("gate_proj", "up_proj", "down_proj") if getattr(blk.mlp, "_md_denoise", False) else ()
        move_obj(blk.mlp, f"blk{i}.mlp", skip=mlp_skip)
    return moved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=1)
    ap.add_argument("--m-pad", type=int, default=32)
    ap.add_argument("--prefix", type=int, default=512)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument(
        "--config",
        choices=["dram", "l1", "l1_md"],
        default="l1_md",
        help="dram: DRAM weights+standard matmul; l1: L1 weights+standard matmul; l1_md: L1+matmul_decode (default)",
    )
    args = ap.parse_args()
    weights_l1 = args.config != "dram"

    device = ttnn.CreateDevice(device_id=args.device_id, l1_small_size=24576, trace_region_size=134_217_728)
    config = GemmaConfig.gemma_300m()
    HIDDEN, HEAD_D = config.width, config.head_dim
    n_layers = args.layers or config.depth

    weights = _synthetic_weights(config)
    chunk = ExpertChunkSlice(
        config, weights, device, layer_range=(0, n_layers), max_seq_len=args.m_pad + args.prefix + 64
    )

    if weights_l1:
        moved = _move_weights_to_l1(chunk)
        print(
            f"[weights-l1] moved {len(moved)} denoise weights DRAM->L1: {sorted(set(m.split('.',1)[1] for m in moved))}"
        )

    def mk(shape, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.randn(*shape).bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    hidden = mk([1, 1, args.m_pad, HIDDEN])
    adarms_cond = mk([1, 1, 1, HIDDEN])
    _k_dt = (
        ttnn.bfloat16
        if os.environ.get("PI0_ROPE_FUSED_QK", "").lower() in ("1", "true", "yes", "on")
        else ttnn.bfloat8_b
    )
    prefix_kv = [
        (mk([1, 1, args.prefix, HEAD_D], dtype=_k_dt), mk([1, 1, args.prefix, HEAD_D], dtype=ttnn.bfloat8_b))
        for _ in range(n_layers)
    ]

    # Production TIER A: the 6 per-layer modulations (sa1,ta,ga,sf1,tf,gf) only depend
    # on the timestep schedule, so they're precomputed ONCE per denoise step on host and
    # reused across all layers (mirrors pipeline_1x8._precompute_block_and_final_mods).
    # The block then skips its on-device mod-Dense matmul. This is the validated path.
    W = HIDDEN
    cond = torch.randn(1, W).to(torch.bfloat16)
    precomputed_mods = []
    for i in range(n_layers):
        p = f"model.layers.{i}."
        fw = torch.cat(
            [weights[f"{p}input_layernorm.dense.weight"], weights[f"{p}post_attention_layernorm.dense.weight"]], dim=0
        ).to(torch.bfloat16)
        mod = torch.nn.functional.linear(cond, fw)  # (1, 6W)
        parts = [mod[:, k * W : (k + 1) * W] for k in range(6)]
        parts[0] = parts[0] + 1.0  # sa1
        parts[3] = parts[3] + 1.0  # sf1
        tup = tuple(
            ttnn.from_torch(
                t.unsqueeze(1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            for t in parts
        )
        precomputed_mods.append(tup)

    print(
        f"[run] config={args.config}, {n_layers} layers, M_pad={args.m_pad}, PREFIX={args.prefix}, dev={args.device_id}"
    )
    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=precomputed_mods)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=precomputed_mods)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for _ in range(N_WARMUP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    times = []
    for i in range(N_ITER):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        ms = (time.perf_counter() - t0) * 1000.0
        times.append(ms)
        print(f"  iter {i+1:2d}: {ms:.2f} ms")

    ttnn.ReadDeviceProfiler(device)
    ttnn.release_trace(device, tid)
    ttnn.CloseDevice(device)
    print("=" * 50)
    print(
        f"single-chip denoise step: avg={statistics.mean(times):.2f} ms  "
        f"min={min(times):.2f}  std={statistics.stdev(times):.3f}  ({n_layers} layers)"
    )
    print("=" * 50)


if __name__ == "__main__":
    main()
