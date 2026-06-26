# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Simplified single-chip denoise step, L1 weights + matmul_decode MLP only, driven by M.

One config (the production fast path): L1-resident weights + PI0_MD_DENOISE matmul_decode.
2 layers, 2-cam prefix=768. Parametrized by M (the suffix/action token count):

  M=32  ->  32x32 tile, matmul_decode active. Reference: ~0.33 ms device-kernel
            (full 2-layer step, device 9).
  M=16  ->  16x32 tile. matmul_decode MLP currently FAILS: its width-sharded
            activation needs a tile-aligned (32-row) shard, so _md_shard_a raises
            TT_FATAL "Physical shard shape (16, 512) must be tile {32, 32} sized!".

Wall-clock is printed for a quick sanity check; the authoritative 0.33 ms figure is
the tracy device-kernel total. To capture it:

  python -m tracy -p -r -n md_m32 -m pytest \
      models/experimental/pi0_5/tests/perf/test_denoise_md_by_m.py -k "M32" --device-id 9
  python _parse_ops_perop.py            # device-kernel total ~= 327 us

  python -m tracy -p -r -n md_m16 -m pytest \
      models/experimental/pi0_5/tests/perf/test_denoise_md_by_m.py -k "M16" --device-id 9
"""

# Env must be set before any model import (mirrors _bench_denoise_1chip.py prod defaults).
import os as _os

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
    "PI0_MD_DENOISE": "1",  # L1 + matmul_decode: the only config this test runs
}
for _k, _v in _PROD_DEFAULTS.items():
    _os.environ.setdefault(_k, _v)

import statistics
import time

import pytest
import torch
import ttnn

from models.experimental.pi0_5.common.configs import GemmaConfig
from models.experimental.pi0_5.tt.tt_bh_glx.expert_slice import ExpertChunkSlice

N_WARMUP = 3
N_ITER = 10
N_LAYERS = 2
PREFIX = 768  # 2-cam: 2x256 image + 256 language


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
    """Move every persistent denoise weight DRAM->L1 (skip the unused DRAM gate/up/down
    originals when matmul_decode already holds L1-sharded copies)."""
    L1 = ttnn.L1_MEMORY_CONFIG

    def move_obj(obj, skip=()):
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

    move_obj(chunk)
    for blk in chunk.blocks:
        move_obj(blk)
        move_obj(blk.attention)
        mlp_skip = ("gate_proj", "up_proj", "down_proj") if getattr(blk.mlp, "_md_denoise", False) else ()
        move_obj(blk.mlp, skip=mlp_skip)


def _build_precomputed_mods(weights, hidden_size, device):
    W = hidden_size
    cond = torch.randn(1, W).bfloat16()
    mods = []
    for i in range(N_LAYERS):
        p = f"model.layers.{i}."
        fw = torch.cat(
            [weights[f"{p}input_layernorm.dense.weight"], weights[f"{p}post_attention_layernorm.dense.weight"]], dim=0
        ).bfloat16()
        mod = torch.nn.functional.linear(cond, fw)
        parts = [mod[:, k * W : (k + 1) * W] for k in range(6)]
        parts[0] = parts[0] + 1.0  # sa1
        parts[3] = parts[3] + 1.0  # sf1
        mods.append(
            tuple(
                ttnn.from_torch(
                    t.unsqueeze(1).contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=device,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                for t in parts
            )
        )
    return mods


def _run_denoise(device, M):
    config = GemmaConfig.gemma_300m()
    HIDDEN, HEAD_D = config.width, config.head_dim
    torch.manual_seed(42)
    weights = _synthetic_weights(config)

    chunk = ExpertChunkSlice(config, weights, device, layer_range=(0, N_LAYERS), max_seq_len=M + PREFIX + 64)
    _move_weights_to_l1(chunk)

    def mk(shape, dtype=ttnn.bfloat16):
        return ttnn.from_torch(torch.randn(*shape).bfloat16(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)

    # M drives the logical row count only; the tile stays the default 32x32 so the
    # path through attention/layernorm is identical to production. For M=16 the MLP
    # keeps M=16 (tile=16 granularity) and matmul_decode is reached genuinely.
    hidden = mk([1, 1, M, HIDDEN])
    adarms_cond = mk([1, 1, 1, HIDDEN])
    prefix_kv = [
        (mk([1, 1, PREFIX, HEAD_D], dtype=ttnn.bfloat8_b), mk([1, 1, PREFIX, HEAD_D], dtype=ttnn.bfloat8_b))
        for _ in range(N_LAYERS)
    ]
    mods = _build_precomputed_mods(weights, HIDDEN, device)

    # Warm up + trace capture (this is where M=16 raises in the matmul_decode MLP).
    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=mods)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=mods)
    ttnn.end_trace_capture(device, tid, cq_id=0)

    for _ in range(N_WARMUP):
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)

    times_ms = []
    for _ in range(N_ITER):
        t0 = time.perf_counter()
        ttnn.execute_trace(device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(device)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    ttnn.ReadDeviceProfiler(device)  # flush device-kernel data for tracy
    ttnn.release_trace(device, tid)

    avg, std = statistics.mean(times_ms), statistics.stdev(times_ms)
    print(f"\n  M={M}  {N_LAYERS} layers  prefix={PREFIX}: wall-clock avg={avg:.3f} ms std={std:.3f}")
    return avg


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576, "trace_region_size": 134_217_728}], indirect=True)
@pytest.mark.parametrize("M", [32, 16], ids=["M32", "M16"])
def test_denoise_md(device, M):
    """L1 + matmul_decode denoise step. M=32 runs (~0.33 ms device-kernel); M=16 is
    expected to fail in matmul_decode (width-sharded activation needs 32-row shards)."""
    avg_ms = _run_denoise(device, M)
    assert avg_ms < 2.0, f"wall-clock {avg_ms:.3f} ms unexpectedly high"
