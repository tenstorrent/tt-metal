# SPDX-FileCopyrightText: 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full denoise forward timing benchmark — M=32 (32×32 tiles) vs M=16 (16×32 tiny tiles).

Production config: L1 weights + matmul_decode MLP (PI0_MD_DENOISE=1), 2-cam prefix=768.

M=32 reference: ~147.70 µs/layer (device 9, 2-cam, from _bench_denoise_1chip.py).
M=16 tiny tile: same forward, MLP pads M=16→32 internally for matmul_decode.

Run:
  pytest test_denoise_tiny_tile_bench.py -v -s --device-id 9
"""

# MUST set env before any model import — mirrors _bench_denoise_1chip.py prod defaults.
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
    "PI0_MD_DENOISE": "1",
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
PREFIX = 768  # 2-cam: 2×256 image + 256 language

# M=32, 2-cam reference (device 9, _bench_denoise_1chip.py --config l1_md --prefix 768)
_REF_US_PER_LAYER = 147.70
# Allow 2.5× slack for chip-to-chip variance; test is a timing sanity check, not a hard SLA.
_SLACK = 2.5


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

    move_obj(chunk, "chunk")
    for i, blk in enumerate(chunk.blocks):
        move_obj(blk, f"blk{i}")
        move_obj(blk.attention, f"blk{i}.attn")
        mlp_skip = ("gate_proj", "up_proj", "down_proj") if getattr(blk.mlp, "_md_denoise", False) else ()
        move_obj(blk.mlp, f"blk{i}.mlp", skip=mlp_skip)
    print(f"\n  [l1] moved {len(moved)}: {sorted(set(n.split('.',1)[1] for n in moved))}")
    return moved


def _build_precomputed_mods(weights, n_layers, hidden_size, device):
    W = hidden_size
    cond = torch.randn(1, W).bfloat16()
    mods = []
    for i in range(n_layers):
        p = f"model.layers.{i}."
        fw = torch.cat(
            [weights[f"{p}input_layernorm.dense.weight"], weights[f"{p}post_attention_layernorm.dense.weight"]], dim=0
        ).bfloat16()
        mod = torch.nn.functional.linear(cond, fw)
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
        mods.append(tup)
    return mods


def _run_denoise_bench(device, m_pad: int, m_tile: int) -> float:
    """Run N_LAYERS denoise forward with trace. Returns mean per-layer µs."""
    config = GemmaConfig.gemma_300m()
    HIDDEN, HEAD_D = config.width, config.head_dim
    torch.manual_seed(42)
    weights = _synthetic_weights(config)

    chunk = ExpertChunkSlice(
        config,
        weights,
        device,
        layer_range=(0, N_LAYERS),
        max_seq_len=m_pad + PREFIX + 64,
    )
    _move_weights_to_l1(chunk)
    for i, blk in enumerate(chunk.blocks):
        md = getattr(blk.mlp, "_md_denoise", False)
        has_gate_md = hasattr(blk.mlp, "_gate_md")
        print(f"  blk{i}.mlp: _md_denoise={md}  has_gate_md={has_gate_md}")

    def mk(shape, dtype=ttnn.bfloat16, tile=None):
        kwargs = dict(dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
        if tile is not None:
            kwargs["tile"] = tile
        return ttnn.from_torch(torch.randn(*shape).bfloat16(), **kwargs)

    hidden = mk([1, 1, m_pad, HIDDEN], tile=ttnn.Tile((m_tile, 32)))
    adarms_cond = mk([1, 1, 1, HIDDEN])
    prefix_kv = [
        (mk([1, 1, PREFIX, HEAD_D], dtype=ttnn.bfloat8_b), mk([1, 1, PREFIX, HEAD_D], dtype=ttnn.bfloat8_b))
        for _ in range(N_LAYERS)
    ]
    precomputed_mods = _build_precomputed_mods(weights, N_LAYERS, HIDDEN, device)

    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=precomputed_mods)
    ttnn.synchronize_device(device)

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    chunk.forward(hidden, adarms_cond, prefix_kv, precomputed_mods=precomputed_mods)
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

    ttnn.release_trace(device, tid)

    avg_us = statistics.mean(times_ms) * 1000.0 / N_LAYERS
    std_us = statistics.stdev(times_ms) * 1000.0 / N_LAYERS if len(times_ms) > 1 else 0.0
    print(
        f"\n  m_pad={m_pad}  tile=[{m_tile},32]  prefix={PREFIX}  "
        f"per-layer: {avg_us:.2f} µs  std={std_us:.2f} µs  "
        f"(ref={_REF_US_PER_LAYER:.2f} µs for M=32)"
    )
    return avg_us


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_md32_timing(device):
    """M=32, Tile([32,32]): production config. Per-layer ≤ 2.5× reference (~147.70 µs)."""
    per_layer_us = _run_denoise_bench(device, m_pad=32, m_tile=32)
    assert per_layer_us < _REF_US_PER_LAYER * _SLACK, (
        f"per-layer {per_layer_us:.2f} µs exceeds {_SLACK}× reference " f"({_REF_US_PER_LAYER * _SLACK:.0f} µs)"
    )


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_denoise_md16_tiny_tile(device):
    """M=16, Tile([16,32]): MLP pads M=16→32 for matmul_decode, then slices output."""
    per_layer_us = _run_denoise_bench(device, m_pad=16, m_tile=16)
    assert per_layer_us < _REF_US_PER_LAYER * _SLACK, (
        f"per-layer {per_layer_us:.2f} µs exceeds {_SLACK}× reference " f"({_REF_US_PER_LAYER * _SLACK:.0f} µs)"
    )
