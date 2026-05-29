# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sweep-tuned matmul presets for the Mistral-4 prefill path (seq_len=32, Mt=1).

Numbers came from ``tests/test_prefill_matmul_sweep.py`` on Blackhole P150.
Each preset captures the fastest PCC-passing config the sweep found.

All shapes have M=32 (Mt=1) so per_core_M=1 always; subblocks cap at h=1.
"""

from __future__ import annotations

import ttnn

# Reuse the dataclass + helpers from the vision config — they're not actually
# vision-specific, just shared infrastructure.
from models.experimental.mistral_small_4_119b.tt.vision_matmul_config import (
    MatmulShape,
    VisionMatmulPreset,
    create_memory_configs,
    create_program_config,
    vision_linear as prefill_linear,  # alias for clarity at call sites
)

TILE = 32

# (M, K, N) for the unique prefill matmul shapes that have a sweep winner.
Q_A_PROJ_SHAPE = (32, 4096, 1024)
Q_B_PROJ_SHAPE = (32, 1024, 4096)
KV_A_PROJ_SHAPE = (32, 4096, 320)
KV_B_PROJ_SHAPE = (32, 256, 6144)
O_PROJ_SHAPE = (32, 4096, 4096)
GATE_ROUTER_SHAPE = (32, 4096, 128)
ROUTING_PROJ_SHAPE = (32, 128, 128)
LM_HEAD_SHAPE = (32, 4096, 131072)


def _build_preset(
    mesh_device: ttnn.MeshDevice,
    shape_tuple: tuple,
    layout: str,
    grid: tuple,
    in0_block_w: int,
) -> VisionMatmulPreset:
    """Generic preset builder. M, K, N from shape_tuple. Layout is e.g. 'l1/dram/ws'."""
    shape = MatmulShape(*shape_tuple)
    dev = mesh_device.compute_with_storage_grid_size()
    gx, gy = min(dev.x, grid[0]), min(dev.y, grid[1])
    cores = gx * gy
    nt = shape.N // TILE + (1 if shape.N % TILE else 0)
    per_core_N = max(1, nt // cores)
    cfg = {
        "family": "1D",
        "layout": layout,
        "grid": (gx, gy),
        "in0_block_w": in0_block_w,
        "per_core_M": max(1, shape.M // TILE),
        "per_core_N": per_core_N,
    }
    in0_mem, in1_mem, out_mem = create_memory_configs(shape, cfg)
    return VisionMatmulPreset(
        shape=shape,
        program_config=create_program_config(cfg),
        in0_memory_config=in0_mem,
        in1_memory_config=in1_mem,
        out_memory_config=out_mem,
    )


def build_q_a_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D l1/dram/ws on 4×4 grid, in0_block_w=16 → 13.2 µs / 20.3 TFLOPs.
    4.3× speedup over default (57 µs / 4.7 TFLOPs)."""
    return _build_preset(mesh_device, Q_A_PROJ_SHAPE, "l1/dram/ws", (4, 4), 16)


def build_q_b_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D dram/dram/ws on 8×8 grid, in0_block_w=2 → 20.1 µs / 13.4 TFLOPs."""
    return _build_preset(mesh_device, Q_B_PROJ_SHAPE, "dram/dram/ws", (8, 8), 2)


def build_o_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D l1/dram/dram on 8×4 grid, in0_block_w=8 → 32.3 µs / 33.3 TFLOPs.
    2.3× speedup over default (75 µs / 14.4 TFLOPs). DRAM output (not WS) — sweep
    found writing direct to DRAM was faster than via WS + sharded_to_interleaved."""
    return _build_preset(mesh_device, O_PROJ_SHAPE, "l1/dram/dram", (8, 4), 8)


def build_kv_a_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D l1/dram/ws on 1×10 grid, in0_block_w=16 → 11.7 µs / 7.2 TFLOPs.
    Grid forced by Nt=10 (= N=320/32); other choices have too few cores."""
    return _build_preset(mesh_device, KV_A_PROJ_SHAPE, "l1/dram/ws", (1, 10), 16)


def build_kv_b_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D dram/dram/ws on 8×4 grid, in0_block_w=1 → 9.4 µs / 10.7 TFLOPs.
    Kt=8 only divides {1,2,4,8}; w=1 is the inner-K loop unrolled."""
    return _build_preset(mesh_device, KV_B_PROJ_SHAPE, "dram/dram/ws", (8, 4), 1)


def build_gate_router_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D ws/dram/ws on 4×1 grid, in0_block_w=16 → 7.6 µs / 4.4 TFLOPs.
    Nt=4 caps the grid at 4 cores. ws in0 puts each shard's Kt/4=32 tiles in L1."""
    return _build_preset(mesh_device, GATE_ROUTER_SHAPE, "ws/dram/ws", (4, 1), 16)


def build_routing_proj_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D l1/dram/ws on 1×4 grid, in0_block_w=4 → 1.2 µs / 0.9 TFLOPs.
    Tiny matmul (Kt=4, Nt=4); mostly launch overhead."""
    return _build_preset(mesh_device, ROUTING_PROJ_SHAPE, "l1/dram/ws", (1, 4), 4)


def build_lm_head_preset(mesh_device: ttnn.MeshDevice) -> VisionMatmulPreset:
    """Sweep winner: 1D dram/dram/ws on 8×8 grid, in0_block_w=1 → 1017 µs / 33.8 TFLOPs.

    **4.2× speedup** over the default ttnn.linear (4,301 µs at 8 TFLOPs).
    in0_block_w=1 is counter-intuitive — at the extreme N=131072 (Nt=4096),
    smaller K-blocks keep the kernel issuing fast small computes while the
    256 MB BFP4 weight streams from DRAM in parallel."""
    return _build_preset(mesh_device, LM_HEAD_SHAPE, "dram/dram/ws", (8, 8), 1)


class PrefillMatmulConfigs:
    """Bundle of sweep-tuned prefill presets."""

    def __init__(self, mesh_device: ttnn.MeshDevice):
        self.q_a_proj = build_q_a_proj_preset(mesh_device)
        self.q_b_proj = build_q_b_proj_preset(mesh_device)
        self.kv_a_proj = build_kv_a_proj_preset(mesh_device)
        self.kv_b_proj = build_kv_b_proj_preset(mesh_device)
        self.o_proj = build_o_proj_preset(mesh_device)
        self.gate_router = build_gate_router_preset(mesh_device)
        self.routing_proj = build_routing_proj_preset(mesh_device)
        self.lm_head = build_lm_head_preset(mesh_device)
