# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool SPAN sweep — measures span(T) = prologue + marginal * T (Debin's decomposition).

Runs the same pool config at several workload sizes T (tiles per cluster = output sticks x
in_ntiles_c) in ONE device session, labeling each phase in the craq-sim per-dispatch trace so
qpool_span_report.py can fit the line per config and, across two runs (e.g. num_threads=4 vs 1),
report marginal gain (asymptotic), span gain at each measured T (real), and the crossover T*.

CORES is pinned to 1 so the program envelope IS the per-cluster kernel envelope (the emulator
protocol's median-over-clusters degenerates to the single cluster). Shapes are chosen so output
sticks divide 2*num_threads (the factory TT_FATAL) and halo volume stays under the craq-sim
stall threshold.

SIM CAVEAT: halo+pool quiesce as one dispatch, so the fitted marginal is the combined halo+pool
per-stick cost. Fine for relative A/B; on the emulator, per-kernel zones separate them.

Run via run_qpool.sh span (single run) or run_qpool.sh span-ab (threads A/B with rebuilds).
"""

import os
import sys

import pytest
import torch

import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_qpool_debug import _build_input

# =============================== CONFIG — edit me ===============================
CHANNELS = 64  # keep in_ntiles_c <= 4 (nblocks==1) until the wide-C rework lands
KERNEL = (3, 3)
STRIDE = (2, 2)
PADDING = (1, 1)
# T ladder: input (H, W) per point. Output sticks must divide 2*num_threads (factory TT_FATAL)
# and keep N*H*W*C*2B under the craq-sim halo stall threshold (~24KB safe).
T_LADDER = [(8, 4), (8, 8), (16, 8)]  # -> 8 / 16 / 32 output sticks
WARMUP_ITERS = 1  # per ladder point (absorbs JIT + program cache for that shape)
MEASURED_ITERS = 3  # per ladder point
# =================================================================================


def _label(phase):
    os.environ["TTSIM_PERF_TRACE_NODEID"] = phase


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_span(mesh_device):
    device = mesh_device
    kernel, stride, padding = list(KERNEL), list(STRIDE), list(PADDING)
    in_ntiles_c = (CHANNELS + 31) // 32

    grid = device.compute_with_storage_grid_size()
    core_grid = ttnn.num_cores_to_corerangeset(1, grid, True)  # CORES pinned to 1

    for in_h, in_w in T_LADDER:
        tensor_height = in_h * in_w
        assert tensor_height % 32 == 0, f"{in_h}x{in_w}: N*H*W must be a multiple of 32"
        out_h = (in_h - kernel[0] + 2 * padding[0]) // stride[0] + 1
        out_w = (in_w - kernel[1] + 2 * padding[1]) // stride[1] + 1
        out_sticks = out_h * out_w
        t_tiles = out_sticks * in_ntiles_c

        mem_config = ttnn.create_sharded_memory_config(
            shape=(1, 1, tensor_height, CHANNELS),
            core_grid=core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        _label(f"setup_t{t_tiles}")
        x_nhwc = _build_input("random", 1, in_h, in_w, CHANNELS, 0, 0).to(torch.bfloat16)
        x = ttnn.from_torch(x_nhwc.reshape(1, 1, tensor_height, CHANNELS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        x = x.to(device, mem_config)

        def run_once():
            out = ttnn.experimental.quasar.max_pool2d(
                input_tensor=x,
                batch_size=1,
                input_h=in_h,
                input_w=in_w,
                channels=CHANNELS,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=[1, 1],
            )
            ttnn.synchronize_device(device)
            out.deallocate()

        print(
            f"\nQPOOL-SPAN: T={t_tiles} tiles ({out_sticks} sticks x {in_ntiles_c} ctiles) in={in_h}x{in_w}x{CHANNELS}",
            flush=True,
        )
        for _ in range(WARMUP_ITERS):
            _label(f"warmup_t{t_tiles}")
            run_once()
        for i in range(MEASURED_ITERS):
            _label(f"t{t_tiles}_i{i}")
            run_once()
        _label(f"teardown_t{t_tiles}")
        x.deallocate()
