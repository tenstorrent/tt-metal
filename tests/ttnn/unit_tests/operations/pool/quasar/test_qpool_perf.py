# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Quasar pool PERF harness for craq-sim — warmup + N measured iterations of ONE pool config,
controlled by the CONFIG block below. Run via the sibling run_qpool_perf.sh, which enables the
sim's per-dispatch perf trace and prints the average pool kernel duration in SIM CLOCKS.

How it measures: the craq-sim per-dispatch trace (TTSIM_PERF_TRACE_PER_DISPATCH) emits one TSV
row per program dispatch with per-engine instruction/stall counters and a `clocks` duration
column (added on craq-sim branch wransom/qsr-csr-timeout-count). This test labels each phase by
setting the TTSIM_PERF_TRACE_NODEID env var, which the in-process sim re-reads at every kernel
launch — so warmup rows and each measured iteration's rows are tagged in the TSV. The report
script then averages the measured iterations.

WHAT THE NUMBERS MEAN: sim clocks from a functional simulator — NOT silicon performance. Use
them only for RELATIVE A/B on the same sim build (e.g. pool compute num_threads 1 vs 2 vs 4),
where a real work split shows up as fewer clocks / lower per-engine stalls. Absolute
cycle counts and any cross-sim-build comparison are meaningless.

The first (warmup) iteration absorbs JIT compilation and program-cache population; measured
iterations replay the cached program, mirroring the steady state.
"""

import os

import pytest
import torch

import ttnn

# =============================== CONFIG — edit me ===============================
OP = "max"  # "max" | "avg"
BATCH = 1
IN_H, IN_W, CHANNELS = 16, 8, 64  # input spatial dims + channels
# ^ SIM WARNING: keep N*H*W <= 128 sticks — bigger shapes hit the open craq-sim DFB bug.
KERNEL = (3, 3)
STRIDE = (2, 2)
PADDING = (1, 1)
CORES = 0  # height-shard core count; 0 = grid-adaptive max, 1 = single cluster

WARMUP_ITERS = 1  # absorb JIT + program-cache population (rows labeled "warmup")
MEASURED_ITERS = 3  # averaged by the report (rows labeled "iter0", "iter1", ...)
# =================================================================================

SIM_MAX_STICKS = 128


def _label(phase):
    """Tag subsequent sim perf-trace rows; the sim re-reads this env at kernel launch."""
    os.environ["TTSIM_PERF_TRACE_NODEID"] = phase


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
def test_qpool_perf(mesh_device):
    device = mesh_device
    is_max = OP == "max"
    kernel, stride, padding = list(KERNEL), list(STRIDE), list(PADDING)

    tensor_height = BATCH * IN_H * IN_W
    assert tensor_height % 32 == 0 and CHANNELS % 32 == 0, "N*H*W and C must be multiples of 32"
    if os.environ.get("TT_METAL_SIMULATOR") and tensor_height > SIM_MAX_STICKS:
        pytest.fail(
            f"CONFIG hits the open craq-sim DFB bug: N*H*W={tensor_height} sticks > {SIM_MAX_STICKS} "
            f"(would stall or corrupt in the sim). Shrink the shape."
        )

    torch.manual_seed(0)
    x_nhwc = torch.rand((BATCH, IN_H, IN_W, CHANNELS)).to(torch.bfloat16)

    grid = device.compute_with_storage_grid_size()
    height_tiles = tensor_height // 32
    if CORES:
        assert height_tiles % CORES == 0, f"CORES={CORES} must divide height tiles ({height_tiles})"
        num_cores = CORES
    else:
        num_cores = max(c for c in range(1, grid.x * grid.y + 1) if height_tiles % c == 0)
    shard_height = (height_tiles // num_cores) * 32
    mem_config = ttnn.create_sharded_memory_config(
        shape=(1, 1, shard_height, CHANNELS),
        core_grid=ttnn.num_cores_to_corerangeset(num_cores, grid, True),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )

    _label("setup")
    x = ttnn.from_torch(x_nhwc.reshape(1, 1, tensor_height, CHANNELS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = x.to(device, mem_config)

    def run_once():
        if is_max:
            out = ttnn.experimental.quasar.max_pool2d(
                input_tensor=x,
                batch_size=BATCH,
                input_h=IN_H,
                input_w=IN_W,
                channels=CHANNELS,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=[1, 1],
            )
        else:
            out = ttnn.experimental.quasar.avg_pool2d(
                input_tensor=x,
                batch_size=BATCH,
                input_h=IN_H,
                input_w=IN_W,
                channels=CHANNELS,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                output_layout=ttnn.TILE_LAYOUT,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ttnn.init_device_compute_kernel_config(
                    device.arch(), math_fidelity=ttnn.MathFidelity.LoFi
                ),
            )
        ttnn.synchronize_device(device)
        out.deallocate()

    print(
        f"\nQPOOL-PERF: op={OP} in={BATCH}x{IN_H}x{IN_W}x{CHANNELS} k={kernel} s={stride} "
        f"p={padding} cores={num_cores} shard={shard_height}x{CHANNELS} "
        f"warmup={WARMUP_ITERS} measured={MEASURED_ITERS}"
    )
    for _ in range(WARMUP_ITERS):
        _label("warmup")
        run_once()
    for i in range(MEASURED_ITERS):
        _label(f"iter{i}")
        run_once()
    _label("teardown")
