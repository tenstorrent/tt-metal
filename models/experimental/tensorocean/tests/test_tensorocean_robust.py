# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Nothing a step computes may come from prepare() or from an earlier run.

prepare() sees step 0's inputs. Then, for other steps (same mesh, new tracer values, f and mask from another seed),
before every run the new per-step inputs are copied into the same natural DRAM buffers, both outputs are filled
with NaN, and a separate program fills all of L1 on every core with NaN; the outputs must match the float64
reference of that step. Eager runs and trace replays are checked, and two runs on the same inputs must agree
bit for bit. Sizes cover the single program, the three-program path, and the level counts that once broke it.
"""
import pytest
import torch
import ttnn
from loguru import logger

from models.experimental.tensorocean.reference import optimized_ttnn as ref
from models.experimental.tensorocean.tests.common import PCC_MIN, RMS_REL_MAX, metrics, reference_outputs
from models.experimental.tensorocean.tt import tensorocean as opt
from models.experimental.tensorocean.tt.natural_io import STEP_INPUTS, natural_host, upload_natural

NAN_FILL = r"""
#include "api/dataflow/dataflow_api.h"
void kernel_main() {
    volatile uint32_t* p = (volatile uint32_t*)get_write_ptr(0);
    const uint32_t n = get_arg_val<uint32_t>(0);
    for (uint32_t i = 0; i < n; ++i) p[i] = 0x7fc00000u;  // quiet NaN
}
"""


def _nan_fill_l1(device, nbytes=1400 * 1024):
    """A program whose one circular buffer covers most of L1 on every core, filled with NaN."""
    g = device.compute_with_storage_grid_size()
    cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    args = ttnn.RuntimeArgs()
    for x in range(g.x):
        for y in range(g.y):
            args[x][y] = [nbytes // 4]
    kernel = ttnn.KernelDescriptor(
        kernel_source=NAN_FILL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=cores,
        compile_time_args=[],
        runtime_args=args,
        config=ttnn.ReaderConfigDescriptor(),
    )
    cb = ttnn.CBDescriptor(
        total_size=nbytes,
        core_ranges=cores,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.float32, page_size=nbytes)],
    )
    prog = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[cb])
    dummy = ttnn.from_torch(
        torch.zeros(32, 32),
        dtype=ttnn.float32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    return lambda: ttnn.generic_op([dummy, dummy], prog)


@pytest.mark.parametrize("device_params", [{"l1_small_size": 32768, "trace_region_size": 64 << 20}], indirect=True)
@pytest.mark.parametrize(
    "n, levels, fused",
    [
        (100, 100, True),  # the single program
        (100, 100, False),  # the three-program path
        (8, 3, True),  # fewer levels than core rows
        (64, 37, True),  # uneven levels per core row
        (30, 120, True),  # too many levels for the single program: three programs, 13 levels per core
        (8, 119, True),  # three programs, last core row with fewer levels than a pass
        (100, 200, True),  # three programs, two passes of 16 levels
    ],
)
def test_tensorocean_robust(device, n, levels, fused):
    host0 = ref.make_inputs(n, levels, 0)
    s = opt.prepare(host0, n, levels, device, fused=fused)
    clobber = _nan_fill_l1(device)

    def new_step(seed):
        other = ref.make_inputs(n, levels, seed)
        h = dict(host0)
        for k in STEP_INPUTS:
            h[k] = other[k]
        nat = upload_natural(natural_host(h), device)
        for k in STEP_INPUTS:
            ttnn.copy(nat[k], s["nat"][k])
            ttnn.deallocate(nat[k])
        return h

    def poison():
        for o in s["outs"]:
            z = ttnn.from_torch(
                torch.full(list(o.shape), float("nan")),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.copy(z, o)
            ttnn.deallocate(z)
        clobber()

    def check(h, tag):
        got = [ttnn.to_torch(o).float() for o in s["outs"]]
        for part, t, g in zip(("even", "odd"), reference_outputs(h, n), got):
            r = metrics(t, g)
            logger.info(f"N={n} L={levels} fused={s['fused']} {tag} {part}: pcc {r['pcc']:.12f} rms {r['rms_rel']:.2e}")
            assert r["finite"] and r["pcc"] >= PCC_MIN and r["rms_rel"] <= RMS_REL_MAX
        return got

    for seed in (1, 2):
        h = new_step(seed)
        poison()
        opt.run(s)
        first = check(h, f"eager step {seed}")
    poison()
    opt.run(s)
    again = check(h, "eager repeat")
    assert all(torch.equal(a, b) for a, b in zip(first, again)), "two runs on the same inputs differ"

    tid = ttnn.begin_trace_capture(device, cq_id=0)
    opt.run(s)
    ttnn.end_trace_capture(device, tid, cq_id=0)
    try:
        for seed in (3, 4):
            h = new_step(seed)
            poison()
            for _ in range(3):
                ttnn.execute_trace(device, tid, cq_id=0, blocking=True)
            check(h, f"trace step {seed}")
    finally:
        ttnn.release_trace(device, tid)
