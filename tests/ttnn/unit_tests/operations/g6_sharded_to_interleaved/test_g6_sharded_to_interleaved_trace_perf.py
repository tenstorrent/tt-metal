# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Unprofiled host time per op of blocking trace replays, per plan knob setting. DO NOT DELETE.

The program's cores end at different times and a second writer RISC adds a kernel binary per core
range, so profiled kernel time alone cannot judge `num_writer_riscs` (lesson RB-053). Prints
`TRACE_REPLAY <case> riscs=<n> rotate=<b> <ns>/op`; correctness is checked before and after.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path

import pytest
import ttnn

import ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved_program_descriptor as pd
from ttnn.operations.g6_sharded_to_interleaved import SUPPORTED, g6_sharded_to_interleaved

_spec = importlib.util.spec_from_file_location(
    "g6_s2i_acceptance", Path(__file__).with_name("test_g6_sharded_to_interleaved.py")
)
_acceptance = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acceptance)

TARGET = {
    "dtype": [ttnn.bfloat16, ttnn.bfloat8_b],
    "layout": [ttnn.TILE_LAYOUT],
    "in_scheme": ["width_sharded", "height_sharded"],
    "buffer": ["dram", "l1"],
}
OPS_PER_TRACE = 20
REPLAYS = 50


P1_TO_DRAM = {**_acceptance.P1_PANOPTIC, "buffer": "dram"}
SCENARIOS = {"p2": _acceptance.P2_FLUX, "p1": _acceptance.P1_PANOPTIC, "p1_to_dram": P1_TO_DRAM}


def _replay_ns_per_op(device, run_op):
    trace_id = ttnn.begin_trace_capture(device, cq_id=0)
    for _ in range(OPS_PER_TRACE):
        run_op()
    ttnn.end_trace_capture(device, trace_id, cq_id=0)
    ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
    start = time.perf_counter()
    for _ in range(REPLAYS):
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    per_op_ns = (time.perf_counter() - start) * 1e9 / (REPLAYS * OPS_PER_TRACE)
    ttnn.release_trace(device, trace_id)
    return per_op_ns


@pytest.mark.use_module_device({"trace_region_size": 1 << 20})
def test_warmup(device):
    """The first trace replay of a module reads several us slow; absorb it here."""
    case = _acceptance._Case(device, _acceptance.P2_FLUX, salt=29)
    output = ttnn.allocate_tensor_on_device(
        case.tt_input.shape, case.tt_input.dtype, ttnn.TILE_LAYOUT, device, case.output_mc
    )
    descriptor = pd.create_program_descriptor(case.tt_input, output, case.output_mc)
    ttnn.generic_op([case.tt_input, output], descriptor)
    _replay_ns_per_op(device, lambda: ttnn.generic_op([case.tt_input, output], descriptor))
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


@pytest.mark.use_module_device({"trace_region_size": 1 << 20})
@pytest.mark.parametrize("rotate", [False, True, None], ids=["inorder", "rotated", "auto"])
@pytest.mark.parametrize("num_writer_riscs", [1, 2])
@pytest.mark.parametrize("scenario", list(SCENARIOS), ids=list(SCENARIOS))
def test_trace_replay_host_time(device, monkeypatch, scenario, num_writer_riscs, rotate):
    for axis, values in TARGET.items():
        monkeypatch.setitem(SUPPORTED, axis, values)
    monkeypatch.setattr(pd, "NUM_WRITER_RISCS", num_writer_riscs)
    monkeypatch.setattr(pd, "ROTATE_WRITE_ORDER", rotate)
    label = scenario
    scenario = SCENARIOS[scenario]

    case = _acceptance._Case(device, scenario, salt=30)
    output = g6_sharded_to_interleaved(case.tt_input, case.output_mc)
    case.check_output(output)

    # Replays rewrite the same output: the allocation stays outside the capture.
    descriptor = pd.create_program_descriptor(case.tt_input, output, case.output_mc)
    per_op_ns = _replay_ns_per_op(device, lambda: ttnn.generic_op([case.tt_input, output], descriptor))

    print(f"\nTRACE_REPLAY {label} riscs={num_writer_riscs} rotate={rotate} {per_op_ns:.0f} ns/op")
    case.check_output(output, "output after replays")
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)


_EMPTY_KERNEL = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'


@pytest.mark.use_module_device({"trace_region_size": 1 << 20})
@pytest.mark.parametrize("args", ["none", "per_core", "common"])
@pytest.mark.parametrize("scenario", ["p2", "p1"])
def test_dispatch_floor(device, scenario, args):
    """Replay cost of an empty kernel on the input's shard grid: which runtime-arg form costs per core."""
    case = _acceptance._Case(device, SCENARIOS[scenario], salt=31)
    grid = case.tt_input.memory_config().shard_spec.grid
    rt_args = ttnn.RuntimeArgs()
    if args == "per_core":
        for core in ttnn.corerange_to_cores(grid, None, True):
            rt_args[core.x][core.y] = [core.x, core.y, 0, 0, 0, 0, 0, 0]
    kernel = ttnn.KernelDescriptor(
        kernel_source=_EMPTY_KERNEL,
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=grid,
        runtime_args=rt_args,
        common_runtime_args=[0] * 8 if args == "common" else [],
        config=ttnn.WriterConfigDescriptor(),
    )
    descriptor = ttnn.ProgramDescriptor(
        kernels=[kernel], semaphores=[], cbs=[ttnn.cb_descriptor_from_sharded_tensor(0, case.tt_input)]
    )
    output = ttnn.allocate_tensor_on_device(
        case.tt_input.shape, case.tt_input.dtype, ttnn.TILE_LAYOUT, device, case.output_mc
    )
    ttnn.generic_op([case.tt_input, output], descriptor)
    per_op_ns = _replay_ns_per_op(device, lambda: ttnn.generic_op([case.tt_input, output], descriptor))
    print(f"\nDISPATCH_FLOOR {scenario} args={args} {per_op_ns:.0f} ns/op")
    ttnn.deallocate(output)
    ttnn.deallocate(case.tt_input)
