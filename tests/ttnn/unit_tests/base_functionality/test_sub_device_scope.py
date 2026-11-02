# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""ttnn.sub_device: operations record the sub-device they run on, and launch holds them to it.

The contract, on the example operation (the reference migration for authors). Each migrated operation tests itself
in its own directory with the checks in tests/ttnn/sub_device_testing.py.
"""

import pytest
import torch
import ttnn
from tests.ttnn.sub_device_testing import (
    check_an_unnamed_op_is_refused_on_a_split_device,
    check_runs_on_the_scoped_sub_device,
    compute_and_last_column,
    core_range,
    grid_of,
    large_input,
    program_placements,
    to_device,
)
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch


OPS = [
    ("composite_example", lambda x: ttnn.composite_example(x), lambda t: t.float()),
]


def a_program_on_the_whole_grid(device):
    """What an op that sizes itself from compute_with_storage_grid_size() builds: kernels on every core of the grid."""
    x, y = grid_of(device)
    kernel = ttnn.KernelDescriptor(
        kernel_source='#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n',
        source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
        core_ranges=core_range(0, 0, x - 1, y - 1),
        config=ttnn.WriterConfigDescriptor(),
    )
    return ttnn.ProgramDescriptor(kernels=[kernel])


@skip_for_slow_dispatch()
def test_a_program_sized_from_the_whole_grid_spans_both_sub_devices(device, expect_error):
    """The problem the contract solves: once the grid is split, an op that sizes itself from the whole grid cannot run.
    generic_op records no sub-device, so the program runs as built."""
    io = [to_device(torch.zeros(1, 1, 32, 32), device), to_device(torch.zeros(1, 1, 32, 32), device)]
    program = a_program_on_the_whole_grid(device)
    ttnn.generic_op(io, program)  # one sub-device: it runs
    with compute_and_last_column(device):
        with expect_error(RuntimeError, "Programs must be executed on a single sub-device"):
            ttnn.generic_op(io, program)
            ttnn.synchronize_device(device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("name, op, reference", OPS, ids=[name for name, _, _ in OPS])
def test_outside_a_scope_an_unnamed_op_is_refused_on_a_split_device(device, name, op, reference, expect_error):
    check_an_unnamed_op_is_refused_on_a_split_device(device, op, reference, expect_error)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("sub_device_id", [0, 1])
@pytest.mark.parametrize("name, op, reference", OPS, ids=[name for name, _, _ in OPS])
def test_ops_run_on_the_scoped_sub_device(device, sub_device_id, name, op, reference):
    check_runs_on_the_scoped_sub_device(device, sub_device_id, op, reference)


def test_an_op_that_records_no_sub_device_raises_inside_a_scope(device, expect_error):
    x = to_device(torch.randn(1, 1, 64, 128, dtype=torch.bfloat16), device)
    with ttnn.sub_device(0):
        # The multiple-return example records no sub-device: it stays the reference for an op that has not opted in.
        with expect_error(RuntimeError, "does not record the sub-device it runs on"):
            ttnn.composite_example_multiple_return(x, return_output1=True, return_output2=True)
    # Outside the scope it runs as before.
    ttnn.composite_example_multiple_return(x, return_output1=True, return_output2=True)


@skip_for_slow_dispatch()
def test_a_tensor_created_from_the_host_inside_a_scope_is_converted_on_the_host(device):
    """With no manager loaded, from_torch with a mesh mapper converts on the device, with a tilize that records no
    sub-device and a typecast. Inside a scope it converts on the host, as it does under a manager: tensors created from
    the host are not checked."""
    torch_input = torch.randn(1, 1, 64, 64, dtype=torch.float32)
    with ttnn.sub_device(0):
        x, placements = program_placements(
            lambda: ttnn.from_torch(
                torch_input,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(device),
            )
        )
    assert placements == []
    assert torch.equal(ttnn.to_torch(ttnn.get_device_tensors(x)[0]), torch_input.to(torch.bfloat16))


@skip_for_slow_dispatch()
def test_the_scope_is_part_of_the_program_cache_key(device):
    """composite_example runs the example op twice: one program, built once per sub-device."""
    torch_input = large_input()
    x = to_device(torch_input, device)

    def example_on(sub_device_id):
        with ttnn.sub_device(sub_device_id):
            return ttnn.composite_example(x)

    with compute_and_last_column(device):
        entries = device.num_program_cache_entries()
        for _ in range(2):
            for sub_device_id in (0, 1):
                y, placements = program_placements(lambda: example_on(sub_device_id))
                assert placements == [sub_device_id, sub_device_id]
                assert_with_pcc(torch_input.float(), ttnn.to_torch(y).float(), 0.999)
        assert device.num_program_cache_entries() == entries + 2


@skip_for_slow_dispatch()
def test_scopes_nest_and_the_innermost_applies(device):
    x = to_device(large_input(), device)
    with compute_and_last_column(device):
        with ttnn.sub_device(0):
            _, outer = program_placements(lambda: ttnn.composite_example(x))
            with ttnn.sub_device(ttnn.SubDeviceId(1)):
                _, inner = program_placements(lambda: ttnn.composite_example(x))
            _, outer_again = program_placements(lambda: ttnn.composite_example(x))
    assert (outer, inner, outer_again) == ([0, 0], [1, 1], [0, 0])


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_a_trace_captured_in_a_scope_replays_on_its_sub_device(device, expect_error):
    """A trace is captured under the assignment it was warmed up with: the scope is part of each program's key, so a
    warm-up on another sub-device does not serve it."""
    torch_input = large_input()
    x = to_device(torch_input, device)
    with compute_and_last_column(device):
        with ttnn.sub_device(1):
            ttnn.composite_example(x)  # compile
            trace_id = ttnn.begin_trace_capture(device, cq_id=0)
            y = ttnn.composite_example(x)
            ttnn.end_trace_capture(device, trace_id, cq_id=0)
        ttnn.execute_trace(device, trace_id, cq_id=0, blocking=True)
        assert_with_pcc(torch_input.float(), ttnn.to_torch(y).float(), 0.999)
        ttnn.release_trace(device, trace_id)

        other = to_device(torch.randn(1, 1, 64, 64, dtype=torch.bfloat16), device)
        with ttnn.sub_device(0):
            ttnn.composite_example(other)  # warmed up on sub-device 0 only
        device.set_program_cache_misses_allowed(False)
        try:
            with ttnn.sub_device(1):
                with expect_error(RuntimeError, "cache misses are forbidden"):
                    ttnn.composite_example(other)
        finally:
            device.set_program_cache_misses_allowed(True)
