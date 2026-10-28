# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# Command lists need no warm-up run: the first call of every op below happens while recording.

import sys

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

SHAPE = (1, 1, 64, 128)


@pytest.fixture
def builder(device):
    builder = ttnn.experimental.CommandListBuilder(device)
    yield builder
    if builder.is_recording:
        builder.__exit__(None, None, None)
    builder.deallocate()


@pytest.fixture
def command_lists(device):
    # Lists hold a raw device reference, so release them before the device closes.
    lists = []
    yield lists
    for command_list in lists:
        command_list.deallocate()


def allocate_input(device, shape=SHAPE):
    return ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16, ttnn.TILE_LAYOUT, device)


def write_input(torch_input, device_input):
    ttnn.copy_host_to_device_tensor(ttnn.from_torch(torch_input, layout=ttnn.TILE_LAYOUT), device_input)


def op_chain(x):
    # Intermediates stay allocated so nothing else is placed at the addresses replays write.
    relu = ttnn.relu(x)
    exp = ttnn.exp(relu)
    out = ttnn.softmax(exp, dim=-1)
    return out, [relu, exp]


def torch_op_chain(x):
    return torch.softmax(torch.exp(torch.relu(x.float())), dim=-1)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("blocking", [True, False])
def test_record_build_replay(device, builder, command_lists, blocking):
    device_input = allocate_input(device)

    with builder:
        out, intermediates = op_chain(device_input)
    assert not builder.is_recording

    command_list = builder.build(cq_id=0)
    command_lists.append(command_list)
    assert command_list.cq_id == 0

    # Replays read whatever the input buffer holds, so new data needs no update_args.
    for _ in range(3):
        torch_input = torch.randn(SHAPE, dtype=torch.bfloat16)
        write_input(torch_input, device_input)
        command_list.enqueue(blocking=blocking, check_dram_overlap=True)
        assert_with_pcc(torch_op_chain(torch_input), ttnn.to_torch(out).float(), pcc=0.99)


@skip_for_slow_dispatch()
def test_dram_overlap_check_rejects_partial_overlap(device, builder, command_lists, expect_error):
    # 64 and 32 tiles give different per-bank sizes on both Wormhole (12 banks) and Blackhole (8 banks).
    device_input = allocate_input(device, shape=(1, 1, 256, 256))
    with builder:
        out = ttnn.relu(device_input)
    command_list = builder.build()
    command_lists.append(command_list)

    # The smaller tensor lands in the range `out` just freed and fills only part of it.
    ttnn.deallocate(out)
    smaller = allocate_input(device, shape=(1, 1, 128, 256))
    with expect_error(RuntimeError, "overlaps a live allocation"):
        command_list.enqueue(check_dram_overlap=True)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_build_on_two_command_queues(device, builder, command_lists):
    device_input = allocate_input(device)
    with builder:
        out, intermediates = op_chain(device_input)
    command_lists.append(builder.build(cq_id=0))
    command_lists.append(builder.build(cq_id=1))
    # Built lists outlive the builder.
    builder.deallocate()

    for command_list, cq_id in zip(command_lists, [0, 1]):
        assert command_list.cq_id == cq_id
        torch_input = torch.randn(SHAPE, dtype=torch.bfloat16)
        write_input(torch_input, device_input)
        ttnn.synchronize_device(device)
        command_list.enqueue(blocking=True)
        assert_with_pcc(torch_op_chain(torch_input), ttnn.to_torch(out).float(), pcc=0.99)


@skip_for_slow_dispatch()
def test_command_list_keeps_device_alive(device, builder, command_lists):
    device_input = allocate_input(device)
    with builder:
        out = ttnn.relu(device_input)
    refs_before = sys.getrefcount(device)
    command_lists.append(builder.build())
    assert sys.getrefcount(device) == refs_before + 1


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"num_command_queues": 2}], indirect=True)
def test_build_defaults_to_current_command_queue(device, builder, command_lists):
    device_input = allocate_input(device)
    with builder:
        out, intermediates = op_chain(device_input)

    command_lists.append(builder.build())
    assert command_lists[-1].cq_id == 0

    with ttnn.command_queue(1):
        command_lists.append(builder.build())
    assert command_lists[-1].cq_id == 1


@skip_for_slow_dispatch()
def test_clear_and_rerecord(device, builder, command_lists, expect_error):
    device_input = allocate_input(device)
    with builder:
        first_out, first_intermediates = op_chain(device_input)

    builder.clear()
    with expect_error(RuntimeError, "Cannot build an empty CommandList"):
        builder.build()

    with builder:
        out = ttnn.relu(device_input)
    command_list = builder.build()
    command_lists.append(command_list)

    torch_input = torch.randn(SHAPE, dtype=torch.bfloat16)
    write_input(torch_input, device_input)
    command_list.enqueue()
    assert_with_pcc(torch.relu(torch_input.float()), ttnn.to_torch(out).float(), pcc=0.99)


@skip_for_slow_dispatch()
def test_legacy_op_is_rejected(device, builder, command_lists, expect_error):
    device_input = allocate_input(device)
    with builder:
        relu = ttnn.relu(device_input)
        # binary_ng still uses a legacy ProgramFactory; command lists are Metal 2.0 only.
        with expect_error(RuntimeError, "Command lists only support Metal 2.0 programs") as exc_info:
            ttnn.add(relu, relu)
        assert "cannot be recorded into a command list" in str(exc_info.value)
        assert builder.is_recording
        out = ttnn.exp(relu)

    # The rejected op left no trace: the list holds relu and exp.
    command_list = builder.build()
    command_lists.append(command_list)
    torch_input = torch.randn(SHAPE, dtype=torch.bfloat16)
    write_input(torch_input, device_input)
    command_list.enqueue()
    assert_with_pcc(torch.exp(torch.relu(torch_input.float())), ttnn.to_torch(out).float(), pcc=0.99)


@skip_for_slow_dispatch()
def test_exception_ends_recording(device, builder, expect_error):
    device_input = allocate_input(device)
    with expect_error(ValueError, "user error"):
        with builder:
            raise ValueError("user error")
    assert not builder.is_recording

    # Not recording: the op runs eagerly.
    torch_input = torch.randn(SHAPE, dtype=torch.bfloat16)
    write_input(torch_input, device_input)
    out = ttnn.relu(device_input)
    assert_with_pcc(torch.relu(torch_input.float()), ttnn.to_torch(out).float(), pcc=0.99)


@skip_for_slow_dispatch()
def test_empty_build_fails(builder, expect_error):
    with expect_error(RuntimeError, "Cannot build an empty CommandList"):
        builder.build()


@skip_for_slow_dispatch()
def test_clear_or_deallocate_while_recording_fails(builder, expect_error):
    with builder:
        with expect_error(RuntimeError, "Cannot clear a recording CommandListBuilder"):
            builder.clear()
        with expect_error(RuntimeError, "Cannot deallocate a recording CommandListBuilder"):
            builder.deallocate()


@skip_for_slow_dispatch()
def test_nested_recording_fails(builder, expect_error):
    with builder:
        with expect_error(RuntimeError, "CommandListBuilder is already recording"):
            with builder:
                pass
        assert builder.is_recording


@skip_for_slow_dispatch()
def test_one_builder_per_device(device, builder, expect_error):
    with expect_error(RuntimeError, "Only one CommandListBuilder may exist for a MeshDevice"):
        ttnn.experimental.CommandListBuilder(device)


@skip_for_slow_dispatch()
@pytest.mark.parametrize("device_params", [{"trace_region_size": 200000}], indirect=True)
def test_trace_capture_fails_while_builder_exists(device, builder, expect_error):
    with expect_error(RuntimeError, "Cannot begin trace capture while a CommandListBuilder or CommandList exists"):
        ttnn.begin_trace_capture(device, cq_id=0)
