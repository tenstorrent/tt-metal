# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

# TTNN command-list parameters on the reduction ops: record with CmdListParam, update_args, replay.

import pytest
import torch
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc
from tt_py_test_utils_common.utility_functions import skip_for_slow_dispatch

ACCUMULATION_SHAPE = (1, 1, 64, 128)
EMA_SHAPE = (1, 2, 32, 64)
ALPHA = 0.8


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


def to_device(device, torch_tensor):
    return ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)


def torch_ema(x, alpha):
    # The recurrence starts from zero, so out_0 = (1 - alpha) * x_0.
    out = torch.empty_like(x, dtype=torch.float32)
    prev = torch.zeros_like(x[..., 0], dtype=torch.float32)
    for t in range(x.shape[-1]):
        out[..., t] = alpha * prev + (1 - alpha) * x[..., t]
        prev = out[..., t]
    return out


# cumsum and cumprod share one factory but name their input keyword "input" and "input_tensor".
@skip_for_slow_dispatch()
@pytest.mark.parametrize(
    "op, keyword, torch_op",
    [(ttnn.cumsum, "input", torch.cumsum), (ttnn.cumprod, "input_tensor", torch.cumprod)],
    ids=["cumsum", "cumprod"],
)
def test_update_args_moves_accumulation_to_the_new_input(device, builder, command_lists, op, keyword, torch_op):
    if device.arch() == ttnn.device.Arch.QUASAR and op is ttnn.cumprod:
        pytest.skip("Quasar cumprod permutes this input, which cannot be rebound as a command-list parameter")
    # Values near 1 keep the running product in range.
    torch_a = torch.rand(ACCUMULATION_SHAPE, dtype=torch.bfloat16) + 0.5
    torch_b = torch.rand(ACCUMULATION_SHAPE, dtype=torch.bfloat16) + 0.5
    input_a = to_device(device, torch_a)
    input_b = to_device(device, torch_b)

    x = ttnn.experimental.CmdListParam(input_a, "x")
    with builder:
        out = op(**{keyword: x}, dim=-1)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.enqueue(blocking=True)
    assert_with_pcc(torch_op(torch_a.float(), dim=-1), ttnn.to_torch(out).float(), 0.999)

    command_list.update_args({x: input_b})
    command_list.enqueue(blocking=True)
    assert_with_pcc(torch_op(torch_b.float(), dim=-1), ttnn.to_torch(out).float(), 0.999)


@skip_for_slow_dispatch()
def test_update_args_moves_ema_input_and_preallocated_output(device, builder, command_lists):
    torch_a = torch.rand(EMA_SHAPE, dtype=torch.bfloat16)
    torch_b = torch.rand(EMA_SHAPE, dtype=torch.bfloat16)
    input_a = to_device(device, torch_a)
    input_b = to_device(device, torch_b)
    out_a = to_device(device, torch.zeros(EMA_SHAPE, dtype=torch.bfloat16))
    out_b = to_device(device, torch.zeros(EMA_SHAPE, dtype=torch.bfloat16))

    x = ttnn.experimental.CmdListParam(input_a, "x")
    out = ttnn.experimental.CmdListParam(out_a, "out")
    with builder:
        ttnn.ema(input_tensor=x, alpha=ALPHA, out=out)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.enqueue(blocking=True)
    assert_with_pcc(torch_ema(torch_a.float(), ALPHA), ttnn.to_torch(out_a).float(), 0.999)

    command_list.update_args({x: input_b, out: out_b})
    command_list.enqueue(blocking=True)
    assert_with_pcc(torch_ema(torch_b.float(), ALPHA), ttnn.to_torch(out_b).float(), 0.999)


# alpha is a compile-time argument, so it is not a command-list parameter. The op records it with its default.
@skip_for_slow_dispatch()
def test_ema_alpha_is_not_a_command_list_parameter(device, builder, command_lists):
    torch_a = torch.rand(EMA_SHAPE, dtype=torch.bfloat16)
    x = ttnn.experimental.CmdListParam(to_device(device, torch_a), "x")
    alpha = ttnn.experimental.CmdListParam(ALPHA, "alpha")
    with builder:
        out = ttnn.ema(input_tensor=x, alpha=alpha)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.enqueue(blocking=True)
    assert_with_pcc(torch_ema(torch_a.float(), ALPHA), ttnn.to_torch(out).float(), 0.999)


# The ops below are compared against an eager call on the new input rather than a torch golden: what is under test is
# that update_args points the recorded program at the new tensor.
def distinct_rows(shape, seed):
    # Every row is a permutation of 0..W-1 (exact in bfloat16 up to 256), so argmax and topk have no ties.
    generator = torch.Generator().manual_seed(seed)
    rows = torch.stack([torch.randperm(shape[-1], generator=generator) for _ in range(shape[-2])])
    return rows.reshape(shape).to(torch.bfloat16)


@skip_for_slow_dispatch()
def test_update_args_moves_argmax_to_the_new_input(device, builder, command_lists):
    shape = (1, 1, 32, 64)
    input_a = ttnn.from_torch(distinct_rows(shape, 0), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    input_b = ttnn.from_torch(distinct_rows(shape, 1), dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    expected_b = ttnn.to_torch(ttnn.argmax(input_b, dim=-1))

    x = ttnn.experimental.CmdListParam(input_a, "x")
    with builder:
        out = ttnn.argmax(input_tensor=x, dim=-1)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.update_args({x: input_b})
    command_list.enqueue(blocking=True)
    assert torch.equal(ttnn.to_torch(out), expected_b)


@skip_for_slow_dispatch()
def test_update_args_moves_topk_to_the_new_input(device, builder, command_lists):
    shape = (1, 1, 32, 64)
    input_a = to_device(device, distinct_rows(shape, 0))
    input_b = to_device(device, distinct_rows(shape, 1))
    expected_values, expected_indices = (ttnn.to_torch(t) for t in ttnn.topk(input_b, k=32, dim=-1))

    x = ttnn.experimental.CmdListParam(input_a, "x")
    with builder:
        values, indices = ttnn.topk(input_tensor=x, k=32, dim=-1)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.update_args({x: input_b})
    command_list.enqueue(blocking=True)
    assert torch.equal(ttnn.to_torch(values), expected_values)
    assert torch.equal(ttnn.to_torch(indices), expected_indices)


# topk picks the single-core factory (Metal 2.0, bindable) or the multi-core one (still a ProgramDescriptor factory)
# from the input width. A recording that reaches the multi-core factory fails, with or without a CmdListParam.
@skip_for_slow_dispatch()
def test_topk_recording_is_rejected_when_the_multi_core_factory_is_selected(device, builder, expect_error):
    if device.arch() == ttnn.device.Arch.QUASAR:
        pytest.skip("Quasar has no multi-core topk factory")
    device_input = to_device(device, distinct_rows((1, 1, 32, 8192), 0).float() % 256)
    with builder:
        with expect_error(RuntimeError, "Command lists only support Metal 2.0 programs"):
            ttnn.topk(input_tensor=ttnn.experimental.CmdListParam(device_input, "x"), k=32, dim=-1)


@skip_for_slow_dispatch()
def test_update_args_moves_sampling_to_the_new_input(device, builder, command_lists):
    shape = (1, 1, 32, 64)
    seed = 2024
    k = ttnn.from_torch(
        torch.tensor([10, 15, 20, 25, 30] * 6 + [10, 20]), device=device, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
    )
    p = ttnn.from_torch(
        torch.tensor([0.0, 0.3, 0.5, 0.7, 0.9] * 6 + [0.1, 0.8]),
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    temp = ttnn.from_torch(torch.ones(32), device=device, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT)
    indices = ttnn.from_torch(
        torch.arange(shape[-1], dtype=torch.int32).expand(shape),
        device=device,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
    )
    values_a = to_device(device, torch.randn(shape, dtype=torch.bfloat16))
    values_b = to_device(device, torch.randn(shape, dtype=torch.bfloat16))
    # The same seed gives the same samples (see operations/reduce/test_sampling.py).
    expected_b = ttnn.to_torch(ttnn.sampling(values_b, indices, k=k, p=p, temp=temp, seed=seed))

    x = ttnn.experimental.CmdListParam(values_a, "x")
    with builder:
        out = ttnn.sampling(input_values_tensor=x, input_indices_tensor=indices, k=k, p=p, temp=temp, seed=seed)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.update_args({x: values_b})
    command_list.enqueue(blocking=True)
    assert torch.equal(ttnn.to_torch(out), expected_b)


@skip_for_slow_dispatch()
def test_update_args_moves_moe_to_the_new_input(device, builder, command_lists):
    N, C, H, W, k, E, e = 1, 1, 32, 64, 32, 8, 2
    expert_mask = torch.zeros([N, C, 1, W], dtype=torch.bfloat16)
    expert_mask[:, :, :, E:] = float("-inf")
    topk_mask = torch.zeros([N, C, 1, k], dtype=torch.bfloat16)
    topk_mask[:, :, :, e:] = float("-inf")

    def make_input(seed):
        torch.manual_seed(seed)
        t = torch.randn([N, C, H, W], dtype=torch.bfloat16)
        t[:, :, :, E:] = 0  # padded input
        return to_device(device, t)

    input_a, input_b = make_input(0), make_input(1)
    ttnn_expert_mask, ttnn_topk_mask = to_device(device, expert_mask), to_device(device, topk_mask)
    expected_b = ttnn.to_torch(ttnn.moe(input_b, ttnn_expert_mask, ttnn_topk_mask, k))

    x = ttnn.experimental.CmdListParam(input_a, "x")
    with builder:
        out = ttnn.moe(input_tensor=x, expert_mask_tensor=ttnn_expert_mask, topk_mask_tensor=ttnn_topk_mask, k=k)
    command_list = builder.build()
    command_lists.append(command_list)

    command_list.update_args({x: input_b})
    command_list.enqueue(blocking=True)
    assert torch.equal(ttnn.to_torch(out), expected_b)
