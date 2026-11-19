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
    # The recurrence starts from zero, so out_0 = (1 - alpha) * x_0. The op's docstring says out_0 = x_0, which is
    # not what the kernel computes (see operations/reduce/test_ema.py).
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
