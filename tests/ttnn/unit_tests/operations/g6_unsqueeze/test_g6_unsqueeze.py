# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_unsqueeze (immutable spec for the implementer).

The op moves values, so it is graded bit for bit (int32 views; a NaN never equals
itself as a float). Scope is limited to the production cases and the cases the
spec's rules require (BB-22): p1 [1, 512] and p2 [1, 256] with dim 2, dim -1,
special bit patterns, poisoned tile padding, one regime-pinned `streaming` case
(forced through the program-descriptor knobs on the p1 shape), the plan report,
and the host-side refusals.
"""

import pytest
import torch
import ttnn

from ttnn.operations.g6_unsqueeze import g6_unsqueeze
import ttnn.operations.g6_unsqueeze.g6_unsqueeze_program_descriptor as g6u_pd

P1 = (1, 512)
P2 = (1, 256)

_SPECIAL_NAN_BITS = (0x7FC12345, 0xFFFC5433 - 2**32, 0x7F800001)
_F32 = torch.finfo(torch.float32)
_SPECIAL_VALUES = (float("inf"), float("-inf"), -0.0, 1e-40, -1e-45, _F32.max, -_F32.max, _F32.tiny)


def torch_reference(x, dim):
    return torch.unsqueeze(x, dim)


def make_input(shape, special=False):
    torch.manual_seed(42)
    x = torch.randn(shape, dtype=torch.float32)
    if special:
        bits = x.view(-1).view(torch.int32)
        specials = torch.cat(
            [
                torch.tensor(_SPECIAL_NAN_BITS, dtype=torch.int32),
                torch.tensor(_SPECIAL_VALUES, dtype=torch.float32).view(torch.int32),
            ]
        )
        pos = torch.arange(0, bits.numel(), 3)
        bits[pos] = specials[torch.arange(pos.numel()) % specials.numel()]
    return x


def to_device(x, device, memory_config=None):
    return ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG if memory_config is None else memory_config,
    )


def assert_bits_equal(actual, expected):
    assert list(actual.shape) == list(expected.shape), f"shape {list(actual.shape)} != {list(expected.shape)}"
    got = actual.contiguous().view(torch.int32)
    want = expected.contiguous().view(torch.int32)
    wrong = got != want
    assert not bool(wrong.any()), f"{int(wrong.sum())} of {wrong.numel()} values differ bitwise"


def run_and_check(x, tt_x, dim):
    out = g6_unsqueeze(tt_x, dim)
    assert out.layout == ttnn.TILE_LAYOUT
    assert out.dtype == ttnn.float32
    mc = out.memory_config()
    assert mc.memory_layout == ttnn.TensorMemoryLayout.INTERLEAVED
    assert mc.buffer_type == ttnn.BufferType.DRAM
    assert out.buffer_address() != tt_x.buffer_address(), "output aliases the input buffer"
    assert_bits_equal(ttnn.to_torch(out), torch_reference(x, dim))
    return out


def _require_round_trip(tt_x, x):
    back = ttnn.to_torch(tt_x)
    if not torch.equal(back.contiguous().view(torch.int32), x.contiguous().view(torch.int32)):
        pytest.skip("harness: host round trip does not keep every input bit")


@pytest.mark.parametrize("shape", [P1, P2], ids=["p1_1x512", "p2_1x256"])
def test_production(device, shape):
    x = make_input(shape)
    run_and_check(x, to_device(x, device), 2)


def test_dim_minus_one(device):
    x = make_input(P1)
    run_and_check(x, to_device(x, device), -1)


def test_dim_keyword(device):
    x = make_input(P2)
    tt_x = to_device(x, device)
    out = g6_unsqueeze(tt_x, dim=2)
    assert_bits_equal(ttnn.to_torch(out), torch_reference(x, 2))


def test_special_bit_patterns(device):
    x = make_input(P1, special=True)
    tt_x = to_device(x, device)
    _require_round_trip(tt_x, x)
    run_and_check(x, tt_x, 2)


def test_poisoned_padding(device):
    x = make_input(P2)
    tt_x = ttnn.fill_implicit_tile_padding(to_device(x, device), float("nan"))
    padded = tt_x.cpu().to_torch_with_padded_shape()
    if not bool(torch.isnan(padded[..., 1:, :]).all()):
        pytest.skip("harness: NaN did not reach the input tile padding")
    run_and_check(x, tt_x, 2)


@pytest.mark.parametrize("shape", [P1, P2], ids=["p1_1x512", "p2_1x256"])
def test_production_plan_is_resident(device, shape):
    x = make_input(shape)
    plan = g6u_pd.select_plan(to_device(x, device))
    assert plan.regime == "resident"
    assert plan.buffer_depth == 1
    tensor_w_tiles = (shape[-1] + 31) // 32
    assert plan.num_cores > 1, "a multi-tile input must be spread across cores"
    assert plan.core_w_tiles_max * plan.num_cores >= tensor_w_tiles


def test_streaming_regime_pinned(device, monkeypatch):
    # 16 tiles on 2 cores = 8 tiles per core, blocks of 3 -> 3/3/2 (ragged last block), depth 2.
    monkeypatch.setattr(g6u_pd, "MAX_CORES", 2)
    monkeypatch.setattr(g6u_pd, "MAX_BLOCK_W_TILES", 3)
    x = make_input(P1, special=True)
    tt_x = to_device(x, device)
    _require_round_trip(tt_x, x)
    plan = g6u_pd.select_plan(tt_x)
    assert plan.regime == "streaming"
    assert plan.num_cores == 2
    assert plan.block_w_tiles == 3
    assert plan.buffer_depth == 2
    run_and_check(x, tt_x, 2)


@pytest.mark.parametrize("dim", [0, 1, -2, -3, 3, -4])
def test_refuses_dim(device, expect_error, dim):
    tt_x = to_device(make_input(P1), device)
    with expect_error((ValueError, RuntimeError), ""):  # host refusal; message not pinned
        g6_unsqueeze(tt_x, dim)


@pytest.mark.parametrize(
    "shape,dim",
    [((1, 1, 1, 512), 4), ((2, 512), 2), ((1, 64, 512), 3)],
    ids=["rank4", "leading_dim_2", "two_rows_rank3"],
)
def test_refuses_shape(device, expect_error, shape, dim):
    tt_x = to_device(make_input(shape), device)
    with expect_error((ValueError, RuntimeError), ""):  # host refusal; message not pinned
        g6_unsqueeze(tt_x, dim)


def test_refuses_empty(device, expect_error):
    try:
        tt_x = to_device(torch.zeros((1, 0), dtype=torch.float32), device)
    except Exception as e:  # noqa: BLE001 - harness cannot build the case
        pytest.skip(f"harness: cannot create an empty TILE tensor: {e}")
    with expect_error((ValueError, RuntimeError), ""):  # host refusal; message not pinned
        g6_unsqueeze(tt_x, 2)


def test_refuses_l1_input(device, expect_error):
    tt_x = to_device(make_input(P1), device, memory_config=ttnn.L1_MEMORY_CONFIG)
    with expect_error((ValueError, RuntimeError), ""):  # host refusal; message not pinned
        g6_unsqueeze(tt_x, 2)
