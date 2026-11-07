# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 atan cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_ulp
from tests.ttnn.unit_tests.operations.eltwise.eltwise_test_utils import (
    generate_bfloat16_bits,
    flush_to_zero,
    to_tt_tensor,
    SMALLEST_NORMAL_BF16,
)


@pytest.mark.parametrize("generated_first", [True, False])
def test_atan_with_sqrt_in_sfpu_chain(device, generated_first):
    """atan chained with stock sqrt, both orders: each step stores what its op stores standalone."""
    generated = (ttnn.UnaryWithParam(ttnn.UnaryOpType.ATAN), lambda tensor: ttnn.atan(tensor))
    stock = (ttnn.UnaryWithParam(ttnn.UnaryOpType.SQRT), lambda tensor: ttnn.sqrt(tensor))
    first, second = (generated, stock) if generated_first else (stock, generated)
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16)
    tt_in = to_tt_tensor(input_tensor, device)

    def words(tensor):
        return ttnn.to_torch(tensor).view(torch.int16)

    for step, standalone in (generated, stock):
        assert torch.equal(
            words(ttnn.unary_chain(tt_in, [step])), words(standalone(tt_in))
        ), f"{step} is not its op's kernel"

    result = words(ttnn.unary_chain(tt_in, [first[0], second[0]]))
    middle = ttnn.to_torch(first[1](tt_in))
    expected = words(second[1](to_tt_tensor(middle, device)))
    # A finite normal first result sits in DEST exactly as stored, so both runs feed the second op the same input.
    checked = torch.isfinite(middle) & (middle.abs() >= SMALLEST_NORMAL_BF16)
    assert checked.sum() >= 64
    mismatched = checked & (result != expected)
    assert not mismatched.any(), (
        f"{int(mismatched.sum())} of {int(checked.sum())} lanes differ from the standalone second op, "
        f"first at its input {middle[mismatched][0].item()!r}"
    )


def test_atan_bf16_several_tiles_per_core(device):
    """Every core runs atan on dozens of tiles in turn; each call records the kernel's body into replay slots
    and reloads LREG4-7 again, so every tile, not only a core's first, must match torch."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("ttnn.atan keeps its stock kernel off Blackhole")
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16).repeat(8, 8)
    grid = device.compute_with_storage_grid_size()
    assert input_tensor.numel() // 1024 >= 4 * grid.x * grid.y
    golden = flush_to_zero(ttnn.get_golden_function(ttnn.atan)(input_tensor, device=device))
    result = flush_to_zero(ttnn.to_torch(ttnn.atan(to_tt_tensor(input_tensor, device))))
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_atan_bf16_nonzero_dest_index(device):
    """atan as the post-activation of a sharded FPU add, which runs it on DEST tiles 0-7 of each eight-tile
    batch: seven calls in eight have a DEST index other than 0. Adding +0 leaves every input as it is."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("ttnn.atan keeps its stock kernel off Blackhole")
    input_tensor = generate_bfloat16_bits(dtype=torch.bfloat16).reshape(1, 1, 256, 256)
    memory = ttnn.create_sharded_memory_config(
        input_tensor.shape, ttnn.CoreGrid(y=1, x=1), ttnn.ShardStrategy.HEIGHT, ttnn.ShardOrientation.ROW_MAJOR
    )
    tt_x, tt_zero = (
        to_tt_tensor(tensor, device, memory_config=memory) for tensor in (input_tensor, torch.zeros_like(input_tensor))
    )
    output = ttnn.add(
        tt_x,
        tt_zero,
        dtype=ttnn.bfloat16,
        memory_config=memory,
        fast_and_approximate_mode=True,  # the FPU add, eight tiles per DEST batch
        activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.ATAN)],
    )
    golden = flush_to_zero(ttnn.get_golden_function(ttnn.atan)(input_tensor, device=device))
    result = flush_to_zero(ttnn.to_torch(output))
    assert_with_ulp(expected_result=golden, actual_result=result, ulp_threshold=1)


def test_atan_bf16_bfloat8_b(device):
    """bfloat8_b in and out. The SFPU reads Bfp8_b data in DEST as BF16, so this runs the BF16 kernel. Each output
    is within two bfloat8_b steps of [1, 2) of torch's value on the input as stored: one for the output's
    shared-exponent rounding, one for the kernel."""
    if not ttnn.device.is_blackhole(device):
        pytest.skip("ttnn.atan keeps its stock kernel off Blackhole")
    torch.manual_seed(0)
    input_tensor = torch.empty(1, 1, 256, 256).uniform_(-8, 8).to(torch.bfloat16)
    tt_in = ttnn.from_torch(input_tensor, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    stored = ttnn.to_torch(tt_in).to(torch.float64)
    output = ttnn.atan(tt_in)
    assert output.dtype == ttnn.bfloat8_b
    golden = ttnn.get_golden_function(ttnn.atan)(stored, device=device)
    error = (ttnn.to_torch(output).to(torch.float64) - golden).abs()
    assert (
        error.max() <= 2.0**-5
    ), f"max error {error.max().item()} at input {stored.flatten()[error.argmax()].item()}"


def test_atan_bf16_unflushed_zero_and_subnormal_inputs(device):
    """-0 and subnormal inputs (kept by TT-NN's copy into DEST) store +0."""
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.atan keeps its stock kernel on {board}")
    words = torch.cat([torch.arange(0x80), torch.arange(0x8000, 0x8080)]).to(torch.int16)
    x = words.view(torch.bfloat16).repeat(4).reshape(1, 1, 32, 32)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    raw = ttnn.to_torch(ttnn.atan(tt_x)).reshape(-1).view(torch.int16).to(torch.int32) & 0xFFFF
    assert torch.all(raw == 0x0000)


def test_atan_bf16_exhaustive_inputs(device):
    """All 65,536 BF16 inputs: torch's result as stored, within STEPS BF16 steps.

    SKIP lists inputs the LLK sweep judges instead. MODEL: per board, the kernel's offset in BF16 steps from torch's word.
    """
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    if board not in ("blackhole",):
        pytest.skip(f"ttnn.atan keeps its stock kernel on {board}")
    STEPS = 1
    SKIP = [(0x0002, 0x007F), (0x8002, 0x807F)]
    MODEL = {
        -1: ("3DB8 3E28 3E47-3E48 3E60 3E74 3E8E 3E9B 3EA6 3EB0 4002 " "4013 BEEA BF07 C08D"),
        1: ("3EEA 3F07 408D BDB8 BE28 BE47-BE48 BE60 BE74 BE8E BE9B " "BEA6 BEB0 C002 C013"),
    }
    bits = torch.arange(1 << 16, dtype=torch.int32)
    x = bits.to(torch.int16).view(torch.bfloat16).reshape(1, 1, 256, 256)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tiny = torch.finfo(torch.bfloat16).tiny

    def stored(t):
        t = t.to(torch.bfloat16).to(torch.float64).flatten()
        return torch.where(t.abs() < tiny, torch.zeros_like(t), t)

    def ordered(t):
        # BF16 words in value order, so that adjacent finite values differ by one step.
        words = t.to(torch.bfloat16).view(torch.int16).to(torch.int32)
        return torch.where(words < 0, -(words & 0x7FFF), words)

    actual = stored(ttnn.to_torch(ttnn.atan(tt_x)))
    x = x.flatten().to(torch.float64)
    reference = (torch.atan(x)).to(torch.float64).flatten()
    expected = stored(reference)
    finite = torch.isfinite(expected) & torch.isfinite(actual)
    ok = (actual == expected) | (finite & ((ordered(actual) - ordered(expected)).abs() <= STEPS))
    # A NaN result reaches L1 as a NaN or as an infinity of either sign.
    ok |= torch.isnan(reference) & (torch.isnan(actual) | torch.isinf(actual))
    # The SFPU flushes a result below the smallest normal, or within 2^-12 above it, to 0.
    ok |= (actual == 0) & (reference.abs() < tiny * (1 + 2.0**-12))
    listed = torch.zeros_like(ok)
    for first, last in SKIP:
        listed |= (bits >= first) & (bits <= last)
        ok |= (bits >= first) & (bits <= last)
    shift = torch.zeros_like(bits)
    for offset, spans in MODEL.items():
        for span in spans.split():
            first, _, last = span.partition("-")
            shift[(bits >= int(first, 16)) & (bits <= int(last or first, 16))] = offset
    moved = ordered(expected) + shift
    word = torch.where(moved < 0, (-moved) | 0x8000, moved)
    held = stored(word.to(torch.int16).view(torch.bfloat16))
    normal = ((bits & 0x7F80) != 0) & ((bits & 0x7F80) != 0x7F80)
    ok = torch.where(normal & ~listed & torch.isfinite(expected), actual == held, ok)
    assert ok.all(), (
        f"{(~ok).sum().item()} of {ok.numel()} inputs off torch and stock; first input 0x{bits[~ok][0].item():04x} "
        f"stored {actual[~ok][0].item()!r}, torch {reference[~ok][0].item()!r}"
    )
