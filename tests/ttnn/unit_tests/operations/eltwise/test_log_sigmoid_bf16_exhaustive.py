# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""
Exhaustive BF16 accuracy of ttnn.log_sigmoid.

Every BF16 bit pattern is an input. The reference is torch.nn.functional.logsigmoid(x) in float64,
rounded once to BF16 with subnormal results flushed to zero. The SFPU may read a subnormal input
as zero, so torch is evaluated both at the input and at the input with subnormals flushed, and an
output may match either. The BF16 pack stores NaN as +inf and -0 as +0, so classes are compared as
stored. Each output must have the reference's class and a pure ULP error, |reference - output| /
ulp(rounded reference), below 1.
The inputs in DECLARED are stored as the class given there, the first row that holds; where
that differs from torch, it is the class the TT-NN op this kernel replaces stores.
Each run logs one ULP line: the largest pure ULP error against torch and the outputs of
another class, for this program and for the path it replaces on the same input.
"""

import pytest
import torch
import ttnn

from models.common.utility_functions import run_for_blackhole, run_for_wormhole_b0
from tests.ttnn.utils_for_testing import assert_with_pcc, generate_all_bfloat16_bitpatterns

SMALLEST_NORMAL = 2.0**-126
CLASS_CODES = {"inf": 0, "-inf": 1, "zero": 2, "finite": 3}

# Per board: (inputs, as a torch expression of the input x or of daz, the input as the SFPU reads it
# with subnormals as zero, and the class their outputs are stored as).
DECLARED = {
    "blackhole": [
        ("torch.isfinite(daz) & (daz > 3.3895313892515355e+38)", "zero"),  # finite x > 3.389531e+38
    ],
}


def _flush(t):
    return torch.where(t.abs() < SMALLEST_NORMAL, torch.zeros_like(t), t)


def _reference(x):
    return torch.nn.functional.logsigmoid(x)


def _round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _flush(bits.view(torch.float32).to(torch.bfloat16))


def _stored_classes(t):
    """The CLASS_CODES of each value as stored: NaN as +inf, either zero as zero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, CLASS_CODES["finite"], dtype=torch.int8)
    classes[t == 0] = CLASS_CODES["zero"]
    classes[(t == float("inf")) | torch.isnan(t)] = CLASS_CODES["inf"]
    classes[t == float("-inf")] = CLASS_CODES["-inf"]
    return classes


def _pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _stored_classes(rounded) == _stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    return torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))


def _versus_torch(x64, output):
    """The largest pure ULP error against torch over outputs of torch's stored class, and the number
    of outputs of another class."""
    ulp = torch.minimum(_pure_ulp(_reference(x64), output), _pure_ulp(_reference(_flush(x64)), output))
    mismatched = torch.isinf(ulp)
    return (ulp[~mismatched].max().item() if (~mismatched).any() else 0.0), int(mismatched.sum())


@run_for_blackhole("the generated kernel runs on Blackhole only")
def test_log_sigmoid_exhaustive_bfloat16(device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x)).to(torch.bfloat16)
    # The path this program replaces, on the same input (see the device-perf test).
    stock = ttnn.to_torch(
        ttnn.unary_chain(
            tt_x, [ttnn.UnaryWithParam(ttnn.UnaryOpType.LOGSIGMOID), ttnn.UnaryWithParam(ttnn.UnaryOpType.IDENTITY)]
        )
    ).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    x64 = x.to(torch.float64)
    (ours, ours_classes), (old, old_classes) = _versus_torch(x64, actual), _versus_torch(x64, stock)
    print(
        f"ULP log_sigmoid {board} ours={ours:.3f} stock={old:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={old_classes}"
    )
    expected = torch.full(x.shape, -1, dtype=torch.int8)
    for inputs, stored in DECLARED[board]:
        lanes = eval(inputs, {"torch": torch, "x": x64, "daz": _flush(x64), "SMALLEST_NORMAL": SMALLEST_NORMAL})
        expected = torch.where(lanes & (expected < 0), CLASS_CODES[stored], expected)
    declared = expected >= 0
    wrong = declared & (_stored_classes(actual) != expected)
    assert not wrong.any(), (
        f"{wrong.sum().item()} declared inputs of the wrong class; first x={x[wrong][0].item()}, "
        f"got {actual[wrong][0].item()}"
    )

    ulp = torch.minimum(_pure_ulp(_reference(x64), actual), _pure_ulp(_reference(_flush(x64)), actual))
    ulp = torch.where(declared, torch.zeros_like(ulp), ulp)
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}: expected {_reference(x64).flatten()[worst].item()}, "
        f"got {actual.flatten()[worst].item()}"
    )


# Inputs well inside the range the generated kernel is fitted on, for the calls below.
LOW, HIGH = -10.0, 10.0


def _inputs(shape):
    generator = torch.Generator().manual_seed(0)
    return (LOW + (HIGH - LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))).to(torch.bfloat16)


def _on_device(x, device, **kwargs):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **kwargs)


@run_for_blackhole("the generated kernel runs on Blackhole only")
@pytest.mark.parametrize("placement", ["row_major", "height_sharded", "unaligned", "cached"])
def test_log_sigmoid_every_placement_matches_interleaved_tiles(device, placement):
    """The generated kernel computes each element alone, so placement must not change a result."""
    x = _inputs((1, 1, 64, 96))
    expected = ttnn.to_torch(ttnn.log_sigmoid(_on_device(x, device)))
    if placement == "row_major":
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    elif placement == "height_sharded":
        shard = ttnn.create_sharded_memory_config(
            x.shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
        )
        tt_x = _on_device(x, device, memory_config=shard)
    elif placement == "unaligned":
        x, expected = x[..., :33, :65].contiguous(), expected[..., :33, :65]
        tt_x = _on_device(x, device)
    else:
        tt_x = _on_device(x, device)
        ttnn.log_sigmoid(tt_x)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x))
    assert torch.equal(actual, expected)


@run_for_blackhole("the generated kernel runs on Blackhole only")
@pytest.mark.parametrize("call", ["float32", "bfloat16_to_float32"])
def test_log_sigmoid_keeps_its_own_path_elsewhere(device, call):
    """A call the generated kernel does not serve runs the op's existing path and matches its golden."""
    x = _inputs((1, 1, 64, 64))
    golden = ttnn.get_golden_function(ttnn.log_sigmoid)
    if call == "float32":
        actual = ttnn.log_sigmoid(
            ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        )
        expected = golden(x.float())
    elif call == "bfloat16_to_float32":
        # Only the route is checked: the generated kernel does not compile with FP32 DEST, so any
        # result shows that the op's own kernel ran.
        output = ttnn.from_torch(torch.zeros(x.shape), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
        assert ttnn.log_sigmoid(_on_device(x, device), output_tensor=output).dtype == ttnn.float32
        return
    else:
        # Any value other than the parameter's default keeps the op's own kernel.
        actual = ttnn.log_sigmoid(_on_device(x, device), **{call: 0.125})
        expected = golden(x.float(), **{call: 0.125})
    assert_with_pcc(expected, ttnn.to_torch(actual).float(), 0.999)


# Per board that keeps TT-NN's own path: (input, that path's output, the generated kernel's output)
# BF16 words at inputs where the two differ, as measured on the board.
KEPT = {
    "wormhole_b0": [
        (0x4092, 0xBC30, 0xBC2A),
        (0x40A8, 0xBBB1, 0xBBAB),
        (0x4091, 0xBC35, 0xBC2F),
        (0x4131, 0xB77F, 0xB784),
    ],
}


@run_for_wormhole_b0("TT-NN's own path is kept on this board")
def test_log_sigmoid_keeps_its_own_path(device):
    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    words = torch.tensor([word for word, _, _ in KEPT[board]], dtype=torch.int32).to(torch.int16)
    x = words.view(torch.bfloat16).reshape(1, -1).expand(32, -1).contiguous()
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.log_sigmoid(tt_x)).to(torch.bfloat16)[0].view(torch.int16).to(torch.int32) & 0xFFFF
    expected = torch.tensor([word for _, word, _ in KEPT[board]], dtype=torch.int32)
    assert torch.equal(actual, expected), f"expected TT-NN's own path {expected.tolist()}, got {actual.tolist()}"
