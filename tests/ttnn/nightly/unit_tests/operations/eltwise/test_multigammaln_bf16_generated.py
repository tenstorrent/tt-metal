# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generated BF16 multigammaln cases, beside the unary category suites."""

import torch
import pytest
import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc, generate_all_bfloat16_bitpatterns
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole


# Exhaustive BF16 accuracy of ttnn.multigammaln: every BF16 bit pattern against torch.special.multigammaln(x, 4) in float64.
#
# The SFPU may read a subnormal input as zero, so an output may match torch at the input or at the input with
# subnormals flushed; the pack stores NaN as +inf and -0 as +0, so classes compare as stored.
# MULTIGAMMALN_DECLARED gives the class stored where it is the replaced TT-NN op's, not torch's.
MULTIGAMMALN_SMALLEST_NORMAL = 2.0**-126


MULTIGAMMALN_CLASS_CODES = {"inf": 0, "-inf": 1, "zero": 2, "finite": 3}


# Per board: (inputs, as a torch expression of the input x or of daz, the input as the SFPU reads it
# with subnormals as zero, and the class their outputs are stored as).
MULTIGAMMALN_DECLARED = {
    "blackhole": [
        (
            "torch.isfinite(daz) & (daz > -0.00341796875) & (daz < 0.00244140625)",
            "inf",
        ),  # -0.003417969 < finite x < 0.002441406
        ("torch.isfinite(daz) & (daz >= 1.0384593717069655e+36)", "inf"),  # finite x >= 1.038459e+36
    ],
    "wormhole_b0": [
        (
            "torch.isfinite(daz) & (daz > -0.00341796875) & (daz < 0.00244140625)",
            "inf",
        ),  # -0.003417969 < finite x < 0.002441406
        ("torch.isfinite(daz) & (daz >= 1.0384593717069655e+36)", "inf"),  # finite x >= 1.038459e+36
    ],
}


def _multigammaln_flush(t):
    return torch.where(t.abs() < MULTIGAMMALN_SMALLEST_NORMAL, torch.zeros_like(t), t)


def _multigammaln_reference(x):
    return torch.special.multigammaln(x, 4)


def _multigammaln_old_path(x, memory_config=None):
    result = ttnn.lgamma(x, memory_config=memory_config)
    result = ttnn.add(
        result,
        ttnn.lgamma(ttnn.subtract(x, 0.5, memory_config=memory_config), memory_config=memory_config),
        memory_config=memory_config,
    )
    result = ttnn.add(
        result,
        ttnn.lgamma(ttnn.subtract(x, 1.0, memory_config=memory_config), memory_config=memory_config),
        memory_config=memory_config,
    )
    result = ttnn.add(
        result,
        ttnn.lgamma(ttnn.subtract(x, 1.5, memory_config=memory_config), memory_config=memory_config),
        memory_config=memory_config,
    )
    result = ttnn.add(result, 3.434189657547, memory_config=memory_config)
    return result


def _multigammaln_round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _multigammaln_flush(bits.view(torch.float32).to(torch.bfloat16))


def _multigammaln_stored_classes(t):
    """The CLASS_CODES of each value as stored: NaN as +inf, either zero as zero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, MULTIGAMMALN_CLASS_CODES["finite"], dtype=torch.int8)
    classes[t == 0] = MULTIGAMMALN_CLASS_CODES["zero"]
    classes[(t == float("inf")) | torch.isnan(t)] = MULTIGAMMALN_CLASS_CODES["inf"]
    classes[t == float("-inf")] = MULTIGAMMALN_CLASS_CODES["-inf"]
    return classes


def _multigammaln_pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _multigammaln_round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _multigammaln_stored_classes(rounded) == _multigammaln_stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    return torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))


def _multigammaln_versus_torch(x64, output):
    """The largest pure ULP error against torch over outputs of torch's stored class, and the number
    of outputs of another class."""
    ulp = torch.minimum(
        _multigammaln_pure_ulp(_multigammaln_reference(x64), output),
        _multigammaln_pure_ulp(_multigammaln_reference(_multigammaln_flush(x64)), output),
    )
    mismatched = torch.isinf(ulp)
    return (ulp[~mismatched].max().item() if (~mismatched).any() else 0.0), int(mismatched.sum())


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
def test_multigammaln_exhaustive_bfloat16(device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.multigammaln(tt_x)).to(torch.bfloat16)
    # The path this program replaces, on the same input (see the device-perf test).
    stock = ttnn.to_torch(_multigammaln_old_path(tt_x)).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    x64 = x.to(torch.float64)
    (ours, ours_classes), (old, old_classes) = _multigammaln_versus_torch(x64, actual), _multigammaln_versus_torch(
        x64, stock
    )
    print(
        f"ULP multigammaln {board} ours={ours:.3f} stock={old:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={old_classes}"
    )
    expected = torch.full(x.shape, -1, dtype=torch.int8)
    for inputs, stored in MULTIGAMMALN_DECLARED[board]:
        lanes = eval(
            inputs,
            {
                "torch": torch,
                "x": x64,
                "daz": _multigammaln_flush(x64),
                "SMALLEST_NORMAL": MULTIGAMMALN_SMALLEST_NORMAL,
            },
        )
        expected = torch.where(lanes & (expected < 0), MULTIGAMMALN_CLASS_CODES[stored], expected)
    declared = expected >= 0
    wrong = declared & (_multigammaln_stored_classes(actual) != expected)
    assert not wrong.any(), (
        f"{wrong.sum().item()} declared inputs of the wrong class; first x={x[wrong][0].item()}, "
        f"got {actual[wrong][0].item()}"
    )

    ulp = torch.minimum(
        _multigammaln_pure_ulp(_multigammaln_reference(x64), actual),
        _multigammaln_pure_ulp(_multigammaln_reference(_multigammaln_flush(x64)), actual),
    )
    ulp = torch.where(declared, torch.zeros_like(ulp), ulp)
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}: expected {_multigammaln_reference(x64).flatten()[worst].item()}, "
        f"got {actual.flatten()[worst].item()}"
    )


# Inputs well inside the range the generated kernel is fitted on, for the calls below.
MULTIGAMMALN_LOW, MULTIGAMMALN_HIGH = 1.5009765625, 50.0


def _multigammaln_inputs(shape):
    generator = torch.Generator().manual_seed(0)
    return (
        MULTIGAMMALN_LOW
        + (MULTIGAMMALN_HIGH - MULTIGAMMALN_LOW) * (0.05 + 0.9 * torch.rand(shape, generator=generator))
    ).to(torch.bfloat16)


def _multigammaln_on_device(x, device, **kwargs):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, **kwargs)


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("placement", ["row_major", "height_sharded", "unaligned", "cached"])
def test_multigammaln_every_placement_matches_interleaved_tiles(device, placement):
    """The generated kernel computes each element alone, so placement must not change a result."""
    x = _multigammaln_inputs((1, 1, 64, 96))
    expected = ttnn.to_torch(ttnn.multigammaln(_multigammaln_on_device(x, device)))
    if placement == "row_major":
        tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    elif placement == "height_sharded":
        shard = ttnn.create_sharded_memory_config(
            x.shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
        )
        tt_x = _multigammaln_on_device(x, device, memory_config=shard)
    elif placement == "unaligned":
        x, expected = x[..., :33, :65].contiguous(), expected[..., :33, :65]
        tt_x = _multigammaln_on_device(x, device)
    else:
        tt_x = _multigammaln_on_device(x, device)
        ttnn.multigammaln(tt_x)
    actual = ttnn.to_torch(ttnn.multigammaln(tt_x))
    assert torch.equal(actual, expected)


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
def test_multigammaln_many_tiles_per_core_match_torch(device):
    """One core holds 128 tiles, so the kernel's per-call state carries across many tiles of one program."""
    x = _multigammaln_inputs((1, 1, 2048, 64))
    shard = ttnn.create_sharded_memory_config(
        x.shape, core_grid=ttnn.CoreGrid(y=1, x=1), strategy=ttnn.ShardStrategy.HEIGHT
    )
    actual = ttnn.to_torch(ttnn.multigammaln(_multigammaln_on_device(x, device, memory_config=shard))).to(
        torch.bfloat16
    )
    x64 = x.to(torch.float64)
    ulp = torch.minimum(
        _multigammaln_pure_ulp(_multigammaln_reference(x64), actual),
        _multigammaln_pure_ulp(_multigammaln_reference(_multigammaln_flush(x64)), actual),
    )
    assert ulp.max().item() < 1.0, f"{(ulp >= 1.0).sum().item()} of {ulp.numel()} outputs at or beyond 1 ulp of torch"


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("call", ["float32"])
def test_multigammaln_keeps_its_own_path_elsewhere(device, call):
    """A call the generated kernel does not serve runs the op's existing path and matches its golden."""
    x = _multigammaln_inputs((1, 1, 64, 64))
    golden = ttnn.get_golden_function(ttnn.multigammaln)
    actual = ttnn.multigammaln(ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device))
    expected = golden(x.float())
    assert_with_pcc(expected, ttnn.to_torch(actual).float(), 0.999)


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("call", ["float32", "chain"])
def test_multigammaln_refuses_calls_its_kernel_does_not_serve(device, call, expect_error):
    """UnaryOpType.MULTIGAMMALN has no kernel of its own: outside a lone op on BF16 data the unary program refuses it."""
    x = _multigammaln_inputs((1, 1, 32, 32))
    ops = [ttnn.UnaryWithParam(ttnn.UnaryOpType.MULTIGAMMALN)]
    if call == "float32":
        tt_x = ttnn.from_torch(x.float(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device)
    else:
        tt_x, ops = _multigammaln_on_device(x, device), ops + [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]
    with expect_error(RuntimeError, "MULTIGAMMALN runs only"):
        ttnn.unary_chain(tt_x, ops)
