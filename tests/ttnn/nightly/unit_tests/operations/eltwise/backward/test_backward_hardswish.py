# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc, generate_all_bfloat16_bitpatterns


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_hardswish(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)

    tt_output_tensor_on_device = ttnn.hardswish_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.hardswish_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass


# Exhaustive BF16 accuracy of ttnn.hardswish_bw.
#
# Every BF16 bit pattern is paired with each gradient in GRADS. The reference is torch autograd
# of torch.nn.functional.hardswish(x) in float64, rounded once to BF16 with subnormal results flushed to zero.
# The SFPU may read a subnormal operand as zero, so torch is evaluated both at the operands and at
# the operands with subnormals flushed, and an output may match either. The BF16 compute and pack
# path stores NaN as +inf and -0 as +0, so classes are compared as stored. Each output must have
# the reference's class and a pure ULP error, |reference - output| / ulp(rounded reference),
# below 1. A +0 where torch's own float64 result is below the smallest normal is torch's result as
# stored, since the BF16 output cannot hold a subnormal. Each case logs one ULP line: the largest
# pure ULP error against torch and the lanes of another class, for the output and for the
# composite's on the same operands.
# At a NaN input the output is grad, as the composite this program replaces returns; torch returns NaN.
HARDSWISH_BW_GRADS = [
    "1",
    "-1",
    "0.5",
    "3",
    "random0",
    "random1",
    "0",
    "-0",
    "inf",
    "-inf",
    "nan",
    "1.1754943508222875e-38",
    "-1.1754943508222875e-38",
    "3.3895313892515355e+38",
    "-3.3895313892515355e+38",
]
HARDSWISH_BW_SMALLEST_NORMAL = 2.0**-126


def _hardswish_bw_flush(t):
    return torch.where(t.abs() < HARDSWISH_BW_SMALLEST_NORMAL, torch.zeros_like(t), t)


def _hardswish_bw_grad_like(x, grad):
    if grad.startswith("random"):
        generator = torch.Generator().manual_seed(int(grad.removeprefix("random")))
        return torch.randn(x.shape, generator=generator).to(torch.bfloat16)
    return torch.full(x.shape, float(grad), dtype=torch.bfloat16)


def _hardswish_bw_torch_reference(grad, x, flush):
    with torch.enable_grad():
        x64, g = x.to(torch.float64), grad.to(torch.float64)
        if flush:
            x64, g = _hardswish_bw_flush(x64), _hardswish_bw_flush(g)
        x64.requires_grad_(True)
        torch.nn.functional.hardswish(x64).backward(g)
        return x64.grad.detach()


def _hardswish_bw_reference(grad, x, flush):
    """Torch, except at a NaN input, where the replaced composite's result is kept."""
    g = _hardswish_bw_flush(grad.to(torch.float64)) if flush else grad.to(torch.float64)
    return torch.where(torch.isnan(x.to(torch.float64)), g, _hardswish_bw_torch_reference(grad, x, flush))


def _hardswish_bw_round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _hardswish_bw_flush(bits.view(torch.float32).to(torch.bfloat16))


def _hardswish_bw_stored_classes(t):
    """0 +inf (or NaN), 1 -inf, 2 zero of either sign, 3 finite nonzero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, 3, dtype=torch.int8)
    classes[t == 0] = 2
    classes[(t == float("inf")) | torch.isnan(t)] = 0
    classes[t == float("-inf")] = 1
    return classes


def _hardswish_bw_versus_torch(g, x, output):
    """The largest pure ULP error against torch over lanes of torch's stored class, and the number
    of lanes of another class."""
    ulp = torch.minimum(
        _hardswish_bw_pure_ulp(_hardswish_bw_torch_reference(g, x, False), output),
        _hardswish_bw_pure_ulp(_hardswish_bw_torch_reference(g, x, True), output),
    )
    mismatched = torch.isinf(ulp)
    return (ulp[~mismatched].max().item() if (~mismatched).any() else 0.0), int(mismatched.sum())


def _hardswish_bw_pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _hardswish_bw_round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _hardswish_bw_stored_classes(rounded) == _hardswish_bw_stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    ulp = torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))
    # A +0 where torch's own result is subnormal is torch's result as stored: BF16 cannot store a subnormal.
    flushed = (actual.to(torch.float64) == 0) & (reference.abs() < HARDSWISH_BW_SMALLEST_NORMAL)
    return torch.where(flushed, torch.zeros_like(ulp), ulp)


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("grad", HARDSWISH_BW_GRADS)
def test_hardswish_bw_exhaustive_bfloat16(grad, device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    g = _hardswish_bw_grad_like(x, grad)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.hardswish_bw(tt_g, tt_x)[0]).to(torch.bfloat16)
    # The composite on the same operands: a gradient in L1 beside an input in DRAM keeps it.
    tt_g_l1 = ttnn.from_torch(
        g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    stock = ttnn.to_torch(ttnn.hardswish_bw(tt_g_l1, tt_x)[0]).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    (ours, ours_classes), (composite, composite_classes) = _hardswish_bw_versus_torch(
        g, x, actual
    ), _hardswish_bw_versus_torch(g, x, stock)
    print(
        f"ULP hardswish_bw {board} ours={ours:.3f} stock={composite:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={composite_classes} grad={grad}"
    )
    ulp = torch.minimum(
        _hardswish_bw_pure_ulp(_hardswish_bw_reference(g, x, False), actual),
        _hardswish_bw_pure_ulp(_hardswish_bw_reference(g, x, True), actual),
    )
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}, grad={g.flatten()[worst].item()}: "
        f"expected {_hardswish_bw_reference(g, x, False).flatten()[worst].item()}, got {actual.flatten()[worst].item()}"
    )


def _hardswish_bw_torch_gradient(grad, x):
    """Torch autograd in float32."""
    with torch.enable_grad():
        x = x.to(torch.float32).requires_grad_(True)
        torch.nn.functional.hardswish(x).backward(grad.to(torch.float32))
        return x.grad


def _hardswish_bw_device_operations(call):
    """The result of ``call``, or the error it raised, and the device operations it launched, as graph
    capture names them."""
    result, error = None, None
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = call()
    except RuntimeError as raised:
        error = raised
    trace = ttnn.graph.extract_calltrace(ttnn.graph.end_graph_capture())
    return result, error, trace


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize(
    "call",
    ["interleaved_tiles", "broadcast_grad", "row_major", "mixed_layouts", "grad_in_l1", "height_sharded", "float32"],
)
def test_hardswish_bw_runs_the_fused_program_only_where_it_applies(call, device):
    """BF16 operands of one shape in interleaved tiles with one placement run the fused program, and
    match torch; every other call keeps the composite, which broadcasts and takes any layout, placement
    and dtype."""
    shape = (1, 2, 32, 64)
    generator = torch.Generator().manual_seed(0)
    x = (4 * torch.randn(shape, generator=generator)).to(torch.bfloat16)
    g = torch.randn((1, 1, 32, 64) if call == "broadcast_grad" else shape, generator=generator).to(torch.bfloat16)
    dtype = ttnn.float32 if call == "float32" else ttnn.bfloat16
    layouts = {"row_major": (ttnn.ROW_MAJOR_LAYOUT,) * 2, "mixed_layouts": (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)}
    grad_layout, input_layout = layouts.get(call, (ttnn.TILE_LAYOUT,) * 2)
    shard = ttnn.create_sharded_memory_config(
        shape, core_grid=ttnn.CoreGrid(y=1, x=2), strategy=ttnn.ShardStrategy.HEIGHT
    )
    memory = {"grad_in_l1": (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG), "height_sharded": (shard, shard)}
    grad_memory, input_memory = memory.get(call, (ttnn.DRAM_MEMORY_CONFIG,) * 2)
    tt_g = ttnn.from_torch(g, dtype=dtype, layout=grad_layout, device=device, memory_config=grad_memory)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=input_layout, device=device, memory_config=input_memory)

    result, error, trace = _hardswish_bw_device_operations(lambda: ttnn.hardswish_bw(tt_g, tt_x)[0])

    assert ("UnaryBackwardDeviceOperation" in trace) == (call == "interleaved_tiles"), trace
    if call != "interleaved_tiles":
        # The composite's own results, or its refusal of a call (some refuse row-major operands), are
        # its tests' to judge; here it only has to be the program that ran.
        assert error is not None or list(result.shape) == list(shape)
        return
    assert error is None, error
    expected = _hardswish_bw_torch_gradient(g, x)
    assert_with_pcc(expected, ttnn.to_torch(result).float(), 0.999)
