# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import pytest
import ttnn
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
from tt_py_test_utils_common.utility_functions import run_for_wormhole_b0_or_blackhole
from tests.ttnn.utils_for_testing import assert_with_pcc, generate_all_bfloat16_bitpatterns


@pytest.mark.parametrize(
    "input_shapes",
    (
        (torch.Size([1, 1, 32, 32])),
        (torch.Size([1, 1, 320, 384])),
        (torch.Size([1, 3, 320, 384])),
    ),
)
def test_bw_sigmoid(input_shapes, device):
    in_data, input_tensor = data_gen_with_range(input_shapes, -1, 1, device, True)
    grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device, True)

    tt_output_tensor_on_device = ttnn.sigmoid_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.sigmoid_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_pcc(tt_output_tensor_on_device, golden_tensor, 0.90)
    assert comp_pass


# Exhaustive BF16 accuracy of ttnn.sigmoid_bw.
#
# Every BF16 bit pattern is paired with each gradient in GRADS. The reference is grad times
# torch.exp(-x.abs()) / (1.0 + torch.exp(-x.abs())) ** 2, the exact derivative of torch.sigmoid(x), in
# float64 at every finite input (float64 autograd rounds it, as where the forward rounds to its
# limit), and torch autograd at the others, rounded once to BF16 with subnormal results flushed to
# zero.
# The SFPU may read a subnormal operand as zero, so torch is evaluated both at the operands and at
# the operands with subnormals flushed, and an output may match either. The BF16 compute and pack
# path stores NaN as +inf and -0 as +0, so classes are compared as stored. Each output must have
# the reference's class and a pure ULP error, |reference - output| / ulp(rounded reference),
# below 1. A +0 where torch's own float64 result is below the smallest normal is torch's result as
# stored, since the BF16 output cannot hold a subnormal. Each case logs one ULP line: the largest
# pure ULP error against torch and the lanes of another class, for the output and for
# main's single program's on the same operands.
# The output is 0 wherever |x| >= 131.698, where even exp(-|x|) * 2^64 is below the smallest normal,
# for a finite grad, as main's single program computes it; torch keeps the exact product.
SIGMOID_BW_GRADS = [
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
    "-nan",
    "1.1754943508222875e-38",
    "-1.1754943508222875e-38",
    "3.3895313892515355e+38",
    "-3.3895313892515355e+38",
]
SIGMOID_BW_SMALLEST_NORMAL = 2.0**-126


def _sigmoid_bw_flush(t):
    return torch.where(t.abs() < SIGMOID_BW_SMALLEST_NORMAL, torch.zeros_like(t), t)


def _sigmoid_bw_grad_like(x, grad):
    if grad.startswith("random"):
        generator = torch.Generator().manual_seed(int(grad.removeprefix("random")))
        return torch.randn(x.shape, generator=generator).to(torch.bfloat16)
    if grad == "-nan":
        # torch.full rounds every NaN to 0x7FC0: the sign-set NaN, 0xFFC0, is written by its bits.
        return torch.full(x.shape, -64, dtype=torch.int16).view(torch.bfloat16)
    return torch.full(x.shape, float(grad), dtype=torch.bfloat16)


def _sigmoid_bw_torch_reference(grad, x, flush):
    with torch.enable_grad():
        x64, g = x.to(torch.float64), grad.to(torch.float64)
        if flush:
            x64, g = _sigmoid_bw_flush(x64), _sigmoid_bw_flush(g)
        x64.requires_grad_(True)
        torch.sigmoid(x64).backward(g)
        autograd, x64 = x64.grad, x64.detach()
        return torch.where(
            torch.isfinite(x64), g * (torch.exp(-x64.abs()) / (1.0 + torch.exp(-x64.abs())) ** 2), autograd
        )


def _sigmoid_bw_reference(grad, x, flush, board):
    """Torch, except on the declared lanes in the module docstring."""
    result = _sigmoid_bw_torch_reference(grad, x, flush)
    x32 = x.to(torch.float32)
    result = torch.where(torch.isfinite(grad) & (x32.abs() >= 131.69798278808594), torch.zeros_like(result), result)
    return result


def _sigmoid_bw_round_to_bfloat16(t):
    """Round float64 to BF16 once (round-to-odd into float32, then nearest-even), then flush."""
    f32 = t.to(torch.float32)
    back = f32.to(torch.float64)
    inexact = torch.isfinite(t) & (back != t)
    bits = f32.view(torch.int32) - (inexact & (back.abs() > t.abs())).to(torch.int32)
    bits = bits | inexact.to(torch.int32)
    return _sigmoid_bw_flush(bits.view(torch.float32).to(torch.bfloat16))


def _sigmoid_bw_stored_classes(t):
    """0 +inf (or NaN), 1 -inf, 2 zero of either sign, 3 finite nonzero."""
    t = t.to(torch.float64)
    classes = torch.full(t.shape, 3, dtype=torch.int8)
    classes[t == 0] = 2
    classes[(t == float("inf")) | torch.isnan(t)] = 0
    classes[t == float("-inf")] = 1
    return classes


def _sigmoid_bw_versus_torch(g, x, output):
    """The largest pure ULP error against torch over lanes of torch's stored class, and the number
    of lanes of another class."""
    ulp = torch.minimum(
        _sigmoid_bw_pure_ulp(_sigmoid_bw_torch_reference(g, x, False), output),
        _sigmoid_bw_pure_ulp(_sigmoid_bw_torch_reference(g, x, True), output),
    )
    mismatched = torch.isinf(ulp)
    return (ulp[~mismatched].max().item() if (~mismatched).any() else 0.0), int(mismatched.sum())


def _sigmoid_bw_pure_ulp(reference, actual):
    """Pure ULP error, infinite where the stored class differs."""
    rounded = _sigmoid_bw_round_to_bfloat16(reference).to(torch.float64)
    magnitude = rounded.abs()
    exponent = torch.floor(torch.log2(torch.where(magnitude > 0, magnitude, torch.ones_like(magnitude))))
    spacing = torch.where(magnitude > 0, 2.0 ** (exponent.clamp(min=-126) - 7), torch.full_like(magnitude, 2.0**-133))
    # The numerator is flushed only where the correctly rounded result is zero (post-round flush).
    golden = torch.where(magnitude == 0, torch.zeros_like(reference), reference)
    ulp = ((golden - actual.to(torch.float64)).abs().to(torch.float32) / spacing.to(torch.float32)).to(torch.float64)
    same_class = _sigmoid_bw_stored_classes(rounded) == _sigmoid_bw_stored_classes(actual)
    ulp = torch.where(torch.isfinite(rounded), ulp, torch.zeros_like(ulp))
    ulp = torch.where(same_class, ulp, torch.full_like(ulp, float("inf")))
    # A +0 where torch's own result is subnormal is torch's result as stored: BF16 cannot store a subnormal.
    flushed = (actual.to(torch.float64) == 0) & (reference.abs() < SIGMOID_BW_SMALLEST_NORMAL)
    return torch.where(flushed, torch.zeros_like(ulp), ulp)


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize("grad", SIGMOID_BW_GRADS)
def test_sigmoid_bw_exhaustive_bfloat16(grad, device):
    x = generate_all_bfloat16_bitpatterns(torch.bfloat16)
    g = _sigmoid_bw_grad_like(x, grad)

    tt_x = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    tt_g = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    actual = ttnn.to_torch(ttnn.sigmoid_bw(tt_g, tt_x)[0]).to(torch.bfloat16)
    # Main's single program on the same operands: a gradient in L1 beside an input in DRAM keeps it.
    tt_g_l1 = ttnn.from_torch(
        g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.L1_MEMORY_CONFIG
    )
    stock = ttnn.to_torch(ttnn.sigmoid_bw(tt_g_l1, tt_x)[0]).to(torch.bfloat16)

    board = "blackhole" if ttnn.device.is_blackhole(device) else "wormhole_b0"
    (ours, ours_classes), (composite, composite_classes) = _sigmoid_bw_versus_torch(
        g, x, actual
    ), _sigmoid_bw_versus_torch(g, x, stock)
    print(
        f"ULP sigmoid_bw {board} ours={ours:.3f} stock={composite:.3f} "
        f"ours_class_mismatches={ours_classes} stock_class_mismatches={composite_classes} grad={grad}"
    )
    ulp = torch.minimum(
        _sigmoid_bw_pure_ulp(_sigmoid_bw_reference(g, x, False, board), actual),
        _sigmoid_bw_pure_ulp(_sigmoid_bw_reference(g, x, True, board), actual),
    )
    worst = ulp.argmax()
    assert ulp.max().item() < 1.0, (
        f"{(ulp >= 1.0).sum().item()} outputs at or beyond 1 ulp or of the wrong class; worst at "
        f"x={x.flatten()[worst].item()}, grad={g.flatten()[worst].item()}: "
        f"expected {_sigmoid_bw_reference(g, x, False, board).flatten()[worst].item()}, got {actual.flatten()[worst].item()}"
    )


# The calls the fused program serves.
SIGMOID_BW_FUSED_CALLS = (
    "interleaved_tiles",
    "interleaved_l1",
    "height_sharded",
    "width_sharded",
    "block_sharded",
    "height_sharded_dram",
    "width_sharded_dram",
    "nd_sharded",
    "nd_sharded_dram",
    "sharded_to_interleaved",
    "interleaved_to_sharded",
    "mixed_placements",
)


def _sigmoid_bw_torch_gradient(grad, x):
    """Torch autograd in float32, or the exact derivative where it is declared."""
    with torch.enable_grad():
        x = x.to(torch.float32).requires_grad_(True)
        torch.sigmoid(x).backward(grad.to(torch.float32))
        x64 = x.detach().to(torch.float64)
        exact = (grad.to(torch.float64) * (torch.exp(-x64.abs()) / (1.0 + torch.exp(-x64.abs())) ** 2)).to(
            torch.float32
        )
        return torch.where(torch.isfinite(x64), exact, x.grad)


def _sigmoid_bw_device_operations(call):
    """The result of ``call``, or the error it raised, and the device operations it launched, as graph
    capture names them."""
    result, error = None, None
    ttnn.graph.begin_graph_capture(ttnn.graph.RunMode.NORMAL)
    try:
        result = call()
    except RuntimeError as raised:
        error = raised
    graph = ttnn.graph.end_graph_capture()
    trace = ttnn.graph.extract_calltrace(graph)
    # The device operation's arguments name the op type it runs.
    launched = " ".join(str(argument) for node in graph for argument in node.get("arguments", []))
    return result, error, trace, launched


def _sigmoid_bw_placements(shape):
    """Per call, the grad's and the input's memory configs and the output config it passes, if any."""
    height, width = shape[0] * shape[1] * shape[2], shape[3]
    L1, DRAM = ttnn.BufferType.L1, ttnn.BufferType.DRAM
    layout = ttnn.TensorMemoryLayout

    def cores(x, y):
        return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(x - 1, y - 1))})

    def sharded(strategy, buffer, grid, shard):
        return ttnn.MemoryConfig(strategy, buffer, ttnn.ShardSpec(grid, shard, ttnn.ShardOrientation.ROW_MAJOR))

    def nd(buffer, grid):
        spec = ttnn.NdShardSpec(
            ttnn.Shape([1, 1, shape[2], shape[3] // 2]),
            grid,
            ttnn.ShardOrientation.ROW_MAJOR,
            ttnn.ShardDistributionStrategy.ROUND_ROBIN_1D,
        )
        return ttnn.MemoryConfig(buffer, spec)

    height_l1 = sharded(layout.HEIGHT_SHARDED, L1, cores(2, 1), (height // 2, width))
    block_l1 = sharded(layout.BLOCK_SHARDED, L1, cores(2, 2), (height // 2, width // 2))
    width_dram = sharded(layout.WIDTH_SHARDED, DRAM, cores(2, 1), (height, width // 2))
    dram, l1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    return {
        "interleaved_l1": (l1, l1, None),
        "height_sharded": (height_l1, height_l1, None),
        "width_sharded": (*(sharded(layout.WIDTH_SHARDED, L1, cores(2, 1), (height, width // 2)),) * 2, None),
        "block_sharded": (block_l1, block_l1, None),
        "height_sharded_dram": (*(sharded(layout.HEIGHT_SHARDED, DRAM, cores(2, 1), (height // 2, width)),) * 2, None),
        "width_sharded_dram": (width_dram, width_dram, None),
        "nd_sharded": (nd(L1, cores(2, 2)), nd(L1, cores(2, 2)), None),
        "nd_sharded_dram": (nd(DRAM, cores(2, 1)), nd(DRAM, cores(2, 1)), None),
        "sharded_to_interleaved": (height_l1, height_l1, dram),
        "interleaved_to_sharded": (dram, dram, block_l1),
        "mixed_placements": (height_l1, width_dram, l1),
        "grad_in_l1": (l1, dram, None),
        "sharded_output_without_spec": (height_l1, height_l1, ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG),
    }


@run_for_wormhole_b0_or_blackhole("the generated kernel exists for Blackhole and Wormhole only")
@pytest.mark.parametrize(
    "call",
    [
        "interleaved_tiles",
        "interleaved_l1",
        "height_sharded",
        "width_sharded",
        "block_sharded",
        "height_sharded_dram",
        "width_sharded_dram",
        "nd_sharded",
        "nd_sharded_dram",
        "sharded_to_interleaved",
        "interleaved_to_sharded",
        "mixed_placements",
        "broadcast_grad",
        "row_major",
        "mixed_layouts",
        "grad_in_l1",
        "sharded_output_without_spec",
        "float32",
    ],
)
def test_sigmoid_bw_runs_the_fused_program_only_where_it_applies(call, device):
    """BF16 operands of one shape in tiles, interleaved or sharded in L1 or DRAM, with one placement or
    an output config run the fused program and match torch; every other call keeps main's single program,
    which broadcasts and takes any layout, placement and dtype."""
    shape = (1, 2, 64, 128)
    generator = torch.Generator().manual_seed(0)
    x = (4 * torch.randn(shape, generator=generator)).to(torch.bfloat16)
    g = torch.randn((1, 1, 64, 128) if call == "broadcast_grad" else shape, generator=generator).to(torch.bfloat16)
    dtype = ttnn.float32 if call == "float32" else ttnn.bfloat16
    layouts = {"row_major": (ttnn.ROW_MAJOR_LAYOUT,) * 2, "mixed_layouts": (ttnn.ROW_MAJOR_LAYOUT, ttnn.TILE_LAYOUT)}
    grad_layout, input_layout = layouts.get(call, (ttnn.TILE_LAYOUT,) * 2)
    placements = _sigmoid_bw_placements(shape)
    shard = placements["height_sharded"][0]
    grad_memory, input_memory, output_memory = placements.get(
        call, (ttnn.DRAM_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG, None)
    )
    tt_g = ttnn.from_torch(g, dtype=dtype, layout=grad_layout, device=device, memory_config=grad_memory)
    tt_x = ttnn.from_torch(x, dtype=dtype, layout=input_layout, device=device, memory_config=input_memory)
    memory = {} if output_memory is None else {"memory_config": output_memory}

    result, error, trace, launched = _sigmoid_bw_device_operations(lambda: ttnn.sigmoid_bw(tt_g, tt_x, **memory)[0])

    assert ("UnaryBackwardOpType::SIGMOID_BW_BF16" in launched) == (call in SIGMOID_BW_FUSED_CALLS), trace
    if call not in SIGMOID_BW_FUSED_CALLS:
        # Main's single program's refusal of a call (some refuse row-major operands) is its tests' to judge; a result it
        # returns is held to torch as the fused program's is.
        if error is None:
            assert list(result.shape) == list(shape)
            # Only where torch's result is finite: at NaN inputs and poles main's single program stores its own values.
            expected = _sigmoid_bw_torch_gradient(g.expand(shape), x)
            finite = torch.isfinite(expected)
            assert_with_pcc(expected[finite], ttnn.to_torch(result).float()[finite], 0.999)
        return
    assert error is None, error
    expected = _sigmoid_bw_torch_gradient(g, x)
    assert_with_pcc(expected, ttnn.to_torch(result).float(), 0.999)
    if call != "interleaved_tiles" and call in SIGMOID_BW_FUSED_CALLS:
        # A layout moves tiles, not arithmetic: the result is where the call asks for it, and is the
        # interleaved call's bit for bit, so the exhaustive test's accuracy holds in every layout.
        placement = tt_x.memory_config()
        if output_memory is not None:
            placement = ttnn.from_torch(
                g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=output_memory
            ).memory_config()
        assert result.memory_config() == placement
        interleaved = ttnn.sigmoid_bw(
            ttnn.from_torch(g, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
            ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device),
        )[0]
        assert torch.equal(ttnn.to_torch(result).view(torch.int16), ttnn.to_torch(interleaved).view(torch.int16))
