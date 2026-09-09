# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Refinement 5 — the `dtype x output_dtype` cartesian and the compute config.

tilize does no arithmetic, so "precision" here means something narrower and
stricter than it does for a compute op: the ONLY error a correct implementation
may show is the OUTPUT FORMAT's own quantization. Everything else must be
bit-identical, and the assertions below say so with `torch.equal` rather than a
tolerance wherever the pair is exactly representable.

Three cells in this matrix are not list widenings and are pinned individually,
because each one is a specific piece of compute-config plumbing that a later
refinement could silently drop while every other cell stayed green:

  1. `fp32 -> fp32` must be BIT-IDENTICAL. It needs all three of
     `Fp32Mode::Lossless`, `fp32_dest_acc_en=true` and
     `UnpackToDestMode::UnpackToDestFp32` on cb_input_rows; with any one
     missing the datum still round-trips through SrcA's tf32 and comes back
     ~1e-3 off. `test_tilize_fp32_lossless_needs_all_three_legs` asserts the
     plan/descriptor carries all three, so a regression is caught at the
     descriptor and not only in the values.
  2. `uint8` must be BIT-IDENTICAL, and only is because of the SrcB ALU-format
     repair (`needs_srcb_alu_format_repair`). Without it every output datum is
     zero on Wormhole B0 — see the EXCLUSIONS comment in tilize.py for the
     mechanism.
  3. `fp32 -> bf16 / bfp` must NOT be tagged UnpackToDestFp32: fast tilize is
     live there and static_asserts the tag is absent.

Device comes from the directory conftest's module-scoped `device` fixture.
"""

from __future__ import annotations

import pytest
import torch
import ttnn

from models.common.utility_functions import comp_allclose
from tests.ttnn.utils_for_testing import check_with_pcc
from ttnn.operations.tilize import tilize
from ttnn.operations.tilize.tilize_program_descriptor import (
    CB_INPUT_ROWS,
    create_program_descriptor,
    derive_plan,
)

_DRAM = ttnn.DRAM_MEMORY_CONFIG
_L1 = ttnn.L1_MEMORY_CONFIG

# The torch dtype a readback of each output format arrives in / is compared at.
# Block float has no torch form and reads back as bf16.
_TORCH = {
    ttnn.bfloat16: torch.bfloat16,
    ttnn.bfloat8_b: torch.bfloat16,
    ttnn.bfloat4_b: torch.bfloat16,
    ttnn.float32: torch.float32,
    ttnn.uint32: torch.int32,
    ttnn.uint16: torch.int32,
    ttnn.int32: torch.int32,
    ttnn.uint8: torch.uint8,
}

# (mode, threshold) per transition. Mirrors the golden suite's
# `helpers._transition_tolerance`: exact wherever the output format represents
# the input values losslessly, and a PCC floor only for the genuinely lossy
# block-float targets. Kept in this file rather than imported so the unit suite
# does not depend on the external benchmark.
# MEASURED against a host-side `ttnn.from_torch(..., TILE_LAYOUT)` conversion of the
# same tensor (probes/probe_042.py, probe_043.py): into `bfloat8_b` the op is exactly
# as good as the format allows (op PCC 0.999971 == host 0.999971), while into
# `bfloat4_b` the DEVICE packer's 3-bit-mantissa rounding costs ~0.009 PCC against the
# host (0.984 vs 0.993). That gap is why the bfp4 floor here has so little headroom;
# it is not reachable from the op (no packer rounding-mode field exists on
# `ComputeConfigDescriptor`, and the fp32-DEST x precise-pack sweep moves bfp4 by
# <2e-4).
_EXACT = ("exact", None)
_TOLERANCE = {
    (ttnn.bfloat16, ttnn.bfloat8_b): ("pcc", 0.99),
    (ttnn.bfloat16, ttnn.bfloat4_b): ("pcc", 0.98),
    (ttnn.float32, ttnn.bfloat8_b): ("pcc", 0.99),
    (ttnn.float32, ttnn.bfloat4_b): ("pcc", 0.98),
    (ttnn.float32, ttnn.bfloat16): ("pcc", 0.999),
}

# The 17 legal (dtype, output_dtype) pairs reachable on non-Blackhole silicon:
# the full TARGET cartesian minus the float<->int crosses, minus the integer
# width changes, minus fp8_e4m3 (Blackhole-only input).
PAIRS = [
    (ttnn.bfloat16, ttnn.bfloat16),
    (ttnn.bfloat16, ttnn.float32),
    (ttnn.bfloat16, ttnn.bfloat8_b),
    (ttnn.bfloat16, ttnn.bfloat4_b),
    (ttnn.float32, ttnn.bfloat16),
    (ttnn.float32, ttnn.float32),
    (ttnn.float32, ttnn.bfloat8_b),
    (ttnn.float32, ttnn.bfloat4_b),
    (ttnn.uint32, ttnn.uint32),
    (ttnn.uint32, ttnn.int32),
    (ttnn.int32, ttnn.uint32),
    (ttnn.int32, ttnn.int32),
    (ttnn.uint16, ttnn.uint16),
    (ttnn.uint8, ttnn.uint8),
]
PAIR_IDS = [f"{str(i)[9:].lower()}_to_{str(o)[9:].lower()}" for i, o in PAIRS]


def make_input(dtype, shape, distribution="randn"):
    """Source tensor per dtype. Integers span their full width where the width
    is small enough for it to matter — a uint8 test on [0,100) would not notice
    the high bit being dropped."""
    if dtype == ttnn.uint8:
        return torch.randint(0, 256, shape, dtype=torch.uint8)
    if dtype in (ttnn.uint32, ttnn.uint16):
        return torch.randint(0, 60000, shape, dtype=torch.int32)
    if dtype == ttnn.int32:
        return torch.randint(-(2**30), 2**30, shape, dtype=torch.int32)
    if dtype == ttnn.float32:
        t = (
            torch.rand(shape, dtype=torch.float32)
            if distribution == "rand"
            else torch.randn(shape, dtype=torch.float32)
        )
        return t
    t = torch.rand(shape) if distribution == "rand" else torch.randn(shape)
    return t.bfloat16()


def _to_device(torch_tensor, device, dtype, memory_config=_DRAM):
    return ttnn.from_torch(
        torch_tensor, dtype=dtype, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=memory_config
    )


def align_dtype(got, expected):
    """Cast `expected` to the readback's own dtype.

    The readback of an UNSIGNED tensor arrives as torch's own unsigned type
    (`torch.uint32` / `torch.uint16`), which torch refuses to PROMOTE against a
    signed one — so even `got != expected` raises. A same-width int->uint cast
    is the two's-complement reinterpretation the device performs, which is
    precisely what `int32 -> uint32` means in TARGET ("a signedness bit_cast at
    the same width").
    """
    return expected if got.dtype == expected.dtype else expected.to(got.dtype)


def check_pair(got, expected, in_dtype, out_dtype, context=""):
    """Assert the transition's own contract: bit-identity unless the OUTPUT
    format is lossy, in which case its PCC floor.

    See `align_dtype` for why the expected tensor is re-cast first.
    """
    expected = align_dtype(got, expected)
    mode, threshold = _TOLERANCE.get((in_dtype, out_dtype), _EXACT)
    if mode == "exact":
        assert torch.equal(got, expected), (
            f"{context}{in_dtype} -> {out_dtype} is exactly representable and must be BIT-IDENTICAL; "
            f"{int((got != expected).sum())} of {got.numel()} elements differ, "
            f"max_abs={float((got.to(torch.float64) - expected.to(torch.float64)).abs().max()):.6g}"
        )
        return
    passing, message = check_with_pcc(expected.float(), got.float(), threshold)
    assert passing, f"{context}{in_dtype} -> {out_dtype}: {message}"


# ---------------------------------------------------------------------------
# 1. The cartesian
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((1, 1, 32, 32), id="single_tile"),
        pytest.param((1, 1, 64, 128), id="multi_tile"),
        pytest.param((2, 3, 64, 96), id="leading_fold"),
    ],
)
@pytest.mark.parametrize("in_dtype,out_dtype", PAIRS, ids=PAIR_IDS)
def test_tilize_dtype_pair(device, shape, in_dtype, out_dtype):
    """Every legal (dtype, output_dtype) pair, on the plain interleaved path."""
    torch.manual_seed(0)
    torch_input = make_input(in_dtype, shape)
    tt_out = tilize(_to_device(torch_input, device, in_dtype), dtype=out_dtype)

    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert tt_out.dtype == out_dtype, f"output dtype must be the requested {out_dtype}, got {tt_out.dtype}"
    assert list(tt_out.shape) == list(shape)
    check_pair(ttnn.to_torch(tt_out), torch_input.to(_TORCH[out_dtype]), in_dtype, out_dtype)


# ---------------------------------------------------------------------------
# 2. The precision matrix (the op's single authoritative precision test)
# ---------------------------------------------------------------------------

_MATRIX_SHAPES = [
    pytest.param((1, 1, 32, 32), id="32x32_small"),
    pytest.param((1, 1, 32, 64), id="32x64"),
    pytest.param((1, 1, 64, 128), id="64x128"),
    pytest.param((1, 1, 128, 512), id="128x512"),
    pytest.param((1, 1, 256, 2048), id="256x2048_large"),
    pytest.param((1, 1, 32, 48), id="32x48_W_non_aligned"),
    pytest.param((1, 1, 48, 64), id="48x64_H_non_aligned"),
    pytest.param((1, 1, 48, 80), id="48x80_both_non_aligned"),
]


@pytest.mark.parametrize("distribution", [pytest.param("rand", id="uniform"), pytest.param("randn", id="normal")])
@pytest.mark.parametrize("in_dtype,out_dtype", PAIRS, ids=PAIR_IDS)
@pytest.mark.parametrize("shape", _MATRIX_SHAPES)
def test_tilize_precision_matrix(device, shape, in_dtype, out_dtype, distribution):
    """All metrics printed for every cell; the assertion is the transition's own
    contract (bit-identity, or the output format's PCC floor).

    Non-tile-aligned shapes go through the padding path — padding is opt-in, so
    they are called with `pad_value=0`, and the LOGICAL readback is still the
    input, which is what is compared here.
    """
    torch.manual_seed(42)
    torch_input = make_input(in_dtype, shape, distribution)
    non_aligned = shape[-2] % 32 or shape[-1] % 32
    kwargs = {"pad_value": 0} if non_aligned else {}

    tt_out = tilize(_to_device(torch_input, device, in_dtype), dtype=out_dtype, **kwargs)
    got = ttnn.to_torch(tt_out)
    expected = align_dtype(got, torch_input.to(_TORCH[out_dtype]))

    a, e = got.to(torch.float64), expected.to(torch.float64)
    abs_err = (a - e).abs()
    rel_rms = float(abs_err.pow(2).mean().sqrt() / e.pow(2).mean().sqrt().clamp(min=1e-10))
    _, allclose_msg = comp_allclose(e, a)
    print(
        f"\n[precision-matrix] shape={tuple(shape)} {in_dtype}->{out_dtype} dist={distribution}"
        f"\n  max_abs   = {float(abs_err.max()):.6g}"
        f"\n  median    = {float(abs_err.median()):.6g}"
        f"\n  p99       = {float(torch.quantile(abs_err.flatten().float(), 0.99)):.6g}"
        f"\n  rel_rms   = {rel_rms:.6g}"
        f"\n  n_differ  = {int((got != expected).sum())}/{got.numel()}"
        f"\n  {allclose_msg}"
    )
    check_pair(got, expected, in_dtype, out_dtype, context=f"shape={tuple(shape)} dist={distribution}: ")


# ---------------------------------------------------------------------------
# 3. The three descriptor legs the values alone would not pin
# ---------------------------------------------------------------------------


def _plan_and_config(device, in_dtype, out_dtype, shape=(1, 1, 64, 128), compute_kernel_config=None):
    torch_input = make_input(in_dtype, shape)
    tt_in = _to_device(torch_input, device, in_dtype)
    tt_out = tilize(tt_in, dtype=out_dtype, compute_kernel_config=compute_kernel_config)
    grid = device.compute_with_storage_grid_size()
    plan = derive_plan(tt_in, tt_out, low_l1=False, grid=grid)
    descriptor = create_program_descriptor(tt_in, tt_out, compute_kernel_config=compute_kernel_config)
    compute = [k for k in descriptor.kernels if "compute" in str(k.kernel_source)]
    return plan, descriptor, (compute[0].config if compute else None), torch_input, tt_out


def test_tilize_fp32_lossless_needs_all_three_legs(device):
    """fp32 -> fp32 carries Lossless + fp32 DEST + UnpackToDestFp32 together.

    Asserted at the descriptor, not only in the values: a refinement that drops
    one leg would turn a bit-identical relay into a ~1e-3 one, and only this
    assertion says WHICH leg went missing.
    """
    plan, _, config, torch_input, tt_out = _plan_and_config(device, ttnn.float32, ttnn.float32)
    assert plan.lossless_fp32, "fp32 -> fp32 must take the lossless path"
    assert plan.fp32_dest_acc_en, "Fp32Mode::Lossless static_asserts DST_ACCUM_MODE"
    modes = list(config.unpack_to_dest_mode)
    assert (
        modes[CB_INPUT_ROWS] == ttnn.UnpackToDestMode.UnpackToDestFp32
    ), "cb_input_rows must be tagged UnpackToDestFp32 or the unpacker truncates fp32 to tf32 via SrcA"
    assert torch.equal(ttnn.to_torch(tt_out), torch_input), "fp32 -> fp32 must be bit-identical"


@pytest.mark.parametrize(
    "out_dtype",
    [ttnn.bfloat16, ttnn.bfloat8_b, ttnn.bfloat4_b],
    ids=["to_bf16", "to_bfp8", "to_bfp4"],
)
def test_tilize_fp32_narrowing_is_not_tagged(device, out_dtype):
    """fp32 -> a NARROWER float must NOT tag cb_input_rows.

    Fast tilize is available there (the output is not Float32) and its own
    static_assert requires `UnpackToDestMode::Default` on the input CB —
    combining the two silently corrupts the output. So the tag is not merely
    unnecessary here, it is forbidden.
    """
    plan, _, config, _, _ = _plan_and_config(device, ttnn.float32, out_dtype)
    assert not plan.lossless_fp32
    modes = list(config.unpack_to_dest_mode)
    assert not modes or modes[CB_INPUT_ROWS] == ttnn.UnpackToDestMode.Default


def test_tilize_uint8_takes_the_srcb_repair(device):
    """uint8 is bit-exact, and it is the SrcB ALU-format repair that makes it so.

    The plan flag is asserted alongside the values because without the repair
    the failure is TOTAL (every datum zero) rather than approximate — a very
    loud symptom with a very quiet cause.
    """
    plan, _, _, torch_input, tt_out = _plan_and_config(device, ttnn.uint8, ttnn.uint8)
    assert plan.repair_srcb_alu_format, "uint8 input must request the SrcB ALU-format repair"
    assert plan.fp32_dest_acc_en, "the LLK requires fp32 DEST for Int8/UInt8 formats on a Src register"
    assert torch.equal(ttnn.to_torch(tt_out), torch_input)


@pytest.mark.parametrize(
    "in_dtype,out_dtype,expect_fp32_dest",
    [
        (ttnn.bfloat16, ttnn.bfloat16, False),
        (ttnn.bfloat16, ttnn.float32, False),
        (ttnn.float32, ttnn.bfloat16, True),
        (ttnn.float32, ttnn.float32, True),
        (ttnn.uint32, ttnn.uint32, False),
        (ttnn.uint16, ttnn.uint16, False),
        (ttnn.uint8, ttnn.uint8, True),
    ],
    ids=["bf16", "bf16_to_fp32", "fp32_to_bf16", "fp32", "uint32", "uint16", "uint8"],
)
def test_tilize_fp32_dest_acc_is_derived(device, in_dtype, out_dtype, expect_fp32_dest):
    """fp32 DEST is turned on exactly where the datapath requires it — and the
    bfloat16 diagonal (the Phase 0 path) still gets the 16-bit DEST it always
    had, so no prior cell silently changed its compute config."""
    plan, _, _, _, _ = _plan_and_config(device, in_dtype, out_dtype)
    assert plan.fp32_dest_acc_en == expect_fp32_dest


# ---------------------------------------------------------------------------
# 4. compute_kernel_config
# ---------------------------------------------------------------------------


def test_tilize_compute_kernel_config_defaults_are_unchanged(device):
    """Passing nothing reproduces the pre-refinement descriptor: HiFi4, no
    approx, no full sync, fp32 DEST from the dtype pair alone."""
    _, _, config, _, _ = _plan_and_config(device, ttnn.bfloat16, ttnn.bfloat16)
    assert config.math_fidelity == ttnn.MathFidelity.HiFi4
    assert not config.math_approx_mode
    assert not config.dst_full_sync_en
    assert not config.fp32_dest_acc_en


@pytest.mark.parametrize(
    "fidelity",
    [ttnn.MathFidelity.HiFi4, ttnn.MathFidelity.HiFi3, ttnn.MathFidelity.HiFi2, ttnn.MathFidelity.LoFi],
    ids=["HiFi4", "HiFi3", "HiFi2", "LoFi"],
)
@pytest.mark.parametrize("dst_full_sync_en", [False, True], ids=["half_sync", "full_sync"])
@pytest.mark.parametrize("fp32_acc", [False, True], ids=["bf16_acc", "fp32_acc"])
def test_tilize_compute_kernel_config_is_honored(device, fidelity, dst_full_sync_en, fp32_acc):
    """Every user knob reaches the descriptor AND the result is unchanged.

    Both halves matter. tilize does no arithmetic, so no setting here may move a
    single bit — but `dst_full_sync_en=True` DOES change the code path (it
    disables fast tilize), which is exactly the kind of silent path switch that
    must still produce identical bytes.
    """
    shape = (1, 1, 64, 128)
    cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=fidelity, fp32_dest_acc_en=fp32_acc, dst_full_sync_en=dst_full_sync_en
    )
    torch.manual_seed(3)
    torch_input = make_input(ttnn.bfloat16, shape)
    tt_in = _to_device(torch_input, device, ttnn.bfloat16)

    baseline = ttnn.to_torch(tilize(tt_in))
    tuned = tilize(tt_in, compute_kernel_config=cfg)
    descriptor = create_program_descriptor(tt_in, tuned, compute_kernel_config=cfg)
    config = [k for k in descriptor.kernels if "compute" in str(k.kernel_source)][0].config

    assert config.math_fidelity == fidelity
    assert bool(config.dst_full_sync_en) == dst_full_sync_en
    assert bool(config.fp32_dest_acc_en) == fp32_acc
    assert torch.equal(ttnn.to_torch(tuned), baseline), "no compute-config setting may change a byte re-lay's bytes"


def test_tilize_compute_kernel_config_cannot_disable_required_fp32_dest(device):
    """`fp32_dest_acc_en=False` is not honoured where the dtype pair requires it.

    Not a knob the caller is entitled to turn: at fp32 -> fp32 a 16-bit DEST is
    a WRONG ANSWER from a value-preserving op (and fails the helper's own
    static_assert), not a cheaper approximation.
    """
    cfg = ttnn.WormholeComputeKernelConfig(fp32_dest_acc_en=False)
    _, _, config, torch_input, tt_out = _plan_and_config(device, ttnn.float32, ttnn.float32, compute_kernel_config=cfg)
    assert config.fp32_dest_acc_en
    assert torch.equal(ttnn.to_torch(tt_out), torch_input)


# ---------------------------------------------------------------------------
# 5. The new dtypes crossed with the refinements that came before
# ---------------------------------------------------------------------------

_CROSS_PAIRS = [
    pytest.param(ttnn.float32, ttnn.float32, id="fp32"),
    pytest.param(ttnn.bfloat16, ttnn.float32, id="bf16_to_fp32"),
    pytest.param(ttnn.bfloat16, ttnn.bfloat8_b, id="bf16_to_bfp8"),
    pytest.param(ttnn.uint32, ttnn.uint32, id="uint32"),
    pytest.param(ttnn.uint16, ttnn.uint16, id="uint16"),
    pytest.param(ttnn.uint8, ttnn.uint8, id="uint8"),
]


@pytest.mark.parametrize("in_dtype,out_dtype", _CROSS_PAIRS)
@pytest.mark.parametrize(
    "shape,pad_value",
    [
        pytest.param((1, 1, 33, 50), 0, id="hw_non_aligned_zero"),
        pytest.param((1, 1, 48, 64), 7, id="h_non_aligned_positive"),
        pytest.param((1, 1, 32, 40), -3, id="w_non_aligned_negative"),
    ],
)
def test_tilize_dtype_x_padding(device, in_dtype, out_dtype, shape, pad_value):
    """Padding (Refinement 2) at every element width.

    `pad_fill_word` encodes the fill in the INPUT dtype's bit pattern at the
    input's element width, so this is the test that would catch a fill written
    at the wrong width (a negative integer fill truncating, an fp32 fill
    written as 16 bits). Both the logical view and the pad region are checked.
    """
    torch.manual_seed(1)
    torch_input = make_input(in_dtype, shape)
    if in_dtype in (ttnn.uint8, ttnn.uint16) and pad_value < 0:
        # An unsigned dtype narrower than 32 bits refuses a negative fill (see
        # EXCLUSIONS); `test_tilize_unsigned_negative_pad_is_refused` pins that.
        pad_value = abs(pad_value)
    tt_out = tilize(_to_device(torch_input, device, in_dtype), dtype=out_dtype, pad_value=pad_value)

    compare_dtype = _TORCH[out_dtype]
    check_pair(ttnn.to_torch(tt_out), torch_input.to(compare_dtype), in_dtype, out_dtype, context="logical view: ")

    padded_shape = [int(d) for d in list(shape)]
    padded_shape[-2] = ((padded_shape[-2] + 31) // 32) * 32
    padded_shape[-1] = ((padded_shape[-1] + 31) // 32) * 32
    expected = torch.nn.functional.pad(
        torch_input.to(compare_dtype),
        tuple(j for i in reversed(range(len(shape))) for j in (0, padded_shape[i] - shape[i])),
        value=pad_value,
    )
    check_pair(tt_out.cpu().to_torch_with_padded_shape(), expected, in_dtype, out_dtype, context="pad region: ")


@pytest.mark.parametrize("in_dtype,out_dtype", _CROSS_PAIRS)
def test_tilize_dtype_x_height_sharded(device, in_dtype, out_dtype):
    """A native (zero-copy) L1 shard at every element width.

    The shard's CB inherits the tensor's page size, so an element-width bug
    shows up here as a shifted shard rather than as a wrong value.
    """
    shape = (1, 1, 256, 128)
    grid = device.compute_with_storage_grid_size()
    if grid.x < 4:
        pytest.skip("needs a 4-wide compute grid")
    cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    mem = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(cores, [64, 128], ttnn.ShardOrientation.ROW_MAJOR),
    )
    torch.manual_seed(2)
    torch_input = make_input(in_dtype, shape)
    tt_in = _to_device(torch_input, device, in_dtype, memory_config=mem)
    tt_out = tilize(tt_in, memory_config=mem, dtype=out_dtype)
    check_pair(ttnn.to_torch(tt_out), torch_input.to(_TORCH[out_dtype]), in_dtype, out_dtype)


@pytest.mark.parametrize("in_dtype,out_dtype", _CROSS_PAIRS)
def test_tilize_dtype_x_low_l1_ab(device, in_dtype, out_dtype):
    """`low_l1` is a buffering knob, so both settings must be BIT-IDENTICAL to
    each other at every dtype — including the lossy block-float target, where
    "same PCC" would not be enough (a different block width must not move a
    single shared exponent)."""
    shape = (1, 1, 32, 2048)
    torch.manual_seed(4)
    torch_input = make_input(in_dtype, shape)
    tt_in = _to_device(torch_input, device, in_dtype)
    wide = ttnn.to_torch(tilize(tt_in, dtype=out_dtype))
    narrow = ttnn.to_torch(tilize(tt_in, dtype=out_dtype, low_l1=True))
    assert torch.equal(wide, narrow), (
        f"low_l1 changed the result at {in_dtype} -> {out_dtype}: "
        f"{int((wide != narrow).sum())} of {wide.numel()} elements differ"
    )


@pytest.mark.parametrize("in_dtype,out_dtype", _CROSS_PAIRS)
@pytest.mark.parametrize("tile_h", [16, 4, 1], ids=["tile16", "tile4", "tile1"])
def test_tilize_dtype_x_tiny_tile(device, in_dtype, out_dtype, tile_h):
    """Tiny tiles (Refinement 4) at every element width. A sub-32 tile drops off
    the fast-tilize path entirely, so this is the combination that exercises the
    REGULAR tilize LLK for the float dtypes as well as the integer ones."""
    shape = (1, 1, 64, 128)
    torch.manual_seed(5)
    torch_input = make_input(in_dtype, shape)
    if tile_h == 16 and out_dtype in (ttnn.bfloat8_b, ttnn.bfloat4_b):
        pytest.skip(
            "block-float output at tile_height=16 is an EXCLUSIONS cell; "
            "test_tilize_block_float_x_tiny_tile pins that refusal"
        )
    tt_out = tilize(_to_device(torch_input, device, in_dtype), dtype=out_dtype, tile=ttnn.Tile([tile_h, 32]))
    assert int(tt_out.tile.tile_shape[0]) == tile_h
    check_pair(ttnn.to_torch(tt_out), torch_input.to(_TORCH[out_dtype]), in_dtype, out_dtype)


@pytest.mark.parametrize("out_dtype", [ttnn.bfloat8_b, ttnn.bfloat4_b], ids=["bfp8", "bfp4"])
@pytest.mark.parametrize("tile_h", [32, 16, 8, 4, 2, 1], ids=lambda h: f"tile{h}")
def test_tilize_block_float_x_tiny_tile(device, out_dtype, tile_h, expect_error):
    """Block-float OUTPUT at every tile height — and the ONE height that is
    refused.

    `tile_height == 16` is the only height that is `partial_face` (< 32 tall)
    while still having a FULL 16-row face, and `llk_pack.h`'s partial-face BFP
    MOP is written for sub-16-row faces: it packs one face instead of
    `num_faces`. The cell is in EXCLUSIONS, so the refusal is the registry's,
    and this test pins BOTH halves — that 16 is refused, and that every other
    height still produces a correct block-float tile. Without the second half an
    over-broad exclusion would look identical.
    """
    from ttnn.operations._op_contract import ExcludedCell

    shape = (1, 1, 64, 128)
    torch.manual_seed(6)
    torch_input = make_input(ttnn.bfloat16, shape)
    tt_in = _to_device(torch_input, device, ttnn.bfloat16)
    if tile_h == 16:
        with expect_error(ExcludedCell, "unsupported combination"):
            tilize(tt_in, dtype=out_dtype, tile=ttnn.Tile([tile_h, 32]))
        return
    tt_out = tilize(tt_in, dtype=out_dtype, tile=ttnn.Tile([tile_h, 32]))
    check_pair(ttnn.to_torch(tt_out), torch_input.to(_TORCH[out_dtype]), ttnn.bfloat16, out_dtype)


@pytest.mark.parametrize(
    "in_dtype,refused",
    [
        pytest.param(ttnn.uint8, True, id="uint8_refused"),
        pytest.param(ttnn.uint16, True, id="uint16_refused"),
        pytest.param(ttnn.uint32, False, id="uint32_allowed"),
        pytest.param(ttnn.int32, False, id="int32_allowed"),
    ],
)
def test_tilize_unsigned_negative_pad(device, in_dtype, refused, expect_error):
    """A negative fill on a narrow UNSIGNED dtype is refused; uint32 and int32
    are not.

    The asymmetry is the assertion. `uint32` is left in because its
    two's-complement bits ARE the negative value asked for when reinterpreted
    at the same width, which is verifiable; `uint16` / `uint8` cannot be shown
    right at any width and are refused. An exclusion that swept up all unsigned
    dtypes on principle would pass a test that only checked the refusals.
    """
    from ttnn.operations._op_contract import ExcludedCell

    shape = (1, 1, 33, 50)
    torch.manual_seed(8)
    torch_input = make_input(in_dtype, shape)
    tt_in = _to_device(torch_input, device, in_dtype)
    if refused:
        with expect_error(ExcludedCell, "unsupported combination"):
            tilize(tt_in, pad_value=-3)
        return
    tt_out = tilize(tt_in, pad_value=-3)
    check_pair(ttnn.to_torch(tt_out), torch_input, in_dtype, in_dtype, context="logical view: ")


def test_tilize_empty_tensor_is_a_zero_work_dispatch(device):
    """A 0-element input still dispatches, and dispatches exactly once.

    An empty tensor has no output tiles, so the column solve has no target to
    divide by — an unguarded empty grid is a host-side ZeroDivisionError rather
    than a wrong answer. The plan degenerates to one core owning ZERO blocks:
    every kernel's block loop runs no iterations and no NoC access is issued
    against the zero-page buffers.
    """
    torch_input = torch.rand(0, dtype=torch.float32)
    tt_in = _to_device(torch_input, device, ttnn.float32)
    tt_out = tilize(tt_in, pad_value=0.0, output_padded_shape=[32, 0])

    assert tt_out.layout == ttnn.TILE_LAYOUT
    assert list(tt_out.shape) == [0]
    assert ttnn.to_torch(tt_out).numel() == 0

    grid = device.compute_with_storage_grid_size()
    plan = derive_plan(tt_in, tt_out, low_l1=False, grid=grid, pad_value=0.0)
    assert plan.num_blocks_total == 0, "an empty grid must carry no blocks"
    assert max(a[2] for a in plan.assignment) == 0, "no core may be handed work"


def test_tilize_retile_with_a_cast_is_refused(device, expect_error):
    """A re-tile carries no packer, so it cannot carry a `dtype=` cast.

    `EXCLUSIONS` is what turns that into the registry's own refusal rather than
    a RuntimeError from the plan; the identity re-tile (in == out height) is
    used because it is the one retile geometry that runs on every arch.
    """
    from ttnn.operations._op_contract import ExcludedCell

    torch_input = make_input(ttnn.bfloat16, (1, 1, 64, 128))
    tt_in = ttnn.from_torch(
        torch_input, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_DRAM
    )
    with expect_error(ExcludedCell, "unsupported combination"):
        tilize(tt_in, dtype=ttnn.float32, tile=ttnn.Tile([32, 32]))
    # ...and the no-cast identity re-tile is still fine.
    out = tilize(tt_in, dtype=ttnn.bfloat16, tile=ttnn.Tile([32, 32]))
    assert torch.equal(ttnn.to_torch(out), torch_input)


def test_fp8_input_support_follows_the_silicon(device, expect_error):
    """`fp8_e4m3` is the ONE arch-conditional value in SUPPORTED, and the
    registry — not the kernels — is what states it.

    On Wormhole the value is not merely untested, it is UNREACHABLE: TT-Metal
    refuses the allocation itself —
    `distributed_tensor_apis.cpp:47: mesh_device.arch() == tt::ARCH::BLACKHOLE,
    "FP8_E4M3 is only supported on Blackhole hardware"` — so no caller can even
    build the input tensor, let alone hand it to this op. A SUPPORTED entry for
    it here would be a claim about an input that cannot exist.

    `tilize.py` therefore builds `SUPPORTED["dtype"]` from an arch-independent
    list plus `fp8_e4m3` when `ARCH_HAS_FP8_TILIZE`. This pins BOTH halves so
    neither can drift:

      * the claim tracks the silicon (present iff the datapath is), so a scored
        run never reports fp8 cells as SUPPORTED-but-unexercised on a box that
        skips or refuses every one of them, and
      * where the claim is absent, the op's own gate refuses an fp8 request with
        `UnsupportedAxisValue` — checked below against a *spec*, since a live
        fp8 tensor is unconstructible here.

    Every OTHER dtype in the axis is arch-independent (the kernels name no
    format), which is why this is one test and not a matrix.
    """
    from ttnn.operations.tilize import ARCH_HAS_FP8_TILIZE, SUPPORTED

    is_blackhole = "blackhole" in str(ttnn.get_arch_name()).lower()
    assert ARCH_HAS_FP8_TILIZE == is_blackhole, "the fp8 capability flag must be read off the arch, not hardcoded"
    assert (
        ttnn.fp8_e4m3 in SUPPORTED["dtype"]
    ) == ARCH_HAS_FP8_TILIZE, "SUPPORTED['dtype'] must claim fp8_e4m3 exactly where the datapath exists"
    # The arch-independent half of the axis is claimed unconditionally.
    for dt in (ttnn.bfloat16, ttnn.float32, ttnn.uint32, ttnn.int32, ttnn.uint16, ttnn.uint8):
        assert dt in SUPPORTED["dtype"]

    if ARCH_HAS_FP8_TILIZE:
        pytest.skip("fp8_e4m3 is claimed here; its VALUES are covered by the dtype matrix")

    # Off-Blackhole: the framework refuses the tensor before the op is reachable.
    torch_input = torch.randn(1, 1, 32, 64, dtype=torch.float32)
    with expect_error(RuntimeError, "FP8_E4M3 is only supported on Blackhole"):
        ttnn.from_torch(
            torch_input,
            dtype=ttnn.fp8_e4m3,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_DRAM,
        )

    # ...and the op's registry gate refuses the request in its own voice, which
    # is what a Blackhole-authored call replayed here would hit. A dtype= cast TO
    # fp8 exercises the same axis (`output_dtype` is derived from `dtype=`) with
    # no unconstructible input needed.
    from ttnn.operations._op_contract import UnsupportedAxisValue

    tt_in = _to_device(torch.randn(1, 1, 32, 64, dtype=torch.bfloat16), device, ttnn.bfloat16)
    with expect_error(UnsupportedAxisValue, "output_dtype="):
        tilize(tt_in, dtype=ttnn.fp8_e4m3)
