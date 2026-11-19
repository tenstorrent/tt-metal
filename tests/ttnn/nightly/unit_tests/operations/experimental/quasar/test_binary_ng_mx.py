# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
binary_ng with MX (OCP microscaling) tensors on Quasar's native factory, checked exactly.

The unpacker decodes MX to Float16_b, the kernel computes in bfloat16, and the packer encodes MX. The oracle takes the
same steps from the bytes that the device holds (binary_ng_mx_test_utils.py): the reference decoder of
mx_reference_codec.py, the kernel's bfloat16 model, and the reference encoder. It never reads an MX value through the
ttnn MX codec. Each case first asserts that the native factory ran its op (the op bound kernels_qsr/ sources only),
and then compares the output with the oracle: 0 differing elements.

- test_add_mx_in_mx_out, test_add_bf16_in_mx_out, test_add_mx_in_bf16_out: every MX dtype through the unpacker and
  the packer, on the FPU kernel (add), in DRAM.
- test_multiply_mx_sfpu: the SFPU kernel (multiply) with MX.
- test_add_mx_placement: each MX tile size (1,056, 544 and 288 B) at each placement, with full batches.
- test_add_mx_outer_broadcast: an outer-dim broadcast, which the native slice holds.
- test_mx_program_cache: each op of a sequence of two (bfloat16 then MX, or two MX dtypes) builds its own program.
- test_mx_special_codes: the special codes of tt-regen-spec 4.2 for every MX dtype, through the unpacker and through
  the packer.
- test_mx_classes_run_native: an activation on a, on b or on the result, a float32 or another MX format on the other
  side, and a bfloat16 or float32 scalar with an MX output, on the FPU and the SFPU kernel; each MX format as the
  output of the classes that change the packer's encode.
- test_mx_fp32_in_special_codes: float32 into the packer: exact ties, values beside them, and the special blocks.
- test_mx_gate_refuses: the native gate refuses each MX case outside its slice, and the op raises.

Run on ttsim, the Quasar simulator. On Wormhole and Blackhole only
test_mx_rejected_off_quasar runs.
"""

import hashlib
import math

import pytest
import torch

import ttnn
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar import binary_ng_mx_test_utils as mx
from tests.ttnn.nightly.unit_tests.operations.experimental.quasar.binary_ng_quasar_test_utils import (
    _fpu_add_bf16,
    _fpu_aligned_sum,
    _sfpu_multiply_bf16,
)
from tt_py_test_utils_common.utility_functions import is_quasar

quasar_only = pytest.mark.skipif(not is_quasar(), reason="MX formats are Quasar only")

# One interleaved shape small enough for the simulator, and one that spreads tiles over several cores.
_SHAPES = [(32, 32), (4 * 32, 8 * 32)]
# 1,024 tiles: 32 on each of the 32 clusters, so each compute thread runs a full batch of MX tiles.
_SWEEP_SHAPE = (32 * 32, 32 * 32)


def _mx_input(device, x, dtype, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """(device tensor, bfloat16 values as the unpacker gives them) of an MX operand that ttnn.from_torch builds."""
    return mx.mx_operand(device, ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT), memory_config)


def _bf16_input(device, x, memory_config=ttnn.DRAM_MEMORY_CONFIG):
    """(device tensor, its bfloat16 values) of a bfloat16 operand."""
    t = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=memory_config)
    return t, ttnn.to_torch(t)


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_mx_in_mx_out(device, dtype, shape):
    torch.manual_seed(0)
    ta, a = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype)
    tb, b = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype)
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb)
    mx.assert_native(before, "add")

    assert tc.dtype == dtype
    wrong = mx.mx_output_wrong(tc, _fpu_add_bf16(a, b))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_bf16_in_mx_out(device, dtype, shape):
    # MX that the packer produces must decode by 4.2.
    torch.manual_seed(0)
    ta, a = _bf16_input(device, torch.randn(shape, dtype=torch.bfloat16))
    tb, b = _bf16_input(device, torch.randn(shape, dtype=torch.bfloat16))
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=dtype)
    mx.assert_native(before, "add")

    assert tc.dtype == dtype
    wrong = mx.mx_output_wrong(tc, _fpu_add_bf16(a, b))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


@quasar_only
@pytest.mark.parametrize("dtype", ttnn.MX_DTYPES)
@pytest.mark.parametrize("shape", _SHAPES)
def test_add_mx_in_bf16_out(device, dtype, shape):
    torch.manual_seed(0)
    ta, a = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype)
    tb, b = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype)
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=ttnn.bfloat16)
    mx.assert_native(before, "add")

    assert tc.dtype == ttnn.bfloat16
    wrong = mx.bf16_output_wrong(tc, _fpu_add_bf16(a, b))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


@quasar_only
@pytest.mark.parametrize("direction", ["mx_in_mx_out", "bf16_in_mx_out"])
def test_multiply_mx_sfpu(device, direction):
    """multiply runs the SFPU kernel, which drops the low 16 bits of the float32 product. One MX dtype is enough:
    the unpacker gives the kernel Float16_b for every MX dtype."""
    torch.manual_seed(0)
    if direction == "mx_in_mx_out":
        ta, a = _mx_input(device, torch.randn(_SWEEP_SHAPE, dtype=torch.float32), ttnn.mxint8)
        tb, b = _mx_input(device, torch.randn(_SWEEP_SHAPE, dtype=torch.float32), ttnn.mxint8)
    else:
        ta, a = _bf16_input(device, torch.randn(_SWEEP_SHAPE, dtype=torch.bfloat16))
        tb, b = _bf16_input(device, torch.randn(_SWEEP_SHAPE, dtype=torch.bfloat16))
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.multiply(ta, tb, dtype=ttnn.mxint8)
    mx.assert_native(before, "multiply")

    assert tc.dtype == ttnn.mxint8
    wrong = mx.mx_output_wrong(tc, _sfpu_multiply_bf16(a, b))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


def _dram_width(device, shard_shape):
    """Width shards over every DRAM bank. The shard grid of a DRAM tensor is in DRAM-bank coordinates."""
    g = device.dram_grid_size()
    banks = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))})
    spec = ttnn.ShardSpec(banks, shard_shape, ttnn.ShardOrientation.ROW_MAJOR)
    return ttnn.MemoryConfig(ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.DRAM, spec)


def _l1_height(shard_shape, cores):
    x0, y0, x1, y1 = cores
    return ttnn.create_sharded_memory_config(
        shard_shape,
        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange((x0, y0), (x1, y1))}),
        strategy=ttnn.ShardStrategy.HEIGHT,
        use_height_and_width_as_shard_shape=True,
    )


# The memory configs of the placements, built on the device.
_MEMORY = {
    "dram": lambda device: ttnn.DRAM_MEMORY_CONFIG,
    "l1": lambda device: ttnn.L1_MEMORY_CONFIG,
    # Two DRAM shards of 512 tiles, one per DRAM bank of the simulator.
    "dram_width": lambda device: _dram_width(device, [32 * 32, 16 * 32]),
    # Shards of 33 tiles on two cores: 32 borrowed, and 1 through the tail rings. The second shard holds 32 real tiles.
    "l1_height_33": lambda device: _l1_height([33 * 32, 32], (0, 0, 0, 1)),
}

# placement -> (shape, a, b, output).
# dram_interleaved: the linear reader and writer walk over DRAM pages, which are larger than the tile (64 B
# alignment), at a full batch (the conversion cases above hold at most 1 tile per cluster).
# l1_interleaved: the native factory borrows each operand's slice of its L1 bank.
# l1_height_uneven: borrowed shards, tail rings, and a last shard that holds fewer real tiles than the others.
# dram_width: DRAM shards over the NoC; a DRAM page of an MX tile is larger than the tile (64 B alignment).
# mixed: a and the output borrow their L1 slices, and b moves over the NoC from its DRAM pages.
_PLACEMENTS = {
    "dram_interleaved": (_SWEEP_SHAPE, "dram", "dram", "dram"),
    "l1_interleaved": (_SWEEP_SHAPE, "l1", "l1", "l1"),
    "l1_height_uneven": ((65 * 32, 32), "l1_height_33", "l1_height_33", "l1_height_33"),
    "dram_width": (_SWEEP_SHAPE, "dram_width", "dram_width", "dram_width"),
    "mixed": (_SWEEP_SHAPE, "l1", "dram", "l1"),
}


@quasar_only
@pytest.mark.parametrize("placement", list(_PLACEMENTS))
@pytest.mark.parametrize("dtype", [ttnn.mxint8, ttnn.mxfp4, ttnn.mxint2])
def test_add_mx_placement(device, dtype, placement):
    """The three MX tile sizes (mxint8 1,056 B, mxfp4 544 B, mxint2 288 B) through the reader, the rings and the
    writer at each placement, MX in and MX out, on the FPU kernel."""
    shape, a_mem, b_mem, out_mem = _PLACEMENTS[placement]
    torch.manual_seed(0)
    ta, a = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype, _MEMORY[a_mem](device))
    tb, b = _mx_input(device, torch.randn(shape, dtype=torch.float32), dtype, _MEMORY[b_mem](device))
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb, memory_config=_MEMORY[out_mem](device))
    mx.assert_native(before, f"add at {placement}")

    assert tc.dtype == dtype
    wrong = mx.mx_output_wrong(tc, _fpu_add_bf16(a, b))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


@quasar_only
def test_add_mx_outer_broadcast(device):
    """An outer-dim broadcast stays in the native slice: the gate refuses a subtile broadcast only. b has 1 where a has
    2 in dim 0, so the reader reads each tile of b twice through b's Map."""
    torch.manual_seed(0)
    ta, a = _mx_input(device, torch.randn((2, 2, 2 * 32, 2 * 32), dtype=torch.float32), ttnn.mxint8)
    tb, b = _mx_input(device, torch.randn((1, 2, 2 * 32, 2 * 32), dtype=torch.float32), ttnn.mxint8)
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb)
    mx.assert_native(before, "add with an outer broadcast")

    assert tc.dtype == ttnn.mxint8 and tuple(tc.shape) == tuple(ta.shape)
    wrong = mx.mx_output_wrong(tc, _fpu_add_bf16(a, b.expand(a.shape)))
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


# sequence -> the (input dtype, output dtype) of two adds of one shape that run one after the other on one device.
_CACHE_SEQUENCES = {
    # The same bfloat16 inputs to bfloat16, then to mxint8: only the output dtype differs, so it must be in the key.
    "bf16_then_mxint8": ((ttnn.bfloat16, ttnn.bfloat16), (ttnn.bfloat16, ttnn.mxint8)),
    # Two MX dtypes one after the other on one device (design risk 6: the ALU format state).
    "mxint8_then_mxfp4": ((ttnn.mxint8, ttnn.mxint8), (ttnn.mxfp4, ttnn.mxfp4)),
}


@quasar_only
@pytest.mark.parametrize("sequence", list(_CACHE_SEQUENCES))
def test_mx_program_cache(device, sequence):
    """Each add of the sequence builds its own program (the program cache grows by one) and gives the exact output."""
    shape = (4 * 32, 8 * 32)
    torch.manual_seed(0)
    x, y = torch.randn(shape, dtype=torch.float32), torch.randn(shape, dtype=torch.float32)
    for step, (in_dtype, out_dtype) in enumerate(_CACHE_SEQUENCES[sequence]):
        if in_dtype == ttnn.bfloat16:
            ta, a = _bf16_input(device, x.to(torch.bfloat16))
            tb, b = _bf16_input(device, y.to(torch.bfloat16))
        else:
            ta, a = _mx_input(device, x, in_dtype)
            tb, b = _mx_input(device, y, in_dtype)
        what = f"add {step} ({in_dtype} to {out_dtype})"
        entries = device.num_program_cache_entries()
        before = mx.kernel_sources()
        tc = ttnn.experimental.quasar.add(ta, tb, dtype=out_dtype)
        assert device.num_program_cache_entries() == entries + 1, f"{what} did not build its own program"
        mx.assert_native(before, what)

        assert tc.dtype == out_dtype
        golden = _fpu_add_bf16(a, b)
        wrong = mx.bf16_output_wrong(tc, golden) if out_dtype == ttnn.bfloat16 else mx.mx_output_wrong(tc, golden)
        assert not wrong, f"{what}: {len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


# The special codes of tt-regen-spec 4.2 for every MX dtype in both directions: each format has its own special
# values, and the unpacker's decode and the packer's encode carry every MX dtype. Each case runs add(a, 0), so the
# output is a as the device holds it. Codes and bits are written as raw bits, because a float input cannot reach some
# of them.
# - "<dtype>_in" writes the MX bytes of a and reads a bfloat16 output: the unpacker's decode alone. Tile t holds every
#   element code of the dtype, in code order, at the scale _DECODE_SCALES[t] in each block: the decode flush at S = 0,
#   1 and 2, the middle, the overflow to Inf at 253 and 254, and S = 255 (every datum NaN). So the case decodes the
#   largest code, the MX float Inf, NaN and subnormal codes, and the MXINT code -2^w at each of these scales.
# - "<dtype>_out" writes the bfloat16 bits of a and reads an MX output: the packer's encode alone. Tile 0 holds every
#   value of the dtype at S = 127, tile 1 the midpoint of each pair of neighbors and the bfloat16 values on each side of
#   it (ties to even, up, down), and tile 2 the blocks of _special_blocks. Each block of tiles 0 and 1 starts with the
#   largest value of the dtype, so that its scale is 127. The case also compares the 32 scale bytes of each tile with
#   the reference encoder's: a value compare cannot see every scale choice (an S = 0 and an S = 1 block can decode to
#   the same values), and no block of these tiles is all zero.
_DECODE_SCALES = (0, 1, 2, 126, 127, 128, 253, 254, 255)
_SPECIAL_CASES = {
    f"{mx.MX_FORMAT[d]}_{direction}": (d, direction) for direction in ("in", "out") for d in ttnn.MX_DTYPES
}


def _special_mx_tiles(dtype):
    """The bytes of one MX tile per scale of _DECODE_SCALES: element 32 b + i of a tile holds the code
    (32 b + i) mod 2^bits, and every block of tile t has the scale _DECODE_SCALES[t]."""
    codes = (torch.arange(32 * 32) % (1 << mx.element_bits(dtype))).reshape(32, 32)
    return torch.cat([mx.pack_tile([scale] * 32, codes, dtype) for scale in _DECODE_SCALES])


def _bf16_step(x, step):
    """The bfloat16 value `step` steps away from zero (toward zero if negative) from each element of `x`."""
    return (x.to(torch.bfloat16).view(torch.int16) + step).view(torch.bfloat16).float()


def _cyclic_blocks(anchor, values):
    """32 blocks: each the anchor, then the next 31 of `values`, cyclically."""
    values = list(values)
    return [[anchor] + [values[(31 * b + i) % len(values)] for i in range(31)] for b in range(32)]


def _special_blocks(top, smallest, vmax):
    """The special blocks of the encode, each from element 0 of its block. top: the power of 2 of the binade of the
    largest value vmax; smallest: the smallest positive value (the element subnormal step, or the MXINT LSB)."""
    inf, nan = math.inf, math.nan
    ladder = [1.0, 1.25, 1.5, 1.75, 1.875, 1.9375, 1.96875, 1.984375, 1.9921875]  # up to the top of a binade
    low = [1.0 + k / 128 for k in range(0, 128, 9)]  # 15 values in [1, 2)
    return [
        [inf, -inf] + [0.0] * 30,  # Inf and zeros only: S = 254, Inf saturates
        [nan, -nan] * 16,  # NaNs only: S = 255
        [nan, -nan, vmax, -vmax] + [vmax / 2**k for k in range(1, 29)],  # a NaN beside finite values
        [inf, -inf, vmax, -vmax] + [-vmax / 2**k for k in range(1, 29)],  # Inf beside finite values saturates
        [r * top * 2.0**100 for r in ladder] + [-r * top * 2.0**100 for r in ladder],  # saturation at a high scale
        # The bottom of the normal range: the S = 1 wrap of MXINT, and the clamp of S to 0 of an MX float.
        [r * 2.0**-126 for r in ladder] + [-r * 2.0**-126 for r in ladder] + [r * 2.0**-126 for r in low[:14]],
        [vmax] + [j * smallest / 4 for j in range(1, 32)],  # quarter steps of the smallest value: subnormals and ties
        [2.0**-125, -(2.0**-125)] + [r * 2.0**-126 for r in low],  # the lowest scales
    ]


def _blocks_tile(blocks, generator):
    """One 32x32 bfloat16 tile: block b holds the values blocks[b], repeated to fill its 32 elements, so that they alone
    set its scale. The blocks after them hold random values."""
    tile = torch.randn((32, 32), generator=generator).to(torch.bfloat16)
    for b, values in enumerate(blocks):
        rows, cols = mx.block_cells(b)
        filled = [values[i % len(values)] for i in range(32)]
        tile[rows, cols] = torch.tensor(filled, dtype=torch.float32).to(torch.bfloat16)
    return tile


def _special_bf16_tiles(dtype):
    """Three 32x32 bfloat16 tiles side by side: every value of the dtype, the midpoints and their neighbors, and the
    special blocks."""
    values = mx.code_values(dtype)
    pos = values[values.isfinite() & (values >= 0)].unique()  # sorted, 0 to the largest finite value
    vmax, smallest = float(pos[-1]), float(pos[1])
    mid = (pos[:-1] + pos[1:]) / 2
    near = torch.cat([_bf16_step(mid, 0), _bf16_step(mid, 1), _bf16_step(mid, -1)])
    g = torch.Generator().manual_seed(1)
    tiles = [
        _blocks_tile(_cyclic_blocks(vmax, pos.tolist() + (-pos).tolist()), g),
        _blocks_tile(_cyclic_blocks(vmax, near.tolist() + (-near).tolist()), g),
        _blocks_tile(_special_blocks(2.0 ** math.floor(math.log2(vmax)), smallest, vmax), g),
    ]
    return torch.cat(tiles, dim=1)


class _WrongAsPredicted(Exception):
    """The output differs from the oracle on exactly the predicted elements."""


def _digest(indices):
    """(count, the first 16 hex digits of the sha256) of a sorted list of indices."""
    return len(indices), hashlib.sha256(",".join(map(str, indices)).encode()).hexdigest()[:16]


_EXACT = (*_digest([]), [])

# case -> (count, sha256 prefix) of the sorted flat indices where the device's values differ from 4.2, and the
# [tile, block] of each scale byte that differs, as measured on ttsim v1.16.0. A case not here is exact.
_SPECIAL_WRONG = {}


def _special_marks(case):
    if case not in _SPECIAL_WRONG:
        return []
    return [pytest.mark.xfail(raises=_WrongAsPredicted, strict=True, reason=f"ttsim differs from 4.2 on {case}")]


@quasar_only
@pytest.mark.parametrize("codes", [pytest.param(c, marks=_special_marks(c)) for c in _SPECIAL_CASES])
def test_mx_special_codes(device, codes):
    dtype, direction = _SPECIAL_CASES[codes]
    if direction == "in":
        raw = _special_mx_tiles(dtype)
        shape = (32, 32 * len(_DECODE_SCALES))
        ta, a = mx.mx_operand(device, mx.mx_host_tensor(raw, dtype, shape), ttnn.DRAM_MEMORY_CONFIG)
        tb, b = mx.mx_operand(device, mx.mx_host_tensor(torch.zeros_like(raw), dtype, shape), ttnn.DRAM_MEMORY_CONFIG)
        out_dtype = ttnn.bfloat16
    else:
        x = _special_bf16_tiles(dtype)
        shape = tuple(x.shape)
        ta, a = _bf16_input(device, x)
        tb, b = _bf16_input(device, torch.zeros_like(x))
        out_dtype = dtype
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=out_dtype)
    mx.assert_native(before, f"add on {codes}")

    golden = _fpu_add_bf16(a, b)
    if direction == "in":
        wrong, scales = mx.bf16_output_wrong(tc, golden), []
    else:
        got, want = mx.device_bytes(tc), mx.encode(golden.float(), dtype)
        wrong = mx.wrong_elements(mx.decode(got, dtype, shape), mx.decode(want, dtype, shape))
        scales = torch.nonzero(mx.scale_bytes(got, dtype) != mx.scale_bytes(want, dtype)).tolist()
    found, predicted = (*_digest(wrong), scales), _SPECIAL_WRONG.get(codes, _EXACT)
    assert found == predicted, (
        f"{codes}: {len(wrong)} values differ from 4.2 (first at {wrong[:16]}), scale bytes differ at {scales[:16]}; "
        f"found {found}, predicted {predicted}"
    )
    if wrong or scales:
        raise _WrongAsPredicted(f"{codes}: {len(wrong)} values and {len(scales)} scale bytes differ, as predicted")


def _random(device, dtype, shape=(2 * 32, 2 * 32)):
    return ttnn.from_torch(torch.randn(shape, dtype=torch.float32), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)


_RELU = [ttnn.UnaryWithParam(ttnn.UnaryOpType.RELU)]


def _add_mx(device, **kwargs):
    return ttnn.experimental.quasar.add(_random(device, ttnn.mxint8), _random(device, ttnn.mxint8), **kwargs)


# refusal -> (the admitted control, the refused op). The control is the refused op without the feature that the gate
# refuses, so the case shows that the feature, not MX, sends the op away. No factory takes the refused op, so it raises.
_GATE_CASES = {
    "row_broadcast": (
        _add_mx,
        lambda d: ttnn.experimental.quasar.add(_random(d, ttnn.mxint8), _random(d, ttnn.mxint8, (1, 2 * 32))),
    ),
    "scalar": (_add_mx, lambda d: ttnn.experimental.quasar.add(_random(d, ttnn.mxint8), 2.0)),
    "mx_and_bf16": (
        _add_mx,
        lambda d: ttnn.experimental.quasar.add(_random(d, ttnn.mxint8), _random(d, ttnn.bfloat16)),
    ),
    "bf16_and_mx": (
        _add_mx,
        lambda d: ttnn.experimental.quasar.add(_random(d, ttnn.bfloat16), _random(d, ttnn.mxint8)),
    ),
    "two_mx_dtypes": (
        _add_mx,
        lambda d: ttnn.experimental.quasar.add(_random(d, ttnn.mxint8), _random(d, ttnn.mxfp8_e4m3)),
    ),
    # multiply runs the SFPU kernel, which keeps the scalar in a's format for an MX a, so only the MX refusal of a
    # scalar beside an MX input sends it away.
    "sfpu_scalar": (
        lambda d: ttnn.experimental.quasar.multiply(_random(d, ttnn.mxint8), _random(d, ttnn.mxint8)),
        lambda d: ttnn.experimental.quasar.multiply(_random(d, ttnn.mxint8), 2.0),
    ),
}

# The message of the factory selection when no factory takes an op.
_NO_FACTORY = "is not supported for a"


@quasar_only
@pytest.mark.parametrize("refusal", list(_GATE_CASES))
def test_mx_gate_refuses(device, expect_error, refusal):
    control, refused = _GATE_CASES[refusal]
    torch.manual_seed(0)
    before = mx.kernel_sources()
    control(device)
    mx.assert_native(before, f"{refusal}: the control")

    before = mx.kernel_sources()
    with expect_error(RuntimeError, _NO_FACTORY):
        refused(device)
    mx.assert_not_native(before, refusal)


def _sfpu_square_bf16(x):
    """SQUARE on the SFPU in a 16-bit DST: the float32 square with its low 16 bits dropped."""
    return (x.float() * x.float()).contiguous().view(torch.int32).bitwise_and(-65536).view(torch.float32)


# The kernel's bfloat16 result for each op: the FPU add rounds twice, the SFPU multiply truncates.
_MX_KERNEL = {"add": _fpu_add_bf16, "multiply": _sfpu_multiply_bf16}
_SQUARE = [ttnn.UnaryWithParam(ttnn.UnaryOpType.SQUARE)]
# class -> (a's format, the output format, the activations, the scalar or None), with mxint8 on the MX side. An
# activation runs in Float16_b beside an MX operand: a RELU on the result is the packer's (PACK_RELU), and a SQUARE on
# the result runs on the SFPU before the packer.
_MX_CLASS_SPECS = {
    "relu_a": (ttnn.mxint8, ttnn.mxint8, {"input_tensor_a_activations": _RELU}, None),
    "relu_b": (ttnn.mxint8, ttnn.mxint8, {"input_tensor_b_activations": _RELU}, None),
    "relu_post": (ttnn.mxint8, ttnn.mxint8, {"activations": _RELU}, None),
    "square_post": (ttnn.mxint8, ttnn.mxint8, {"activations": _SQUARE}, None),
    "bf16_in_relu_post": (ttnn.bfloat16, ttnn.mxint8, {"activations": _RELU}, None),
    "bf16_out_relu_post": (ttnn.mxint8, ttnn.bfloat16, {"activations": _RELU}, None),
    "fp32_out": (ttnn.mxint8, ttnn.float32, {}, None),
    "fp32_in": (ttnn.float32, ttnn.mxint8, {}, None),
    "other_mx_out": (ttnn.mxint8, ttnn.mxfp8_e4m3, {}, None),
    "scalar_bf16_in": (ttnn.bfloat16, ttnn.mxint8, {}, 0.75),
    "scalar_fp32_in": (ttnn.float32, ttnn.mxint8, {}, 0.1),
}


def _mx_classes():
    """case -> (op, a's format, the output format, the activations, the scalar or None). Each class runs on add (FPU)
    and multiply (SFPU) with mxint8; add also runs SQUARE on a. The classes that change the packer's encode run each MX
    format as the output, on add: fp32 in (a 32-bit DST into the packer), PACK_RELU into the packer, and another MX
    format out. For another format out, mxint8 goes to each other format, and mxfp4 to mxint8, so that the input and
    the output tiles differ in size in both directions. The format loop skips the pairs that the classes run already.
    The fp32 scalar is one that bf16 cannot hold, so a scalar packed as bf16 shows."""
    classes = {f"{op}-{name}": (op, *spec) for op in ("add", "multiply") for name, spec in _MX_CLASS_SPECS.items()}
    classes["add-square_a"] = ("add", ttnn.mxint8, ttnn.mxint8, {"input_tensor_a_activations": _SQUARE}, None)
    for d in ttnn.MX_DTYPES:
        name = mx.MX_FORMAT[d]
        if d != ttnn.mxint8:
            classes[f"add-fp32_in-{name}"] = ("add", ttnn.float32, d, {}, None)
            classes[f"add-bf16_in_relu_post-{name}"] = ("add", ttnn.bfloat16, d, {"activations": _RELU}, None)
        if d != ttnn.mxfp8_e4m3:
            classes[f"add-other_mx_out-{name}"] = ("add", ttnn.mxfp4 if d == ttnn.mxint8 else ttnn.mxint8, d, {}, None)
    return classes


_MX_CLASSES = _mx_classes()


def _mx_class_model(op, a, b, a_dtype, out_dtype, activations):
    """The op's result from the device operands a and b, before the packer. An MX or bfloat16 input runs the bfloat16
    kernel; a float32 input runs the float32 SFPU op with rounding to nearest even. A 32-bit DST keeps the FPU's
    aligned sum unrounded and the SFPU's float32 product."""
    pre_a = activations.get("input_tensor_a_activations", [])
    if pre_a and pre_a[0].op_type == ttnn.UnaryOpType.SQUARE:
        a = _sfpu_square_bf16(a)
    elif pre_a:
        a = torch.relu(a)
    if activations.get("input_tensor_b_activations"):
        b = torch.relu(b)
    if a_dtype == ttnn.float32:
        result = a.float() + b.float() if op == "add" else a.float() * b.float()
    elif out_dtype == ttnn.float32:
        result = _fpu_aligned_sum(a, b) if op == "add" else a.float() * b.float()
    else:
        result = _MX_KERNEL[op](a.float(), b.float()).float()
    post = activations.get("activations", [])
    if post and post[0].op_type == ttnn.UnaryOpType.SQUARE:
        return _sfpu_square_bf16(result)
    return torch.relu(result) if post else result


@quasar_only
@pytest.mark.parametrize("case", list(_MX_CLASSES))
def test_mx_classes_run_native(device, case):
    """Each class runs native, and its output equals the oracle exactly. The MX output goes through the reference
    encoder, the bfloat16 and float32 outputs compare directly."""
    op, a_dtype, out_dtype, activations, scalar = _MX_CLASSES[case]
    torch.manual_seed(0)
    if a_dtype in ttnn.MX_DTYPES:
        ta, a = _mx_input(device, torch.randn(_SHAPES[1], dtype=torch.float32), a_dtype)
        tb, b = _mx_input(device, torch.randn(_SHAPES[1], dtype=torch.float32), a_dtype)
    else:
        cfg = dict(dtype=a_dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ta = ttnn.from_torch(torch.randn(_SHAPES[1], dtype=torch.float32), **cfg)
        tb = ttnn.from_torch(torch.randn(_SHAPES[1], dtype=torch.float32), **cfg)
        a, b = ttnn.to_torch(ta), ttnn.to_torch(tb)
    if scalar is not None:
        held = torch.float32 if a_dtype == ttnn.float32 else torch.bfloat16
        tb, b = scalar, torch.full(_SHAPES[1], scalar, dtype=held)
    before = mx.kernel_sources()
    tc = getattr(ttnn.experimental.quasar, op)(ta, tb, dtype=out_dtype, **activations)
    mx.assert_native(before, case)

    assert tc.dtype == out_dtype
    golden = _mx_class_model(op, a, b, a_dtype, out_dtype, activations)
    if out_dtype in ttnn.MX_DTYPES:
        wrong = mx.mx_output_wrong(tc, golden)
    else:
        wrong = mx.wrong_elements(ttnn.to_torch(tc).float(), golden.float())
    assert not wrong, f"{len(wrong)} elements differ from the oracle, first at {wrong[:8]}"


# fp32 in, MX out: the packer encodes MX from the 32-bit DST, where a value can lie between two bf16 values. Each case
# runs add(a, 0) with fp32 a, so the output is a as the packer encodes it. Tile 0 holds the midpoint of each pair of
# neighboring values of the dtype at S = 127 (an exact tie after the alignment), tile 1 the fp32 values one step above
# and below each midpoint (nonzero bits below the tie), and tile 2 the special blocks of _special_blocks. Every block of
# tiles 0 and 1 starts with the largest value of the dtype, so that its scale is 127.
# The case compares the values and the 32 scale bytes of each tile with the reference encoder's. No block holds an fp32
# subnormal: the SFPU add flushes it to zero before the packer (the narrowing special values of the native module show
# the flush), so add(a, 0) cannot bring one to the packer.
_FP32_IN_SPECIAL = {mx.MX_FORMAT[d]: d for d in (ttnn.mxint8, ttnn.mxfp8_e4m3)}
# case -> (count, sha256 prefix) of the sorted flat indices where the device's values differ from 4.2, and the
# [tile, block] of each scale byte that differs, as measured on ttsim v1.16.0. A case not here is exact. mxfp8_e4m3: 16
# values one fp32 step above a tie in the element's subnormal range round down to even, as if the packer drops the fp32
# bits below the tie first; 4.2 rounds them up.
_FP32_IN_SPECIAL_WRONG = {"mxfp8_e4m3": (16, "db591b9f290b70f1", [])}


def _fp32_step(x, step):
    """The fp32 value `step` steps away from zero (toward zero if negative) from each element of `x`."""
    return (x.float().contiguous().view(torch.int32) + step).view(torch.float32)


def _special_fp32_tiles(dtype):
    """Three 32x32 fp32 tiles side by side: the midpoints, their fp32 neighbors, and the special blocks."""
    values = mx.code_values(dtype)
    pos = values[values.isfinite() & (values >= 0)].unique()
    vmax, smallest = float(pos[-1]), float(pos[1])
    mid = ((pos[:-1].double() + pos[1:].double()) / 2).float()
    near = torch.cat([_fp32_step(mid, 1), _fp32_step(mid, -1)])
    g = torch.Generator().manual_seed(2)

    def tile(blocks):
        t = torch.randn((32, 32), generator=g)
        for b, block in enumerate(blocks):
            rows, cols = mx.block_cells(b)
            t[rows, cols] = torch.tensor([block[i % len(block)] for i in range(32)], dtype=torch.float32)
        return t

    special = _special_blocks(2.0 ** math.floor(math.log2(vmax)), smallest, vmax)
    tiles = [
        tile(_cyclic_blocks(vmax, mid.tolist() + (-mid).tolist())),
        tile(_cyclic_blocks(vmax, near.tolist() + (-near).tolist())),
        tile(special),
    ]
    return torch.cat(tiles, dim=1)


def _fp32_in_marks(case):
    if case not in _FP32_IN_SPECIAL_WRONG:
        return []
    return [pytest.mark.xfail(raises=_WrongAsPredicted, strict=True, reason=f"ttsim differs from 4.2 on {case}")]


@quasar_only
@pytest.mark.parametrize("codes", [pytest.param(c, marks=_fp32_in_marks(c)) for c in _FP32_IN_SPECIAL])
def test_mx_fp32_in_special_codes(device, codes):
    dtype = _FP32_IN_SPECIAL[codes]
    x = _special_fp32_tiles(dtype)
    shape = tuple(x.shape)
    cfg = dict(dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ta, tb = ttnn.from_torch(x, **cfg), ttnn.from_torch(torch.zeros_like(x), **cfg)
    a = ttnn.to_torch(ta).float()
    before = mx.kernel_sources()
    tc = ttnn.experimental.quasar.add(ta, tb, dtype=dtype)
    mx.assert_native(before, f"add on fp32 {codes}")

    got, want = mx.device_bytes(tc), mx.encode(a, dtype)
    wrong = mx.wrong_elements(mx.decode(got, dtype, shape), mx.decode(want, dtype, shape))
    scales = torch.nonzero(mx.scale_bytes(got, dtype) != mx.scale_bytes(want, dtype)).tolist()
    found, predicted = (*_digest(wrong), scales), _FP32_IN_SPECIAL_WRONG.get(codes, _EXACT)
    assert found == predicted, (
        f"{codes}: {len(wrong)} values differ from 4.2 (first at {wrong[:16]}), scale bytes differ at {scales[:16]}; "
        f"found {found}, predicted {predicted}"
    )
    if wrong or scales:
        raise _WrongAsPredicted(f"{codes}: {len(wrong)} values and {len(scales)} scale bytes differ, as predicted")


@pytest.mark.skipif(is_quasar(), reason="checks the non-Quasar guard")
@pytest.mark.parametrize("dtype", [ttnn.mxfp8_e4m3, ttnn.mxint8])
def test_mx_rejected_off_quasar(device, expect_error, dtype):
    x = torch.randn((32, 32), dtype=torch.float32)
    with expect_error(RuntimeError, "is not supported on arch"):
        ttnn.from_torch(x, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device)
