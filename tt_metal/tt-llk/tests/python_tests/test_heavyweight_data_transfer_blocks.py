# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-side guards for the heavyweight golden's data-transfer blocks.

No kernel, no device. Each transfer boundary in
``helpers/golden_generator/heavyweight/data_transfer_blocks`` encodes a hardware rule --
how much mantissa survives, which format pairs are legal, what Dest can hold, how the
packer narrows a ReLU threshold -- and a wrong rule makes the golden confidently wrong.
These tests pin each rule in isolation, with every expected value written out here rather
than computed by the code under test.

They pin the rules as documented; agreement with silicon is established end to end by the
tests that use the golden.
"""

import warnings

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generator.heavyweight.data_transfer_blocks import (
    BlackholeDataTransferBlocks,
    EdgeMaskMode,
    PackEdgeMask,
    QuasarDataTransferBlocks,
    WormholeDataTransferBlocks,
    apply_relu,
)
from helpers.golden_generator.heavyweight.data_transfer_blocks.l1_codec import (
    tile_bytes_for,
)
from helpers.llk_params import PackerReluType, StochasticRounding

QUASAR = QuasarDataTransferBlocks()
WORMHOLE = WormholeDataTransferBlocks()
BLACKHOLE = BlackholeDataTransferBlocks()

TILE = 1024
ONE_TILE_GEOMETRY = dict(num_faces=4, face_r_dim=16)
# Two faces of a single row: a 1x32 tile, the smallest shape where BFP's
# 16-exponent minimum pads the stride beyond the datum count.
ONE_ROW_TILE_GEOMETRY = dict(num_faces=2, face_r_dim=1)


def _tile_of(*values):
    """A full tile whose leading datums are `values` and the rest zero."""
    tile = torch.zeros(TILE, dtype=torch.float32)
    tile[: len(values)] = torch.tensor(values, dtype=torch.float32)
    return tile


def _bits(t):
    return t.float().view(torch.int32)


# ---------------------------------------------------------------------------
# L1 -> src register precision


def test_float32_into_srca_truncates_to_ten_mantissa_bits():
    # 1 + 2^-10 + 2^-11 would round up to 1 + 2^-9; the unpacker truncates.
    l1 = QUASAR.pack_to_l1(
        _tile_of(1 + 2**-10 + 2**-11, -(1 + 2**-10 + 2**-23)), DataFormat.Float32
    )
    src = QUASAR.l1_to_srcA(l1, DataFormat.Float32)
    assert src[0].item() == 1 + 2**-10
    assert src[1].item() == -(1 + 2**-10)


def test_float16_b_src_keeps_ten_mantissa_bits_not_seven():
    """Float16_b is an alias for Tf32 in a src register, not a bf16 truncation."""
    l1 = QUASAR.pack_to_l1(_tile_of(1 + 2**-10), DataFormat.Float32)
    src = QUASAR.l1_to_srcA(l1, DataFormat.Float32, DataFormat.Float16_b)
    assert src[0].item() == 1 + 2**-10


def test_float16_src_saturates_and_flushes_outside_its_range():
    l1 = QUASAR.pack_to_l1(
        _tile_of(70000.0, -70000.0, 2**-15, 2**-14), DataFormat.Float32
    )
    src = QUASAR.l1_to_srcA(l1, DataFormat.Float32, DataFormat.Float16).float()
    assert src[0].item() == float("inf")
    assert src[1].item() == float("-inf")
    assert src[2].item() == 0.0  # fp16 subnormal: flushed
    assert src[3].item() == 2**-14  # smallest fp16 normal: kept


@pytest.mark.parametrize("dest_acc", [False, True])
def test_float32_into_srcs_keeps_every_bit(dest_acc):
    x = _tile_of(1 + 2**-23, -(1 + 2**-10 + 2**-11), 3.14159265)
    l1 = QUASAR.pack_to_l1(x, DataFormat.Float32, use_srcs=True, dest_acc=dest_acc)
    srcs = QUASAR.l1_to_srcS(l1, DataFormat.Float32, dest_acc=dest_acc)
    assert torch.equal(_bits(srcs), _bits(x))


@pytest.mark.parametrize(
    "l1_format, dest_acc, expected",
    [
        (DataFormat.Float32, False, DataFormat.Float32),
        (DataFormat.Float32, True, DataFormat.Float32),
        (DataFormat.Float16_b, True, DataFormat.Float16_b),
        (DataFormat.Float16, True, DataFormat.Float16),
        (DataFormat.Fp8_e4m3, True, DataFormat.Float32),
        (DataFormat.MxFp8R, True, DataFormat.Float16_b),
        (DataFormat.MxFp4, False, DataFormat.Float16_b),
        (DataFormat.Int8, True, DataFormat.Int8),
    ],
)
def test_srcs_storage_format(l1_format, dest_acc, expected):
    assert QUASAR.srcs_format(l1_format, dest_acc) == expected


# ---------------------------------------------------------------------------
# Legal and illegal format pairs, per architecture


@pytest.mark.parametrize(
    "l1_format, src_format",
    [
        (DataFormat.Int32, None),  # Int32 reaches Dest or SrcS only
        (DataFormat.Int32, DataFormat.Int32),
        (DataFormat.Float32, DataFormat.Float32),  # a 19-bit datum cannot hold fp32
        (DataFormat.Int8, DataFormat.Float16),  # an integer does not become a float
        (DataFormat.Float16_b, DataFormat.Int16),
    ],
)
def test_quasar_rejects_illegal_srca_pairs(l1_format, src_format):
    l1 = QUASAR.pack_to_l1(torch.ones(TILE), l1_format)
    with pytest.raises(ValueError):
        QUASAR.l1_to_srcA(l1, l1_format, src_format)


@pytest.mark.parametrize(
    "l1_format, src_format",
    [
        (DataFormat.Float32, DataFormat.Tf32),
        (DataFormat.Float32, DataFormat.Float16),
        (DataFormat.MxFp8P, DataFormat.Float16_b),
        (DataFormat.Int8, DataFormat.Int8),
        (DataFormat.UInt8, DataFormat.UInt8),
    ],
)
def test_quasar_accepts_legal_srca_pairs(l1_format, src_format):
    l1 = QUASAR.pack_to_l1(torch.ones(TILE), l1_format)
    QUASAR.l1_to_srcA(l1, l1_format, src_format)


@pytest.mark.parametrize(
    "l1_format, src_format",
    [
        (DataFormat.Float16_b, DataFormat.Float16),
        (DataFormat.Float16, DataFormat.Tf32),
    ],
)
def test_srcs_rejects_converting_fp16_under_dest_acc(l1_format, src_format):
    l1 = QUASAR.pack_to_l1(torch.ones(TILE), l1_format, use_srcs=True, dest_acc=True)
    with pytest.raises(ValueError):
        QUASAR.l1_to_srcS(l1, l1_format, src_format, dest_acc=True)


@pytest.mark.parametrize(
    "blocks, l1_format",
    [
        (QUASAR, DataFormat.Bfp8_b),  # Quasar has no block float
        (QUASAR, DataFormat.Bfp4_b),
        (WORMHOLE, DataFormat.MxFp8R),  # MX is Quasar-only
        (WORMHOLE, DataFormat.Fp8_e4m3),  # Wormhole's only fp8 is Lf8
        (BLACKHOLE, DataFormat.MxInt8),
    ],
    ids=lambda v: getattr(v, "name", type(v).__name__),
)
def test_a_format_the_architecture_cannot_hold_is_refused(blocks, l1_format):
    assert not blocks.supports(l1_format)
    with pytest.raises(ValueError, match="cannot read"):
        blocks.src_format(l1_format)


@pytest.mark.parametrize(
    "blocks, l1_format",
    [(QUASAR, DataFormat.Tf32), (WORMHOLE, DataFormat.Bfp8)],
    ids=lambda v: getattr(v, "name", type(v).__name__),
)
def test_a_real_format_without_a_codec_says_so(blocks, l1_format):
    """Not a KeyError, and not a claim that the hardware cannot hold it."""
    assert not blocks.supports(l1_format)
    with pytest.raises(ValueError, match="no L1 codec"):
        blocks.src_format(l1_format)


def test_wormhole_without_a_pair_table_still_rejects_a_non_src_format():
    l1 = WORMHOLE.pack_to_l1(torch.ones(TILE), DataFormat.Float16_b)
    WORMHOLE.l1_to_srcA(l1, DataFormat.Float16_b, DataFormat.Tf32)
    with pytest.raises(ValueError):
        WORMHOLE.l1_to_srcA(l1, DataFormat.Float16_b, DataFormat.Int8)


def test_a_tensor_is_not_an_l1_buffer():
    with pytest.raises(TypeError):
        QUASAR.l1_to_srcA(torch.ones(TILE), DataFormat.Float16_b)


# ---------------------------------------------------------------------------
# Dest width and storage


@pytest.mark.parametrize(
    "l1_format, dest_acc, expected",
    [
        (DataFormat.Float16_b, False, DataFormat.Float16_b),
        (DataFormat.Float16, False, DataFormat.Float16),
        (DataFormat.MxFp8R, False, DataFormat.Float16_b),
        (DataFormat.Float16_b, True, DataFormat.Float32),
        (DataFormat.Float32, True, DataFormat.Float32),
        (DataFormat.Int8, False, DataFormat.Int8),
        (DataFormat.Int8, True, DataFormat.Int32),
    ],
)
def test_dest_format_follows_the_input_and_dest_acc(l1_format, dest_acc, expected):
    assert QUASAR.dest_format_for(l1_format, dest_acc) == expected


@pytest.mark.parametrize("l1_format", [DataFormat.Float32, DataFormat.Tf32])
def test_a_wide_float_input_with_16_bit_dest_does_not_guess_the_family(l1_format):
    with pytest.raises(ValueError):
        QUASAR.dest_format_for(l1_format, False)


@pytest.mark.parametrize(
    "dest_format, dest_acc",
    [
        (DataFormat.Float32, False),
        (DataFormat.Float16_b, True),
        (DataFormat.Int32, False),
    ],
)
def test_a_dest_format_that_contradicts_dest_acc_is_refused(dest_format, dest_acc):
    with pytest.raises(ValueError):
        QUASAR.resolve_dest_format(dest_format, DataFormat.Float16_b, dest_acc)


def test_a_float16_dest_holds_no_denormal():
    dest = QUASAR.src_to_dest(
        torch.tensor([2**-15, 2**-14, -(2**-20)]), DataFormat.Float16
    )
    assert dest.float().tolist() == [0.0, 2**-14, 0.0]


def test_dest_rounds_on_every_accumulating_write():
    """256 + 1 is not a bf16, so each pass lands back on 256 instead of reaching 259."""
    dest = QUASAR.src_to_dest(torch.tensor([256.0]), DataFormat.Float16_b)
    for _ in range(3):
        dest = QUASAR.src_to_dest(
            torch.tensor([1.0]), DataFormat.Float16_b, current=dest
        )
    assert dest.float().item() == 256.0


# ---------------------------------------------------------------------------
# Dest -> src feedback


def test_float16_b_dest_feeds_back_seven_mantissa_bits():
    back = QUASAR.dest_to_srcA(torch.tensor([1 + 2**-7 + 2**-9]), DataFormat.Float16_b)
    assert back.item() == 1 + 2**-7


def test_int16_dest_is_carried_unchanged():
    """Transport through the bf16 label, not a conversion: 257 must not become 256."""
    back = QUASAR.dest_to_srcA(
        torch.tensor([257.0, 1001.0, -32767.0]), DataFormat.Int16
    )
    assert back.tolist() == [257.0, 1001.0, -32767.0]


def test_int32_dest_saturates_to_int8_magnitude():
    back = QUASAR.dest_to_srcA(
        torch.tensor([300.0, -300.0, 127.0, -128.0, 5.0]), DataFormat.Int32
    )
    assert back.tolist() == [127.0, -127.0, 127.0, -127.0, 5.0]


def test_float32_dest_feeds_back_ten_mantissa_bits():
    back = QUASAR.dest_to_srcA(
        torch.tensor([1 + 2**-10 + 2**-11]), DataFormat.Float32, DataFormat.Tf32
    )
    assert back.item() == 1 + 2**-10


def test_float32_dest_into_a_float16_src_rebiases_into_the_fp16_range():
    back = QUASAR.dest_to_srcA(
        torch.tensor([1e-6, 1e5, 1.5]), DataFormat.Float32, DataFormat.Float16
    ).float()
    assert back.tolist() == [0.0, float("inf"), 1.5]


# ---------------------------------------------------------------------------
# L1 codecs: partial and multi-tile geometry


@pytest.mark.parametrize("num_faces, face_r_dim", [(1, 1), (1, 8), (2, 4), (4, 16)])
def test_partial_tiles_round_trip(num_faces, face_r_dim):
    n = num_faces * face_r_dim * 16
    x = (
        torch.arange(n, dtype=torch.float32) % 256 - 128
    ) / 8  # <= 8 bits: exact in bf16
    geometry = dict(num_faces=num_faces, face_r_dim=face_r_dim)
    l1 = QUASAR.pack_to_l1(x, DataFormat.Float16_b, **geometry)
    assert len(l1) == 2 * n
    back = QUASAR.unpack_from_l1(l1, DataFormat.Float16_b, **geometry)
    assert torch.equal(back.float(), x)


def test_multi_tile_buffers_round_trip_tile_by_tile():
    x = torch.randn(3 * TILE, generator=torch.Generator().manual_seed(0))
    l1 = QUASAR.pack_to_l1(x, DataFormat.Float32, **ONE_TILE_GEOMETRY)
    assert len(l1) == 3 * 4 * TILE
    back = QUASAR.unpack_from_l1(l1, DataFormat.Float32, **ONE_TILE_GEOMETRY)
    assert torch.equal(_bits(back), _bits(x))


@pytest.mark.parametrize(
    "l1_format, geometry, expected_bytes",
    [
        # BFP holds at least 16 exponents: 48 bytes for a 1x32 tile, not 34.
        (DataFormat.Bfp8_b, ONE_ROW_TILE_GEOMETRY, 48),
        # MX dense vs the SrcS per-slice layout, and its 32-bit variant.
        (DataFormat.MxFp8R, ONE_TILE_GEOMETRY, 1056),
        (DataFormat.MxFp8R, dict(ONE_TILE_GEOMETRY, use_srcs=True), 1152),
        (
            DataFormat.MxFp8R,
            dict(ONE_TILE_GEOMETRY, use_srcs=True, dest_acc=True),
            1280,
        ),
    ],
)
def test_tile_stride_accounts_for_padding_and_layout(
    l1_format, geometry, expected_bytes
):
    assert tile_bytes_for(l1_format, **geometry) == expected_bytes


def test_multi_tile_bfp8_reads_every_tile_at_the_padded_stride():
    """A datum-count stride (34 B) would misalign every tile after the first."""
    geometry = ONE_ROW_TILE_GEOMETRY
    tiles = [torch.full((32,), float(v)) for v in (0.5, -2.0, 8.0)]
    l1 = WORMHOLE.pack_to_l1(torch.cat(tiles), DataFormat.Bfp8_b, **geometry)
    assert len(l1) == 3 * 48
    back = WORMHOLE.unpack_from_l1(l1, DataFormat.Bfp8_b, **geometry).float()
    assert back.tolist() == torch.cat(tiles).tolist()


@pytest.mark.parametrize("datums", [TILE + 1, 2 * TILE - 1, TILE // 2])
def test_a_partial_tile_is_refused_rather_than_dropped(datums):
    with pytest.raises(ValueError, match="whole number"):
        QUASAR.pack_to_l1(torch.ones(datums), DataFormat.Float32)


def test_an_explicit_tile_count_packs_a_prefix_but_not_past_the_end():
    x = torch.arange(2 * TILE, dtype=torch.float32)
    l1 = QUASAR.pack_to_l1(x, DataFormat.Float32, tile_count=1)
    assert QUASAR.unpack_from_l1(l1, DataFormat.Float32).tolist() == x[:TILE].tolist()
    with pytest.raises(ValueError, match="tile_count=3"):
        QUASAR.pack_to_l1(x, DataFormat.Float32, tile_count=3)


# ---------------------------------------------------------------------------
# Packer ReLU and its 16-bit threshold field


def test_min_threshold_compares_against_the_truncated_bf16_threshold():
    """0.3 truncates to 0.298828125 in the register. A rounding bf16 cast would
    give 0.30078125 and wrongly zero 0.2995."""
    out = apply_relu(
        torch.tensor([0.29, 0.2995, 0.31, -1.0]),
        PackerReluType.MinThresholdRelu,
        0.3,
        DataFormat.Float16_b,
    )
    assert out.tolist() == pytest.approx([0.0, 0.2995, 0.31, 0.0])


def test_a_32_bit_dest_reads_the_threshold_as_bf16_too():
    out = apply_relu(
        torch.tensor([0.2995]), PackerReluType.MinThresholdRelu, 0.3, DataFormat.Float32
    )
    assert out.tolist() == pytest.approx([0.2995])


def test_a_float16_dest_reads_the_threshold_as_rounded_fp16():
    """0.3 rounds to fp16 0.30004883, so float32 0.3 sits below it and is zeroed."""
    out = apply_relu(
        torch.tensor([0.3, 0.3001]),
        PackerReluType.MinThresholdRelu,
        0.3,
        DataFormat.Float16,
    )
    assert out.tolist() == pytest.approx([0.0, 0.3001])


def test_max_threshold_clamps_to_zero_and_the_threshold():
    out = apply_relu(
        torch.tensor([-1.0, 0.5, 2.0]), PackerReluType.MaxThresholdRelu, 1.0
    )
    assert out.tolist() == [0.0, 0.5, 1.0]


def test_zero_relu_and_no_relu():
    x = torch.tensor([-1.0, 2.0])
    assert apply_relu(x, PackerReluType.ZeroRelu).tolist() == [0.0, 2.0]
    assert apply_relu(x, PackerReluType.NoRelu).tolist() == [-1.0, 2.0]


# ---------------------------------------------------------------------------
# Packer edge mask


ROW = [1.0] * 16
FIRST_ONLY = [1.0] + [0.0] * 15


@pytest.mark.parametrize(
    "masked_when_set, word, expected_row",
    [
        # Quasar: a set bit masks. 0xFFFE masks datums 1..15, 0x0000 passes all.
        (True, 0xFFFE, FIRST_ONLY),
        (True, 0x0000, ROW),
        (True, 0xFFFF, [0.0] * 16),
        # Wormhole/Blackhole: a set bit keeps. 0x0001 keeps datum 0, 0xFFFF passes all.
        (False, 0x0001, FIRST_ONLY),
        (False, 0xFFFF, ROW),
        (False, 0x0000, [0.0] * 16),
    ],
)
def test_mask_polarity(masked_when_set, word, expected_row):
    out = PackEdgeMask(masks=(word,)).apply(
        torch.ones(32), masked_when_set=masked_when_set
    )
    assert out.tolist() == expected_row * 2


@pytest.mark.parametrize(
    "blocks, keep_first_word",
    [(QUASAR, 0xFFFE), (WORMHOLE, 0x0001), (BLACKHOLE, 0x0001)],
    ids=["quasar", "wormhole", "blackhole"],
)
def test_dest_to_l1_uses_the_architectures_polarity(blocks, keep_first_word):
    """The register value each architecture's reduce uses to keep column 0."""
    dest = blocks.src_to_dest(torch.ones(TILE), DataFormat.Float16_b)
    l1 = blocks.dest_to_l1(
        dest,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        edge_mask=PackEdgeMask(masks=(keep_first_word,)),
    )
    back = blocks.unpack_from_l1(l1, DataFormat.Float16_b).float()
    assert back.tolist() == FIRST_ONLY * (TILE // 16)


def test_negative_saturate_mode_replaces_with_minus_inf():
    mask = PackEdgeMask(masks=(0x0001,), mode=EdgeMaskMode.NEG_SATURATE)
    out = mask.apply(torch.ones(16), masked_when_set=True)
    assert out[0].item() == float("-inf")
    assert out[1:].tolist() == [1.0] * 15


def test_the_selector_picks_one_mask_per_row():
    mask = PackEdgeMask(masks=(0x0000, 0xFFFF), select=[0, 1, 1, 0])
    out = mask.apply(torch.ones(64), masked_when_set=True)
    assert out.tolist() == ROW + [0.0] * 16 + [0.0] * 16 + ROW


@pytest.mark.parametrize(
    "word, rows_on_mask_1",
    [
        (0x00000000, []),  # EDGE_MASK_FACE_ALL_ROWS_MASK_0
        (0x00000001, [0]),  # EDGE_MASK_FACE_ROW0_MASK_1
        (0x00010001, [0, 8]),  # EDGE_MASK_FACE_ROW0_ROW8_MASK_1
        (0x55555555, list(range(16))),  # EDGE_MASK_FACE_ALL_ROWS_MASK_1
    ],
)
def test_face_select_words_expand_to_one_selector_per_row(word, rows_on_mask_1):
    mask = PackEdgeMask.from_face_select_words((0xFFFF, 0x0000), [word] * 4)
    face = [1 if row in rows_on_mask_1 else 0 for row in range(16)]
    assert list(mask.select) == face * 4


def test_quasar_reduce_col_keeps_rows_0_and_8_of_each_face():
    """Quasar's REDUCE_COL config: mask0 ALL, mask1 NONE on rows 0 and 8."""
    mask = PackEdgeMask.from_face_select_words((0xFFFF, 0x0000), [0x00010001] * 4)
    out = mask.apply(torch.ones(TILE), masked_when_set=True).reshape(64, 16)
    kept_rows = [r for r in range(64) if out[r].sum() == 16]
    assert kept_rows == [0, 8, 16, 24, 32, 40, 48, 56]
    assert out.sum().item() == 8 * 16


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(masks=(0xFFFF,) * 5),  # at most four masks
        dict(masks=()),
        dict(masks=(0x10000,)),  # each mask is 16 bits
        dict(masks=(0xFFFF, 0x0000), select=2),  # selects a mask that does not exist
        dict(masks=(0xFFFF, 0x0000), select=[0, 1, 2]),
        dict(masks=(0xFFFF,), select=[-1]),
    ],
)
def test_an_out_of_range_edge_mask_is_refused(kwargs):
    with pytest.raises(ValueError):
        PackEdgeMask(**kwargs)


def test_a_selector_shorter_than_the_rows_is_refused():
    with pytest.raises(ValueError, match="rows"):
        PackEdgeMask(masks=(0xFFFF,), select=[0]).apply(torch.ones(32))


def test_data_that_is_not_whole_rows_is_refused():
    with pytest.raises(ValueError, match="rows"):
        PackEdgeMask(masks=(0xFFFF,)).apply(torch.ones(20))


# ---------------------------------------------------------------------------
# The pack block end to end


def test_dest_to_l1_applies_relu_then_packs():
    dest = QUASAR.src_to_dest(_tile_of(-2.0, 0.5, 3.0), DataFormat.Float16_b)
    l1 = QUASAR.dest_to_l1(
        dest,
        DataFormat.Float16_b,
        DataFormat.Float16_b,
        relu_type=PackerReluType.MaxThresholdRelu,
        relu_threshold=1.0,
    )
    back = QUASAR.unpack_from_l1(l1, DataFormat.Float16_b).float()
    assert back[:3].tolist() == [0.0, 0.5, 1.0]


def test_stochastic_rounding_warns_rather_than_pretending_to_reproduce_it():
    dest = QUASAR.src_to_dest(torch.ones(TILE), DataFormat.Float16_b)
    with pytest.warns(UserWarning, match="cannot be reproduced"):
        QUASAR.dest_to_l1(
            dest,
            DataFormat.Float16_b,
            DataFormat.Float16_b,
            stoch_rnd=StochasticRounding.Pack,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        QUASAR.dest_to_l1(dest, DataFormat.Float16_b, DataFormat.Float16_b)


# An integer Dest narrows by saturation, not by wrapping. A plain torch cast
# gives 200 -> -56 for Int8, and a float beyond int32's range casts to an
# undefined value; the hardware clamps instead.
INTEGER_SATURATION_CASES = [
    (DataFormat.Int8, 200.0, 127),
    (DataFormat.Int8, -200.0, -127),
    # Sign-magnitude: the most negative two's-complement value has no encoding.
    (DataFormat.Int8, -128.0, -127),
    (DataFormat.Int16, 40000.0, 32767),
    (DataFormat.Int16, -40000.0, -32767),
    # Beyond float32's ability to even name int32's max, so the clamp has to
    # happen in a wider type or the cast is undefined.
    (DataFormat.Int32, 3e38, 2147483647),
    (DataFormat.Int32, -3e38, -2147483647),
    # Unsigned uses the full range, with no sign-magnitude adjustment.
    (DataFormat.UInt8, 300.0, 255),
    (DataFormat.UInt8, -5.0, 0),
]


@pytest.mark.parametrize(
    "dest_format, value, expected",
    INTEGER_SATURATION_CASES,
    ids=[f"{c[0].name}-{c[1]:g}" for c in INTEGER_SATURATION_CASES],
)
def test_integer_dest_saturates_instead_of_wrapping(dest_format, value, expected):
    got = QUASAR._to_dest_storage(torch.tensor([value]), dest_format)
    assert got.item() == expected


@pytest.mark.parametrize(
    "value, expected",
    [(float("nan"), 0), (float("inf"), 127), (float("-inf"), -127)],
    ids=["nan", "inf", "-inf"],
)
def test_non_finite_into_an_integer_dest_is_not_undefined(value, expected):
    """A NaN survives a clamp and makes the cast undefined, so it is mapped
    before the clamp rather than left to torch."""
    got = QUASAR._to_dest_storage(torch.tensor([value]), DataFormat.Int8)
    assert got.item() == expected


# An integer Dest accumulates exactly on the device. float32 holds integers
# only to 2^24, so accumulating there drops the low bits of a wide Dest value
# before it is narrowed back -- and can push an in-range sum past INT32_MAX,
# where the saturation clamp then pins it.
INTEGER_ACCUMULATE_CASES = [
    (16777217, 1, 16777218),
    (16777217, 0, 16777217),
    (-16777217, -1, -16777218),
    (2147483000, 600, 2147483600),
    # Genuinely out of range, so this one does saturate.
    (2000000000, 300000000, 2147483647),
]


@pytest.mark.parametrize(
    "current, addend, expected",
    INTEGER_ACCUMULATE_CASES,
    ids=[f"{c[0]}+{c[1]}" for c in INTEGER_ACCUMULATE_CASES],
)
def test_integer_dest_accumulates_exactly(current, addend, expected):
    got = QUASAR.src_to_dest(
        torch.tensor([addend], dtype=torch.int32),
        DataFormat.Int32,
        torch.tensor([current], dtype=torch.int32),
    )
    assert got.item() == expected


# A derived Dest format gets the same checks as an explicit one. Deriving it is
# not a reason to skip them: an Int32 input with accumulation off would
# otherwise produce a 32-bit Dest that resolve_dest_format rejects when the
# caller passes the identical format by hand.
@pytest.mark.parametrize(
    "blocks, l1_format, dest_acc",
    [
        (QUASAR, DataFormat.Int32, False),  # 32-bit Dest, accumulation off
        (QUASAR, DataFormat.Int16, True),  # Int16 cannot drive a 32-bit Dest
        (WORMHOLE, DataFormat.UInt16, False),  # readable from L1, not a Dest format
        (WORMHOLE, DataFormat.UInt32, False),
    ],
    ids=["int32-no-acc", "int16-acc", "wh-uint16", "wh-uint32"],
)
def test_a_derived_integer_dest_is_checked_like_an_explicit_one(
    blocks, l1_format, dest_acc
):
    with pytest.raises(ValueError):
        blocks.dest_format_for(l1_format, dest_acc)


@pytest.mark.parametrize(
    "l1_format, dest_acc, expected",
    [
        (DataFormat.Int8, False, DataFormat.Int8),
        (DataFormat.Int8, True, DataFormat.Int32),
        (DataFormat.UInt8, False, DataFormat.UInt8),
        (DataFormat.UInt8, True, DataFormat.Int32),
        (DataFormat.Int16, False, DataFormat.Int16),
        (DataFormat.Int32, True, DataFormat.Int32),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_the_legal_integer_dest_pairings_still_resolve(l1_format, dest_acc, expected):
    assert QUASAR.dest_format_for(l1_format, dest_acc) == expected
