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
        (DataFormat.Bfp8_b, dict(num_faces=2, face_r_dim=1), 48),
        # MX dense vs the SrcS per-slice layout, and its 32-bit variant.
        (DataFormat.MxFp8R, dict(num_faces=4, face_r_dim=16), 1056),
        (DataFormat.MxFp8R, dict(num_faces=4, face_r_dim=16, use_srcs=True), 1152),
        (
            DataFormat.MxFp8R,
            dict(num_faces=4, face_r_dim=16, use_srcs=True, dest_acc=True),
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
    geometry = dict(num_faces=2, face_r_dim=1)
    tiles = [torch.full((32,), float(v)) for v in (0.5, -2.0, 8.0)]
    l1 = WORMHOLE.pack_to_l1(torch.cat(tiles), DataFormat.Bfp8_b, **geometry)
    assert len(l1) == 3 * 48
    back = WORMHOLE.unpack_from_l1(l1, DataFormat.Bfp8_b, **geometry).float()
    assert back.tolist() == torch.cat(tiles).tolist()


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


def test_a_set_bit_keeps_the_datum_and_a_clear_bit_zeroes_it():
    mask = PackEdgeMask(masks=(0x00FF,), select=0, mode=EdgeMaskMode.ZERO)
    out = mask.apply(torch.ones(32))
    expected = ([1.0] * 8 + [0.0] * 8) * 2  # bit i % 16
    assert out.tolist() == expected


def test_negative_saturate_mode_replaces_with_minus_inf():
    mask = PackEdgeMask(masks=(0xFFFE,), mode=EdgeMaskMode.NEG_SATURATE)
    out = mask.apply(torch.ones(16))
    assert out[0].item() == float("-inf")
    assert out[1:].tolist() == [1.0] * 15


def test_the_per_datum_selector_picks_a_mask_per_datum():
    mask = PackEdgeMask(masks=(0xFFFF, 0x0000), select=[0, 1] * 8)
    assert mask.apply(torch.ones(16)).tolist() == [1.0, 0.0] * 8


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(masks=(0xFFFF,) * 5),  # at most four masks
        dict(masks=()),
        dict(masks=(0x10000,)),  # each mask is 16 bits
    ],
)
def test_an_out_of_range_edge_mask_is_refused(kwargs):
    with pytest.raises(ValueError):
        PackEdgeMask(**kwargs)


def test_a_selector_shorter_than_the_data_is_refused():
    with pytest.raises(ValueError):
        PackEdgeMask(masks=(0xFFFF,), select=[0] * 4).apply(torch.ones(16))


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
