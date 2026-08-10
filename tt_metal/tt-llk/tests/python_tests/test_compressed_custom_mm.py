# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-tree experimental LLK compressed_custom_mm (tt-metal#47554 / tt-blaze#1971), pending
# promotion into tt_llk_blackhole/llk_lib/experimental/. Include path (shared with custom_mm) is REUSED here; it must
# be repointed to the canonical header on promotion. Primitives verified byte-identical to tt-blaze main as of this
# writing.
#
# compressed_custom_mm is the BFP-compressed-in1 sibling of custom_mm. Same documented contract as custom_mm
# (llk_math_compressed_custom_mm.h / llk_unpack_AB_compressed_custom_mm.h header banners):
#   in0 tile shape: [{1, 2, 4, 8}, 32]   (partial-row tile -> SrcB, bf16, reused across output width)
#   in1 tile shape: [32, 32]             (full tile -> SrcA, BFP-compressed: Bfp8_b / Bfp4_b / Bfp2_b)
#   rt_dim: 1
#   ct_dim: any integer from 1 to 16
#   kt_dim: even number from 2 to 256 (inclusive)
#   fidelity: LoFi only (math init takes no MathFidelity template)
#   throttle: not supported
#
# Difference vs custom_mm: BOTH primitives take an extra base_address_meta argument (a buffer of packed 3-bit
# per-tile compression-format codes). We route it through the harness's optional third operand (buffer_C). For this
# compile-green advance test the metadata content is a placeholder; exact numerical agreement is validated only on
# Blackhole hardware.
#
# Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is pending
# Blackhole hardware/CI; this host is Wormhole.

import torch
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import generate_combination, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CRK_TILE_DIMM,
    IN0_FACE_R_DIM,
    NUM_FACES,
    TILE_COUNT,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# in0 (SrcB) is bf16; in1 (SrcA) is the BFP-compressed operand. Sweep the three BFP widths the LLK's exponent-section
# MOP config supports. unpack_A == in0, unpack_B == in1 (same operand->buffer mapping as custom_mm_test.cpp).
IN1_COMPRESSED_FORMATS = [DataFormat.Bfp8_b, DataFormat.Bfp4_b, DataFormat.Bfp2_b]

# Kernel per-tile format codes (helpers.compressed_utils.FMT_CODE). 0 == "zero tile" == skip.
_FMT_CODE = {DataFormat.Bfp8_b: 3, DataFormat.Bfp4_b: 2, DataFormat.Bfp2_b: 1}


def _encode_meta(fmt_code, ct_dim, kt_dim):
    # Same packing as test_matmul_custom_compressed.encode_meta: 10 tiles per u32, tile i at
    # bits [3i+3 : 3i+4] (format) and bit 3i+2 (use_b), with bits [1:0] of each word carrying the
    # previous tile's format so the unpacker's 5-bit sliding window sees (prev, use_b, curr).
    total = kt_dim * ct_dim
    meta = [0] * ((total + 9) // 10)
    prev_fmt = 0
    for i in range(total):
        u, j = divmod(i, 10)
        if j == 0:
            meta[u] |= prev_fmt & 0b11
        use_b = 1 if (i % ct_dim) == 0 else 0
        meta[u] |= use_b << (3 * j + 2)
        meta[u] |= fmt_code << (3 * j + 3)
        prev_fmt = fmt_code
    return meta


# generate_combination tuple layout (len 7, same_src_format=False):
#   (unpack_A_src, unpack_A_dst, unpack_B_src, unpack_B_dst, pack_src, pack_dst, math)
COMPRESSED_MM_FORMATS = generate_combination(
    [
        (
            DataFormat.Float16_b,  # in0 (SrcB) L1 format
            DataFormat.Float16_b,  # in0 unpack dst
            in1_fmt,  # in1 (SrcA) L1 format -- BFP-compressed
            DataFormat.Float16_b,  # in1 unpack dst (decompressed to bf16 in SrcA)
            DataFormat.Float16_b,  # pack src
            DataFormat.Float16_b,  # pack dst (output)
            DataFormat.Float16_b,  # math format (LoFi bf16)
        )
        for in1_fmt in IN1_COMPRESSED_FORMATS
    ]
)

# Honor the header contract. ct_dim in the documented allowed set, kt_dim even (2..256), in0 rows in {1, 2, 4, 8}.
CT_DIMS = [1, 2, 4, 8, 16]
KT_DIMS = [2, 4]
IN0_ROWS = [1, 2, 4, 8]


def _grid():
    combos = []
    for ct in CT_DIMS:
        for kt in KT_DIMS:
            for rows in IN0_ROWS:
                combos.append((ct, kt, rows))
    return combos


@parametrize(
    formats=COMPRESSED_MM_FORMATS,
    ct_kt_rows=_grid(),
)
def test_compressed_custom_mm(
    formats,
    ct_kt_rows,
    boot_mode=BootMode.DEFAULT,
):
    ct_dim, kt_dim, in0_rows = ct_kt_rows
    rt_dim = 1  # compressed_custom_mm contract: rt_dim is always 1
    output_tile_cnt = rt_dim * ct_dim

    torch_format = format_dict[formats.output_format]

    # in0 is [rt_dim*32, kt_dim*32] (partial rows modeled inside the LLK; host feeds full 32-row tiles and the kernel
    # only unpacks the top in0_rows rows of each in0 face). in1 is [kt_dim*32, ct_dim*32].
    input_A_dimensions = [rt_dim * 32, kt_dim * 32]
    input_B_dimensions = [kt_dim * 32, ct_dim * 32]

    in0_format = formats.unpack_A_src  # bf16
    in1_format = formats.unpack_B_src  # BFP-compressed

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=in0_format,
        input_dimensions_A=input_A_dimensions,
        stimuli_format_B=in1_format,
        input_dimensions_B=input_B_dimensions,
        spec_A=spec,
        spec_B=spec,
    )

    # LoFi golden: standard matmul with the compressed-format rounding of in1 (input_B_format = the BFP format) and
    # bf16 (LoFi) rounding of in0. NOTE: this does not model compressed_custom_mm's exact packed output tile layout
    # (split_acc/dense_packing are off here). Exact numerical agreement is validated only on Blackhole hardware.
    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_A,
        src_B,
        formats.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=input_A_dimensions,
        input_B_dimensions=input_B_dimensions,
        tilize=True,
        input_A_format=in0_format,
        input_B_format=in1_format,
    )

    tilized_A = tilize_block(
        src_A, dimensions=input_A_dimensions, stimuli_format=in0_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=input_B_dimensions, stimuli_format=in1_format
    )

    # Per-tile compression metadata buffer read by both primitives as base_address_meta. The codes are CONTROL FLOW,
    # not just numerics: code 0 means "zero tile", for which the unpacker emits no UNPACR at all and math takes the
    # STALLWAIT(SRCB_VLD) branch. An all-zero buffer therefore hangs Math/Packer forever. Every tile here is the one
    # BFP format under test.
    meta_words = _encode_meta(_FMT_CODE[in1_format], ct_dim, kt_dim)
    meta_buffer = torch.zeros(1024, dtype=torch.int64)
    meta_buffer[: len(meta_words)] = torch.tensor(meta_words, dtype=torch.int64)

    configuration = TestConfig(
        "sources/compressed_custom_mm_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(),
            TILE_COUNT(output_tile_cnt),
            CRK_TILE_DIMM(ct_dim, rt_dim, kt_dim),
            IN0_FACE_R_DIM(in0_rows),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            in0_format,
            tilized_B.flatten(),
            in1_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=output_tile_cnt,
            buffer_C=meta_buffer,
            stimuli_C_format=DataFormat.UInt32,
            tile_count_C=1,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
