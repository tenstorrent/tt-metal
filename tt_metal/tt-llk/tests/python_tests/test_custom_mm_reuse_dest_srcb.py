# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test for the Blackhole-only experimental LLK custom_mm_reuse_dest_srcb
# (PR #53297): the second matmul of a fused chain whose in0 operand is moved out
# of DEST (where a preceding custom_mm left its output) into SrcB instead of
# being unpacked from L1.  Only the weights (in1) are unpacked, into SrcA.
#
# The reuse LLK is meaningless without a DEST-resident in0 in exactly the layout
# custom_mm's Dense layout produces, so the C++ driver runs the whole documented
# chain:  producer custom_mm (DenseSplitAcc)  ->  consumer custom_mm_reuse_dest_srcb.
#
# Golden: the chained LoFi matmul (A0 @ B0) @ B1, with the producer result
# narrowed to the DEST format before the consumer reads it. Only the first
# IN0_TILE_R_DIM (=8) rows of each output tile are defined; the device packs them
# as two 8-row faces (cols 0..15, then cols 16..31), and only those are asserted.

import torch
from conftest import blackhole_only
from helpers.custom_mm_utils import pack_in0_faces
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    CUSTOM_MM_REUSE_CFG,
    NUM_FACES,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test

# Chain geometry.  in0 tile height is the full-height 8 case (MOVD2B copies
# all 8 rows, no zero-pad subtlety), with a 2-tile-deep producer.
IN0_TILE_R_DIM = 8
PRODUCER_KT = 2  # producer inner dim in tiles (its kt_dim)
# (consumer kt_dim, consumer nt_dim). kt_dim == producer ct_dim == # DEST in0 tiles;
# odd kt_dim runs the unpack's replay tail, and nt_dim >= 2 its halo-mode windows.
# kt_dim * 32 in0 rows from DEST row 0 must stay below the accumulator at row 256.
REUSE_KT_NT = [(kt, nt) for kt in range(1, 9) for nt in (1, 2, 3, 4)]

# LoFi only for both custom_mm and the reuse LLK.
MATH_FIDELITY = MathFidelity.LoFi

# 16-bit float operands keep the chained matmul well inside format tolerance.
REUSE_FORMATS = input_output_formats([DataFormat.Float16_b])


def _tile(t, dims, fmt, face_r_dim=16):
    return tilize_block(t, dimensions=dims, stimuli_format=fmt, face_r_dim=face_r_dim)


@blackhole_only
# NOTE: a single-axis @parametrize(formats=...) passes formats as a 1-tuple
# (pytest does not unpack a comma-less argname), so we add the dest_acc axis.
# With >1 axis the framework unpacks the value tuple and `formats` is the object.
@parametrize(
    formats=REUSE_FORMATS,
    dest_acc=[DestAccumulation.No],
    reuse_kt_nt=REUSE_KT_NT,
)
def test_custom_mm_reuse_dest_srcb(formats, dest_acc, reuse_kt_nt):
    REUSE_KT, REUSE_NT = reuse_kt_nt
    torch_format = format_dict[formats.output_format]
    in_fmt = formats.input_format

    # Producer contracting dim (its K) in tiles; kept at PRODUCER_KT.
    m = IN0_TILE_R_DIM
    kp = PRODUCER_KT * 32  # producer contraction width in datums
    kt = REUSE_KT * 32  # producer output width == consumer contraction width
    nt = REUSE_NT * 32  # consumer output width

    torch.manual_seed(0)
    # in0 producer input A0 : [m, kp]   (in0 tile shape [8, 32])
    A0 = torch.rand((m, kp), dtype=torch.float32)
    # producer weights B0   : [kp, kt]  (32x32 tiles)
    B0 = torch.rand((kp, kt), dtype=torch.float32)
    # consumer weights B1   : [kt, nt]  (32x32 tiles)
    B1 = torch.rand((kt, nt), dtype=torch.float32)

    # Golden: chained LoFi matmul (A0 @ B0) @ B1. The producer result is narrowed
    # to the DEST format (Float16_b, dest_acc off) before the consumer reads it.
    def _lofi_mm(a, b, rows, k, n):
        return (
            MatmulGolden()(
                a,
                b,
                formats.output_format,
                MathFidelity.LoFi,
                input_A_dimensions=[rows, k],
                input_B_dimensions=[k, n],
                tilize=False,
                input_A_format=in_fmt,
                input_B_format=in_fmt,
            )
            .reshape(rows, n)
            .to(torch_format)
        )

    in0 = _lofi_mm(A0, B0, m, kp, kt)  # [m, kt] == producer output in DEST
    golden = _lofi_mm(in0.float(), B1, m, kt, nt)  # [m, nt]

    # Hand all three operands over as packed L1 images.  in0 (A0) is a dense run of
    # 8-row faces with no tile geometry write_matrix can stride by; the weights are
    # full tiles in k-major / n-minor order, the order the SrcA walks read them.
    packer = StimuliConfig.get_packer(in_fmt)
    A0_bytes = pack_in0_faces(A0, PRODUCER_KT, in_fmt)
    B0_bytes = bytes(packer(_tile(B0.flatten(), dims=[kp, kt], fmt=in_fmt)))
    B1_bytes = bytes(packer(_tile(B1.flatten(), dims=[kt, nt], fmt=in_fmt)))

    configuration = TestConfig(
        "sources/custom_mm_reuse_dest_srcb_test.cpp",
        formats,
        templates=[
            CUSTOM_MM_REUSE_CFG(
                in0_tile_r_dim=IN0_TILE_R_DIM,
                producer_kt=PRODUCER_KT,
                reuse_kt=REUSE_KT,
                reuse_nt=REUSE_NT,
            ),
        ],
        runtimes=[
            NUM_FACES(num_faces=4, num_faces_A=2, num_faces_B=4),
        ],
        variant_stimuli=StimuliConfig(
            A0_bytes,
            in_fmt,
            B0_bytes,
            in_fmt,
            formats.output_format,
            tile_count_A=PRODUCER_KT,  # in0 spans PRODUCER_KT tiles along K
            tile_count_B=PRODUCER_KT * REUSE_KT,
            tile_count_res=REUSE_NT,
            buffer_C=B1_bytes,
            stimuli_C_format=in_fmt,
            tile_count_C=REUSE_KT * REUSE_NT,
        ),
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Each output tile is packed as two 8-row faces (cols 0..15, then cols 16..31);
    # the rest of the tile is undefined. Reorder the defined rows to [m, nt].
    per_tile = res_tensor.reshape(REUSE_NT, -1)[:, : 2 * m * 16]
    device = per_tile.reshape(REUSE_NT, 2, m, 16).permute(2, 0, 1, 3).reshape(m, nt)

    assert passed_test(
        golden, device, formats.output_format
    ), "Assert against golden failed (defined lanes)"
