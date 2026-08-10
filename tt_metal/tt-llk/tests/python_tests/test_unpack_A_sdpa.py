# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# ADVANCE TEST: covers demo-fork experimental LLK unpack_A_sdpa (tt-metal#47554 / tt-blaze#1971), pending promotion.
# Include path (shadow -I) repoint on promotion. Primitive verified vs tt-blaze main as of this writing.
#
# This test is pinned to the DEMO-fork primitive header
# (models/demos/deepseek_v3_b1/kernel_includes/tt_llk/tt_llk_blackhole/llk_lib/llk_unpack_A_sdpa.h) via the shadow -I
# already registered in helpers/test_config.py, so it does not depend on the tt-blaze migration. The comparison phase
# found the demo-fork header byte-identical to tt-blaze main except for the copyright-holder comment (USA, Inc. vs
# AI ULC), so the numerics validated here match the canonical primitive.
#
# unpack_A_sdpa documented contract (llk_unpack_A_sdpa.h):
#   - It is init/mop-config + a dummy-SrcB-valid helper only; it has no per-tile execute of its own.
#     * _llk_unpack_A_sdpa_init_<num_tiles, BType>(...)  programs a SrcA-only UNPACR MOP.
#     * the base llk_unpack_A execute streams the operand tile into SrcA under that MOP.
#     * _llk_unpack_A_sdpa_set_srcb_dummy_valid_()       injects STALL_UNPACK + a UNPACR_NOP SET_DVALID on SrcB
#       (ZEROSRC, no real data) so a downstream dual-source eltwise's math preamble STALLWAIT(SRCB_VLD) does not stall.
#       This is unpacker-side self-satisfied (no MATH-waits-on-SFPU handshake), so an isolated kernel does not deadlock.
#   - num_faces: the init helper LLK_ASSERTs {1, 2, 4}, but the paired SDPA mop config hard-asserts == 2, so the only
#     instantiable shape is a 16x32 tiny tile (num_faces == 2). See the comparison report's header-vs-#1971 note.
#
# To exercise unpack_A_sdpa with a validatable NUMERIC golden, it is paired with the demo-fork math SDPA column-
# broadcast SrcB-reuse op: the column source is seeded into DEST via a plain A2D datacopy, MOVD2B'd into SrcB by the
# math preamble (which waits on the dummy SrcB valid unpack_A_sdpa injects), then multiplied against the SrcA operand.
# The golden validates NUMERICS only (a plain column-broadcast MUL).
#
# This advance test exercises the MUL (softmax-scale) instantiation, LoFi, on a single 16x32 tiny tile.
#
# Blackhole-only. Deliverable here is compile-green (compile-producer). On-device numerical verification is
# pending Blackhole hardware/CI; this host is Wormhole.

import torch
from helpers.device import BootMode
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import NUM_FACES, TILE_COUNT
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

# LoFi-only, bf16-natural path. Keep the grid tiny for the advance test.
SDPA_FORMATS = input_output_formats([DataFormat.Float16_b])


@parametrize(
    formats=SDPA_FORMATS,
)
def test_unpack_A_sdpa(
    formats,
    boot_mode=BootMode.DEFAULT,
):
    # A single-axis @parametrize passes the value as a 1-tuple; unwrap it.
    if isinstance(formats, tuple):
        (formats,) = formats

    # Single 16x32 tiny tile, num_faces == 2 (the only shape the SDPA mop config accepts).
    num_faces = 2
    face_r_dim = 16
    tile_rows = 16
    tile_dims = [tile_rows, 32]
    dimensions = [tile_rows, 32]
    tile_cnt = 1

    torch_format = format_dict[formats.output_format]

    # buffer_A = operand tile (streamed into SrcA by the unpack_A_sdpa MOP). buffer_B = column-source tile; only
    # column 0 of each face matters (the value MOVD2B/SRCB_BCAST_COL fans across the row), but the whole tile is a
    # valid column-source seed for DEST.
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dimensions,
        tile_dimensions=tile_dims,
    )

    tilized_A = tilize_block(
        src_A,
        dimensions,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    )
    tilized_B = tilize_block(
        src_B,
        dimensions,
        formats.input_format,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    )

    # Golden built in UNTILIZED (row-major) space, then compared to the untilized result read back from L1. The
    # broadcast is computed in tilized space (that is where the column-0 value lives), untilized, then row-expanded to
    # the operand width. Doing the eltwise on untilized tensors avoids untilizing the golden — during compile-producer
    # the golden generators are dummies whose eltwise output is a full-tile 1024-element zero tensor, and untilizing
    # that against a 16x32 shape would spuriously fail.
    broadcast_golden = get_golden_generator(BroadcastGolden)
    src_B_bcast_tilized = broadcast_golden(
        BroadcastType.Column,
        tilized_B.flatten(),
        formats.input_format,
        num_faces=num_faces,
        tile_cnt=tile_cnt,
        face_r_dim=face_r_dim,
        input_format=formats.input_format,
    )

    src_B_bcast = untilize_block(
        src_B_bcast_tilized,
        formats.input_format,
        dimensions,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        MathOperation.Elwmul,
        src_A,
        src_B_bcast,
        formats.output_format,
        MathFidelity.LoFi,
        input_format=formats.input_format,
        input_format_B=formats.input_format,
    )

    configuration = TestConfig(
        "sources/unpack_A_sdpa_test.cpp",
        formats,
        runtimes=[
            NUM_FACES(num_faces, num_faces, num_faces),
            TILE_COUNT(tile_cnt),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            formats.input_format,
            tilized_B.flatten(),
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt,
            num_faces=num_faces,
            face_r_dim=face_r_dim,
            tile_dimensions=tile_dims,
            use_dense_tile_dimensions=True,
        ),
        dest_acc=DestAccumulation.No,
        boot_mode=boot_mode,
    )

    res_from_L1 = configuration.run().result

    res_from_L1 = untilize_block(
        res_from_L1,
        formats.output_format,
        dimensions,
        num_faces=num_faces,
        tile_dimensions=tile_dims,
        face_r_dim=face_r_dim,
    ).flatten()

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "Assert against golden failed"
