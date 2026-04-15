# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    DataCopyGolden,
    TransposeGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DataCopyType,
    DestAccumulation,
    ImpliedMathFormat,
    Transpose,
    UnpackerEngine,
    format_dict,
)
from helpers.param_config import (
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_TRANSPOSE_FACES,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    UNPACKER_ENGINE_SEL,
)
from helpers.utils import passed_test

DATACOPY_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float32,
        DataFormat.Int32,
    ],
    same=True,
)


@pytest.mark.quasar
@parametrize(
    formats=DATACOPY_FORMATS,
    dest_acc=[DestAccumulation.Yes, DestAccumulation.No],
    math_transpose_faces=[Transpose.No, Transpose.Yes],
    implied_math_format=[ImpliedMathFormat.No],
)
def test_transpose_dest_quasar(
    formats,
    dest_acc,
    math_transpose_faces,
    implied_math_format,
):
    if formats.input_format.is_32_bit() and dest_acc == DestAccumulation.No:
        pytest.skip("Skip 32-bit dest with DestAccumulation.No")

    if (
        not formats.input_format.is_32_bit()
        and not math_transpose_faces == Transpose.Yes
    ):
        pytest.skip("Skip 16-bit dest with math_transpose_faces = Transpose.No")

    data_copy_type = DataCopyType.A2D
    input_dimensions = [64, 64]
    num_faces = 4

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    if formats.input_format == DataFormat.Int32:
        src_A = (torch.arange(0, src_A.numel()) * 10000).reshape_as(src_A)
        src_B = (torch.arange(0, src_B.numel()) * 10000).reshape_as(src_B)

    if formats.input_format == DataFormat.Float32:
        src_A = (
            torch.arange(0, src_A.numel(), dtype=torch.float32) * 10000.0
        ).reshape_as(src_A)
        src_B = (
            torch.arange(0, src_B.numel(), dtype=torch.float32) * 10000.0
        ).reshape_as(src_B)

    generate_datacopy_golden = get_golden_generator(DataCopyGolden)
    datacopy_tensor = generate_datacopy_golden(
        src_A,
        formats.output_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
    )

    t_matrix = get_golden_generator(TransposeGolden)
    golden_tensor = t_matrix.transpose_within_faces_multi_tile(
        datacopy_tensor,
        formats.output_format,
        num_tiles=tile_cnt_A,
        untilize=False,
        input_dimensions=input_dimensions,
    )
    if math_transpose_faces == Transpose.Yes:
        golden_tensor = t_matrix.transpose_faces_multi_tile(
            golden_tensor,
            formats.output_format,
            num_tiles=tile_cnt_A,
            tilize=False,
            input_dimensions=input_dimensions,
        )

    unpack_to_dest = True
    configuration = TestConfig(
        "sources/quasar/transpose_dest_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(implied_math_format),
            DATA_COPY_TYPE(data_copy_type),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest if unpack_to_dest else UnpackerEngine.UnpA
            ),
            DEST_SYNC(),
            MATH_TRANSPOSE_FACES(math_transpose_faces),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            DEST_INDEX(),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=unpack_to_dest,
        dest_acc=dest_acc,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    # Silence the large per-tile error dump from passed_test; we print a compact hex dump below on Int32 failures.
    passed = passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=False
    )
    if (not passed) and formats.output_format == DataFormat.Int32:
        # Dump a few mismatches in both decimal and hex (uint32) to spot hi16/lo16 mixing.
        diff = (golden_tensor != res_tensor).flatten()
        idx = torch.nonzero(diff, as_tuple=False).flatten()
        max_dump = 32

        def _u32_hex(x: torch.Tensor) -> list[str]:
            x64 = x.to(torch.int64) & 0xFFFFFFFF
            return [f"0x{int(v):08x}" for v in x64.tolist()]

        idx_sel = idx[:max_dump]
        g_sel = golden_tensor.flatten()[idx_sel]
        r_sel = res_tensor.flatten()[idx_sel]

        print("\nInt32 mismatch dump (first few):")
        for i, g, r, gh, rh in zip(
            idx_sel.tolist(),
            g_sel.tolist(),
            r_sel.tolist(),
            _u32_hex(g_sel),
            _u32_hex(r_sel),
        ):
            print(f"  idx={i:5d}  golden={int(g):12d} ({gh})  res={int(r):12d} ({rh})")

    assert passed, "Assert against golden failed"
