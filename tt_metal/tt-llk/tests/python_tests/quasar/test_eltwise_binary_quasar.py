# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import (
    generate_unary_input_dimensions,
    input_output_formats,
    parametrize,
)
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import generate_stimuli
from helpers.test_config import BootMode, TestConfig
from helpers.test_variant_parameters import (
    ACC_TO_DEST,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    NUM_TILES_IN_BLOCK,
    OUTPUT_TILE_CNT,
    TEST_FACE_DIMS,
)
from helpers.tile_shape import construct_tile_shape
from helpers.utils import passed_test, tolerances


def print_diff_summary(
    golden_tensor: torch.Tensor,
    res_tensor: torch.Tensor,
    output_format: DataFormat,
    num_faces: int = 4,
    face_size: int = 256,  # 16*16
    max_samples: int = 16,
) -> None:
    """Concise mismatch report. Per-face breakdown surfaces face-arrangement bugs
    (e.g. FP4-2x SrcA only reads lower nibble under non-mmul ops)."""
    g = golden_tensor.type(format_dict[output_format]).flatten()
    r = res_tensor.type(format_dict[output_format]).flatten()
    n = min(g.numel(), r.numel())
    g, r = g[:n], r[:n]

    tol = tolerances[output_format]
    is_close = torch.isclose(g, r, rtol=tol.rtol, atol=tol.atol)
    is_nan = torch.isnan(g) & torch.isnan(r)
    mismatch = ~(is_close | is_nan)
    n_bad = int(mismatch.sum())

    print(
        f"\n--- diff summary [{output_format.name}, atol={tol.atol}, rtol={tol.rtol}] ---"
    )
    print(f"  total={n}  mismatch={n_bad}  ratio={n_bad / max(n, 1):.4f}")
    if n_bad == 0:
        return

    abs_diff = (g.float() - r.float()).abs()
    max_idx = int(abs_diff.argmax())
    print(
        f"  abs_diff max={float(abs_diff.max()):.4g} mean={float(abs_diff.mean()):.4g}"
        f"  worst@{max_idx}: g={float(g[max_idx]):.4g} r={float(r[max_idx]):.4g}"
    )

    elems_per_tile = face_size * num_faces
    n_tiles = n // elems_per_tile
    if n_tiles >= 1:
        per_face = []
        for f in range(num_faces):
            start = f * face_size
            end = start + face_size
            face_mismatch = mismatch[start:end].sum().item()
            per_face.append(f"f{f}={face_mismatch}/{face_size}")
        print(f"  tile 0 per-face mismatches: {'  '.join(per_face)}")

    bad_idx = torch.where(mismatch)[0][:max_samples].tolist()
    print(f"  first {len(bad_idx)} mismatches (idx, golden, result, |diff|):")
    for i in bad_idx:
        print(
            f"    [{i:5d}]  g={float(g[i]):+.4f}  r={float(r[i]):+.4f}  |diff|={float(abs_diff[i]):.4f}"
        )


ELTWISE_DIMENSIONS = [
    (dest_sync, dims, DestAccumulation.No)
    for dest_sync in (DestSync.Half, DestSync.Full)
    for dims in generate_unary_input_dimensions(DestAccumulation.No, dest_sync)
]


# For acc_to_dest setting, accumulate two result tiles into dest. Can be extended.
def get_num_tiles_per_accumulation(acc_to_dest: bool) -> int:
    return 2 if acc_to_dest else 1


_TILE_SHAPE = construct_tile_shape()


def valid_acc_to_dest(dest_sync_dims_dest_acc) -> list:
    """Pick the acc_to_dest modes worth running for a given input size.

    acc_to_dest=True accumulates `get_num_tiles_per_accumulation(True)` result tiles into
    dest, so it only makes sense when the tile count is a non-zero multiple of that.
    """
    _, input_dimensions, _ = dest_sync_dims_dest_acc
    total_tiles = (
        input_dimensions[0] * input_dimensions[1]
    ) // _TILE_SHAPE.total_tile_size()

    per_acc = get_num_tiles_per_accumulation(True)
    if total_tiles >= per_acc and total_tiles % per_acc == 0:
        return [False, True]
    return [False]


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.MxFp8R,
            DataFormat.MxFp8P,
            DataFormat.MxFp4,
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
            DataFormat.Float16_b,
            DataFormat.Float16,
        ],
    ),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    # Math fidelity only affects multiplication; for add/sub only LoFi is meaningful.
    math_fidelity=lambda mathop: (
        [MathFidelity.LoFi]
        if mathop in [MathOperation.Elwadd, MathOperation.Elwsub]
        else [
            MathFidelity.LoFi,
            MathFidelity.HiFi2,
            MathFidelity.HiFi3,
            MathFidelity.HiFi4,
        ]
    ),
    implied_math_format=lambda formats: (
        [
            ImpliedMathFormat.No,
            ImpliedMathFormat.Yes,
        ]
        if not formats.input_format.is_mx_format()
        else [ImpliedMathFormat.Yes]
    ),
    dest_sync_dims_dest_acc=ELTWISE_DIMENSIONS,
    acc_to_dest=valid_acc_to_dest,
    num_faces=[4],
)
def test_eltwise_binary(
    formats,
    mathop,
    math_fidelity,
    implied_math_format,
    dest_sync_dims_dest_acc,
    acc_to_dest,
    num_faces,
    boot_mode=BootMode.DEFAULT,
):
    dest_sync_mode, input_dimensions, dest_acc = dest_sync_dims_dest_acc

    num_tiles_per_accumulation = get_num_tiles_per_accumulation(acc_to_dest)

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
        output_format=formats.output_format,
    )

    tile_cnt_res = src_A.numel() // (
        _TILE_SHAPE.total_tile_size() * num_tiles_per_accumulation
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_format=formats.input_format,
        acc_to_dest=acc_to_dest,
        tile_shape=_TILE_SHAPE,
        num_tiles_per_accumulation=num_tiles_per_accumulation,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            DEST_SYNC(dest_sync_mode),
            ACC_TO_DEST(acc_to_dest),
        ],
        runtimes=[
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_res),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            NUM_TILES_IN_BLOCK(num_tiles_per_accumulation),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            formats.input_format,
            src_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_res,
            num_faces=num_faces,
        ),
        # Determine unpack_to_dest based on format and accumulation mode
        unpack_to_dest=(
            formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
        ),
        dest_acc=dest_acc,
        boot_mode=boot_mode,
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting
        # This ensures Python-side format inference uses Float16_b for MX internal math
        disable_format_inference=(implied_math_format == ImpliedMathFormat.Yes),
    )

    res_from_L1 = configuration.run().result

    # Verify results match golden
    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    assert passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    ), "Assert against golden failed"


# 2x-packed FP4 register-format variants for ELWMUL. L1 stays MxFp4; the unpacker
# produces MxFp4_2x_A/B in the src registers. CONFIRMED FAIL on current Quasar HW
# (sim run 2026-05-11): every Dest element reads as exactly 0.0, all 4 faces, 100% diff.
#
# Why: tt_instruction_issue.sv:6985 sets op_mmul = MVMUL||MVMULDI||GAPOOL only. ELWMUL
# does NOT assert op_mmul, so tt_srca_format_mux.sv:57's FP4-2x branch is skipped and
# the else branch (line 60-62) reads the SrcA datum via the t.fp.expo/sign/man union
# fields. But the unpacker stored the data in the t.fp4.datum[0/1] sub-region of the
# union when format=MxFp4_2x_A/B, so t.fp.* reads the unused-padding bits = all zeros.
# SrcA effectively reads as 0.0 -> ELWMUL produces 0.0 for every element.
#
# Verif cctb-fpu-directed.yml has zero elwmul-mxfp4 directed tests; only MVMUL/MVMULDI/
# GAPOOL have MxFp4_2x coverage. Kept as regression marker: if HW support is added
# (op_mmul list expanded, or a separate FP4-2x elwise path wired in the format mux),
# this test should start producing non-zero results that match the golden.
ELTWISE_BINARY_MXFP4_2X_FORMATS = [
    InputOutputFormat(
        DataFormat.MxFp4,
        DataFormat.Float16,
        register_format_hint=DataFormat.MxFp4_2x_A,
    ),
    InputOutputFormat(
        DataFormat.MxFp4,
        DataFormat.Float16_b,
        register_format_hint=DataFormat.MxFp4_2x_B,
    ),
]


@pytest.mark.quasar
# @pytest.mark.xfail(
#     reason=(
#         "ELWMUL does not honor MXFP4_2x sub-datum expansion on current Quasar HW: "
#         "tt_srca_format_mux.sv FP4-2x branch is op_mmul-gated, and "
#         "tt_instruction_issue.sv:6985 op_mmul = MVMUL||MVMULDI||GAPOOL only. "
#         "Regression marker: should pass if HW gains elwise FP4-2x support."
#     ),
#     strict=True,
# )
@parametrize(
    format=ELTWISE_BINARY_MXFP4_2X_FORMATS,
    dest_sync_mode=[DestSync.Half, DestSync.Full],
)
def test_eltwise_binary_mxfp4_2x_elwmul(
    format, dest_sync_mode, boot_mode=BootMode.DEFAULT
):
    dest_acc = DestAccumulation.No
    input_dimensions = [32, 32]
    num_faces = 4
    mathop = MathOperation.Elwmul
    math_fidelity = MathFidelity.LoFi
    acc_to_dest = False
    tile_shape = construct_tile_shape()

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=input_dimensions,
        output_format=format.output_format,
    )

    tile_cnt_res = src_A.numel() // tile_shape.total_tile_size()

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    golden_tensor = generate_golden(
        mathop,
        src_A,
        src_B,
        format.output_format,
        math_fidelity,
        input_format=format.input_format,
        acc_to_dest=acc_to_dest,
        tile_shape=tile_shape,
        num_tiles_per_accumulation=1,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_test.cpp",
        format,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
            ACC_TO_DEST(acc_to_dest),
        ],
        runtimes=[
            INPUT_TILE_CNT(tile_cnt_A),
            OUTPUT_TILE_CNT(tile_cnt_res),
            NUM_FACES(num_faces),
            TEST_FACE_DIMS(),
            NUM_TILES_IN_BLOCK(1),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            format.input_format,
            src_B,
            format.input_format,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_A,
            tile_count_res=tile_cnt_res,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
        # Inference must run so register_format_hint flows to unpack_A_dst/unpack_B_dst.
        disable_format_inference=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[format.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor,
        res_tensor,
        format.output_format,
        print_errors=False,
        print_pcc=False,
    )

    if not test_passed:
        print_diff_summary(
            golden_tensor, res_tensor, format.output_format, num_faces=num_faces
        )

    assert test_passed, "Assert against golden failed"
