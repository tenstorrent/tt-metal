# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from typing import List

import pytest
import torch
from helpers.data_format_inference import data_formats, infer_data_formats
from helpers.format_config import DataFormat, FormatConfig, InputOutputFormat
from helpers.golden_generators import (
    DataCopyGolden,
    MatmulGolden,
    PackGolden,
    get_golden_generator,
    quantize_mx_tensor_chunked,
)
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    MathFidelity,
    PackerReluType,
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
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    NUM_FACES,
    RELU_CONFIG,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test


def print_diff_summary(
    golden_tensor: torch.Tensor,
    res_tensor: torch.Tensor,
    output_format: DataFormat,
    num_faces: int = 4,
    face_size: int = 256,  # 16*16
    max_samples: int = 16,
) -> None:
    """Concise mismatch report. Per-face breakdown surfaces face-arrangement bugs
    (e.g. EN_X2 matmul Dest face layout vs DataCopyGolden expectations)."""
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

    # Per-face mismatch counts (single tile assumed; for multi-tile callers can pass
    # face_size = num_elems_per_tile and num_faces accordingly).
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

    bad_idx = torch.where(mismatch)[0].tolist()
    print(f"  first {len(bad_idx)} mismatches (idx, golden, result, |diff|):")
    for i in bad_idx:
        print(
            f"    [{i:5d}]  g={float(g[i]):+.4f}  r={float(r[i]):+.4f}  |diff|={float(abs_diff[i]):.4f}"
        )


def generate_qsr_pack_combinations(
    formats_list: List[FormatConfig],
):
    """
    Generate pack combinations for Quasar pack tests.

    Args:
        formats_list: List of input/output format pairs

    Returns:
        List of (format, dest_acc, input_dimensions, relu_type) tuples
    """

    def is_supported_format_conversion(in_fmt, out_fmt):
        """Check if the format conversion is supported by packer. These format conversions are NOT dependent on the dest register mode."""
        # Skip if mixing integer and non-integer formats
        if in_fmt.is_integer() ^ out_fmt.is_integer():
            return False
        # If input format is Int16, output format must also be Int16, and vice versa
        if (in_fmt == DataFormat.Int16) ^ (out_fmt == DataFormat.Int16):
            return False
        return True

    def get_dest_acc_modes(in_fmt):
        """Determine valid dest register modes depending on the input format."""
        # Int16 requires 16bit mode dest register
        if in_fmt == DataFormat.Int16:
            return (DestAccumulation.No,)
        # Int32, Float32 (unpack_to_dest) requires 32bit mode dest register
        if in_fmt.is_32_bit():
            return (DestAccumulation.Yes,)
        return (DestAccumulation.No, DestAccumulation.Yes)

    def is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
        """Check if the format conversion is supported by packer. These format conversions are dependent on the dest register mode."""
        # Upcasting to Float32/Int32 requires dest_acc enabled
        if (
            out_fmt.is_32_bit()
            and not in_fmt.is_32_bit()
            and dest_acc == DestAccumulation.No
        ):
            return False
        # Int8<->UInt8 conversion requires dest_acc enabled
        if (
            dest_acc == DestAccumulation.No
            and in_fmt in (DataFormat.Int8, DataFormat.UInt8)
            and in_fmt != out_fmt
        ):
            return False
        return True

    dimensions_cache = {
        (dest_acc, dest_sync): tuple(
            generate_unary_input_dimensions(dest_acc, dest_sync)
        )
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes)
        for dest_sync in (DestSync.Half, DestSync.Full)
    }

    all_relu_types = [
        PackerReluType.NoRelu,
        PackerReluType.ZeroRelu,
        PackerReluType.MinThresholdRelu,
        PackerReluType.MaxThresholdRelu,
    ]

    dest_sync_modes = (DestSync.Half, DestSync.Full)

    combinations = []
    for fmt in formats_list:
        in_fmt, out_fmt = fmt.input_format, fmt.output_format

        if not is_supported_format_conversion(in_fmt, out_fmt):
            continue

        # Threshold ReLU modes are not supported for integer pack_src formats
        # (mirroring the pytest.skip guard in the test body).
        relu_types = (
            [PackerReluType.NoRelu, PackerReluType.ZeroRelu]
            if in_fmt.is_integer()
            else all_relu_types
        )
        for dest_acc in get_dest_acc_modes(in_fmt):
            if is_supported_dest_mode_dependent_conversion(in_fmt, out_fmt, dest_acc):
                for dest_sync in dest_sync_modes:
                    for dimensions in dimensions_cache[(dest_acc, dest_sync)]:
                        for relu_type in relu_types:
                            combinations.append(
                                (fmt, dest_acc, dest_sync, dimensions, relu_type)
                            )

    return combinations


PACK_FORMATS = input_output_formats(
    [
        DataFormat.Float16_b,
        DataFormat.Float16,
        DataFormat.Float32,
        DataFormat.Int32,
        DataFormat.Int8,
        DataFormat.UInt8,
        DataFormat.Int16,
        DataFormat.MxFp4,
        DataFormat.MxInt8,
        DataFormat.MxInt4,
        DataFormat.MxInt2,
    ]
)


@pytest.mark.quasar
@parametrize(
    formats_dest_acc_sync_dims_relu=generate_qsr_pack_combinations(PACK_FORMATS),
)
def test_pack_quasar(formats_dest_acc_sync_dims_relu, boot_mode=BootMode.DEFAULT):
    (formats, dest_acc, dest_sync_mode, input_dimensions, relu_type) = (
        formats_dest_acc_sync_dims_relu[0]
    )

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    unpack_to_dest = (
        formats.input_format.is_32_bit() and dest_acc == DestAccumulation.Yes
    )
    data_formats = infer_data_formats(
        input_format=formats.input_format,
        output_format=formats.output_format,
        is_fp32_dest_acc_en=dest_acc,
        unpacking_to_dest=unpack_to_dest,
    )

    # HW flow with relu: unpack input -> dest -> apply relu in pack_src
    # space -> pack to output (one MX quantization, block scale derived at pack
    # time from post-relu values). DataCopyGolden, given an MX output format,
    # does a pre-relu MxInt4 quantization that HW doesn't do. That extra
    # quantization can shift values across the relu threshold, producing
    # divergence from HW that grows with threshold-relu (most visible for
    # MxFp4 -> MxInt4 + MaxThresholdRelu). For MX outputs we route through
    # pack_src instead and apply the single output MX quantization ourselves
    # after relu. Non-MX outputs keep the existing path (saturate_integer etc.).
    num_faces = 4
    generate_golden = get_golden_generator(DataCopyGolden)
    datacopy_out_format = (
        data_formats.pack_src
        if formats.output_format.is_mx_format()
        else formats.output_format
    )
    golden_tensor = generate_golden(
        src_A,
        datacopy_out_format,
        num_faces=num_faces,
        input_dimensions=input_dimensions,
        input_format=formats.input_format,
    )

    tensor_average = (
        torch.mean(golden_tensor).item()
        if not formats.output_format.is_integer()
        else 0.0
    )

    relu_config = PackGolden.generate_relu_config(
        relu_type,
        relu_threshold=tensor_average,
        intermediate_format=data_formats.pack_src,
    )

    golden_tensor = PackGolden.apply_relu(
        golden_tensor,
        relu_config,
        data_formats.pack_src,
    )

    # Single output MX quantization, after relu — matches HW's pack-time
    # block-scale derivation from post-relu values.
    if formats.output_format.is_mx_format():
        golden_tensor = quantize_mx_tensor_chunked(
            golden_tensor.to(torch.bfloat16), formats.output_format
        )

    configuration = TestConfig(
        "sources/quasar/pack_quasar_test.cpp",
        formats,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            RELU_CONFIG(relu_config),
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
        boot_mode=boot_mode,
        disable_format_inference=(formats.input_format.is_mx_format()),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor,
        res_tensor,
        formats.output_format,
    )

    # Same method as test_pack.py for original ReLu testing and threshold tolerance issue
    # Could be adjusted to have Golden model HW behaviour for pack ReLu activation which applies ReLu on datum format in dst and then converts to output format
    # Check issue/request: https://github.com/tenstorrent/tt-llk/issues/1391
    if (
        not test_passed
        and relu_type
        in [
            PackerReluType.MinThresholdRelu,
            PackerReluType.MaxThresholdRelu,
        ]
        and PackGolden.is_relu_threshold_tolerance_issue(
            golden_tensor,
            res_tensor,
            relu_config,
            data_formats.pack_src,
            # MxInt4's lattice step is 0.25 * block_scale, so values that
            # disagree across a threshold can sit ~0.5 apart while still both
            # being "near the threshold" relative to the format's resolution.
            # The default rtol/atol of 0.01 is calibrated for finer-precision
            # formats (Bfp8_b etc.) and misses these legitimate near-threshold
            # flips for MxInt4. Use the format's tolerance entries.
            **(
                {"atol": 0.5, "rtol": 0.35}
                if formats.output_format == DataFormat.MxInt4
                else {}
            ),
        )
    ):
        test_passed = True

    assert test_passed, "Assert against golden failed"


# 2x-packed FP4 register-format variants for the pack pipeline. L1 stays MxFp4; the
# unpacker produces MxFp4_2x_A/B in the src registers. _A pairs with the FP16 family,
# _B with the FP16_b family. Per TEN-3634 MOV MXFP4_2x was removed from Quasar HW, so
# this path routes through an identity-SrcB matmul (see _llk_math_eltwise_unary_datacopy_x2_).
PACK_QSR_MXFP4_2X_FORMATS = [
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
@parametrize(
    format=PACK_QSR_MXFP4_2X_FORMATS,
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    tile_count=[1, 2, 4, 8],
)
def test_pack_quasar_mxfp4_2x(
    format, dest_sync_mode, tile_count, boot_mode=BootMode.DEFAULT
):
    # The LLK x2 datacopy path runs as a matmul block with ct_dim=tile_count, rt_dim=1:
    # one identity SrcB tile is loaded and reused across N data SrcA tiles via the matmul
    # reuse_a path. SrcA tiles tile-major in L1; one identity tile in L1 for SrcB.
    # dest_acc fixed to No (Float16/Float16_b output).
    dest_acc = DestAccumulation.No
    num_faces = 4

    # Skip variants that exceed Dest register capacity. DestSync.Half holds 8 tiles
    # (Float16), DestSync.Full holds 16. dest_acc=Yes would halve these.
    if dest_sync_mode == DestSync.Half and tile_count > 8:
        pytest.skip(f"tile_count={tile_count} exceeds DestSync.Half capacity")

    # SrcA: tile_count tiles of random MxFp4 data, laid out as a 32 x (32*tile_count)
    # block (1 tile row, tile_count tile cols). SrcB: single 32x32 identity tile.
    src_A_dimensions = [32, 32 * tile_count]
    src_B_dimensions = [32, 32]

    src_A, tile_cnt_A, _, _ = generate_stimuli(
        stimuli_format_A=format.input_format,
        input_dimensions_A=src_A_dimensions,
        stimuli_format_B=format.input_format,
        input_dimensions_B=src_B_dimensions,
        sfpu=False,
        output_format=format.output_format,
    )
    src_B = torch.eye(
        src_B_dimensions[0], src_B_dimensions[1], dtype=format_dict[format.input_format]
    )
    tile_cnt_B = 1

    tilized_A = tilize_block(
        src_A, dimensions=src_A_dimensions, stimuli_format=format.input_format
    )
    tilized_B = tilize_block(
        src_B, dimensions=src_B_dimensions, stimuli_format=format.input_format
    )

    # Quantize through MxFp4 in tilized order then untilize for golden (matches test_matmul_quasar).
    tilized_A_q = quantize_mx_tensor_chunked(
        tilized_A.flatten().to(torch.bfloat16), format.input_format
    ).reshape(tilized_A.shape)
    tilized_B_q = quantize_mx_tensor_chunked(
        tilized_B.flatten().to(torch.bfloat16), format.input_format
    ).reshape(tilized_B.shape)
    src_A_golden = untilize_block(
        tilized_A_q, stimuli_format=format.input_format, dimensions=src_A_dimensions
    )
    src_B_golden = untilize_block(
        tilized_B_q, stimuli_format=format.input_format, dimensions=src_B_dimensions
    )

    formats_config = data_formats(
        input_format=format.input_format,
        input_format_B=format.input_format_B,
        output_format=format.output_format,
        is_fp32_dest_acc_en=dest_acc,
        num_iterations=1,
        unpacking_to_dest=False,
        disable_format_inference=False,
        register_format_hint=format.register_format_hint,
    )[0]
    pack_src_format = formats_config.pack_src

    # Operand order: kernel wires python_src_A->SrcA, python_src_B->SrcB. HW computes
    # Dest = SrcB * SrcA = identity_32x32 * data_32x(32N), shape (32, 32N) = data.
    # In the FP4-2x scheme MatmulGolden models, so golden = MatmulGolden(src_B, src_A).
    generate_golden = get_golden_generator(MatmulGolden)
    golden_tensor = generate_golden(
        src_B_golden,
        src_A_golden,
        format.output_format,
        MathFidelity.LoFi,
        input_A_dimensions=src_B_dimensions,
        input_B_dimensions=src_A_dimensions,
        tilize=True,
        input_A_format=format.input_format,
        input_B_format=format.input_format,
        math_format=pack_src_format,
        dest_acc=dest_acc,
    )

    relu_config = PackGolden.generate_relu_config(
        PackerReluType.NoRelu,
        relu_threshold=0.0,
        intermediate_format=format.output_format,
    )

    configuration = TestConfig(
        "sources/quasar/pack_quasar_mxfp4_2x_test.cpp",
        format,
        templates=[
            IMPLIED_MATH_FORMAT(ImpliedMathFormat.Yes),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TEST_FACE_DIMS(),
            NUM_FACES(num_faces),
            TILE_COUNT(tile_cnt_A),
            RELU_CONFIG(relu_config),
        ],
        variant_stimuli=StimuliConfig(
            tilized_A.flatten(),
            format.input_format,
            tilized_B.flatten(),
            format.input_format,
            format.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=tile_cnt_A,
            num_faces=num_faces,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
        # Inference must run so register_format_hint propagates to unpack_A_dst etc.
        disable_format_inference=False,
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[format.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    test_passed = passed_test(
        golden_tensor, res_tensor, format.output_format, print_errors=False
    )

    if not test_passed:
        print_diff_summary(
            golden_tensor, res_tensor, format.output_format, num_faces=num_faces
        )

    assert test_passed, "Assert against golden failed"
