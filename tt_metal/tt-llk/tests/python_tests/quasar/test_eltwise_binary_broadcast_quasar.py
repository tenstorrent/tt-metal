# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from helpers.constraints import (
    get_valid_dest_accumulation_modes,
    get_valid_math_fidelities,
)
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    BroadcastGolden,
    EltwiseBinaryGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestSync,
    ImpliedMathFormat,
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
    BROADCAST_TYPE,
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    MATH_FIDELITY,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
)
from helpers.utils import passed_test

TILE_ELEMS = 32 * 32
FACE_ELEMS = 16 * 16


def _print_diff(golden, res, max_print=64):
    g = golden.float().flatten()
    r = res.float().flatten()
    diff = (g - r).abs()
    mism = (diff > 0).nonzero(as_tuple=True)[0]
    n = g.numel()
    print(
        f"\n=== DIFF: {len(mism)}/{n} mismatches, max |diff|={diff.max().item():.4f} ==="
    )
    if len(mism) == 0:
        return

    # Per-tile / per-face mismatch counts
    n_tiles = n // TILE_ELEMS
    if n_tiles > 0 and n % TILE_ELEMS == 0:
        print(f"{'tile':>4} {'f0':>5} {'f1':>5} {'f2':>5} {'f3':>5} {'total':>6}")
        for t in range(n_tiles):
            base = t * TILE_ELEMS
            tile_mism = diff[base : base + TILE_ELEMS] > 0
            face_cnts = [
                int(tile_mism[f * FACE_ELEMS : (f + 1) * FACE_ELEMS].sum())
                for f in range(4)
            ]
            print(
                f"{t:>4} {face_cnts[0]:>5} {face_cnts[1]:>5} {face_cnts[2]:>5} {face_cnts[3]:>5} {sum(face_cnts):>6}"
            )

    # First N mismatches
    print(
        f"\n{'idx':>7} {'tile':>4} {'face':>4} {'row':>3} {'col':>3} {'golden':>10} {'result':>10} {'diff':>10}"
    )
    for i in mism[:max_print].tolist():
        t = i // TILE_ELEMS
        within_tile = i % TILE_ELEMS
        f = within_tile // FACE_ELEMS
        within_face = within_tile % FACE_ELEMS
        row, col = within_face // 16, within_face % 16
        print(
            f"{i:>7} {t:>4} {f:>4} {row:>3} {col:>3} "
            f"{g[i].item():>10.4f} {r[i].item():>10.4f} {diff[i].item():>10.4f}"
        )
    if len(mism) > max_print:
        print(f"... ({len(mism) - max_print} more)")


@pytest.mark.quasar
@parametrize(
    formats=input_output_formats(
        [
            DataFormat.Float16_b,
            DataFormat.Float16,
            DataFormat.MxFp4,
            DataFormat.MxInt8,
            DataFormat.MxInt4,
            DataFormat.MxInt2,
        ],
    ),
    dest_acc=lambda formats: get_valid_dest_accumulation_modes(formats),
    mathop=[
        MathOperation.Elwadd,
        MathOperation.Elwsub,
        MathOperation.Elwmul,
    ],
    broadcast_type=[
        BroadcastType.Column,
        BroadcastType.Row,
        BroadcastType.Scalar,
    ],
    math_fidelity=lambda formats, mathop: get_valid_math_fidelities(formats, mathop),
    implied_math_format=lambda formats: (
        [ImpliedMathFormat.No, ImpliedMathFormat.Yes]
        if not formats.input_format.is_mx_format()
        else [ImpliedMathFormat.Yes]
    ),
    dest_sync_mode=[DestSync.Half, DestSync.Full],
    input_dimensions=lambda dest_acc, dest_sync_mode: generate_unary_input_dimensions(
        dest_acc, dest_sync_mode
    ),
)
def test_eltwise_binary_broadcast_quasar(
    formats,
    dest_acc,
    mathop,
    broadcast_type,
    math_fidelity,
    implied_math_format,
    dest_sync_mode,
    input_dimensions,
    boot_mode=BootMode.DEFAULT,
):

    src_A, tile_cnt_A, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=input_dimensions,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=input_dimensions,
    )

    generate_broadcast_golden = get_golden_generator(BroadcastGolden)
    bcast_src_B_tensor = generate_broadcast_golden(
        broadcast_type,
        src_B,
        formats.output_format,
        num_faces=4,
        tile_cnt=tile_cnt_A,
        face_r_dim=16,
        input_format=formats.input_format,
    )

    generate_golden = get_golden_generator(EltwiseBinaryGolden)
    input_format = formats.input_format
    input_format_B = (
        DataFormat.Float16_b
        if formats.input_format.is_mx_format()
        else formats.input_format
    )
    golden_tensor = generate_golden(
        mathop,
        src_A,
        bcast_src_B_tensor,
        formats.output_format,
        math_fidelity,
        input_format=input_format,
        input_format_B=input_format_B,
    )

    configuration = TestConfig(
        "sources/quasar/eltwise_binary_broadcast_quasar_test.cpp",
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            MATH_OP(mathop=mathop),
            IMPLIED_MATH_FORMAT(implied_math_format),
            BROADCAST_TYPE(broadcast_type),
            DEST_SYNC(dest_sync_mode),
        ],
        runtimes=[
            TILE_COUNT(tile_cnt_A),
            NUM_FACES(4),
            TEST_FACE_DIMS(),
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
            num_faces=4,
        ),
        unpack_to_dest=False,
        dest_acc=dest_acc,
        boot_mode=boot_mode,
        # MX formats require disable_format_inference to match C++ IMPLIED_MATH_FORMAT setting.
        disable_format_inference=formats.input_format.is_mx_format(),
    )

    res_from_L1 = configuration.run().result

    assert len(res_from_L1) == len(
        golden_tensor
    ), "Result tensor and golden tensor are not of the same length"

    torch_format = format_dict[formats.output_format]
    res_tensor = torch.tensor(res_from_L1, dtype=torch_format)

    if not passed_test(
        golden_tensor, res_tensor, formats.output_format, print_errors=True
    ):
        _print_diff(golden_tensor, res_tensor)
        assert False, "Assert against golden failed"
