# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Test for tt-llk#1700 item 9: fp32 exp (non-approximate, 32-bit Dest) on special inputs,
# including negative-signed NaN bit patterns, which float('nan') never produces.
import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    ApproximationMode,
    DataCopyType,
    DestAccumulation,
    DestSync,
    MathOperation,
    PerfRunType,
    UnpackerEngine,
)
from helpers.param_config import parametrize, resolve_quasar_sfpu_variant
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    DATA_COPY_TYPE,
    DEST_INDEX,
    DEST_SYNC,
    LOOP_FACTOR,
    MATH_OP,
    NUM_FACES,
    TEST_FACE_DIMS,
    TILE_COUNT,
    TYPECAST_FORMATS,
    UNPACKER_ENGINE_SEL,
)
from helpers.tile_constants import MAX_NUM_FACES
from helpers.utils import result_tensor

TILE_DATUMS = 1024

# (name, input bit pattern, check on the output float)
SPECIAL_LANES = [
    ("-qNaN 0xFFC00000", 0xFFC00000, torch.isnan),
    ("-sNaN 0xFF800001", 0xFF800001, torch.isnan),
    ("-NaN payload 0xFFFFFFFF", 0xFFFFFFFF, torch.isnan),
    ("+qNaN 0x7FC00000", 0x7FC00000, torch.isnan),
    ("+NaN payload 0x7FFFFFFF", 0x7FFFFFFF, torch.isnan),
    ("+Inf", 0x7F800000, lambda y: torch.isinf(y) & (y > 0)),
    ("-Inf", 0xFF800000, lambda y: y == 0),
    ("-100", 0xC2C80000, lambda y: y == 0),
    ("0", 0x00000000, lambda y: y == 1),
    ("1", 0x3F800000, lambda y: torch.isclose(y, torch.tensor(2.7182817))),
]

# Spread each special value over the four faces so a face-dependent path cannot hide it.
LANE_STRIDE = TILE_DATUMS // 4


def _input_tile() -> torch.Tensor:
    bits = torch.zeros(TILE_DATUMS, dtype=torch.int32)  # 0.0 elsewhere: exp(0) = 1
    for i, (_, pattern, _) in enumerate(SPECIAL_LANES):
        for face in range(4):
            bits[face * LANE_STRIDE + i] = torch.tensor(pattern, dtype=torch.int64).to(
                torch.int32
            )
    return bits.view(torch.float32)


@pytest.mark.quasar
@parametrize(dest_sync=[DestSync.Half])
def test_sfpu_exp_special_values_quasar(dest_sync):
    # A lone parametrize argument arrives wrapped in a 1-tuple.
    (dest_sync,) = dest_sync
    mathop = MathOperation.Exp
    formats = InputOutputFormat(DataFormat.Float32, DataFormat.Float32)
    variant = resolve_quasar_sfpu_variant(mathop, formats, DestAccumulation.Yes)
    assert variant is not None, "no Quasar SFPU route for fp32 exp with 32-bit Dest"

    src_A = _input_tile()
    src_B = torch.zeros_like(src_A)

    test_config_kwargs = {
        "test_name": "sources/quasar/eltwise_unary_sfpu_quasar_test.cpp",
        "formats": variant.formats,
        "templates": [
            MATH_OP(mathop=mathop),
            APPROX_MODE(ApproximationMode.No),
            DATA_COPY_TYPE(DataCopyType.A2D),
            UNPACKER_ENGINE_SEL(
                UnpackerEngine.UnpDest
                if variant.unpack_to_dest
                else UnpackerEngine.UnpA
            ),
            DEST_SYNC(dest_sync),
            TYPECAST_FORMATS(),
        ],
        "runtimes": [
            TILE_COUNT(1),
            NUM_FACES(MAX_NUM_FACES),
            TEST_FACE_DIMS(),
            DEST_INDEX(0),
            LOOP_FACTOR(1),
        ],
        "variant_stimuli": StimuliConfig(
            src_A,
            variant.formats.input_format,
            src_B,
            variant.formats.input_format,
            variant.formats.output_format,
            tile_count_A=1,
            tile_count_B=1,
            tile_count_res=1,
            num_faces=MAX_NUM_FACES,
        ),
        "unpack_to_dest": variant.unpack_to_dest,
        "dest_acc": variant.dest_acc,
    }

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=(PerfRunType.L1_TO_L1,),
        test_config_kwargs=test_config_kwargs,
    )
    variant.apply_formats(configuration.formats_config)

    res = result_tensor(
        configuration.run().result, variant.formats.output_format, src_A
    ).to(torch.float32)

    failures = []
    for i, (name, pattern, check) in enumerate(SPECIAL_LANES):
        for face in range(4):
            lane = face * LANE_STRIDE + i
            y = res[lane]
            if not bool(check(y)):
                out_bits = y.view(torch.int32).item() & 0xFFFFFFFF
                failures.append(
                    f"{name}: lane {lane} (face {face}) in 0x{pattern:08X} -> "
                    f"out 0x{out_bits:08X} ({y.item()})"
                )
    assert not failures, "exp special values wrong:\n  " + "\n  ".join(failures)
