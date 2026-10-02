# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Mid-kernel FP32 dest-acc toggle (_llk_set_fp32_dest_acc_) on Quasar.

The kernel is compiled for 16-bit dest and accumulates TILES_PER_PHASE input tiles into dest tile 0
twice: once with 16-bit dest, then, after toggling all three threads to 32-bit dest, once more.

Per phase, tile 0 of A holds x in [1, 2) and the remaining 8 tiles hold 2**-10 (B is all zeros). Each
addend is below half a bf16 ULP of x (2**-8), so:
  - 16-bit dest drops every addend: result == x (round-to-nearest and truncation agree);
  - 32-bit dest keeps them: result == x + 8 * 2**-10 == x + 2**-7, exactly one bf16 ULP above x, so the
    Float32 -> Float16_b pack is exact.
The two expected outputs differ in every element, so a toggle that does not take effect fails phase 1
and a toggle that leaks into phase 0 fails phase 0.
"""

import pytest
import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.llk_params import (
    DestAccumulation,
    DestSync,
    ImpliedMathFormat,
    PerfRunType,
)
from helpers.param_config import parametrize
from helpers.perf.core import create_test_or_perf_config
from helpers.stimuli_config import StimuliConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    IMPLIED_MATH_FORMAT,
    INPUT_TILE_CNT,
    NUM_FACES,
    OUTPUT_TILE_CNT,
    TEST_FACE_DIMS,
    generate_input_dim,
)
from helpers.tile_shape import construct_tile_shape

# Must match TILES_PER_PHASE / NUM_PHASES in fp32_dest_acc_toggle_quasar_test.cpp.
TILES_PER_PHASE = 9
NUM_PHASES = 2
ADDEND = 2.0**-10

_TILE_SHAPE = construct_tile_shape()
_TILE_ELEMS = _TILE_SHAPE.total_tile_size()


def _phase_stimuli(generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    # x in [1, 2 - 2**-7] so x + 2**-7 stays in [1, 2] and is exactly representable in bf16.
    x = (1.0 + torch.rand(_TILE_ELEMS, generator=generator) * (1.0 - 2.0**-7)).to(
        torch.bfloat16
    )
    addends = torch.full(
        ((TILES_PER_PHASE - 1) * _TILE_ELEMS,), ADDEND, dtype=torch.bfloat16
    )
    return x, torch.cat([x, addends])


@pytest.mark.quasar
@parametrize(
    formats=[InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)],
    run_types=[[PerfRunType.L1_TO_L1]],
)
def test_fp32_dest_acc_toggle_quasar(formats, run_types):
    generator = torch.Generator().manual_seed(0)
    x0, phase0_A = _phase_stimuli(generator)
    x1, phase1_A = _phase_stimuli(generator)

    src_A = torch.cat([phase0_A, phase1_A])
    src_B = torch.zeros_like(src_A)
    tile_cnt_in = NUM_PHASES * TILES_PER_PHASE
    tile_cnt_res = NUM_PHASES

    golden_phase0 = x0  # 16-bit dest: every addend is lost
    golden_phase1 = (x1.float() + (TILES_PER_PHASE - 1) * ADDEND).to(torch.bfloat16)
    assert torch.all(
        golden_phase1.float() == x1.float() + 2.0**-7
    ), "stimulus invariant broken"

    input_dimensions = [
        _TILE_SHAPE.total_row_dim() * tile_cnt_in,
        _TILE_SHAPE.total_col_dim(),
    ]
    num_faces = 4

    configuration = create_test_or_perf_config(
        is_perf=False,
        run_types=run_types,
        test_config_kwargs={
            "test_name": "sources/quasar/fp32_dest_acc_toggle_quasar_test.cpp",
            "formats": formats,
            "templates": [
                IMPLIED_MATH_FORMAT(ImpliedMathFormat.No),
                DEST_SYNC(DestSync.Full),
            ],
            "runtimes": [
                generate_input_dim(input_dimensions, input_dimensions),
                INPUT_TILE_CNT(tile_cnt_in),
                OUTPUT_TILE_CNT(tile_cnt_res),
                NUM_FACES(num_faces),
                TEST_FACE_DIMS(),
            ],
            "variant_stimuli": StimuliConfig(
                src_A,
                formats.input_format,
                src_B,
                formats.input_format,
                formats.output_format,
                tile_count_A=tile_cnt_in,
                tile_count_B=tile_cnt_in,
                tile_count_res=tile_cnt_res,
                num_faces=num_faces,
            ),
            "dest_acc": DestAccumulation.No,
        },
    )

    res = torch.tensor(configuration.run().result, dtype=torch.bfloat16)
    assert res.numel() == NUM_PHASES * _TILE_ELEMS, "unexpected result size"

    for phase, golden in enumerate((golden_phase0, golden_phase1)):
        got = res[phase * _TILE_ELEMS : (phase + 1) * _TILE_ELEMS]
        mismatches = int((got != golden).sum())
        assert mismatches == 0, (
            f"phase {phase} ({'32' if phase else '16'}-bit dest): {mismatches}/{_TILE_ELEMS} "
            f"elements differ, first got={got[got != golden][:4].tolist()} "
            f"expected={golden[got != golden][:4].tolist()}"
        )
