# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 Dest ring with Dest tensors of different shapes in one kernel (compute_api/dest.h).

Each iteration i of the kernel reserves three sections of the kernel's one Dest ring through three views:
  y = x_i @ w                  [1 x mm_ct] tiles   x_i: tile row i of A [iterations x kt]; w: B [kt x mm_ct]
  z = x_i[:add_ct] + B_i       [1 x add_ct] tiles  B_i: the first add_ct tiles of B's tile row i
  c = B[i, mm_ct - 1]          [1 x 1] tile, copied
and packs them to iteration i's mm_ct + add_ct + 1 output tiles, in that order. Interleaved, y and z are held in
Dest at once (two sections, released in reserve order); sequential, each tensor is released before the next reserve.
With a 16-bit Dest the score row is [1 x 8], a whole section, as SDPA's QK^T subblock; with a 32-bit Dest a section
holds 4 tiles, so the shapes halve.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import (
    EltwiseBinaryGolden,
    MatmulGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    DestAccumulation,
    MathFidelity,
    MathOperation,
    format_dict,
)
from helpers.param_config import input_output_formats, parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.utils import passed_test
from llk2_common import DEST_HALF_TILES, TILE_DATUMS, TILE_DIM, tilize_operands
from llk2_paths import cpp_source, include_dirs

KT = 4
ITERATIONS = 4  # 12 reserves: every shape lands in both sections


@dataclass
class LLK2_DEST_PARAMS(TemplateParameter):
    mm_ct: int
    add_ct: int
    sections: int
    interleave: bool

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t KT_DIM = {KT};\n"
            f"constexpr std::uint32_t ITERATIONS = {ITERATIONS};\n"
            f"constexpr std::uint32_t MM_CT_DIM = {self.mm_ct};\n"
            f"constexpr std::uint32_t ADD_CT_DIM = {self.add_ct};\n"
            f"#define DEST_SECTIONS {self.sections}\n"
            f"constexpr bool INTERLEAVE = {str(self.interleave).lower()};"
        )


# (dest_acc, sections, interleave)
_RINGS = [
    (DestAccumulation.No, 2, True),
    (DestAccumulation.No, 2, False),
    (DestAccumulation.No, 1, False),
    (DestAccumulation.Yes, 2, True),
]


@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi2],
    formats=input_output_formats([DataFormat.Float16_b], same=True),
    ring=_RINGS,
)
def test_llk2_dest(math_fidelity, formats, ring):
    dest_acc, sections, interleave = ring
    mm_ct = DEST_HALF_TILES[
        dest_acc
    ]  # a whole two-section section: 8 tiles (16-bit Dest), 4 (32-bit)
    add_ct = mm_ct // 2
    torch_format = format_dict[formats.output_format]

    dims_A = [ITERATIONS * TILE_DIM, KT * TILE_DIM]
    dims_B = [KT * TILE_DIM, mm_ct * TILE_DIM]
    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, tile_cnt_A, src_B, tile_cnt_B = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    tilized_A, tilized_B = tilize_operands(
        src_A, dims_A, src_B, dims_B, formats.input_format
    )
    tiles_A = tilized_A.view(ITERATIONS, KT, TILE_DATUMS)
    tiles_B = tilized_B.view(KT, mm_ct, TILE_DATUMS)

    golden_mm = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    ).view(ITERATIONS, mm_ct, TILE_DATUMS)

    golden = []
    for i in range(ITERATIONS):
        golden_add = get_golden_generator(EltwiseBinaryGolden)(
            MathOperation.Elwadd,
            tiles_A[i, :add_ct].flatten(),
            tiles_B[i, :add_ct].flatten(),
            formats.output_format,
            MathFidelity.LoFi,
            input_format=formats.input_format,
            dest_acc=dest_acc,
        )
        golden += [
            golden_mm[i].flatten(),
            torch.as_tensor(golden_add).flatten().to(torch_format),
            tiles_B[i, mm_ct - 1].to(torch_format),
        ]
    golden = torch.cat([g.to(torch_format) for g in golden])

    configuration = TestConfig(
        cpp_source("llk2_dest_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_DEST_PARAMS(mm_ct, add_ct, sections, interleave),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            tilized_A,
            formats.input_format,
            tilized_B,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt_A,
            tile_count_B=tile_cnt_B,
            tile_count_res=ITERATIONS * (mm_ct + add_ct + 1),
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden, torch.tensor(res, dtype=torch_format), formats.output_format
    ), "Assert against golden failed"
