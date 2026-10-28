# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 dataflow buffer (DM) API: Tensors from wait_front / reserve_back feeding llk2::matmul.

The kernel walks OUT_BLOCKS output blocks of [RT, CT] tiles, each the sum over K_BLOCKS of an in0 block [RT, KB]
times an in1 block [KB, CT]. There are no data movement cores in the harness, so this host side plays the reader
kernels: it lays the input blocks out in L1 in the order the kernel pops them (in1's blocks once per output block),
and the kernel marks them all pushed. The result is the matmul of [OUT_BLOCKS * RT, K_BLOCKS * KB] by
[K_BLOCKS * KB, CT] tiles, checked against the same golden as test_llk2_matmul.py.
"""

from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat
from helpers.golden_generators import MatmulGolden, get_golden_generator
from helpers.llk_params import DestAccumulation, MathFidelity, format_dict
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
from llk2_common import DEST_HALF_TILES, TILE_DATUMS, TILE_DIM, blocks_in_pop_order, tilize_operands
from llk2_paths import cpp_source, kernel_test_include_dirs



@dataclass
class LLK2_DFB_DIMS(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kb_dim: int
    k_blocks: int
    out_blocks: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KB_DIM = {self.kb_dim};\n"
            f"constexpr std::uint32_t K_BLOCKS = {self.k_blocks};\n"
            f"constexpr std::uint32_t OUT_BLOCKS = {self.out_blocks};"
        )


# (rt, ct, kb, k_blocks, out_blocks)
_BLOCKINGS = [
    (1, 1, 1, 1, 1),
    (1, 1, 1, 3, 2),
    (2, 2, 1, 2, 2),
    (1, 4, 2, 2, 3),
    (2, 1, 3, 2, 2),
    (4, 2, 1, 2, 2),
]


def _combinations():
    combos = []
    for fmt in input_output_formats(
        [DataFormat.Float16_b, DataFormat.Bfp8_b, DataFormat.Float32], same=True
    ):
        for dest_acc in (DestAccumulation.No, DestAccumulation.Yes):
            if (
                fmt.input_format == DataFormat.Float32
                and dest_acc == DestAccumulation.No
            ):
                continue
            for blocking in _BLOCKINGS:
                rt, ct = blocking[0], blocking[1]
                if rt * ct <= DEST_HALF_TILES[dest_acc]:
                    combos.append((fmt, dest_acc, blocking))
    return combos


@parametrize(
    math_fidelity=[MathFidelity.LoFi, MathFidelity.HiFi4],
    format_dest_acc_and_blocking=_combinations(),
)
def test_llk2_dfb(math_fidelity, format_dest_acc_and_blocking):
    formats, dest_acc, (rt, ct, kb, k_blocks, out_blocks) = format_dest_acc_and_blocking
    torch_format = format_dict[formats.output_format]

    kt = kb * k_blocks
    dims_A = [out_blocks * rt * TILE_DIM, kt * TILE_DIM]
    dims_B = [kt * TILE_DIM, ct * TILE_DIM]

    spec = StimuliSpec.uniform(low=0.0, high=1.0)
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )

    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        formats.output_format,
        math_fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )

    tilized_A, tilized_B = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)

    # The reader kernels' job: block (o, k) of in0, and block k of in1 again for every o.
    order = [(o, k) for o in range(out_blocks) for k in range(k_blocks)]
    in0 = blocks_in_pop_order(tilized_A, out_blocks * rt, kt, rt, kb, order)
    in1 = blocks_in_pop_order(tilized_B, kt, ct, kb, ct, [(k, 0) for _, k in order])

    configuration = TestConfig(
        cpp_source("llk2_dfb_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(math_fidelity),
            DEST_SYNC(),
            LLK2_DFB_DIMS(rt, ct, kb, k_blocks, out_blocks),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            in0,
            formats.input_format,
            in1,
            formats.input_format,
            formats.output_format,
            tile_count_A=in0.numel() // TILE_DATUMS,
            tile_count_B=in1.numel() // TILE_DATUMS,
            tile_count_res=out_blocks * rt * ct,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=kernel_test_include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    assert passed_test(
        golden, torch.tensor(res, dtype=torch_format), formats.output_format
    ), "Assert against golden failed"
