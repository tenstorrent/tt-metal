# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""LLK 2.0 body of ttnn's bmm_large_block_zm_fused_bias_activation on one core.

The kernel (kernels/bmm_large_block_zm_fused_bias_activation.h, driven by llk2_bmm_fused_compute_api_test.cpp)
walks OUT_BLOCKS output blocks of [RT, CT] tiles, each over K_BLOCKS blocks of KB tiles along K, accumulating by
spill + reload or by packer L1 accumulation, then optionally adds a row-broadcast bias, runs an activation from the
pack thread and packs with ReLU. The host plays the reader kernels: it lays in0 / in1 out in the order the kernel
pops them (in1's blocks once per output block). Golden: matmul, + bias row, + activation, + ReLU, from the
harness generators.
"""

import struct
from dataclasses import dataclass

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import (
    RELU_MAX_THRESHOLD,
    BroadcastGolden,
    EltwiseBinaryGolden,
    MatmulGolden,
    PackGolden,
    UnarySFPUGolden,
    get_golden_generator,
)
from helpers.llk_params import (
    BroadcastType,
    DestAccumulation,
    MathFidelity,
    MathOperation,
    PackerReluType,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.stimuli_generator import StimuliSpec, generate_stimuli
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    DEST_SYNC,
    MATH_FIDELITY,
    TemplateParameter,
)
from helpers.tilize_untilize import tilize_block
from helpers.utils import passed_test
from llk2_common import TILE_DATUMS, TILE_DIM, blocks_in_pop_order, tilize_operands, unwrap_case
from llk2_paths import cpp_source, kernel_test_include_dirs

HARDTANH_MIN, HARDTANH_MAX = -1.0, 1.0  # the harness golden's CLAMP_MIN / CLAMP_MAX

# Activation: (llk2::VecFunction enumerator, harness op, param0, param1); params as FP32 values.
ACTIVATIONS = {
    "none": ("None", None, 0.0, 0.0),
    "silu": ("Silu", MathOperation.Silu, 0.0, 0.0),
    "relu_max": ("ReluMax", MathOperation.ReluMax, RELU_MAX_THRESHOLD, 0.0),
    "hardtanh": ("Hardtanh", MathOperation.Hardtanh, HARDTANH_MIN, HARDTANH_MAX),
}


def _fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


@dataclass
class LLK2_BMM_FUSED(TemplateParameter):
    rt_dim: int
    ct_dim: int
    kb_dim: int
    k_blocks: int
    out_blocks: int
    l1_acc: bool
    fuse_bias: bool
    pack_relu: bool
    activation: str

    def convert_to_cpp(self) -> str:
        function, _, p0, p1 = ACTIVATIONS[self.activation]
        return (
            f"constexpr std::uint32_t RT_DIM = {self.rt_dim};\n"
            f"constexpr std::uint32_t CT_DIM = {self.ct_dim};\n"
            f"constexpr std::uint32_t KB_DIM = {self.kb_dim};\n"
            f"constexpr std::uint32_t K_BLOCKS = {self.k_blocks};\n"
            f"constexpr std::uint32_t OUT_BLOCKS = {self.out_blocks};\n"
            f"constexpr bool L1_ACC = {str(self.l1_acc).lower()};\n"
            f"constexpr bool FUSE_BIAS = {str(self.fuse_bias).lower()};\n"
            f"constexpr bool PACK_RELU = {str(self.pack_relu).lower()};\n"
            '#include "resolver/vec_function.h"\n'
            f"constexpr auto ACTIVATION = llk2::VecFunction::{function};\n"
            f"constexpr std::uint32_t ACTIVATION_P0 = {hex(_fp32_bits(p0))};\n"
            f"constexpr std::uint32_t ACTIVATION_P1 = {hex(_fp32_bits(p1))};"
        )


# (format, Dest accumulation, fidelity)
FORMATS = [
    (
        InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b),
        DestAccumulation.No,
        MathFidelity.LoFi,
    ),
    (
        InputOutputFormat(DataFormat.Bfp8_b, DataFormat.Bfp8_b),
        DestAccumulation.No,
        MathFidelity.LoFi,
    ),
    (
        InputOutputFormat(DataFormat.Float32, DataFormat.Float32),
        DestAccumulation.Yes,
        MathFidelity.HiFi4,
    ),
]

# (rt, ct, kb, k_blocks, out_blocks); every one fits a 32-bit Dest section (4 tiles)
BLOCKINGS = [
    (1, 1, 1, 1, 1),
    (2, 2, 1, 2, 1),
    (1, 2, 2, 3, 2),
    (2, 1, 1, 3, 1),
    (1, 4, 1, 2, 2),
    (2, 2, 2, 3, 1),
]


BLOCKINGS_PER_VARIANT = 2  # each variant runs on this many blockings


def _cases():
    """Every (l1_acc, bias, activation, relu) variant per format, each on BLOCKINGS_PER_VARIANT blockings, rotating
    through BLOCKINGS."""
    cases = []
    variants = [
        (l1_acc, bias, act, relu)
        for l1_acc in (False, True)
        for bias in (False, True)
        for act in ACTIVATIONS
        for relu in (False, True)
    ]
    for f, fmt in enumerate(FORMATS):
        for v, variant in enumerate(variants):
            for b in range(BLOCKINGS_PER_VARIANT):
                blocking = BLOCKINGS[(BLOCKINGS_PER_VARIANT * v + b + f) % len(BLOCKINGS)]
                cases.append((fmt, blocking, variant))
    return cases


@parametrize(case=_cases())
def test_llk2_bmm_fused(case):
    (
        (formats, dest_acc, fidelity),
        (rt, ct, kb, k_blocks, out_blocks),
        (l1_acc, bias, act, relu),
    ) = unwrap_case(case)
    inter = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )  # partials and bias
    torch.manual_seed(0)

    kt = kb * k_blocks
    dims_A = [out_blocks * rt * TILE_DIM, kt * TILE_DIM]
    dims_B = [kt * TILE_DIM, ct * TILE_DIM]
    dims_out = [dims_A[0], dims_B[1]]
    out_tiles = out_blocks * rt * ct

    spec = StimuliSpec.uniform(low=-1.0, high=1.0)
    src_A, _, src_B, _ = generate_stimuli(
        stimuli_format_A=formats.input_format,
        input_dimensions_A=dims_A,
        stimuli_format_B=formats.input_format,
        input_dimensions_B=dims_B,
        spec_A=spec,
        spec_B=spec,
    )
    src_bias = (torch.rand(TILE_DIM, ct * TILE_DIM) * 2 - 1).to(format_dict[inter])
    tilized_bias = tilize_block(
        src_bias, dimensions=[TILE_DIM, ct * TILE_DIM], stimuli_format=inter
    ).flatten()

    # Golden, stage by stage in the intermediate format; the last stage in the output format.
    _, sfpu_op, _, _ = ACTIVATIONS[act]
    stages = [bias, sfpu_op is not None, relu]
    last = max([i for i, s in enumerate(stages) if s], default=-1)

    def fmt(stage):
        return formats.output_format if stage >= last else inter

    golden = get_golden_generator(MatmulGolden)(
        src_A,
        src_B,
        fmt(-1),
        fidelity,
        input_A_dimensions=dims_A,
        input_B_dimensions=dims_B,
        tilize=True,
        input_A_format=formats.input_format,
        input_B_format=formats.input_format,
    )
    if bias:
        golden_bias = get_golden_generator(BroadcastGolden)(
            BroadcastType.Row,
            tilized_bias.repeat(out_blocks * rt),
            inter,
            tile_cnt=out_tiles,
        )
        golden = get_golden_generator(EltwiseBinaryGolden)(
            MathOperation.Elwadd,
            golden,
            golden_bias,
            fmt(0),
            MathFidelity.LoFi,
            input_format=inter,
            input_format_B=None,
            dest_acc=dest_acc,
        )
    if sfpu_op is not None:
        golden = get_golden_generator(UnarySFPUGolden)(
            sfpu_op,
            golden,
            fmt(1),
            dest_acc,
            inter,
            dims_out,
            relu_max_threshold=RELU_MAX_THRESHOLD,
        )
    if relu:
        relu_config = PackGolden.generate_relu_config(
            PackerReluType.ZeroRelu, relu_threshold=0.0, intermediate_format=inter
        )
        golden = PackGolden.apply_relu(golden, relu_config, inter)
    golden = torch.as_tensor(golden).to(format_dict[formats.output_format])

    tilized_A, tilized_B = tilize_operands(src_A, dims_A, src_B, dims_B, formats.input_format)

    # The reader kernels' job: in0 block (o, k), and in1 block k again for every o.
    order = [(o, k) for o in range(out_blocks) for k in range(k_blocks)]
    in0 = blocks_in_pop_order(tilized_A, out_blocks * rt, kt, rt, kb, order)
    in1 = blocks_in_pop_order(tilized_B, kt, ct, kb, ct, [(k, 0) for _, k in order])
    # buffer_C: the bias tiles, then the partials buffer's pages (one output block).
    buffer_C = torch.cat([tilized_bias.float(), torch.zeros(rt * ct * TILE_DATUMS)])

    configuration = TestConfig(
        cpp_source("llk2_bmm_fused_compute_api_test.cpp"),
        formats,
        templates=[
            MATH_FIDELITY(fidelity),
            DEST_SYNC(),
            LLK2_BMM_FUSED(rt, ct, kb, k_blocks, out_blocks, l1_acc, bias, relu, act),
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
            tile_count_res=out_tiles,
            buffer_C=buffer_C,
            stimuli_C_format=inter,
            tile_count_C=buffer_C.numel() // TILE_DATUMS,
        ),
        dest_acc=dest_acc,
        compile_time_formats=True,
        include_dirs=kernel_test_include_dirs(),
    )

    res = configuration.run().result
    assert len(res) == len(
        golden
    ), "Result tensor and golden tensor are not of the same length"
    res_tensor = torch.tensor(res, dtype=format_dict[formats.output_format])
    assert passed_test(
        golden, res_tensor, formats.output_format
    ), "Assert against golden failed"
