# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

# Test for the fused o_norm SFPU entry (Kimi-K3 / KDA "o_norm" node):
#
#     o_norm = RMSNorm(o) * gamma2 * sigmoid(g_out)
#
# RMSNorm reduces over the head-dim, which is laid out along the rows of each
# [head_dim, heads] operand block (heads processed in parallel down the columns).
# See ckernel_sfpu_o_norm.h for the full tilization contract.

import struct

import torch
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE, TILE_DIM
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import APPROX_MODE, O_NORM_CONFIG
from helpers.tilize_untilize import tilize_block, untilize_block
from helpers.utils import passed_test

EPS = 1e-6


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _o_norm_golden(
    o: torch.Tensor, gamma2: torch.Tensor, g_out: torch.Tensor, eps: float
) -> torch.Tensor:
    """o_norm = RMSNorm(o) * gamma2 * sigmoid(g_out).

    o, gamma2 and g_out are [head_dim, heads] (head_dim = reduction axis along
    rows). The RMS scale is computed per column (head) over the rows and shared
    across all rows.
    """
    o = o.to(torch.float32)
    gamma2 = gamma2.to(torch.float32)
    g_out = g_out.to(torch.float32)

    mean_sq = torch.mean(o * o, dim=0, keepdim=True)  # (1, heads)
    inv_rms = torch.rsqrt(mean_sq + eps)  # (1, heads)
    return o * inv_rms * gamma2 * torch.sigmoid(g_out)


# The reference kernel is written for 16-bit bf16 DEST (matches the tt-llk PR
# #1674 scope); dest_acc=Yes (32-bit DEST) is left for follow-up.
@parametrize(
    dest_acc=[DestAccumulation.No],
    num_reduce_tiles=[1],
)
def test_o_norm(dest_acc, num_reduce_tiles):
    torch.manual_seed(0)

    formats = InputOutputFormat(DataFormat.Float16_b, DataFormat.Float16_b)
    torch_format = format_dict[formats.input_format]

    # [rows, cols] = [head_dim, heads].
    input_dimensions = [num_reduce_tiles * TILE_DIM, TILE_DIM]
    tile_cnt = input_dimensions[0] * input_dimensions[1] // ELEMENTS_PER_TILE

    stimuli_size = (tile_cnt * ELEMENTS_PER_TILE,)
    o = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
    g_out = torch.empty(stimuli_size, dtype=torch_format).uniform_(-4.0, 4.0)
    # gamma2 is per head-dim (per row), broadcast across heads (columns).
    gamma2_vec = torch.empty(
        (input_dimensions[0], 1), dtype=torch_format
    ).uniform_(-2.0, 2.0)
    gamma2 = gamma2_vec.expand(input_dimensions[0], input_dimensions[1]).contiguous()

    # Untilized 2D views for the golden (row = head-dim, col = head).
    o_2d = o.view(input_dimensions[0], input_dimensions[1])
    g_out_2d = g_out.view(input_dimensions[0], input_dimensions[1])
    golden_tensor = _o_norm_golden(o_2d, gamma2, g_out_2d, EPS)

    # Tilize each operand the way the device consumes it.
    o_tilized = tilize_block(o, input_dimensions, stimuli_format=formats.input_format).flatten()
    gamma2_tilized = tilize_block(
        gamma2.flatten(), input_dimensions, stimuli_format=formats.input_format
    ).flatten()
    g_out_tilized = tilize_block(
        g_out, input_dimensions, stimuli_format=formats.input_format
    ).flatten()

    configuration = TestConfig(
        "sources/o_norm_test.cpp",
        formats,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            O_NORM_CONFIG(num_reduce_tiles=num_reduce_tiles, eps_bits=_f32_bits(EPS)),
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            o_tilized,
            formats.input_format,
            gamma2_tilized,
            formats.input_format,
            formats.output_format,
            tile_count_A=tile_cnt,
            tile_count_B=tile_cnt,
            tile_count_res=tile_cnt,
            buffer_C=g_out_tilized,
            stimuli_C_format=formats.input_format,
            tile_count_C=tile_cnt,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=False,
        disable_format_inference=True,
        compile_time_formats=True,
    )
    res_from_L1 = configuration.run().result

    res_tensor = torch.tensor(res_from_L1, dtype=format_dict[formats.output_format])
    res_tensor = untilize_block(res_tensor, formats.output_format, input_dimensions)

    assert passed_test(
        golden_tensor, res_tensor, formats.output_format
    ), "o_norm result does not match golden"
