# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""L1 LLK tests for hw/ckernels/blackhole/metal/llk_api/experimental/llk_sfpu/ckernel_sfpu_row_norm.h."""

from __future__ import annotations

import struct
from dataclasses import dataclass

import pytest
import torch
from conftest import skip_for_quasar, skip_for_wormhole
from helpers.format_config import DataFormat
from helpers.llk_params import DestAccumulation
from helpers.param_config import input_output_formats
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import TemplateParameter
from helpers.tilize_untilize import tilize_block, untilize_block

pytestmark = [skip_for_wormhole, skip_for_quasar]

EPSILON = 1e-5


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


@dataclass
class RowNormParams(TemplateParameter):
    row_norm_op_mode: int
    row_norm_num_rows: int
    row_norm_tile_h: int
    row_norm_num_tiles: int
    row_norm_scale_sum: bool
    row_norm_inv_width_bits: int
    row_norm_epsilon_bits: int

    def convert_to_cpp(self) -> str:
        return (
            f"constexpr std::uint32_t ROW_NORM_OP_MODE = {self.row_norm_op_mode};\n"
            f"constexpr std::uint32_t ROW_NORM_NUM_ROWS = {self.row_norm_num_rows};\n"
            f"constexpr std::uint32_t ROW_NORM_TILE_H = {self.row_norm_tile_h};\n"
            f"constexpr std::uint32_t ROW_NORM_NUM_TILES = {self.row_norm_num_tiles};\n"
            f"constexpr bool ROW_NORM_SCALE_SUM = {'true' if self.row_norm_scale_sum else 'false'};\n"
            f"constexpr std::uint32_t ROW_NORM_INV_WIDTH_BITS = 0x{self.row_norm_inv_width_bits:08x}u;\n"
            f"constexpr std::uint32_t ROW_NORM_EPSILON_BITS = 0x{self.row_norm_epsilon_bits:08x}u;\n"
        )


def _logical_rows(tiles: torch.Tensor, num_rows: int, tile_h: int) -> torch.Tensor:
    """Extract logical (num_rows, width) view from [num_tiles, 32, 32] tiles."""
    active = tiles[:, :tile_h, :]  # [num_tiles, tile_h, 32]
    num_tiles = active.shape[0]
    if tile_h == num_rows:
        return active.permute(1, 0, 2).reshape(num_rows, num_tiles * 32)
    # Stacked: each tile packs tile_h // num_rows consecutive 32-element slices per row.
    slices_per_tile = tile_h // num_rows
    # Row r occupies tile rows [r, r + num_rows, r + 2*num_rows, ...].
    stacked = active.reshape(num_tiles, slices_per_tile, num_rows, 32)
    return stacked.permute(2, 0, 1, 3).reshape(
        num_rows, num_tiles * slices_per_tile * 32
    )


def _run_row_norm(
    op_mode: int,
    num_rows: int,
    tile_h: int,
    num_tiles: int,
    dest_acc: DestAccumulation,
    tiles_f32: torch.Tensor,
    scale_sum: bool = True,
) -> torch.Tensor:
    width = num_tiles * (tile_h // num_rows) * 32
    inv_width = 1.0 / float(width)
    data_format = (
        DataFormat.Float32 if dest_acc == DestAccumulation.Yes else DataFormat.Float16_b
    )
    torch_dtype = torch.float32 if data_format == DataFormat.Float32 else torch.bfloat16
    tiles_in = tiles_f32.to(torch_dtype)
    src_a = torch.cat(
        [
            tilize_block(
                tiles_in[t].flatten(), [32, 32], stimuli_format=data_format
            ).flatten()
            for t in range(num_tiles)
        ]
    )
    out_tiles = 1 if op_mode == 1 else num_tiles
    config = TestConfig(
        "sources/sfpu_row_norm_test.cpp",
        input_output_formats([data_format], same=True)[0],
        templates=[
            RowNormParams(
                row_norm_op_mode=op_mode,
                row_norm_num_rows=num_rows,
                row_norm_tile_h=tile_h,
                row_norm_num_tiles=num_tiles,
                row_norm_scale_sum=scale_sum,
                row_norm_inv_width_bits=_f32_bits(inv_width),
                row_norm_epsilon_bits=_f32_bits(EPSILON),
            )
        ],
        runtimes=[],
        variant_stimuli=StimuliConfig(
            src_a,
            data_format,
            torch.zeros(1024, dtype=torch_dtype),
            data_format,
            data_format,
            tile_count_A=num_tiles,
            tile_count_B=1,
            tile_count_res=out_tiles,
        ),
        dest_acc=dest_acc,
        unpack_to_dest=(data_format == DataFormat.Float32),
    )
    raw = torch.tensor(config.run().result, dtype=torch_dtype)
    return torch.stack(
        [
            untilize_block(raw[t * 1024 : (t + 1) * 1024], data_format, [32, 32])
            .reshape(32, 32)
            .float()
            for t in range(out_tiles)
        ]
    )


@pytest.mark.parametrize("num_rows", [1, 2, 4, 8])
@pytest.mark.parametrize("tile_h_kind", ["native", "16", "32"])
@pytest.mark.parametrize("op_mode", [0, 1, 2, 3])
@pytest.mark.parametrize("dest_acc", [DestAccumulation.No, DestAccumulation.Yes])
def test_sfpu_row_norm(
    num_rows: int, tile_h_kind: str, op_mode: int, dest_acc: DestAccumulation
) -> None:
    tile_h = num_rows if tile_h_kind == "native" else int(tile_h_kind)
    num_tiles = 2
    torch.manual_seed(42 + num_rows * 17 + tile_h + op_mode)
    # Build input tiles; for op_mode == 1 (reduce_scale_add_rsqrt), inputs represent pre-squared values.
    tiles = torch.randn(num_tiles, 32, 32)
    if op_mode == 1:
        tiles = tiles.square()
    if dest_acc == DestAccumulation.No:
        tiles = tiles.to(torch.bfloat16).float()

    out = _run_row_norm(op_mode, num_rows, tile_h, num_tiles, dest_acc, tiles)
    x = _logical_rows(tiles, num_rows, tile_h)

    atol = 2e-2 if dest_acc == DestAccumulation.No else 1e-4
    rtol = 2e-2 if dest_acc == DestAccumulation.No else 1e-4

    if op_mode == 0:
        got = _logical_rows(out, num_rows, tile_h)
        want = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + EPSILON)
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    elif op_mode == 1:
        want = torch.rsqrt(x.mean(dim=-1) + EPSILON)
        for r in range(num_rows):
            # Replicated across row stride in Face 0 col 0 (and replicated across faces for FACE0_ONLY=false).
            stride = num_rows if tile_h >= 16 else tile_h
            for row_idx in range(r, tile_h, stride):
                assert abs(out[0, row_idx, 0].item() - want[r].item()) <= (
                    atol + rtol * abs(want[r].item())
                )
    else:
        got = _logical_rows(out, num_rows, tile_h)
        mean = x.mean(dim=-1, keepdim=True)
        var = (x - mean).square().mean(dim=-1, keepdim=True)
        want = (x - mean) * torch.rsqrt(var + EPSILON)
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
