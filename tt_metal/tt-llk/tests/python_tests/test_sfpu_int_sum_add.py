# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Operand placement of int_sum's add_int (hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_int_sum.h).

    add_int<APPROX, 8>(dst_offset):  Dest tile 0 += Dest tile dst_offset, in place

It is the kernel behind the compute API's sfpu_add_int(idst, dst_offset), whose default
dst_offset is 2. Four Int32 tiles go into Dest tiles 0..3, each carrying a distinct per-tile
value, so the sum identifies which tile was actually added. The operand tiles must come back
untouched.

Operands stay non-negative and small, so the Int32 sign-magnitude encoding in Dest is the plain
value and no sum can overflow.
"""

import torch
from conftest import skip_for_quasar
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import ApproximationMode, DestAccumulation, format_dict
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    INT_SUM_DST_OFFSET,
    TILE_COUNT,
)

# The Quasar copy of the kernel has its own harness entry (INT_SUM_ADD in sfpu_operations_quasar.h).
pytestmark = [skip_for_quasar]

FORMATS = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

# Four Int32 tiles fill one SyncHalf Dest half. The operand can sit at any of the other three.
TILE_CNT = 4
DST_OFFSETS = [1, 2, 3]


def _build_input_tiles() -> torch.Tensor:
    """TILE_CNT tilized Int32 tiles; tile k holds (k + 1) * 4096 + lane, so every tile differs in every lane."""
    lane = torch.arange(ELEMENTS_PER_TILE, dtype=torch.int32)
    return torch.cat([(k + 1) * 4096 + lane for k in range(TILE_CNT)])


def _tile(tensor: torch.Tensor, k: int) -> torch.Tensor:
    return tensor[k * ELEMENTS_PER_TILE : (k + 1) * ELEMENTS_PER_TILE]


def _run(dst_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one variant, returning (result, input) as flat tilized int32 tensors of TILE_CNT tiles."""
    src_A = _build_input_tiles()
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/sfpu_int_sum_add_test.cpp",
        FORMATS,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            INT_SUM_DST_OFFSET(dst_offset),
        ],
        runtimes=[
            TILE_COUNT(TILE_CNT),
        ],
        variant_stimuli=StimuliConfig(
            src_A,
            FORMATS.input_format,
            src_B,
            FORMATS.input_format,
            FORMATS.output_format,
            tile_count_A=TILE_CNT,
            tile_count_B=TILE_CNT,
            tile_count_res=TILE_CNT,
        ),
        dest_acc=DestAccumulation.Yes,
        unpack_to_dest=True,
    )

    res_from_L1 = configuration.run().result
    res = torch.tensor(res_from_L1, dtype=format_dict[FORMATS.output_format])
    return res, src_A


@parametrize(dst_offset=DST_OFFSETS)
def test_sfpu_int_sum_add(dst_offset):
    (dst_offset,) = dst_offset
    res, src = _run(dst_offset)

    got = _tile(res, 0)
    want = _tile(src, 0) + _tile(src, dst_offset)
    if not bool((got == want).all()):
        # Name the tile that was actually added, if the result is some other tile sum.
        added = [
            k
            for k in range(1, TILE_CNT)
            if bool((got == _tile(src, 0) + _tile(src, k)).all())
        ]
        detail = (
            f"tile 0 was summed with tile {added[0]}"
            if added
            else f"tile 0 came back as {got[:4].tolist()}..."
        )
        raise AssertionError(
            f"add_int(dst_offset={dst_offset}) must add Dest tile {dst_offset} onto tile 0; "
            f"{detail} (expected lanes {want[:4].tolist()}...)"
        )

    for k in range(1, TILE_CNT):
        assert bool(
            (_tile(res, k) == _tile(src, k)).all()
        ), f"add_int(dst_offset={dst_offset}) writes tile 0 only; tile {k} changed"
