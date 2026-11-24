# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Operand placement of int_sum's add_int (hw/ckernels/<arch>/metal/llk_api/llk_sfpu/ckernel_sfpu_int_sum.h).

    add_int<APPROX, ITERATIONS>(dst_offset):  Dest tile idst += Dest tile idst + dst_offset, in place

It is the kernel behind the compute API's sfpu_add_int(idst, dst_offset). Four Int32 tiles go
into Dest tiles 0..3, each carrying a distinct per-tile value, so the sum identifies which tile
was actually added and whether the operand was found relative to idst. Every other tile must
come back untouched.

Operands stay non-negative and small, so the Int32 sign-magnitude encoding in Dest is the plain
value and no sum can overflow.

The operand-tile bound of the compute API wrapper (idst + dst_offset must stay inside the active
Dest half) is checked by test_sfpu_int_sum_add_operand_out_of_bounds_asserts, which expects the
LLK_ASSERT in the driver to fire. That assert is a copy of the wrapper's in sfpu_int_sum.h (the
tt-llk harness cannot include the compute API), so keep the two in step.
"""

from pathlib import Path

import pytest
import torch
from conftest import skip_for_blackhole, skip_for_quasar
from helpers import device as device_module
from helpers.device import LLKAssertException
from helpers.format_config import DataFormat, InputOutputFormat
from helpers.golden_generators import ELEMENTS_PER_TILE
from helpers.llk_params import (
    ApproximationMode,
    DestAccumulation,
    VectorMode,
    format_dict,
)
from helpers.param_config import parametrize
from helpers.stimuli_config import StimuliConfig
from helpers.test_config import TestConfig
from helpers.test_variant_parameters import (
    APPROX_MODE,
    INT_SUM_DST_INDEX,
    INT_SUM_DST_OFFSET,
    ITERATIONS,
    TILE_COUNT,
    VECTOR_MODE,
)

# The Quasar copy of the kernel has its own harness entry (INT_SUM_ADD in sfpu_operations_quasar.h).
pytestmark = [skip_for_quasar]

FORMATS = InputOutputFormat(DataFormat.Int32, DataFormat.Int32)

# Four Int32 tiles fill one SyncHalf Dest half (32-bit Dest). The destination may be any tile
# that leaves room for an operand above it, and the operand may be any tile above the destination.
TILE_CNT = 4
DST_INDICES = [0, 1, 2]


def _dst_offsets(dst_index: int) -> list[int]:
    return [offset for offset in range(1, TILE_CNT) if dst_index + offset < TILE_CNT]


# Per-tile value base. Any stride >= ELEMENTS_PER_TILE keeps tile and lane decodable from a raw value.
TILE_VALUE_STRIDE = 4 * ELEMENTS_PER_TILE


def _build_input_tiles() -> torch.Tensor:
    """TILE_CNT tilized Int32 tiles; tile k holds (k + 1) * TILE_VALUE_STRIDE + lane, so every tile differs in every lane."""
    lane = torch.arange(ELEMENTS_PER_TILE, dtype=torch.int32)
    return torch.cat([(k + 1) * TILE_VALUE_STRIDE + lane for k in range(TILE_CNT)])


def _tile(tensor: torch.Tensor, k: int) -> torch.Tensor:
    return tensor[k * ELEMENTS_PER_TILE : (k + 1) * ELEMENTS_PER_TILE]


def _run(dst_index: int, dst_offset: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one variant, returning (result, input) as flat tilized int32 tensors of TILE_CNT tiles."""
    src_A = _build_input_tiles()
    src_B = torch.zeros_like(src_A)

    configuration = TestConfig(
        "sources/sfpu_int_sum_add_test.cpp",
        FORMATS,
        templates=[
            APPROX_MODE(ApproximationMode.No),
            INT_SUM_DST_INDEX(dst_index),
            INT_SUM_DST_OFFSET(dst_offset),
            ITERATIONS(8),
            VECTOR_MODE(VectorMode.RC),
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


@parametrize(dst_index=DST_INDICES, dst_offset=_dst_offsets)
def test_sfpu_int_sum_add(dst_index, dst_offset):
    res, src = _run(dst_index, dst_offset)
    operand = dst_index + dst_offset
    call = f"add_int(idst={dst_index}, dst_offset={dst_offset})"

    got = _tile(res, dst_index)
    want = _tile(src, dst_index) + _tile(src, operand)
    if not bool((got == want).all()):
        # Name the tile that was actually added, if the result is some other tile sum.
        added = [
            k
            for k in range(TILE_CNT)
            if k != operand
            and bool((got == _tile(src, dst_index) + _tile(src, k)).all())
        ]
        detail = (
            f"tile {dst_index} was summed with tile {added[0]}"
            if added
            else f"tile {dst_index} came back as {got[:4].tolist()}..."
        )
        raise AssertionError(
            f"{call} must add Dest tile {operand} onto tile {dst_index}; "
            f"{detail} (expected lanes {want[:4].tolist()}...)"
        )

    for k in range(TILE_CNT):
        if k == dst_index:
            continue
        assert bool(
            (_tile(res, k) == _tile(src, k)).all()
        ), f"{call} writes tile {dst_index} only; tile {k} changed"


# (dst_index, dst_offset) whose operand tile dst_index + dst_offset is outside the TILE_CNT-tile Dest
# half. The first three sit exactly on the bound (operand tile TILE_CNT), the valid neighbour of each
# being covered by test_sfpu_int_sum_add; the last one wraps a 32-bit idst + dst_offset around to a
# small in-range value, so only a sum widened to 64 bits still rejects it.
OUT_OF_BOUNDS_CASES = [
    (0, TILE_CNT),
    (1, TILE_CNT - 1),
    (2, TILE_CNT - 2),
    (1, 0xFFFFFFFF),
]


# Wormhole only: on Blackhole a tripped LLK_ASSERT leaves the board wedged. The assert cases themselves
# pass, but every kernel that runs afterwards (even in a new pytest process) times out until `tt-smi -r`,
# and re-booting BRISC and clearing the harness's cached state does not recover it. On Wormhole the
# recovery in the `finally` below is enough for the valid tests that follow to pass.
@skip_for_blackhole
@pytest.mark.parametrize("dst_index, dst_offset", OUT_OF_BOUNDS_CASES)
def test_sfpu_int_sum_add_operand_out_of_bounds_asserts(dst_index, dst_offset):
    """An operand tile past the active Dest half must trip the LLK_ASSERT, not be read silently."""
    assert dst_index + dst_offset >= TILE_CNT, "case must be out of bounds"
    try:
        with pytest.raises(  # allow-pytest.raises: no expect_error fixture in LLK suite
            LLKAssertException
        ):
            _run(dst_index, dst_offset)
    finally:
        # The assert leaves the math TRISC parked on ebreak. Make the next test bring BRISC up afresh
        # and reload the TRISC ELFs instead of trusting the harness's cached boot state; a re-booted
        # BRISC counts commands from zero again, so the host-side command counter has to restart too.
        TestConfig.BRISC_ELF_LOADED = False
        TestConfig.LAST_LOADED_ELFS = Path()
        device_module.common_counter = 0
