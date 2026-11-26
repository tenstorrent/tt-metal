# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Acceptance test for g6_interleaved_to_sharded (immutable spec for the implementer).

The op is an exact copy into a new placement: each result is graded by equality
of the bfloat16 bit patterns AND by the memory configuration it carries.

Cases are limited to the production cases and the cases a rule of the spec or
the design requires (BB-22):
- flux58 / flux57: the two production cases (flux57 is outside Phase 0).
- special values, poisoned padding: bit exactness and independence from the
  input's tile padding.
- col_major: core order follows the orientation memory_config names.
- h33_width: one full and one partial tile row per core, the two transfer
  shapes of the design (whole page, valid rows).
- height_multi_shard: height map over several shards and images (refinement).
- program cache: shard spec in the cache key, new input address on a hit.

A case outside SUPPORTED is xfail-strict on the op's support refusal and turns
into a real test when the refinement lands.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from eval.feature_matrix import apply_input_taggers, unsupported_reason
from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_57, FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    check_output,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import (  # type: ignore
    EXCLUSIONS,
    INPUT_TAGGERS,
    SUPPORTED,
    g6_interleaved_to_sharded,
)

SEED = 42


def torch_reference(x: torch.Tensor) -> torch.Tensor:
    return x.clone()


def _variant(base, shape=None, **out_memory):
    scenario = copy.deepcopy(base)
    if shape is not None:
        scenario["shape"] = list(shape)
    scenario["out_memory"].update(out_memory)
    return scenario


def _gated(*scenarios, name):
    reasons = [unsupported_reason(apply_input_taggers(INPUT_TAGGERS, (s,)), SUPPORTED, EXCLUSIONS) for s in scenarios]
    reasons = [r for r in reasons if r is not None]
    marks = [pytest.mark.xfail(reason="; ".join(reasons), strict=True, raises=NotImplementedError)] if reasons else []
    return pytest.param(*scenarios, marks=marks, id=name)


FLUX_58_COL_MAJOR = _variant(FLUX_58, orientation="col_major")
# 2 tile rows (row 0 full, row 1 holds 1 valid row) on each of the 32 cores.
H33_WIDTH = _variant(FLUX_58, shape=(33, 5120), shard_shape=[64, 160])
# 2 images of 2 tile rows (H = 33), 4 height shards of one tile row on 4 cores.
HEIGHT_MULTI_SHARD = _variant(
    FLUX_58,
    shape=(2, 1, 33, 128),
    scheme="height_sharded",
    grid=[[0, 0], [3, 0]],
    shard_shape=[32, 128],
    orientation="row_major",
)
# Same tensor as flux58, 8 width shards of 20 tile columns: a second program
# for one shape, which a cache key without the shard spec would confuse.
FLUX_58_8_CORES = _variant(FLUX_58, grid=[[0, 0], [7, 0]], shard_shape=[32, 640])


def _place(scenario, torch_input, device):
    return create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=build_memory_config(scenario["in_memory"], device),
    )


def _run_and_check(scenario, ttnn_input, device):
    # Graded against the values the device holds, so a host conversion of a
    # NaN payload is not charged to the op.
    expected = torch_reference(ttnn.to_torch(ttnn_input))
    output_config = build_memory_config(scenario["out_memory"], device)
    output = g6_interleaved_to_sharded(ttnn_input, output_config)
    check_output(
        output,
        expected,
        shape=scenario["shape"],
        dtype=scenario["dtype"],
        layout=scenario["layout"],
        memory_config=output_config,
        gate="exact",
    )
    return output


@pytest.mark.parametrize(
    "scenario",
    [
        _gated(FLUX_58, name="flux58-1x5120-width-32x160-8x4-row_major"),
        _gated(FLUX_57, name="flux57-1x1x1x128-height-32x128-1x1-row_major"),
        _gated(FLUX_58_COL_MAJOR, name="flux58-col_major"),
        _gated(H33_WIDTH, name="h33_width-33x5120-width-64x160-8x4"),
        _gated(HEIGHT_MULTI_SHARD, name="height_multi_shard-2x1x33x128-height-32x128-4x1"),
    ],
)
def test_g6_interleaved_to_sharded(scenario, device):
    torch.manual_seed(SEED)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"])
    _run_and_check(scenario, _place(scenario, torch_input, device), device)


@pytest.mark.parametrize("scenario", [_gated(FLUX_58, name="flux58")])
def test_special_values_bit_exact(scenario, device):
    torch.manual_seed(SEED)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"], values="special")
    _run_and_check(scenario, _place(scenario, torch_input, device), device)


@pytest.mark.parametrize("scenario", [_gated(FLUX_58, name="flux58")])
def test_poisoned_padding(scenario, device):
    torch.manual_seed(SEED)
    torch_input = make_torch_input(scenario["shape"], scenario["dtype"])
    input_config = build_memory_config(scenario["in_memory"], device)
    ttnn_input = _place(scenario, torch_input, device)
    ttnn_input = ttnn.fill_implicit_tile_padding(ttnn_input, float("nan"), memory_config=input_config)
    _run_and_check(scenario, ttnn_input, device)


@pytest.mark.parametrize("first,second", [_gated(FLUX_58, FLUX_58_8_CORES, name="flux58-then-8cores")])
def test_program_cache(first, second, device):
    torch.manual_seed(SEED)
    first_input = make_torch_input(first["shape"], first["dtype"])
    first_tensor = _place(first, first_input, device)
    _run_and_check(first, first_tensor, device)
    _run_and_check(second, first_tensor, device)
    # Allocated while the first input is alive: another DRAM address, other values.
    second_input = make_torch_input(first["shape"], first["dtype"])
    second_tensor = _place(first, second_input, device)
    _run_and_check(first, second_tensor, device)
    ttnn.deallocate(first_tensor)
