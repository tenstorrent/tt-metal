# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench for g6_interleaved_to_sharded on the PERF FOCUS case (FLUX 58, p1).

Measurement only: run under `scripts/run_safe_pytest.sh --profile --no-py-zones`. Every test makes
REPEATS op calls of one variant and no readback, so the profiler CSV holds REPEATS GenericOp rows
per test, in parametrize order (scenario-major: every p1 variant, then every p1_640 variant). Correctness of each knob value is graded by
test_g6_interleaved_to_sharded_knobs.py.

"ablate" variants set ABLATE_READS: the NoC reads are dropped while launch, argument loading,
address generation and the barrier stay, which gives the sync-only floor.
"""

from __future__ import annotations

import copy

import pytest
import torch
import ttnn

from eval.golden_tests.g6_interleaved_to_sharded.feature_spec import FLUX_58
from eval.golden_tests.g6_interleaved_to_sharded.helpers import (
    build_memory_config,
    create_ttnn_input_tensor,
    make_torch_input,
)
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded
from ttnn.operations.g6_interleaved_to_sharded import g6_interleaved_to_sharded_program_descriptor as descriptor

REPEATS = 5

# variant -> (KNOBS overrides, ABLATE_READS)
EVEN = {"issuer_split": "even"}
VARIANTS = {
    "default": ({}, False),
    "1issuer-valid_rows": ({"issuers_per_core": 1}, False),  # the Phase 0 / Refinement 1 default
    "1issuer-face_rows": ({"issuers_per_core": 1, "partial_tile_read": "face_rows"}, False),
    "1issuer-whole_tile": ({"issuers_per_core": 1, "partial_tile_read": "whole_tile"}, False),
    "2issuers-face_rows": ({"issuers_per_core": 2, "partial_tile_read": "face_rows"}, False),
    "2issuers-whole_tile": ({"issuers_per_core": 2, "partial_tile_read": "whole_tile"}, False),
    "2issuers-even-ncrisc_noc0-first": ({"issuers_per_core": 2, **EVEN}, False),
    "2issuers-even-brisc_noc1-first": (
        {"issuers_per_core": 2, "issuer_placement": "brisc_noc1+ncrisc_noc0", **EVEN},
        False,
    ),
    "2issuers-even-ncrisc_noc1-first": (
        {"issuers_per_core": 2, "issuer_placement": "ncrisc_noc1+brisc_noc0", **EVEN},
        False,
    ),
    "2issuers-even-brisc_noc0-first": (
        {"issuers_per_core": 2, "issuer_placement": "brisc_noc0+ncrisc_noc1", **EVEN},
        False,
    ),
    "2issuers-noc_balanced-brisc_noc0-first": (
        {"issuers_per_core": 2, "issuer_placement": "brisc_noc0+ncrisc_noc1"},
        False,
    ),
    "1issuer-bank_stagger": ({"issuers_per_core": 1, "bank_stagger": True}, False),
    "2issuers-bank_stagger": ({"issuers_per_core": 2, "bank_stagger": True}, False),
    "ablate-1issuer": ({"issuers_per_core": 1}, True),
    "ablate-2issuers": ({"issuers_per_core": 2}, True),
}


# (32, 640) on 8 cores: 20 tiles per owner, first pages alternate banks 0/4 (bank_stagger's case).
FLUX_58_640 = copy.deepcopy(FLUX_58)
FLUX_58_640["out_memory"]["grid"] = [[0, 0], [7, 0]]
FLUX_58_640["out_memory"]["shard_shape"] = [32, 640]
# The same 8 owners stacked in one column instead of one row.
FLUX_58_640_COL = copy.deepcopy(FLUX_58_640)
FLUX_58_640_COL["out_memory"]["grid"] = [[0, 0], [0, 7]]
# (33, 5120): a full and a partial tile row in every owner's block (10 tiles per owner).
H33 = copy.deepcopy(FLUX_58)
H33["shape"] = [33, 5120]
H33["out_memory"]["shard_shape"] = [64, 160]
SCENARIOS = {"p1": FLUX_58, "p1_640": FLUX_58_640, "p1_640_col": FLUX_58_640_COL, "h33": H33}


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("scenario", list(SCENARIOS))
def test_perf_flux58(scenario, variant, device, monkeypatch):
    case = SCENARIOS[scenario]
    knobs, ablate = VARIANTS[variant]
    for name, value in knobs.items():
        monkeypatch.setitem(descriptor.KNOBS, name, value)
    monkeypatch.setattr(descriptor, "ABLATE_READS", ablate)

    torch.manual_seed(0)
    torch_input = make_torch_input(case["shape"], case["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=case["dtype"],
        layout=case["layout"],
        memory_config=build_memory_config(case["in_memory"], device),
    )
    output_config = build_memory_config(case["out_memory"], device)
    for _ in range(REPEATS):
        output = g6_interleaved_to_sharded(ttnn_input, output_config)
        ttnn.synchronize_device(device)
        assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        output.deallocate()
