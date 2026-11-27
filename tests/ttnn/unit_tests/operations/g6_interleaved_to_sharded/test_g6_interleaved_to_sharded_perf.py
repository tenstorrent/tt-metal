# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Perf bench for g6_interleaved_to_sharded on the PERF FOCUS case (FLUX 58, p1).

Measurement only: run under `scripts/run_safe_pytest.sh --profile --no-py-zones`. Every test makes
REPEATS op calls of one variant and no readback, so the profiler CSV holds REPEATS GenericOp rows
per test, in parametrize order. Correctness of each knob value is graded by
test_g6_interleaved_to_sharded_knobs.py.

"ablate" variants set ABLATE_READS: the NoC reads are dropped while launch, argument loading,
address generation and the barrier stay, which gives the sync-only floor.
"""

from __future__ import annotations

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
VARIANTS = {
    "default": ({}, False),
    "1issuer-face_rows": ({"issuers_per_core": 1, "partial_tile_read": "face_rows"}, False),
    "1issuer-whole_tile": ({"issuers_per_core": 1, "partial_tile_read": "whole_tile"}, False),
    "2issuers-face_rows": ({"issuers_per_core": 2, "partial_tile_read": "face_rows"}, False),
    "2issuers-valid_rows": ({"issuers_per_core": 2, "partial_tile_read": "valid_rows"}, False),
    "2issuers-whole_tile": ({"issuers_per_core": 2, "partial_tile_read": "whole_tile"}, False),
    "ablate-1issuer": ({"issuers_per_core": 1}, True),
    "ablate-2issuers": ({"issuers_per_core": 2}, True),
}


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_perf_flux58(variant, device, monkeypatch):
    knobs, ablate = VARIANTS[variant]
    for name, value in knobs.items():
        monkeypatch.setitem(descriptor.KNOBS, name, value)
    monkeypatch.setattr(descriptor, "ABLATE_READS", ablate)

    torch.manual_seed(0)
    torch_input = make_torch_input(FLUX_58["shape"], FLUX_58["dtype"])
    ttnn_input = create_ttnn_input_tensor(
        torch_input,
        device,
        dtype=FLUX_58["dtype"],
        layout=FLUX_58["layout"],
        memory_config=build_memory_config(FLUX_58["in_memory"], device),
    )
    output_config = build_memory_config(FLUX_58["out_memory"], device)
    for _ in range(REPEATS):
        output = g6_interleaved_to_sharded(ttnn_input, output_config)
        ttnn.synchronize_device(device)
        assert output.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
        output.deallocate()
