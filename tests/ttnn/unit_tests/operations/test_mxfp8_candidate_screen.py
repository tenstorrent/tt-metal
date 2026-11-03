# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Candidate screen: unchanged exhaustive corpus and exact gate, primary configuration only.

Promising candidates still require the complete regression suite and independent
hardware timing confirmation.
"""

import json
import os

import pytest
import torch

import ttnn
from models.common.utility_functions import is_blackhole
from tests.ttnn.unit_tests.operations.mxfp8_reference import (
    PRIMARY,
    corpus,
    kernel_hashes,
    metrics,
    reference,
    run_decode,
)

pytestmark = pytest.mark.skipif(not is_blackhole(), reason="MXFP8 screen requires Blackhole")


def test_mxfp8_candidate_screen(device, tmp_path):
    torch.set_num_threads(4)
    codes, scales = corpus()
    actual, keep_alive = run_decode(device, codes, scales, **PRIMARY)
    for tensor in keep_alive:
        ttnn.deallocate(tensor)
    result = metrics(actual, reference(codes, scales), PRIMARY["dtype"], codes, scales)
    report = dict(
        kernel_hashes=kernel_hashes(),
        valid_values_per_case=int(codes.numel()),
        cases=[dict(config=PRIMARY, primary=True, metrics=result)],
        accuracy_gate=dict(
            path="mxfp8_exact",
            passed=result["mismatches"] == 0,
            observed=dict(mismatches=result["mismatches"], cases=1),
        ),
    )
    destination = os.environ.get("MXFP8_ACCURACY_REPORT", str(tmp_path / "screen.json"))
    with open(destination, "w") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert result["mismatches"] == 0, f"MXFP8 screen mismatches: {result}"
