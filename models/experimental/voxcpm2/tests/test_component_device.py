# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full pinned CUDA-oracle accuracy suite, using the repository device fixture."""

import argparse
import os
from pathlib import Path

import pytest

from models.experimental.voxcpm2.validation.cases import CASES, CAPTURES, DEFAULT_CHECKPOINT
from models.experimental.voxcpm2.validation.manifest import sha256_file
from models.experimental.voxcpm2.validation.replay_component import replay


@pytest.fixture(scope="module")
def replay_assets():
    checkpoint = Path(os.environ.get("VOXCPM2_CHECKPOINT", str(DEFAULT_CHECKPOINT)))
    captures = {}
    missing = []
    if not checkpoint.is_dir():
        missing.append(str(checkpoint))
    for key, (variable, default_path, expected_sha256) in CAPTURES.items():
        directory = Path(os.environ.get(variable, str(default_path)))
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            missing.append(str(manifest))
        else:
            actual_sha256 = sha256_file(manifest)
            assert actual_sha256 == expected_sha256, (
                f"{key} capture manifest does not match the pinned CUDA oracle: "
                f"expected {expected_sha256}, got {actual_sha256}. "
                "Use replay_component for a different capture."
            )
        captures[key] = directory
    if missing:
        message = "Missing VoxCPM2 accuracy assets: " + ", ".join(missing)
        if os.environ.get("VOXCPM2_REQUIRE_DEVICE_TESTS") == "1":
            pytest.fail(message)
        pytest.skip(message + "; set VOXCPM2_REQUIRE_DEVICE_TESTS=1 to require them")
    return checkpoint, captures


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
@pytest.mark.parametrize("device_params", [{"l1_small_size": 262144}], indirect=True)
def test_cuda_reference_component_on_tt(case, device_params, replay_assets, request, tmp_path):
    checkpoint, captures = replay_assets
    # Request after assets are checked: a missing local oracle must not open hardware.
    # The repository fixture owns opening, teardown, --device-id and device_params.
    device = request.getfixturevalue("device")
    args = argparse.Namespace(
        reference=captures[case.capture],
        checkpoint=checkpoint,
        device_id=request.config.getoption("device_id"),
        component=case.component,
        event_index=case.event_index,
        dtype=case.dtype,
        l1_small_size=device_params["l1_small_size"],
        min_pcc=0.99,
        max_relative_rms=case.max_relative_rms,
        max_abs=case.max_abs,
        output=tmp_path / "report.json",
    )
    assert replay(args, device=device) == 0
