#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FRAMEWORK_ROOT = Path(__file__).parent
if str(FRAMEWORK_ROOT) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_ROOT))

from framework.execution_capabilities import (
    VectorRequirement,
    load_execution_capability_profiles,
    requirements_from_module_name,
    requirements_from_vector_data,
    resolve_active_profile,
    select_profile_for_host,
    summarize_vector_file,
    is_requirement_eligible,
)


PROFILES_PATH = Path(__file__).parent / "framework" / "execution_capability_profiles.yaml"


def test_load_profiles():
    profiles = load_execution_capability_profiles(PROFILES_PATH)

    assert "wormhole_n300_2c_host" in profiles
    assert profiles["wormhole_n300_2c_host"].host_match == ("wormhole", "n300", 2)
    assert ("wormhole", "n300", 1) in profiles["wormhole_n300_2c_host"].can_run_hardware_groups
    assert (1, 2) in profiles["wormhole_n300_2c_host"].can_run_mesh_shapes


def test_select_profile_for_host_unique_match():
    profiles = load_execution_capability_profiles(PROFILES_PATH)

    profile = select_profile_for_host(profiles, ("wormhole", "n300", 2))

    assert profile.name == "wormhole_n300_2c_host"


def test_select_profile_for_host_raises_when_missing():
    profiles = load_execution_capability_profiles(PROFILES_PATH)

    with pytest.raises(RuntimeError, match="No execution capability profile matches"):
        select_profile_for_host(profiles, ("wormhole", "unknown", 9))


def test_resolve_active_profile_prefers_explicit_name(monkeypatch):
    profiles = load_execution_capability_profiles(PROFILES_PATH)
    monkeypatch.setenv("TT_SWEEP_CAPABILITY_PROFILE", "wormhole_n150_host")

    profile = resolve_active_profile(profiles=profiles)

    assert profile.name == "wormhole_n150_host"


def test_requirements_from_module_name():
    requirement = requirements_from_module_name("model_traced.add_model_traced.hw_wormhole_n300_1c")

    assert requirement.hardware_groups == frozenset({("wormhole", "n300", 1)})
    assert requirement.mesh_shapes == frozenset()


def test_requirements_from_vector_data_reads_traced_machine_info():
    requirement = requirements_from_vector_data(
        {
            "traced_machine_info": {
                "board_type": "Wormhole",
                "device_series": "n300",
                "card_count": 1,
                "mesh_device_shape": [1, 1],
            }
        }
    )

    assert requirement.hardware_groups == frozenset({("wormhole", "n300", 1)})
    assert requirement.mesh_shapes == frozenset({(1, 1)})


def test_is_requirement_eligible_exact_match_against_declared_sets():
    profiles = load_execution_capability_profiles(PROFILES_PATH)
    profile = profiles["wormhole_n300_2c_host"]

    assert is_requirement_eligible(
        VectorRequirement(
            hardware_groups=frozenset({("wormhole", "n300", 1)}),
            mesh_shapes=frozenset({(1, 1)}),
        ),
        profile,
    )
    assert not is_requirement_eligible(
        VectorRequirement(
            hardware_groups=frozenset({("wormhole", "tt_galaxy_wh", 32)}),
            mesh_shapes=frozenset(),
        ),
        profile,
    )


def test_summarize_vector_file_aggregates_trace_ids_and_requirements(tmp_path):
    vector_path = tmp_path / "model_traced.add_model_traced.hw_wormhole_n300_1c.json"
    vector_path.write_text(
        json.dumps(
            {
                "model_traced": {
                    "abc": {
                        "trace_ids": [3],
                        "traced_machine_info": {
                            "board_type": "Wormhole",
                            "device_series": "n300",
                            "card_count": 1,
                            "mesh_device_shape": [1, 1],
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_vector_file(vector_path)

    assert summary.trace_ids == frozenset({3})
    assert summary.requirement.hardware_groups == frozenset({("wormhole", "n300", 1)})
    assert summary.requirement.mesh_shapes == frozenset({(1, 1)})
