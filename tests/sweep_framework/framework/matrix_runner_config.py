# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Matrix routing configuration for sweep CI.

Used by ``compute_sweep_matrix.py`` to answer three related questions:
1. Which physical runner profile should a job use?
2. Which logical ``test_group_name`` should a routed job belong to?
3. Which GitHub Actions output bucket (``n150-matrix``, ``galaxy-matrix``, ...)
   should that job be emitted into?

This module centralizes routing policy. ``constants.py`` still owns filename
parsing such as ``.mesh_*`` and ``.hw_*`` suffix semantics; this file maps those
already-parsed routing hints to logical test groups and runner profiles.
"""


# ── Run type detection (workflow inputs vs cron schedule) ────────────────────
# ``compute_sweep_matrix.main`` sets batching and which matrix builder to call
# from these maps. Workflow ``SWEEP_NAME`` wins; else ``GITHUB_EVENT_SCHEDULE``
# is matched; default is ``nightly``.

SWEEP_TYPES = {
    "ALL SWEEPS (Lead Models)": "lead_models",
    "ALL SWEEPS (Model Traced)": "model_traced",
    "ALL SWEEPS (Comprehensive)": "comprehensive",
    "ALL SWEEPS (Nightly)": "nightly",
}

SCHEDULE_TYPES = {
    "0 2 * * *": "lead_models",
    "0 3 * * *": "model_traced",
    "0 4 * * 3,6": "comprehensive",
}


# ── Artifact written by ``sweeps_parameter_generator`` ───────────────────────
# Lived next to vector JSON under ``vectors_export/``; lists produced files and
# ``vector_grouping_mode`` so the matrix step does not infer policy from scans alone.

GENERATION_MANIFEST_FILENAME = "generation_manifest.json"


# ── Physical runner profiles ──────────────────────────────────────────────────
# These are the machine-level properties consumed directly by the workflow.
# Multiple logical test groups may point at the same profile, which avoids
# repeating ``runs_on``, ``runner_label``, ``tt_smi_cmd``, and ``arch``.

MATRIX_OUTPUT_KEYS = ("n150", "n300", "p150b", "t3k", "galaxy")

RUNNER_PROFILES = {
    "n150": {
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n150-stable",
        "runner_label": "N150",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "n150",
    },
    "n300": {
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n300-stable",
        "runner_label": "N300",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "n300",
    },
    "n300-llmbox": {
        "arch": "wormhole_b0",
        "runs_on": "tt-ubuntu-2204-n300-llmbox-viommu-stable",
        "runner_label": "n300-llmbox",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "n300",
    },
    "p150b": {
        "arch": "blackhole",
        "runs_on": "tt-ubuntu-2204-p150b-viommu-stable",
        "runner_label": "p150b",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "p150b",
    },
    "t3k": {
        "arch": "wormhole_b0",
        "runs_on": ["config-t3000", "arch-wormhole_b0", "in-service", "pipeline-functional"],
        "runner_label": "config-t3000",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "t3k",
    },
    "galaxy-topology-6u": {
        "arch": "wormhole_b0",
        "runs_on": ["topology-6u", "in-service", "bare-metal"],
        "runner_label": "topology-6u",
        "tt_smi_cmd": "tt-smi -glx_reset_auto",
        "matrix_output_key": "galaxy",
    },
    "galaxy-g04glx03": {
        "arch": "wormhole_b0",
        "runs_on": "g04glx03",
        "runner_label": "g04glx03",
        "tt_smi_cmd": "tt-smi -r",
        "matrix_output_key": "galaxy",
    },
}


# ── Logical test groups ───────────────────────────────────────────────────────
# ``test_group_name`` is the logical identity that appears in matrix rows,
# artifacts, and workflow conditionals. Each group points at one runner profile.

TEST_GROUPS = {
    "wormhole-n150-sweeps": {"runner_profile": "n150"},
    "wormhole-n300-sweeps": {"runner_profile": "n300"},
    "n300-llmbox-ccl": {"runner_profile": "n300-llmbox"},
    "blackhole-p150b-sweeps": {"runner_profile": "p150b"},
    "wormhole-t3k-sweeps": {"runner_profile": "t3k"},
    "wormhole-galaxy-sweeps": {"runner_profile": "galaxy-topology-6u"},
    "lead-models-single-chip": {"runner_profile": "n150"},
    "lead-models-galaxy": {"runner_profile": "galaxy-g04glx03"},
}


# ── Lead models sweep: mesh routing + batching policy ───────────────────────
# Lead-model vectors are still routed by mesh or hardware hints, but unlike
# model-traced runs they collapse into only two logical CI lanes:
# single-chip and galaxy. Mesh-grouped files route through the map below.
# Hardware-grouped files use ``get_lead_models_test_group_name_for_hardware_group``.
# Files with no explicit grouping suffix fall back to ``LEAD_MODELS_DEFAULT_TEST_GROUP``.

LEAD_MODELS_MESH_TEST_GROUPS = {
    "1x1": "lead-models-single-chip",
    "1x2": "lead-models-galaxy",
    "1x4": "lead-models-galaxy",
    "1x8": "lead-models-galaxy",
    "2x4": "lead-models-galaxy",
    "4x8": "lead-models-galaxy",
    "8x4": "lead-models-galaxy",
    "2x16": "lead-models-galaxy",
    "16x2": "lead-models-galaxy",
}

LEAD_MODELS_DEFAULT_TEST_GROUP = "lead-models-single-chip"
LEAD_MODELS_SUITE_NAME = "model_traced"

# Absent entries use the caller-provided fixed ``batch_size``.
LEAD_MODELS_BATCH_POLICY = {
    "lead-models-galaxy": {"parallel_jobs": 3},
}


# ── Model-traced sweep: mesh suffix → logical test group ─────────────────────
# Used when ``generation_manifest`` says the exported vector files were grouped
# by mesh. The values are logical test group names, not raw machine payloads.

MODEL_TRACED_MESH_TEST_GROUPS = {
    "1x1": "wormhole-n150-sweeps",
    "1x2": "wormhole-n300-sweeps",
    "1x4": "wormhole-t3k-sweeps",
    "1x8": "wormhole-galaxy-sweeps",
    "2x4": "wormhole-galaxy-sweeps",
    "4x8": "wormhole-galaxy-sweeps",
    "8x4": "wormhole-galaxy-sweeps",
    "2x16": "wormhole-galaxy-sweeps",
    "16x2": "wormhole-galaxy-sweeps",
}


# ── Routing helpers ───────────────────────────────────────────────────────────
# Resolve ``test_group_name`` / runner payloads for ``compute_*_matrix`` and
# ``main`` output splitting.


def get_runner_config(test_group_name):
    """Build the matrix row payload for a logical test group."""
    group = TEST_GROUPS[test_group_name]
    profile = RUNNER_PROFILES[group["runner_profile"]]
    return {
        "test_group_name": test_group_name,
        "arch": profile["arch"],
        "runs_on": profile["runs_on"],
        "runner_label": profile["runner_label"],
        "tt_smi_cmd": profile["tt_smi_cmd"],
    }


def get_matrix_output_key_for_test_group(test_group_name):
    """Return which ``*-matrix`` output bucket this test group belongs to."""
    group = TEST_GROUPS[test_group_name]
    profile = RUNNER_PROFILES[group["runner_profile"]]
    return profile["matrix_output_key"]


def get_test_group_name_for_hardware_group(hardware_group):
    """Map a parsed hardware tuple to the logical test group used in CI."""
    if hardware_group is None:
        return "wormhole-n150-sweeps"

    board_type, device_series, card_count = hardware_group

    if board_type == "blackhole" or device_series == "p150b":
        return "blackhole-p150b-sweeps"
    if device_series == "tt_galaxy_wh":
        return "wormhole-galaxy-sweeps"
    if device_series == "n300" and card_count == 4:
        return "wormhole-t3k-sweeps"
    if device_series == "n300":
        return "wormhole-n300-sweeps"
    return "wormhole-n150-sweeps"


def get_lead_models_test_group_name_for_hardware_group(hardware_group):
    """Map a parsed hardware tuple to the lead-model CI lane."""
    if hardware_group is None:
        return LEAD_MODELS_DEFAULT_TEST_GROUP

    _, device_series, card_count = hardware_group
    wants_galaxy = device_series == "tt_galaxy_wh" or card_count > 1
    return "lead-models-galaxy" if wants_galaxy else LEAD_MODELS_DEFAULT_TEST_GROUP


# ── Derived: split combined matrix into per-hardware workflow outputs ────────
# The output bucket is a property of the runner profile, so derive these lists
# instead of repeating them by hand in a second mapping.

HW_GROUP_MATRIX_KEYS = {
    key: [
        test_group_name
        for test_group_name in TEST_GROUPS
        if get_matrix_output_key_for_test_group(test_group_name) == key
    ]
    for key in MATRIX_OUTPUT_KEYS
}
