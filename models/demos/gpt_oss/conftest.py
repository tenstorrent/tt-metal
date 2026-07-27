# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import json
import os

import pytest


def pytest_addoption(parser):
    parser.addoption("--skip-model-load", action="store_true", default=False, help="Skip loading the model state dict")
    # Profiling-only overrides (used by gpt_logs/scripts/tracy.sh). Do not affect default runs.
    parser.addoption(
        "--gpt-oss-decode-trace-off",
        action="store_true",
        default=False,
        help="Force enable_decode_trace=False so device profiler (tracy) captures per-op decode timings",
    )
    parser.addoption(
        "--gpt-oss-max-tokens",
        action="store",
        type=int,
        default=0,
        help="Override max_generated_tokens (0 = use parametrized value). Small values speed up profiling.",
    )


@pytest.fixture(scope="session")
def state_dict(request):
    load_model = not request.config.getoption("--skip-model-load")
    model_path = os.getenv("HF_MODEL", None)
    if model_path is None or not load_model:
        # Explicit skip: build weights purely from the ttnn cache.
        return {}
    # Defer the (expensive) HF weight load to create_tt_model, which knows the mesh shape +
    # dtype and can skip it when a warm ttnn cache is already on disk (see
    # ModelArgs.weight_cache_is_complete). Returning None signals "load if needed".
    return None


@pytest.fixture
def test_thresholds(request):
    with open("models/demos/gpt_oss/unit_test_thresholds.json", "r") as f:
        thresholds = json.load(f)
    return thresholds
