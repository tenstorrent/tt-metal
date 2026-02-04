# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))


@pytest.mark.parametrize("repeat_batches", [2])
def test_demo(repeat_batches):
    """
    Stress test the DeepSeek v3 demo with prompts loaded from JSON file, 2x runs.
    Uses only 5 layers (override_num_layers=5) for faster CI execution.
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=56)

    # Run demo with only 5 layers for faster CI execution
    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=128,
        override_num_layers=5,
        repeat_batches=repeat_batches,
    )

    # Check output
    assert len(results["generations"][0]["tokens"]) == 128
