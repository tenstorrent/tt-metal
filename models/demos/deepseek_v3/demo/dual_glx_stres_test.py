# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import pytest

from models.demos.deepseek_v3.demo.demo import run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/proj_sw/user_dev/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/proj_sw/user_dev/deepseek-v3-cache"))


@pytest.mark.timeout(180)
@pytest.parametrize("repeat_batches", [2])
def dual_glx_stres_test(tmp_path, repeat_batches):
    """
    Stress test for dual GLX setup.
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/56_inputs.json"

    # Load the list of prompts
    with open(json_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    result = run_demo(
        prompts,
        model_path=MODEL_PATH,
        cache_dir=str(CACHE_DIR),
        random_weights=False,
        max_new_tokens=128,
        repeat_batches=repeat_batches,
    )

    tokens = result.get("tokens")
    assert isinstance(tokens, list)
    assert len(tokens) == 128, f"Expected 128 tokens, got {len(tokens)}"
