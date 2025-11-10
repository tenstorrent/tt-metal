# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache"))


def dual_glx_stres_test():
    """
    Stress test for dual GLX setup.
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=56)

    result = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=128,
        repeat_batches=2,
    )

    # Check that we got generations back
    assert "generations" in result, "Result should contain 'generations' key"
    assert isinstance(result["generations"], list), "Generations should be a list"
    assert len(result["generations"]) > 0, "Should have at least one generation"

    # Check the first generation's tokens
    first_gen = result["generations"][0]
    tokens = first_gen.get("tokens")
    assert isinstance(tokens, list), "Tokens should be a list"
    assert len(tokens) == 128, f"Expected 128 tokens, got {len(tokens)}"


if __name__ == "__main__":
    dual_glx_stres_test()
