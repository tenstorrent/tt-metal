# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528"))
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))


@pytest.mark.parametrize(
    "max_prompts,repeat_batches,artifact_name",
    [
        pytest.param(256, 1, "dual_demo_full_results", id="full_demo"),
        pytest.param(56, 20, "dual_demo_stress_results", id="stress_demo"),
    ],
)
def test_demo_dual(max_prompts: int, repeat_batches: int, artifact_name: str):
    """
    DeepSeek v3 dual demo test with prompts loaded from JSON file.

    Test variants:
    - full_demo: 256 prompts, 1 batch - tests full prompt capacity
    - stress_demo: 56 prompts, 20 batches - tests stability under repeated execution
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=max_prompts)

    # Run demo
    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=129,
        repeat_batches=repeat_batches,
        enable_trace=True,
    )

    # Check output
    assert len(results["generations"][0]["tokens"]) == 129

    # Save results to JSON for artifact upload
    artifact_dir = Path("generated/artifacts")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    output_file = artifact_dir / f"{artifact_name}.json"

    # Prepare output data structure matching demo.py format
    output_data = {
        "prompts": prompts,
        "generations": [],
        "statistics": results.get("statistics", {}),
    }

    # Add generation results
    for i, gen_result in enumerate(results["generations"]):
        prompt_text = prompts[i] if i < len(prompts) else "[empty prompt]"
        output_data["generations"].append(
            {
                "index": i + 1,
                "prompt": prompt_text,
                "text": gen_result.get("text"),
            }
        )

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\nDemo results saved to: {output_file}")
