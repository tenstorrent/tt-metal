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
    "max_prompts,repeat_batches,max_new_tokens,override_num_layers,enable_trace,artifact_name,profile_decode",
    [
        pytest.param(
            56,
            2,
            128,
            5,
            False,
            None,
            False,
            id="tg_stress",
            marks=pytest.mark.requires_device(["TG"]),
        ),
        pytest.param(
            256,
            1,
            129,
            None,
            True,
            "dual_demo_full_results",
            False,
            id="dual_full_demo",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(2400)],
        ),
        pytest.param(
            56,
            20,
            129,
            None,
            True,
            "dual_demo_stress_results",
            False,
            id="dual_stress_demo",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(5400)],
        ),
        pytest.param(
            512,
            1,
            129,
            None,
            True,
            "quad_demo_full_results",
            False,
            id="quad_full_demo",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        pytest.param(
            56,
            20,
            129,
            None,
            True,
            "quad_demo_stress_results",
            False,
            id="quad_stress_demo",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(5400)],
        ),
        pytest.param(
            1,
            1,
            13,
            5,
            True,
            None,
            True,
            id="profile_decode",
            marks=pytest.mark.timeout(1800),
        ),
    ],
)
def test_demo(
    max_prompts: int,
    repeat_batches: int,
    max_new_tokens: int,
    override_num_layers: int,
    enable_trace: bool,
    artifact_name: str,
    profile_decode: bool,
):
    """
    DeepSeek v3 demo test with prompts loaded from JSON file.

    Test variants:
    - tg_stress (TG): 56 prompts, 2 batches, 5 layers - stress test for CI
    - dual_full_demo (DUAL): 256 prompts, 1 batch - tests full prompt capacity
    - dual_stress_demo (DUAL): 56 prompts, 20 batches - tests stability under repeated execution
    - quad_full_demo (QUAD): 512 prompts, 1 batch - tests full prompt capacity
    - quad_stress_demo (QUAD): 56 prompts, 20 batches - tests stability under repeated execution
    - profile_decode: Profile decode for non-moe and moe layers
    """
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=max_prompts)

    # Run demo
    run_kwargs = dict(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        repeat_batches=repeat_batches,
        profile_decode=profile_decode,
        signpost=True,
    )
    if override_num_layers is not None:
        run_kwargs["override_num_layers"] = override_num_layers
    if enable_trace:
        run_kwargs["enable_trace"] = True

    results = run_demo(**run_kwargs)

    # Check output
    assert len(results["generations"][0]["tokens"]) == max_new_tokens

    # Save results to JSON for artifact upload (QUAD tests only)
    if artifact_name is not None:
        artifact_dir = Path("generated/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_file = artifact_dir / f"{artifact_name}.json"

        output_data = {
            "prompts": prompts,
            "generations": [],
            "statistics": results.get("statistics", {}),
        }

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
