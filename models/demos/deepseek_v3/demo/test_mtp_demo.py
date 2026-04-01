# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(os.environ["DEEPSEEK_V3_HF_MODEL"])
CACHE_DIR = Path(os.environ["DEEPSEEK_V3_CACHE"])


def _assert_demo_outputs_match(baseline: dict, mtp: dict) -> None:
    baseline_gens = baseline["generations"]
    mtp_gens = mtp["generations"]

    assert len(baseline_gens) == len(
        mtp_gens
    ), f"Generation count mismatch: baseline={len(baseline_gens)} mtp={len(mtp_gens)}"

    for idx, (baseline_gen, mtp_gen) in enumerate(zip(baseline_gens, mtp_gens, strict=True)):
        assert baseline_gen["tokens"] == mtp_gen["tokens"], (
            f"Token mismatch for generation[{idx}]: "
            f"baseline={baseline_gen['tokens'][:32]} mtp={mtp_gen['tokens'][:32]}"
        )
        assert baseline_gen["text"] == mtp_gen["text"], (
            f"Text mismatch for generation[{idx}]: " f"baseline={baseline_gen['text']!r} mtp={mtp_gen['text']!r}"
        )


def _artifact_name_for_current_mesh() -> str | None:
    mesh_device = os.getenv("MESH_DEVICE", "").upper()
    if mesh_device == "DUAL":
        return "dual_demo_full_results_mtp"
    if mesh_device == "QUAD":
        return "quad_demo_full_results_mtp"
    return None


def _write_demo_artifact(prompts: list[str], results: dict, artifact_name: str) -> None:
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

    print(f"\nMTP demo smoke results saved to: {output_file}")


@pytest.mark.requires_device(["DUAL", "QUAD"])
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "prompts_file,num_prompts,max_new_tokens",
    [
        pytest.param(
            Path("models/demos/deepseek_v3/demo/test_prompts.json"),
            2,
            32,
            id="smoke_2_prompts_32_tokens",
        ),
    ],
)
def test_mtp_demp_compare_outputs(
    prompts_file: Path,
    num_prompts: int,
    max_new_tokens: int,
    force_recalculate_weight_config: bool,
):
    prompts = load_prompts_from_json(str(prompts_file), max_prompts=num_prompts)

    common_kwargs = dict(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        repeat_batches=1,
        force_recalculate=force_recalculate_weight_config,
        signpost=True,
        sampling_temperature=0.0,
        sampling_top_k=1,
        sampling_top_p=1.0,
    )

    baseline = run_demo(enable_mtp=False, **common_kwargs)
    mtp = run_demo(enable_mtp=True, sample_on_device=False, **common_kwargs)

    _assert_demo_outputs_match(baseline, mtp)

    artifact_name = _artifact_name_for_current_mesh()
    if artifact_name is not None:
        _write_demo_artifact(prompts, mtp, artifact_name)

    # Prompt-level acceptance varies across real demo prompts. Keep this smoke test focused on
    # end-to-end output parity and on proving the MTP path executed and reported its stats.
    # Predictor quality and acceptance thresholds are covered by dedicated reference-based tests.
    mtp_stats = mtp.get("statistics", {})
    mtp_accept_rate = mtp_stats.get("mtp_accept_rate")
    mtp_verifies = mtp_stats.get("mtp_verifies")
    assert mtp_accept_rate is not None, "Expected mtp_accept_rate to be reported for MTP-enabled demo run"
    assert mtp_verifies is not None and mtp_verifies > 0, "Expected the MTP-enabled demo run to execute verify steps"
