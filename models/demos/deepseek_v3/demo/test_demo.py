# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized-stacked")
)
_ds_cache = os.getenv("DEEPSEEK_V3_CACHE")
CACHE_DIR = Path(_ds_cache) if _ds_cache else None
PERF_MARGIN = 0.08
FINAL_DECODE_TPS_PER_USER = "decode_t/s/u"


@lru_cache(maxsize=1)
def get_total_model_layers(model_path: Path) -> int:
    with open(model_path / "config.json", "r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    total_layers = config.get("num_hidden_layers")
    if not isinstance(total_layers, int):
        raise ValueError(f"Expected integer num_hidden_layers in {(model_path / 'config.json')}, got {total_layers!r}")
    return total_layers


def _assert_no_garbage_tokens(results: dict) -> None:
    failures = []
    for i, generation in enumerate(results.get("generations", []), start=1):
        garbage_count = int(generation.get("garbage_token_count", 0) or 0)
        if garbage_count == 0:
            continue
        garbage_checked = int(generation.get("garbage_tokens_checked", 0) or 0)
        garbage_topk = generation.get("garbage_token_topk")
        failures.append(
            f"Generation {i}: garbage_token_count={garbage_count} over {garbage_checked} checked tokens"
            + (f" against teacher top-{garbage_topk}" if garbage_topk is not None else "")
        )
        failures.extend(generation.get("garbage_token_debug", []) or [])

    if failures:
        pytest.fail("Garbage tokens detected during demo:\n" + "\n".join(failures))


def _is_primary_artifact_writer() -> bool:
    for rank_env in ("TT_MESH_HOST_RANK", "OMPI_COMM_WORLD_RANK", "PMI_RANK", "RANK"):
        rank_value = os.getenv(rank_env)
        if rank_value is None:
            continue
        try:
            return int(rank_value) == 0
        except ValueError:
            return False
    return True


def _assert_perf_targets(results: dict, perf_targets: dict[str, float]) -> None:
    statistics = results.get("statistics", {})
    assert statistics, "Expected demo statistics for performance assertion"

    for metric_name, expected in perf_targets.items():
        measured = statistics.get(metric_name)
        assert measured is not None, f"Expected demo statistic {metric_name!r} for performance assertion"

        measured = float(measured)
        lower = expected * (1 - PERF_MARGIN)
        upper = expected * (1 + PERF_MARGIN)
        assert lower <= measured <= upper, (
            f"{metric_name}={measured:.6f} is outside expected range "
            f"[{lower:.6f}, {upper:.6f}] (expected {expected:.6f} +/- {PERF_MARGIN*100:.1f}%)"
        )


def _timestamped_artifact_stem(artifact_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{artifact_name}_{timestamp}"


def _demo_case(
    *,
    max_prompts: int,
    max_users_per_row: int,
    repeat_batches: int,
    max_new_tokens: int,
    override_num_layers: int | None,
    enable_trace: bool,
    sample_on_device: bool,
    artifact_name: str | None,
    profile_decode: bool,
    stop_at_eos: bool | None,
    expect_full_length: bool,
    perf_targets: dict[str, float] | None = None,
    case_id: str,
    marks=None,
):
    return pytest.param(
        {
            "max_prompts": max_prompts,
            "max_users_per_row": max_users_per_row,
            "repeat_batches": repeat_batches,
            "max_new_tokens": max_new_tokens,
            "override_num_layers": override_num_layers,
            "enable_trace": enable_trace,
            "sample_on_device": sample_on_device,
            "artifact_name": artifact_name,
            "profile_decode": profile_decode,
            "stop_at_eos": stop_at_eos,
            "expect_full_length": expect_full_length,
            "perf_targets": perf_targets,
        },
        id=case_id,
        marks=marks,
    )


# Test matrix:
# +------------------+-------------+-------------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+
# | id               | max_prompts | max_users_per_row | repeat_batches | max_new_tokens | override_num_layers | enable_trace | sample_on_device | artifact_name            | profile_decode | stop_at_eos | expect_full_length |
# +------------------+-------------+-------------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+
# | tg_stress        | 56          | 32                | 2              | 128            | 5                   | False        | True             | None                     | False          | False       | True               |
# | tg_upr8          | 32          | 8                 | 2              | 128            | 5                   | False        | True             | None                     | False          | False       | True               |
# | dual_full_demo   | 256         | 32                | 1              | 129            | None                | True         | True             | dual_demo_full_results   | False          | None        | False              |
# | dual_stress_demo | 56          | 32                | 20             | 129            | None                | True         | True             | dual_demo_stress_results | False          | False       | True               |
# | quad_full_demo   | 512         | 32                | 1              | 129            | None                | True         | True             | quad_demo_full_results   | False          | None        | False              |
# | quad_stress_demo | 56          | 32                | 20             | 129            | None                | True         | True             | quad_demo_stress_results | False          | False       | True               |
# | profile_decode   | 1           | 32                | 1              | 13             | 5                   | True         | True             | None                     | True           | False       | True               |
# +------------------+-------------+-------------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+


@pytest.mark.parametrize(
    # update test matrix table above if new test cases are added
    "case",
    [
        _demo_case(
            max_prompts=56,
            max_users_per_row=32,
            repeat_batches=2,
            max_new_tokens=128,
            override_num_layers=5,
            enable_trace=False,
            sample_on_device=True,
            artifact_name=None,
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="tg_stress",
            marks=pytest.mark.requires_device(["TG"]),
        ),
        _demo_case(
            max_prompts=32,
            max_users_per_row=8,
            repeat_batches=2,
            max_new_tokens=128,
            override_num_layers=5,
            enable_trace=False,
            sample_on_device=True,
            artifact_name=None,
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="tg_upr8",
            marks=pytest.mark.requires_device(["TG"]),
        ),
        _demo_case(
            max_prompts=256,
            max_users_per_row=32,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_full_results_32upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            perf_targets={FINAL_DECODE_TPS_PER_USER: 0.986038678353197},
            case_id="dual_full_demo_32upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(2400)],
        ),
        _demo_case(
            max_prompts=64,
            max_users_per_row=8,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=False,
            artifact_name="dual_demo_full_results_8upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            case_id="dual_full_demo_8upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(2400)],
        ),
        _demo_case(
            max_prompts=56,
            max_users_per_row=32,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_stress_results_32upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="dual_stress_demo_32upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=14,
            max_users_per_row=8,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=False,
            artifact_name="dual_demo_stress_results_8upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="dual_stress_demo_8upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=512,
            max_users_per_row=32,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_full_results_32upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            perf_targets={FINAL_DECODE_TPS_PER_USER: 1.0168109351915215},
            case_id="quad_full_demo_32upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        _demo_case(
            max_prompts=128,
            max_users_per_row=8,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=False,
            artifact_name="quad_demo_full_results_8upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            case_id="quad_full_demo_8upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        _demo_case(
            max_prompts=56,
            max_users_per_row=32,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_stress_results_32upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="quad_stress_demo_32upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=14,
            max_users_per_row=8,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=False,
            artifact_name="quad_demo_stress_results_8upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="quad_stress_demo_8upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=1,
            max_users_per_row=32,
            repeat_batches=1,
            max_new_tokens=13,
            override_num_layers=5,
            enable_trace=True,
            sample_on_device=True,
            artifact_name=None,
            profile_decode=True,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="profile_decode",
            marks=pytest.mark.timeout(1800),
        ),
    ],
)
def test_demo(case: dict, force_recalculate_weight_config: bool):
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/demo_aime24_gpqa_short.json"

    # Load prompts from JSON file
    prompts = load_prompts_from_json(json_path, max_prompts=case["max_prompts"])

    # Run demo
    run_kwargs = dict(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=case["max_new_tokens"],
        max_users_per_row=case["max_users_per_row"],
        repeat_batches=case["repeat_batches"],
        enable_trace=case["enable_trace"],
        sample_on_device=case["sample_on_device"],
        profile_decode=case["profile_decode"],
        force_recalculate=force_recalculate_weight_config,
        signpost=True,
    )
    if case["override_num_layers"] is not None:
        run_kwargs["override_num_layers"] = case["override_num_layers"]
    if case["stop_at_eos"] is not None:
        run_kwargs["stop_at_eos"] = case["stop_at_eos"]

    results = run_demo(**run_kwargs)

    requested_system_name = os.getenv("MESH_DEVICE")
    assert requested_system_name is not None, "MESH_DEVICE must be set for demo tests"
    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    expected_generations = min(len(prompts), case["max_users_per_row"] * int(mesh_shape[0]))

    assert len(results["generations"]) == expected_generations

    # Full-demo cases can stop early on EOS; stress/profile cases disable EOS and
    # should always produce the requested token count.
    generated_lengths = [len(generation["tokens"]) for generation in results["generations"]]
    if case["expect_full_length"]:
        assert all(length == case["max_new_tokens"] for length in generated_lengths)
    else:
        assert all(length <= case["max_new_tokens"] for length in generated_lengths)

    # Save artifact from host rank 0 only to avoid multi-host write races.
    if case["artifact_name"] is not None and _is_primary_artifact_writer():
        artifact_dir = Path("generated/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_file = artifact_dir / f"{_timestamped_artifact_stem(case['artifact_name'])}.json"

        output_data = {
            "prompts": prompts if prompts else [],
            "generations": [],
            "statistics": results.get("statistics", {}),
            "model_params": results.get("model_params", {}),
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

    _assert_no_garbage_tokens(results)
    if case["perf_targets"] is not None:
        _assert_perf_targets(results, case["perf_targets"])
