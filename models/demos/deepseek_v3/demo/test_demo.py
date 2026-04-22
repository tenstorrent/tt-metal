# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from functools import lru_cache
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized")
)
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))
TOKEN_MATCHING_GOLDEN_DIR = Path(__file__).with_name("token_matching_goldens")
TOKEN_MATCHING_UPDATE_ENV = "DEEPSEEK_V3_UPDATE_TOKEN_MATCHING_GOLDENS"
TOKEN_MATCHING_SEED = 17
TOKEN_MATCHING_PREFIX_TAIL_TOKENS = 24


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


def _is_env_flag_enabled(env_name: str) -> bool:
    value = os.getenv(env_name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _truncate_for_log(text: str, max_chars: int = 160) -> str:
    compact = text.replace("\n", "\\n")
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


def _token_matching_golden_path(case: dict) -> Path:
    return TOKEN_MATCHING_GOLDEN_DIR / f"{case['case_id']}.json"


def _build_token_matching_payload(case: dict, prompts: list[str], results: dict) -> dict:
    generations = []
    for i, generation in enumerate(results.get("generations", []), start=1):
        prompt_text = prompts[i - 1] if i - 1 < len(prompts) else "[empty prompt]"
        generations.append(
            {
                "index": i,
                "prompt": prompt_text,
                "tokens": [int(token) for token in generation.get("tokens", [])],
                "text": generation.get("text"),
            }
        )

    return {
        "case_id": case["case_id"],
        "mesh_device": os.getenv("MESH_DEVICE", ""),
        "max_new_tokens": case["max_new_tokens"],
        "max_users_per_row": case["max_users_per_row"],
        "sampling": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "seed": TOKEN_MATCHING_SEED,
        },
        "prompts": [generation["prompt"] for generation in generations],
        "generations": generations,
    }


def _write_token_matching_golden(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _assert_token_matching_against_golden(case: dict, prompts: list[str], results: dict) -> None:
    if not case.get("token_matching_golden", False):
        return

    golden_path = _token_matching_golden_path(case)
    run_payload = _build_token_matching_payload(case, prompts, results)
    golden_exists = golden_path.exists()
    should_refresh = _is_env_flag_enabled(TOKEN_MATCHING_UPDATE_ENV)

    if should_refresh or not golden_exists:
        _write_token_matching_golden(golden_path, run_payload)
        action = "updated" if (golden_exists and should_refresh) else "created"
        reason = f" ({TOKEN_MATCHING_UPDATE_ENV}=1)" if should_refresh else ""
        print(f"\nToken-matching golden {action}: {golden_path}{reason}")
        return

    with open(golden_path, "r", encoding="utf-8") as handle:
        golden_payload = json.load(handle)

    failures = []
    expected_generations = golden_payload.get("generations", [])
    actual_generations = run_payload.get("generations", [])

    if len(expected_generations) != len(actual_generations):
        failures.append(
            "generation_count mismatch: " f"golden={len(expected_generations)} current={len(actual_generations)}"
        )

    compared_count = min(len(expected_generations), len(actual_generations))
    for idx in range(compared_count):
        expected = expected_generations[idx]
        actual = actual_generations[idx]

        expected_prompt = str(expected.get("prompt", ""))
        actual_prompt = str(actual.get("prompt", ""))
        if expected_prompt != actual_prompt:
            failures.append(
                f"prompt_mismatch generation[{idx + 1}]: "
                f"golden={_truncate_for_log(expected_prompt)!r} "
                f"current={_truncate_for_log(actual_prompt)!r}"
            )
            continue

        expected_tokens = [int(token) for token in expected.get("tokens", [])]
        actual_tokens = [int(token) for token in actual.get("tokens", [])]
        if expected_tokens == actual_tokens:
            continue

        mismatch_step = next(
            (
                step
                for step, (expected_token, actual_token) in enumerate(zip(expected_tokens, actual_tokens))
                if expected_token != actual_token
            ),
            min(len(expected_tokens), len(actual_tokens)),
        )

        expected_token = expected_tokens[mismatch_step] if mismatch_step < len(expected_tokens) else "<missing>"
        actual_token = actual_tokens[mismatch_step] if mismatch_step < len(actual_tokens) else "<missing>"
        generated_prefix = actual_tokens[:mismatch_step]
        prefix_tail = generated_prefix[-TOKEN_MATCHING_PREFIX_TAIL_TOKENS:]

        failures.append(
            f"generation[{idx + 1}] first_divergence: "
            f"prompt={_truncate_for_log(actual_prompt)!r} "
            f"decode_step={mismatch_step} "
            f"golden_token={expected_token} "
            f"asic_token={actual_token} "
            f"prefix_tail={prefix_tail} "
            f"(prefix_len={len(generated_prefix)}, golden_len={len(expected_tokens)}, current_len={len(actual_tokens)})"
        )

    if failures:
        shown = 30
        details = "\n".join(failures[:shown])
        remainder = (
            "" if len(failures) <= shown else f"\n... {len(failures) - shown} additional mismatch entries omitted."
        )
        pytest.fail(
            "Token-matching divergence against TT golden output.\n"
            f"case={case['case_id']}\n"
            f"golden={golden_path}\n"
            f"{details}{remainder}\n"
            f"To refresh this golden intentionally, rerun with {TOKEN_MATCHING_UPDATE_ENV}=1."
        )


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
    token_matching_golden: bool = False,
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
            "token_matching_golden": token_matching_golden,
            "case_id": case_id,
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
            token_matching_golden=True,
            case_id="dual_full_demo_32upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(2400)],
        ),
        _demo_case(
            max_prompts=24,
            max_users_per_row=8,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_full_results_8upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            token_matching_golden=True,
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
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(9400)],
        ),
        _demo_case(
            max_prompts=56,
            max_users_per_row=8,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_stress_results_8upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="dual_stress_demo_8upr",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(9400)],
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
            token_matching_golden=True,
            case_id="quad_full_demo_32upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        _demo_case(
            max_prompts=512,
            max_users_per_row=8,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_full_results_8upr",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            token_matching_golden=True,
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
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(9400)],
        ),
        _demo_case(
            max_prompts=56,
            max_users_per_row=8,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_stress_results_8upr",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="quad_stress_demo_8upr",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(9400)],
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
def test_demo(case: dict, force_recalculate_weight_config: bool, set_deterministic_env):
    # Path to the external JSON file containing prompts
    json_path = "models/demos/deepseek_v3/demo/test_prompts.json"

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
    if case["token_matching_golden"]:
        run_kwargs.update(
            sampling_temperature=0.0,
            sampling_top_k=1,
            sampling_top_p=1.0,
            sampling_seed=TOKEN_MATCHING_SEED,
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

    # Save results to JSON for artifact upload (QUAD tests only)
    if case["artifact_name"] is not None:
        artifact_dir = Path("generated/artifacts")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_file = artifact_dir / f"{case['artifact_name']}.json"

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
                    "tokens": gen_result.get("tokens", []),
                    "text": gen_result.get("text"),
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDemo results saved to: {output_file}")

    _assert_token_matching_against_golden(case, prompts, results)
    _assert_no_garbage_tokens(results)
