# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unicodedata
from pathlib import Path

import pytest

from models.demos.deepseek_v3.demo.demo import load_prompts_from_json, run_demo

MODEL_PATH = Path(
    os.getenv("DEEPSEEK_V3_HF_MODEL", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized")
)
CACHE_DIR = Path(os.getenv("DEEPSEEK_V3_CACHE", "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/CI"))


def is_allowed_unicode_letter(char: str) -> bool:
    """True for Latin and Greek script letters per Unicode character name."""
    if not char.isalpha():
        return False
    name = unicodedata.name(char, "")
    return name.startswith("LATIN ") or name.startswith("GREEK ")


def find_disallowed_non_ascii_letter(text: str) -> tuple[int, str] | None:
    """Find the first disallowed non-ASCII letter in the text."""
    for char_index, char in enumerate(text):
        if char.isascii() or not char.isalpha() or is_allowed_unicode_letter(char):
            continue
        return char_index, char
    return None


def validate_english_keyboard_output(results: dict) -> None:
    generations = results.get("generations", [])
    for generation_index, generation in enumerate(generations, start=1):
        generated_text = generation.get("text")
        if generated_text is None:
            continue
        if not isinstance(generated_text, str):
            raise ValueError(
                "Generated output text must be a string or None: "
                f"generation_index={generation_index}, got={type(generated_text).__name__}"
            )

        bad_char = find_disallowed_non_ascii_letter(generated_text)
        if bad_char is None:
            continue

        bad_char_index, bad_char = bad_char
        unicode_name = unicodedata.name(bad_char, "UNKNOWN")
        raise ValueError(
            "Generated output contains disallowed non-Latin letter: "
            f"generation_index={generation_index}, char_index={bad_char_index}, "
            f"char={bad_char!r}, codepoint=U+{ord(bad_char):04X}, unicode_name={unicode_name}. "
            "Punctuation/symbols and Latin letters (including accents, e.g. é) are allowed; "
            "other scripts (e.g. CJK) are not."
        )


def _demo_case(
    *,
    max_prompts: int,
    repeat_batches: int,
    max_new_tokens: int,
    override_num_layers: int | None,
    enable_trace: bool,
    sample_on_device: bool,
    artifact_name: str | None,
    profile_decode: bool,
    stop_at_eos: bool | None,
    expect_full_length: bool,
    case_id: str,
    marks=None,
):
    return pytest.param(
        {
            "max_prompts": max_prompts,
            "repeat_batches": repeat_batches,
            "max_new_tokens": max_new_tokens,
            "override_num_layers": override_num_layers,
            "enable_trace": enable_trace,
            "sample_on_device": sample_on_device,
            "artifact_name": artifact_name,
            "profile_decode": profile_decode,
            "stop_at_eos": stop_at_eos,
            "expect_full_length": expect_full_length,
        },
        id=case_id,
        marks=marks,
    )


# Test matrix:
# +------------------+-------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+
# | id               | max_prompts | repeat_batches | max_new_tokens | override_num_layers | enable_trace | sample_on_device | artifact_name            | profile_decode | stop_at_eos | expect_full_length |
# +------------------+-------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+
# | tg_stress        | 56          | 2              | 128            | 5                   | False        | True             | None                     | False          | False       | True               |
# | dual_full_demo   | 256         | 1              | 129            | None                | True         | True             | dual_demo_full_results   | False          | None        | False              |
# | dual_stress_demo | 56          | 20             | 129            | None                | True         | True             | dual_demo_stress_results | False          | False       | True               |
# | quad_full_demo   | 512         | 1              | 129            | None                | True         | True             | quad_demo_full_results   | False          | None        | False              |
# | quad_stress_demo | 56          | 20             | 129            | None                | True         | True             | quad_demo_stress_results | False          | False       | True               |
# | profile_decode   | 1           | 1              | 13             | 5                   | True         | True             | None                     | True           | False       | True               |
# +------------------+-------------+----------------+----------------+---------------------+--------------+------------------+--------------------------+----------------+-------------+--------------------+


@pytest.mark.parametrize(
    # update test matrix table above if new test cases are added
    "case",
    [
        _demo_case(
            max_prompts=56,
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
            max_prompts=256,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_full_results",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            case_id="dual_full_demo",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(2400)],
        ),
        _demo_case(
            max_prompts=56,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="dual_demo_stress_results",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="dual_stress_demo",
            marks=[pytest.mark.requires_device(["DUAL"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=512,
            repeat_batches=1,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_full_results",
            profile_decode=False,
            stop_at_eos=None,
            expect_full_length=False,
            case_id="quad_full_demo",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(3600)],
        ),
        _demo_case(
            max_prompts=56,
            repeat_batches=20,
            max_new_tokens=129,
            override_num_layers=None,
            enable_trace=True,
            sample_on_device=True,
            artifact_name="quad_demo_stress_results",
            profile_decode=False,
            stop_at_eos=False,
            expect_full_length=True,
            case_id="quad_stress_demo",
            marks=[pytest.mark.requires_device(["QUAD"]), pytest.mark.timeout(5400)],
        ),
        _demo_case(
            max_prompts=1,
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
    validate_english_keyboard_output(results)

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
                    "text": gen_result.get("text"),
                }
            )

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\nDemo results saved to: {output_file}")
