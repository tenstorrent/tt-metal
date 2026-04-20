# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.deepseek_v3.demo.demo import run_demo
from models.demos.deepseek_v3.demo.token_accuracy import decompress_lzma_payload
from models.demos.deepseek_v3.utils.config_helpers import DEFAULT_MAX_SEQ_LEN, K_CHUNK_SIZE
from models.demos.deepseek_v3.utils.hf_model_utils import load_tokenizer
from models.demos.deepseek_v3.utils.test_utils import system_name_to_mesh_shape

MODEL_PATH = Path(
    os.getenv(
        "DEEPSEEK_V3_HF_MODEL",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-dequantized",
    )
)
CACHE_DIR = Path(
    os.getenv(
        "DEEPSEEK_V3_CACHE",
        "/mnt/MLPerf/tt_dnn-models/deepseek-ai/DeepSeek-R1-0528-Cache/dev/",
    )
)

REFERENCE_FILE = Path(
    os.getenv(
        "DEEPSEEK_V3_TEACHER_REF",
        str(Path(__file__).with_name("deepseek_r1_teacher_forcing_512.refpt")),
    )
)

GENERATED_OUTPUTS_FILE = Path(__file__).with_name("teacher_forced_generated_outputs.json")


def tile_align(length: int) -> int:
    k_chunk_size = K_CHUNK_SIZE
    aligned_size = max(int(ttnn.TILE_SIZE), k_chunk_size)
    return ((int(length) + aligned_size - 1) // aligned_size) * aligned_size


def resolve_prompt_count(system_name: str) -> int:
    name = system_name.upper()
    if name == "QUAD":
        return 512
    if name == "DUAL":
        return 256
    return 1


def entry_prompt_text(entry: dict, tokenizer) -> str:
    prompt = entry.get("prompt")
    if prompt:
        return str(prompt)
    prompt_tokens = entry.get("prompt_tokens")
    if isinstance(prompt_tokens, torch.Tensor):
        ids = prompt_tokens[0].tolist() if prompt_tokens.dim() == 2 else prompt_tokens.tolist()
        if ids:
            return tokenizer.decode([int(x) for x in ids], skip_special_tokens=False)
    return ""


def entry_generated_text(entry: dict, tokenizer) -> str:
    generated = entry.get("decoded_generated_text")
    if generated:
        return str(generated)
    generated_tokens = entry.get("generated_tokens")
    if isinstance(generated_tokens, torch.Tensor):
        ids = generated_tokens[0].tolist() if generated_tokens.dim() == 2 else generated_tokens.tolist()
        if ids:
            return tokenizer.decode([int(x) for x in ids], skip_special_tokens=True)
    return ""


@pytest.mark.timeout(3600)
@pytest.mark.parametrize("reference_file", [REFERENCE_FILE])
@pytest.mark.parametrize("max_new_tokens", [128], ids=["128"])
def test_demo_teacher_forcing_accuracy(
    reference_file: Path,
    max_new_tokens: int,
    is_ci_env: bool,
    force_recalculate_weight_config: bool,
    tmp_path: Path,
):
    if not reference_file.exists():
        pytest.fail(
            f"Reference file not found at {reference_file}. "
            "Generate it with: python models/demos/deepseek_v3/demo/convert_api_json_to_refpt.py "
            "--input <api-json> --output <refpt-path> --model-path <hf-model>"
        )

    requested_system_name = os.getenv("MESH_DEVICE")
    if requested_system_name is None:
        pytest.fail("Environment variable $MESH_DEVICE is not set. Please set it to DUAL or QUAD.")

    desired_prompt_count = resolve_prompt_count(requested_system_name)
    if is_ci_env:
        desired_prompt_count = 1

    mesh_shape = system_name_to_mesh_shape(requested_system_name.upper())
    mesh_rows = mesh_shape[0]
    if desired_prompt_count % mesh_rows != 0:
        pytest.fail(
            f"Desired prompt count {desired_prompt_count} is not divisible by "
            f"mesh rows {mesh_rows} for {requested_system_name}."
        )
    max_users_per_row = desired_prompt_count // mesh_rows
    num_users = max_users_per_row * mesh_rows

    tokenizer = load_tokenizer(MODEL_PATH)

    payload = torch.load(reference_file, weights_only=False)

    fmt = payload.get("format_version", "")
    if fmt == "multi_prompt_v1_lzma_v1":
        entries = decompress_lzma_payload(payload)
    else:
        entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        pytest.fail("Reference payload must contain non-empty 'entries'.")
    if len(entries) < desired_prompt_count:
        pytest.fail(
            f"Reference contains {len(entries)} prompts, but "
            f"{requested_system_name} requires {desired_prompt_count}."
        )
    entries = entries[:desired_prompt_count]

    max_prompt_len = max(int(entry["tf_prompt_len"]) for entry in entries)
    configured_max_seq_len = tile_align(max_prompt_len + max_new_tokens)
    if configured_max_seq_len > DEFAULT_MAX_SEQ_LEN:
        pytest.skip(
            f"Teacher-forced context needs max_seq_len={configured_max_seq_len}, "
            f"exceeds default {DEFAULT_MAX_SEQ_LEN}."
        )

    run_payload = {
        "format_version": "multi_prompt_v1",
        "num_prompts": len(entries),
        "max_new_tokens": max_new_tokens,
        "token_ids_meta": payload.get("token_ids_meta", {}),
        "entries": entries,
    }
    e0 = entries[0]
    run_payload.update(
        {
            "reference_tokens": e0.get(
                "reference_tokens", torch.cat([e0["prompt_tokens"], e0["generated_tokens"]], dim=1)
            ),
            "prompt_tokens": e0["prompt_tokens"],
            "generated_tokens": e0["generated_tokens"],
            "top5_tokens": e0["top5_tokens"],
            "tf_prompt_len": int(e0["tf_prompt_len"]),
            "prompt": entry_prompt_text(e0, tokenizer),
            "decoded_generated_text": entry_generated_text(e0, tokenizer),
        }
    )

    run_ref_file = tmp_path / "teacher_forcing_multi_prompt.refpt"
    torch.save(run_payload, run_ref_file)

    prompts = [entry_prompt_text(entry, tokenizer) for entry in entries]
    assert len(prompts) == num_users, (
        f"Prompt count {len(prompts)} must match run users {num_users} "
        f"({mesh_rows} rows x {max_users_per_row} users/row) for one-call validation."
    )

    logger.info(
        "Running one teacher-forced call for {} prompts ({} rows x {} users/row)",
        len(prompts),
        mesh_rows,
        max_users_per_row,
    )
    results = run_demo(
        prompts=prompts,
        model_path=MODEL_PATH,
        cache_dir=CACHE_DIR,
        random_weights=False,
        max_new_tokens=max_new_tokens,
        max_seq_len=configured_max_seq_len,
        repeat_batches=1,
        token_accuracy=True,
        reference_file=run_ref_file,
        enable_trace=True,
        force_recalculate=force_recalculate_weight_config,
        stop_at_eos=False,
        sample_on_device=False,
        max_users_per_row=max_users_per_row,
    )

    generations = results.get("generations", [])
    assert len(generations) == len(
        entries
    ), f"Expected {len(entries)} generations (one per prompt), got {len(generations)}"

    output_records: list[dict[str, str | int]] = []
    for idx, (entry, gen) in enumerate(zip(entries, generations)):
        pred_ids = gen.get("predicted_tokens", [])
        tt_text = tokenizer.decode([int(x) for x in pred_ids], skip_special_tokens=True) if pred_ids else ""
        output_records.append(
            {
                "index": idx,
                "prompt": entry_prompt_text(entry, tokenizer),
                "tt_output": tt_text,
                "gt_output": entry_generated_text(entry, tokenizer),
            }
        )
    GENERATED_OUTPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GENERATED_OUTPUTS_FILE.open("w", encoding="utf-8") as f:
        json.dump(output_records, f, ensure_ascii=False, indent=2)
        f.write("\n")
    logger.info("Saved {} prompt outputs to {}", len(output_records), GENERATED_OUTPUTS_FILE)

    statistics = results.get("statistics", {})
    agg_top1 = statistics.get("teacher_forcing_top1")
    agg_top5 = statistics.get("teacher_forcing_top5")
    agg_total = statistics.get("teacher_forcing_total_tokens")

    if agg_top1 is not None and agg_top5 is not None and agg_total is not None and agg_total > 0:
        total_top1 = float(agg_top1)
        total_top5 = float(agg_top5)
        overall_compared = int(agg_total)
        logger.info(
            "Using aggregate accuracy from run_demo: {} tokens, top1={:.2%}, top5={:.2%}",
            overall_compared,
            total_top1,
            total_top5,
        )
    else:
        overall_top1_matches = 0
        overall_top5_matches = 0
        overall_compared = 0
        for idx, (entry, gen) in enumerate(zip(entries, generations)):
            expected_forced = entry["generated_tokens"][0].tolist()[:max_new_tokens]
            got_forced = [int(x) for x in gen.get("tokens", [])]
            assert got_forced == expected_forced, (
                f"Prompt {idx}: teacher-forced token mismatch.\n"
                f"First 20 expected: {expected_forced[:20]}\n"
                f"First 20 got     : {got_forced[:20]}"
            )

            tt_preds = [int(x) for x in gen.get("predicted_tokens", [])]
            top5_tokens = entry["top5_tokens"]
            tf_prompt_len = int(entry["tf_prompt_len"])
            gen_len = int(entry["generated_tokens"].shape[1])
            compared = min(len(tt_preds), gen_len, max_new_tokens)
            assert compared > 0, f"Prompt {idx}: no tokens available to compare"

            for i in range(compared):
                pos = tf_prompt_len + i
                hf_top5 = top5_tokens[pos].tolist()
                hf_top1 = hf_top5[0]
                tt_pred = tt_preds[i]
                if tt_pred == hf_top1:
                    overall_top1_matches += 1
                if tt_pred in hf_top5:
                    overall_top5_matches += 1
                overall_compared += 1

        total_top1 = overall_top1_matches / overall_compared if overall_compared else 0.0
        total_top5 = overall_top5_matches / overall_compared if overall_compared else 0.0

    logger.info(
        "Aggregate teacher-forcing accuracy over {} prompts / {} tokens: top1={:.2%}, top5={:.2%}",
        len(entries),
        overall_compared,
        total_top1,
        total_top5,
    )

    min_expected_top1 = 0.70
    min_expected_top5 = 0.85
    assert total_top1 >= min_expected_top1, (
        f"Aggregate top-1 accuracy {total_top1:.4f} is below minimum {min_expected_top1:.2f} "
        f"across {len(entries)} prompts / {overall_compared} tokens."
    )
    assert total_top5 >= min_expected_top5, (
        f"Aggregate top-5 accuracy {total_top5:.4f} is below minimum {min_expected_top5:.2f} "
        f"across {len(entries)} prompts / {overall_compared} tokens."
    )
