# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.audio.higgs_audio_v2.demo._audio_decode import (
    apply_delay_pattern_to_greedy_audio_tokens,
    initialize_delay_pattern_state,
)
from models.demos.audio.higgs_audio_v2.demo._prompts import (
    DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    DEFAULT_VALIDATION_CASES_PATH,
    build_case_sample,
    load_cases,
    load_reference_audio_manifest,
)
from models.demos.audio.higgs_audio_v2.tt.model import create_higgs_tt_model
from models.demos.audio.higgs_audio_v2.tt.reference import (
    load_audio_tokenizer,
    load_higgs_config,
    load_higgs_tokenizer,
    prepare_inputs_for_generation,
)

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
ACCURACY_MIN_TOKEN_ACCURACY = 0.95
pytestmark = pytest.mark.timeout(600)


@torch.inference_mode()
def _run_reference_case(reference_model, inputs: dict, config, num_audio_steps: int) -> dict:
    generation_outputs = reference_model.generate(
        **inputs,
        do_sample=False,
        use_cache=True,
        max_new_tokens=num_audio_steps + 1,
        return_dict_in_generate=True,
        output_logits=True,
    )
    if len(generation_outputs.logits) == 0:
        raise RuntimeError("PyTorch reference generation did not return any logits.")
    if not generation_outputs.audio_sequences:
        raise RuntimeError("PyTorch reference generation did not return any audio sequences.")

    audio_sequence = generation_outputs.audio_sequences[-1].detach().long().cpu()
    if audio_sequence.shape[1] == 0:
        raise RuntimeError("PyTorch reference generation returned an empty audio sequence.")
    if not torch.equal(
        audio_sequence[:, 0],
        torch.full((config.audio_num_codebooks,), config.audio_stream_bos_id, dtype=torch.long),
    ):
        raise RuntimeError("PyTorch reference generation did not start with the expected audio BOS column.")

    num_delay, num_remaining_delays = initialize_delay_pattern_state(audio_sequence[:, :1], config)
    audio_steps = []
    available_audio_steps = min(num_audio_steps, audio_sequence.shape[1] - 1, len(generation_outputs.logits) - 1)
    for step_idx in range(available_audio_steps):
        audio_logits = generation_outputs.logits[step_idx + 1].detach().float().cpu()
        (
            next_tokens,
            active_mask,
            num_delay,
            num_remaining_delays,
            finished,
        ) = apply_delay_pattern_to_greedy_audio_tokens(
            audio_logits,
            config,
            num_delay,
            num_remaining_delays,
        )
        expected_next_tokens = audio_sequence[:, step_idx + 1]
        if not torch.equal(next_tokens, expected_next_tokens):
            raise AssertionError(
                f"PyTorch reference reconstruction drifted at audio step {step_idx}: "
                f"{next_tokens.tolist()} != {expected_next_tokens.tolist()}"
            )
        audio_steps.append(
            {
                "audio_logits": audio_logits,
                "next_tokens": expected_next_tokens,
                "active_mask": active_mask,
                "finished": finished,
            }
        )
        if finished:
            break

    return {
        "audio_sequence": audio_sequence,
        "audio_steps": audio_steps,
    }


def _prepare_case_inputs(
    case: dict,
    case_manifest_path: Path,
    reference_audio_index: dict,
    tokenizer,
    audio_tokenizer,
    config,
) -> dict:
    sample = build_case_sample(case, base_dir=case_manifest_path.parent, reference_audio_index=reference_audio_index)
    return prepare_inputs_for_generation(
        chat_ml_sample=sample,
        tokenizer=tokenizer,
        audio_tokenizer=audio_tokenizer,
        config=config,
        force_audio_gen=True,
        device="cpu",
    )


def _evaluate_accuracy_case(tt_model, model_inputs: dict, reference_case: dict, config) -> dict:
    tt_model.reset_kv_cache()
    prompt_embeddings, prompt_audio_mask = tt_model.embed_prompt_inputs(
        input_ids=model_inputs["input_ids"],
        audio_in_ids=model_inputs.get("audio_in_ids"),
        audio_in_ids_start=model_inputs.get("audio_in_ids_start"),
        audio_out_ids=model_inputs.get("audio_out_ids"),
        audio_out_ids_start=model_inputs.get("audio_out_ids_start"),
    )
    tt_model.prefill(prompt_embeddings, prompt_audio_mask, return_logits=False)

    current_audio_tokens = torch.full((config.audio_num_codebooks, 1), config.audio_stream_bos_id, dtype=torch.long)
    num_delay, num_remaining_delays = initialize_delay_pattern_state(current_audio_tokens, config)
    current_pos = prompt_embeddings.shape[1]
    current_pos_tt = tt_model.create_current_pos_tensor(current_pos)
    token_matches = 0
    token_total = 0
    steps_run = 0

    try:
        for reference_audio_step in reference_case["audio_steps"]:
            current_embedding = tt_model.embed_audio_tokens(current_audio_tokens)[0]
            _, tt_audio_logits_flat = tt_model.decode_step(
                current_embedding=current_embedding,
                current_pos=current_pos,
                is_audio_token=True,
                return_text_logits=False,
                current_pos_tt=current_pos_tt,
            )
            tt_audio_logits = tt_audio_logits_flat.view(config.audio_num_codebooks, -1)
            ref_next_tokens = reference_audio_step["next_tokens"]
            active_mask = reference_audio_step["active_mask"]
            tt_next_tokens, _, num_delay, num_remaining_delays, _ = apply_delay_pattern_to_greedy_audio_tokens(
                tt_audio_logits,
                config,
                num_delay,
                num_remaining_delays,
            )
            token_matches += int((tt_next_tokens[active_mask] == ref_next_tokens[active_mask]).sum().item())
            token_total += int(active_mask.sum().item())
            current_audio_tokens = ref_next_tokens.unsqueeze(1)
            current_pos += 1
            tt_model.increment_current_pos_tensor(current_pos_tt)
            steps_run += 1
            if reference_audio_step["finished"]:
                break
    finally:
        ttnn.deallocate(current_pos_tt)

    return {
        "prompt_len": int(model_inputs["input_ids"].shape[1]),
        "merged_prompt_len": int(prompt_embeddings.shape[1]),
        "generated_audio_steps": steps_run,
        "token_matches": token_matches,
        "token_total": token_total,
        "token_accuracy": token_matches / token_total if token_total > 0 else 1.0,
    }


@torch.inference_mode()
def run_accuracy_check(
    model_path: str = MODEL_PATH,
    audio_tokenizer_path: str = AUDIO_TOKENIZER_PATH,
    cases_json_path: str | Path = DEFAULT_VALIDATION_CASES_PATH,
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    reference_audio_assets_root: str | Path | None = None,
) -> dict:
    try:
        from boson_multimodal.model.higgs_audio import HiggsAudioModel
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Unable to import `boson_multimodal`. Export "
            "`PYTHONPATH=$HIGGS_AUDIO_REPO:$TT_METAL_HOME${PYTHONPATH:+:$PYTHONPATH}` "
            "before running the Higgs Audio accuracy test."
        ) from exc

    cases_json_path = Path(cases_json_path).resolve()
    validation_cases = load_cases(cases_json_path)
    config = load_higgs_config(model_path)
    tokenizer = load_higgs_tokenizer(model_path)
    audio_tokenizer = load_audio_tokenizer(audio_tokenizer_path, device="cpu")
    reference_audio_index = load_reference_audio_manifest(
        reference_audio_manifest_path,
        assets_root=reference_audio_assets_root,
        require_local=True,
    )
    reference_model = HiggsAudioModel.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cpu")
    reference_model.eval()

    model_inputs_by_case = {}
    reference_by_case = {}
    for case in validation_cases:
        model_inputs = _prepare_case_inputs(
            case=case,
            case_manifest_path=cases_json_path,
            reference_audio_index=reference_audio_index,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            config=config,
        )
        model_inputs_by_case[case["name"]] = model_inputs
        reference_by_case[case["name"]] = _run_reference_case(
            reference_model=reference_model,
            inputs=model_inputs,
            config=config,
            num_audio_steps=int(case["max_audio_steps"]),
        )

    previous_fallback_setting = ttnn.CONFIG.throw_exception_on_fallback
    ttnn.CONFIG.throw_exception_on_fallback = True
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))
    try:
        _, tt_model, _ = create_higgs_tt_model(
            mesh_device,
            model_path,
            optimizations="accuracy",
            use_hf_rope=True,
        )
        case_reports = []
        token_matches = 0
        token_total = 0
        for case in validation_cases:
            case_report = _evaluate_accuracy_case(
                tt_model=tt_model,
                model_inputs=model_inputs_by_case[case["name"]],
                reference_case=reference_by_case[case["name"]],
                config=config,
            )
            case_report.update(
                {
                    "name": case["name"],
                    "mode": case["mode"],
                    "language": case["language"],
                    "max_audio_steps": int(case["max_audio_steps"]),
                }
            )
            token_matches += case_report["token_matches"]
            token_total += case_report["token_total"]
            case_reports.append(case_report)

        return {
            "mode": "accuracy",
            "model_path": model_path,
            "audio_tokenizer_path": audio_tokenizer_path,
            "cases_json_path": str(cases_json_path),
            "reference_audio_manifest_path": str(Path(reference_audio_manifest_path).resolve()),
            "reference_audio_assets_root": str(reference_audio_assets_root) if reference_audio_assets_root else None,
            "token_matches": token_matches,
            "token_total": token_total,
            "token_accuracy": token_matches / token_total if token_total > 0 else 1.0,
            "case_reports": case_reports,
        }
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.CONFIG.throw_exception_on_fallback = previous_fallback_setting


def test_accuracy():
    report = run_accuracy_check()

    assert report["mode"] == "accuracy"
    assert report["token_accuracy"] >= ACCURACY_MIN_TOKEN_ACCURACY
