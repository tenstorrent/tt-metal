# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import time
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.audio.higgs_audio_v2.demo._prompts import (
    DEFAULT_PERFORMANCE_CASES_PATH,
    DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
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
from models.tt_transformers.tt.common import copy_host_to_device

MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
AUDIO_TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"
PERFORMANCE_MIN_TOKENS_PER_SECOND = 60.0
PERFORMANCE_MAX_RTF = 0.5
PERFORMANCE_TRACE_BLOCK_STEPS = 64
PERFORMANCE_TRACE_REGION_SIZE = 400_000_000
pytestmark = pytest.mark.timeout(600)


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


def _select_decode_trace_block_steps(max_audio_steps: int) -> int:
    if max_audio_steps < 1:
        raise ValueError(f"Decode trace block steps must be >= 1, got {max_audio_steps}")
    return min(int(max_audio_steps), PERFORMANCE_TRACE_BLOCK_STEPS)


def _allocate_decode_runtime_state(tt_model, config) -> dict:
    initial_audio_tokens = torch.full(
        (config.audio_num_codebooks, 1),
        config.audio_stream_bos_id,
        dtype=torch.long,
    )
    tt_model._ensure_audio_delay_postprocess_tensors()
    postprocess_host_inputs = tt_model.prepare_audio_decode_delay_state_host(num_delay=0, num_remaining_delays=None)
    postprocess_state_inputs = tuple(copy_host_to_device(postprocess_host_inputs, mesh_device=tt_model.mesh_device))
    finished_state_host_input = tt_model.prepare_audio_decode_finished_state_host(False)
    finished_output = copy_host_to_device((finished_state_host_input,), mesh_device=tt_model.mesh_device)[0]
    host_inputs = tt_model.prepare_audio_decode_raw_token_inputs_host(initial_audio_tokens, current_pos=0)
    device_inputs = tuple(copy_host_to_device(host_inputs, mesh_device=tt_model.mesh_device))
    return {
        "host_inputs": host_inputs,
        "device_inputs": device_inputs,
        "postprocess_host_inputs": postprocess_host_inputs,
        "postprocess_state_inputs": postprocess_state_inputs,
        "finished_state_host_input": finished_state_host_input,
        "finished_output": finished_output,
    }


def _reset_decode_runtime_state(decode_runtime_state: dict) -> None:
    copy_host_to_device(
        host_tensors=decode_runtime_state["host_inputs"],
        device_tensors=decode_runtime_state["device_inputs"],
    )
    copy_host_to_device(
        host_tensors=decode_runtime_state["postprocess_host_inputs"],
        device_tensors=decode_runtime_state["postprocess_state_inputs"],
    )
    copy_host_to_device(
        host_tensors=(decode_runtime_state["finished_state_host_input"],),
        device_tensors=(decode_runtime_state["finished_output"],),
    )


def _release_decode_runtime_state(decode_runtime_state: dict | None) -> None:
    if decode_runtime_state is None:
        return
    for tensor in decode_runtime_state.get("device_inputs", ()):
        if tensor is not None:
            ttnn.deallocate(tensor)
    for tensor in decode_runtime_state.get("postprocess_state_inputs", ()):
        if tensor is not None:
            ttnn.deallocate(tensor)
    finished_output = decode_runtime_state.get("finished_output")
    if finished_output is not None:
        ttnn.deallocate(finished_output)


def _capture_prefill_trace(tt_model, prefill_inputs: dict, decode_runtime_state: dict) -> dict:
    host_inputs = prefill_inputs["host_inputs"]
    device_inputs = copy_host_to_device(host_inputs, mesh_device=tt_model.mesh_device)
    decode_bootstrap_host_inputs = tt_model.prepare_audio_decode_bootstrap_inputs_host(
        current_pos=prefill_inputs["effective_seq_len"]
    )
    decode_bootstrap_device_inputs = tuple(
        copy_host_to_device(decode_bootstrap_host_inputs, mesh_device=tt_model.mesh_device)
    )
    decode_device_inputs = decode_runtime_state["device_inputs"]
    decode_delay_state_inputs = decode_runtime_state["postprocess_state_inputs"]
    decode_finished_output = decode_runtime_state["finished_output"]

    transformed_inputs = tt_model.transform_and_embed_prefill_inputs_device(*device_inputs)
    compile_output = tt_model.ttnn_prefill_forward(
        x=transformed_inputs[0],
        effective_seq_len=prefill_inputs["effective_seq_len"],
        audio_token_mask=transformed_inputs[1],
        inverse_audio_token_mask=transformed_inputs[2],
        page_table=transformed_inputs[3],
        chunk_page_table=transformed_inputs[4],
        return_logits=False,
        release_audio_masks=False,
    )
    tt_model.audio_ttnn_initialize_decode_state_inplace(
        decode_bootstrap_device_inputs[0],
        decode_bootstrap_device_inputs[1],
        decode_bootstrap_device_inputs[2],
        decode_bootstrap_device_inputs[3],
        decode_bootstrap_device_inputs[4],
        decode_bootstrap_device_inputs[5],
        decode_device_inputs[0],
        decode_device_inputs[1],
        decode_device_inputs[2],
        decode_delay_state_inputs[0],
        decode_delay_state_inputs[1],
        decode_finished_output,
    )
    ttnn.synchronize_device(tt_model.mesh_device)
    ttnn.deallocate(compile_output)

    copy_host_to_device(host_tensors=host_inputs, device_tensors=device_inputs)
    copy_host_to_device(host_tensors=decode_bootstrap_host_inputs, device_tensors=decode_bootstrap_device_inputs)
    trace_id = ttnn.begin_trace_capture(tt_model.mesh_device, cq_id=0)
    transformed_inputs = tt_model.transform_and_embed_prefill_inputs_device(*device_inputs)
    trace_output = tt_model.ttnn_prefill_forward(
        x=transformed_inputs[0],
        effective_seq_len=prefill_inputs["effective_seq_len"],
        audio_token_mask=transformed_inputs[1],
        inverse_audio_token_mask=transformed_inputs[2],
        page_table=transformed_inputs[3],
        chunk_page_table=transformed_inputs[4],
        return_logits=False,
        release_audio_masks=False,
    )
    tt_model.audio_ttnn_initialize_decode_state_inplace(
        decode_bootstrap_device_inputs[0],
        decode_bootstrap_device_inputs[1],
        decode_bootstrap_device_inputs[2],
        decode_bootstrap_device_inputs[3],
        decode_bootstrap_device_inputs[4],
        decode_bootstrap_device_inputs[5],
        decode_device_inputs[0],
        decode_device_inputs[1],
        decode_device_inputs[2],
        decode_delay_state_inputs[0],
        decode_delay_state_inputs[1],
        decode_finished_output,
    )
    ttnn.end_trace_capture(tt_model.mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_model.mesh_device)
    return {
        "trace_id": trace_id,
        "trace_output": trace_output,
        "device_inputs": device_inputs,
        "decode_bootstrap_device_inputs": decode_bootstrap_device_inputs,
    }


def _capture_decode_trace(tt_model, decode_runtime_state: dict, *, block_steps: int) -> dict:
    _reset_decode_runtime_state(decode_runtime_state)
    compile_output = tt_model.audio_ttnn_decode_block_from_raw_token_ids_inplace(
        decode_runtime_state["device_inputs"][0],
        decode_runtime_state["device_inputs"][1],
        decode_runtime_state["device_inputs"][2],
        decode_runtime_state["postprocess_state_inputs"][0],
        decode_runtime_state["postprocess_state_inputs"][1],
        block_steps=block_steps,
        tt_finished_output=decode_runtime_state["finished_output"],
    )
    ttnn.synchronize_device(tt_model.mesh_device)
    assert compile_output is decode_runtime_state["finished_output"]

    _reset_decode_runtime_state(decode_runtime_state)
    trace_id = ttnn.begin_trace_capture(tt_model.mesh_device, cq_id=0)
    trace_output = tt_model.audio_ttnn_decode_block_from_raw_token_ids_inplace(
        decode_runtime_state["device_inputs"][0],
        decode_runtime_state["device_inputs"][1],
        decode_runtime_state["device_inputs"][2],
        decode_runtime_state["postprocess_state_inputs"][0],
        decode_runtime_state["postprocess_state_inputs"][1],
        block_steps=block_steps,
        tt_finished_output=decode_runtime_state["finished_output"],
    )
    ttnn.end_trace_capture(tt_model.mesh_device, trace_id, cq_id=0)
    ttnn.synchronize_device(tt_model.mesh_device)
    return {
        "trace_id": trace_id,
        "trace_output": trace_output,
    }


def _release_prefill_trace_state(tt_model, trace_state: dict | None) -> None:
    if trace_state is None:
        return
    trace_output = trace_state.get("trace_output")
    if trace_output is not None:
        ttnn.deallocate(trace_output)
    trace_id = trace_state.get("trace_id")
    if trace_id is not None:
        ttnn.release_trace(tt_model.mesh_device, trace_id)
    for tensor in trace_state.get("device_inputs", ()):
        if tensor is not None:
            ttnn.deallocate(tensor)
    for tensor in trace_state.get("decode_bootstrap_device_inputs", ()):
        if tensor is not None:
            ttnn.deallocate(tensor)


def _release_decode_trace_state(tt_model, trace_state: dict | None) -> None:
    if trace_state is None:
        return
    trace_id = trace_state.get("trace_id")
    if trace_id is not None:
        ttnn.release_trace(tt_model.mesh_device, trace_id)


def _benchmark_case_trace(
    tt_model, model_inputs: dict, config, audio_tps: float, max_audio_steps: int, decode_runtime_state: dict
) -> dict:
    tt_model.reset_kv_cache()
    tt_model.prime_trace_runtime_assets()
    actual_steps = int(max_audio_steps)
    decode_block_steps = int(_select_decode_trace_block_steps(actual_steps))
    if actual_steps % decode_block_steps != 0:
        raise ValueError(
            f"Fixed-horizon traced decode requires horizon divisible by block size, got {actual_steps=} {decode_block_steps=}"
        )

    prefill_inputs = tt_model.prepare_prefill_inputs_trace(
        input_ids=model_inputs["input_ids"],
        audio_in_ids=model_inputs.get("audio_in_ids"),
        audio_in_ids_start=model_inputs.get("audio_in_ids_start"),
        audio_out_ids=model_inputs.get("audio_out_ids"),
        audio_out_ids_start=model_inputs.get("audio_out_ids_start"),
    )
    prefill_trace_state = _capture_prefill_trace(tt_model, prefill_inputs, decode_runtime_state)

    prefill_start = time.perf_counter()
    copy_host_to_device(
        host_tensors=prefill_inputs["host_inputs"],
        device_tensors=prefill_trace_state["device_inputs"],
    )
    copy_host_to_device(
        host_tensors=tt_model.prepare_audio_decode_bootstrap_inputs_host(prefill_inputs["effective_seq_len"]),
        device_tensors=prefill_trace_state["decode_bootstrap_device_inputs"],
    )
    ttnn.execute_trace(tt_model.mesh_device, prefill_trace_state["trace_id"], cq_id=0, blocking=False)
    ttnn.synchronize_device(tt_model.mesh_device, cq_id=0)
    prefill_seconds = time.perf_counter() - prefill_start
    _release_prefill_trace_state(tt_model, prefill_trace_state)

    decode_trace_state = _capture_decode_trace(tt_model, decode_runtime_state, block_steps=decode_block_steps)
    decode_replay_count = actual_steps // decode_block_steps
    try:
        decode_start = time.perf_counter()
        for _ in range(decode_replay_count):
            ttnn.execute_trace(tt_model.mesh_device, decode_trace_state["trace_id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(tt_model.mesh_device, cq_id=0)
        decode_seconds = time.perf_counter() - decode_start
    finally:
        _release_decode_trace_state(tt_model, decode_trace_state)

    audio_seconds = actual_steps / audio_tps if actual_steps > 0 else 0.0
    return {
        "prompt_len": int(model_inputs["input_ids"].shape[1]),
        "merged_prompt_len": int(prefill_inputs["effective_seq_len"]),
        "generated_audio_steps": actual_steps,
        "prefill_seconds": prefill_seconds,
        "decode_seconds": decode_seconds,
        "tokens_per_second": actual_steps / decode_seconds,
        "decode_rtf": decode_seconds / audio_seconds if audio_seconds > 0 else math.inf,
        "rtf": (prefill_seconds + decode_seconds) / audio_seconds if audio_seconds > 0 else math.inf,
        "audio_seconds": audio_seconds,
    }


@torch.inference_mode()
def run_performance_check(
    model_path: str = MODEL_PATH,
    audio_tokenizer_path: str = AUDIO_TOKENIZER_PATH,
    cases_json_path: str | Path = DEFAULT_PERFORMANCE_CASES_PATH,
    reference_audio_manifest_path: str | Path = DEFAULT_REFERENCE_AUDIO_MANIFEST_PATH,
    reference_audio_assets_root: str | Path | None = None,
) -> dict:
    cases_json_path = Path(cases_json_path).resolve()
    performance_cases = load_cases(cases_json_path)
    config = load_higgs_config(model_path)
    tokenizer = load_higgs_tokenizer(model_path)
    audio_tokenizer = load_audio_tokenizer(audio_tokenizer_path, device="cpu")
    reference_audio_index = load_reference_audio_manifest(
        reference_audio_manifest_path,
        assets_root=reference_audio_assets_root,
        require_local=True,
    )

    model_inputs_by_case = {}
    for case in performance_cases:
        model_inputs_by_case[case["name"]] = _prepare_case_inputs(
            case=case,
            case_manifest_path=cases_json_path,
            reference_audio_index=reference_audio_index,
            tokenizer=tokenizer,
            audio_tokenizer=audio_tokenizer,
            config=config,
        )

    previous_fallback_setting = ttnn.CONFIG.throw_exception_on_fallback
    ttnn.CONFIG.throw_exception_on_fallback = True
    mesh_device = ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, 1),
        trace_region_size=PERFORMANCE_TRACE_REGION_SIZE,
    )
    decode_runtime_state = None
    try:
        _, tt_model, _ = create_higgs_tt_model(
            mesh_device,
            model_path,
            optimizations="performance",
            use_hf_rope=False,
        )
        decode_runtime_state = _allocate_decode_runtime_state(tt_model, config)
        case_reports = []
        total_generated_audio_steps = 0
        total_prefill_seconds = 0.0
        total_decode_seconds = 0.0
        total_audio_seconds = 0.0
        for case in performance_cases:
            case_report = _benchmark_case_trace(
                tt_model=tt_model,
                model_inputs=model_inputs_by_case[case["name"]],
                config=config,
                audio_tps=float(audio_tokenizer.tps),
                max_audio_steps=int(case["max_audio_steps"]),
                decode_runtime_state=decode_runtime_state,
            )
            case_report.update(
                {
                    "name": case["name"],
                    "mode": case["mode"],
                    "language": case["language"],
                    "max_audio_steps": int(case["max_audio_steps"]),
                }
            )
            total_generated_audio_steps += case_report["generated_audio_steps"]
            total_prefill_seconds += case_report["prefill_seconds"]
            total_decode_seconds += case_report["decode_seconds"]
            total_audio_seconds += case_report["audio_seconds"]
            case_reports.append(case_report)

        return {
            "mode": "performance",
            "model_path": model_path,
            "audio_tokenizer_path": audio_tokenizer_path,
            "cases_json_path": str(cases_json_path),
            "reference_audio_manifest_path": str(Path(reference_audio_manifest_path).resolve()),
            "reference_audio_assets_root": str(reference_audio_assets_root) if reference_audio_assets_root else None,
            "tokens_per_second": total_generated_audio_steps / total_decode_seconds,
            "rtf": (total_prefill_seconds + total_decode_seconds) / total_audio_seconds,
            "decode_rtf": total_decode_seconds / total_audio_seconds,
            "generated_audio_steps": total_generated_audio_steps,
            "prefill_seconds": total_prefill_seconds,
            "decode_seconds": total_decode_seconds,
            "audio_seconds": total_audio_seconds,
            "case_reports": case_reports,
        }
    finally:
        _release_decode_runtime_state(decode_runtime_state)
        ttnn.close_mesh_device(mesh_device)
        ttnn.CONFIG.throw_exception_on_fallback = previous_fallback_setting


def test_performance():
    report = run_performance_check()

    assert report["mode"] == "performance"
    assert report["tokens_per_second"] >= PERFORMANCE_MIN_TOKENS_PER_SECOND
    assert report["rtf"] < PERFORMANCE_MAX_RTF
