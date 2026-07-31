# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import torch
import ttnn

from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.generator import Llama32Generator
from models.autoports.meta_llama_llama_3_2_1b_instruct.tt.model import MODEL_ID


MODEL_DIR = Path("models/autoports/meta_llama_llama_3_2_1b_instruct")
ARTIFACT_DIR = Path(
    os.getenv(
        "MD_OPT_FULL_MODEL_ARTIFACT_DIR",
        MODEL_DIR / "doc" / "optimized_full_model",
    )
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _prompt_tokens(prompt_len: int) -> list[int]:
    if prompt_len < 1:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    gen = torch.Generator().manual_seed(20260615)
    body = torch.randint(100, 32000, (max(prompt_len - 1, 0),), generator=gen, dtype=torch.long).tolist()
    return [128000] + [int(token) for token in body]


@contextmanager
def _open_t3k_mesh():
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 8))
    try:
        yield mesh_device
    finally:
        ttnn.close_mesh_device(mesh_device)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _build_generator(mesh_device, *, num_layers: int | None = None) -> Llama32Generator:
    return Llama32Generator(
        model_dir=MODEL_DIR,
        mesh_device=mesh_device,
        num_layers=num_layers,
        cache_path=MODEL_DIR / ".ttnn_cache",
    )


@pytest.mark.perf_artifact
def test_optimized_full_model_token_out_no_readback_benchmark():
    prompt_len = int(os.getenv("MD_OPT_FULL_MODEL_PROMPT_LEN", "128"))
    decode_steps = int(os.getenv("MD_OPT_FULL_MODEL_DECODE_STEPS", "128"))

    with _open_t3k_mesh() as mesh_device:
        generator = _build_generator(mesh_device)
        try:
            result = generator.benchmark_token_out_no_readback(
                _prompt_tokens(prompt_len),
                decode_steps=decode_steps,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                warmup_steps=1,
            )
        finally:
            generator.teardown()

    measured = result["counter_deltas"]["measured_decode"]
    assert measured["token_refreshes"] == 0
    assert measured["current_position_refreshes"] == 0
    assert measured["rope_index_refreshes"] == 0
    assert measured["page_table_refreshes"] == 0
    assert measured["readbacks"] == 0
    assert measured["model_trace_replays"] == decode_steps
    assert measured["device_token_feedback_steps"] == decode_steps
    assert result["sampling"]["force_argmax_enabled"] is False

    _write_json(ARTIFACT_DIR / "token_out_no_readback_perf.json", result)


@pytest.mark.perf_artifact
def test_optimized_full_model_reduced_perf_artifact_signposts():
    from tracy import signpost

    prompt_len = int(os.getenv("MD_OPT_FULL_MODEL_PROMPT_LEN", "128"))
    decode_steps = int(os.getenv("MD_OPT_FULL_MODEL_REDUCED_DECODE_STEPS", "8"))
    num_layers = int(os.getenv("MD_OPT_FULL_MODEL_REDUCED_LAYERS", "1"))
    prompt = torch.tensor([_prompt_tokens(prompt_len)], dtype=torch.long)

    with _open_t3k_mesh() as mesh_device:
        generator = _build_generator(mesh_device, num_layers=num_layers)
        page_table = generator.model.make_page_table(
            batch_size=generator.max_batch_size,
            max_seq_len=generator.max_seq_len,
        )
        try:
            generator.prefill_forward(
                prompt,
                page_table=page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            ttnn.synchronize_device(mesh_device)

            prefill_start = time.perf_counter()
            signpost("PERF_FULL_MODEL_PREFILL")
            generator.prefill_forward(
                prompt,
                page_table=page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_FULL_MODEL_PREFILL_END")
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

            token_out = generator.benchmark_token_out_no_readback(
                _prompt_tokens(prompt_len),
                decode_steps=decode_steps,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                warmup_steps=1,
                signpost_labels=("PERF_FULL_MODEL_TOKEN_OUT_DECODE", "PERF_FULL_MODEL_TOKEN_OUT_DECODE_END"),
            )
        finally:
            generator.teardown()

    _write_json(
        ARTIFACT_DIR / "perf_trace_contract.json",
        {
            "model": MODEL_ID,
            "scope": "reduced full-model profiling variant with real weights and terminal token-out path",
            "mesh_shape": [1, 8],
            "fabric_config": "FABRIC_1D_RING",
            "num_layers": num_layers,
            "prompt_len": prompt_len,
            "decode_steps": decode_steps,
            "prefill_signposts": ["PERF_FULL_MODEL_PREFILL", "PERF_FULL_MODEL_PREFILL_END"],
            "decode_signposts": ["PERF_FULL_MODEL_TOKEN_OUT_DECODE", "PERF_FULL_MODEL_TOKEN_OUT_DECODE_END"],
            "host_wall_ms": {
                "reduced_prefill": prefill_ms,
                "reduced_token_out_decode_total": token_out["decode_elapsed_s"] * 1000.0,
                "reduced_token_out_decode_per_token": token_out["decode_latency_ms_mean"],
            },
            "token_out_no_readback": token_out,
            "status": "passed",
        },
    )


@pytest.mark.perf_artifact
def test_optimized_full_model_reduced_eager_profile_signposts():
    from tracy import signpost

    prompt_len = int(os.getenv("MD_OPT_FULL_MODEL_PROMPT_LEN", "128"))
    num_layers = int(os.getenv("MD_OPT_FULL_MODEL_REDUCED_LAYERS", "1"))
    prompt = torch.tensor([_prompt_tokens(prompt_len)], dtype=torch.long)

    with _open_t3k_mesh() as mesh_device:
        generator = _build_generator(mesh_device, num_layers=num_layers)
        page_table = generator.model.make_page_table(
            batch_size=generator.max_batch_size,
            max_seq_len=generator.max_seq_len,
        )
        page_table_tt = generator.model.page_table_to_device(page_table)
        tokens_tt = generator.model.prepare_prefill_tokens_device(prompt)
        generator._apply_sampling_params(
            generator._make_sampling_params(top_k=1, top_p=0.0, temperature=1.0)
        )
        try:
            # Warm kernels outside signposts so the profiler tables reflect steady device work.
            generator.model.prefill_forward_device(tokens_tt, page_table=page_table_tt, start_pos=0, user_id=0)
            host_inputs = generator.model.prepare_decode_inputs_host(
                torch.zeros(1, dtype=torch.long),
                torch.tensor([prompt_len], dtype=torch.long),
                page_table,
            )
            device_inputs = generator.model.copy_decode_inputs_to_device(host_inputs)
            logits = generator.model.decode_forward_device(*device_inputs)
            generator._sample_logits(logits, tt_out_tok=device_inputs[0], enable_trace=False)
            ttnn.synchronize_device(mesh_device)

            prefill_start = time.perf_counter()
            signpost("PERF_FULL_MODEL_PREFILL")
            generator.model.prefill_forward_device(tokens_tt, page_table=page_table_tt, start_pos=0, user_id=0)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_FULL_MODEL_PREFILL_END")
            prefill_ms = (time.perf_counter() - prefill_start) * 1000.0

            host_inputs = generator.model.prepare_decode_inputs_host(
                torch.zeros(1, dtype=torch.long),
                torch.tensor([prompt_len], dtype=torch.long),
                page_table,
            )
            device_inputs = generator.model.copy_decode_inputs_to_device(host_inputs)
            decode_start = time.perf_counter()
            signpost("PERF_FULL_MODEL_EAGER_DECODE")
            logits = generator.model.decode_forward_device(*device_inputs)
            generator._sample_logits(logits, tt_out_tok=device_inputs[0], enable_trace=False)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_FULL_MODEL_EAGER_DECODE_END")
            decode_ms = (time.perf_counter() - decode_start) * 1000.0
        finally:
            generator.teardown()

    _write_json(
        ARTIFACT_DIR / "perf_trace_contract_eager.json",
        {
            "model": MODEL_ID,
            "scope": "reduced eager full-model profiling variant with real weights and split greedy terminal path",
            "mesh_shape": [1, 8],
            "fabric_config": "FABRIC_1D_RING",
            "num_layers": num_layers,
            "prompt_len": prompt_len,
            "prefill_signposts": ["PERF_FULL_MODEL_PREFILL", "PERF_FULL_MODEL_PREFILL_END"],
            "decode_signposts": ["PERF_FULL_MODEL_EAGER_DECODE", "PERF_FULL_MODEL_EAGER_DECODE_END"],
            "host_wall_ms": {
                "reduced_prefill": prefill_ms,
                "reduced_eager_decode": decode_ms,
            },
            "sampling": {
                "greedy": True,
                "force_argmax_enabled": generator.sampling.tt_sampling.force_argmax_sampling,
                "pad_logits_to_power_of_2": generator.sampling.tt_sampling.pad_to_power_of_2,
                "local_logits_width": int(generator.model.per_device_vocab_size),
            },
            "status": "passed",
        },
    )


def test_optimized_full_model_runtime_fallback_audit_no_readback_loop():
    prompt_len = int(os.getenv("MD_OPT_FULL_MODEL_AUDIT_PROMPT_LEN", "128"))
    guarded_steps = int(os.getenv("MD_OPT_FULL_MODEL_AUDIT_STEPS", "2"))
    prompt_ids = _prompt_tokens(prompt_len)

    with _open_t3k_mesh() as mesh_device:
        generator = _build_generator(mesh_device)
        page_table = generator.model.make_page_table(
            batch_size=generator.max_batch_size,
            max_seq_len=generator.max_seq_len,
        )
        try:
            prefill_logits = generator.prefill_forward(
                torch.tensor([prompt_ids], dtype=torch.long),
                page_table=page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[prompt_len],
                return_all_logits=False,
            )
            first_token = int(torch.argmax(prefill_logits[0, -1]).item())
            generator._decode_token_out_device(
                torch.tensor([first_token], dtype=torch.long),
                torch.tensor([prompt_len], dtype=torch.long),
                page_table,
                enable_trace=True,
                reset_inputs=True,
                page_table_changed=True,
            )
            before = dict(generator.trace_audit()["counters"])

            original_from_torch = ttnn.from_torch
            original_to_torch = ttnn.to_torch
            original_copy = ttnn.copy_host_to_device_tensor
            original_sync = ttnn.synchronize_device

            def fail_bridge(*args, **kwargs):
                raise AssertionError("host/device bridge called inside guarded no-readback decode loop")

            ttnn.from_torch = fail_bridge
            ttnn.to_torch = fail_bridge
            ttnn.copy_host_to_device_tensor = fail_bridge
            ttnn.synchronize_device = fail_bridge
            try:
                dummy_token = torch.zeros(1, dtype=torch.long)
                dummy_pos = torch.tensor([prompt_len + 1], dtype=torch.long)
                for _ in range(guarded_steps):
                    generator._decode_token_out_device(
                        dummy_token,
                        dummy_pos,
                        page_table,
                        enable_trace=True,
                        reset_inputs=False,
                        page_table_changed=False,
                    )
            finally:
                ttnn.from_torch = original_from_torch
                ttnn.to_torch = original_to_torch
                ttnn.copy_host_to_device_tensor = original_copy
                ttnn.synchronize_device = original_sync

            ttnn.synchronize_device(mesh_device)
            after = dict(generator.trace_audit()["counters"])
        finally:
            generator.teardown()

    deltas = {key: int(after[key]) - int(before[key]) for key in after}
    assert deltas["token_refreshes"] == 0
    assert deltas["current_position_refreshes"] == 0
    assert deltas["rope_index_refreshes"] == 0
    assert deltas["page_table_refreshes"] == 0
    assert deltas["readbacks"] == 0
    assert deltas["model_trace_replays"] == guarded_steps
    assert deltas["device_token_feedback_steps"] == guarded_steps

    _write_json(
        ARTIFACT_DIR / "runtime_fallback_audit.json",
        {
            "model": MODEL_ID,
            "mesh_shape": [1, 8],
            "fabric_config": "FABRIC_1D_RING",
            "guarded_python_bridges": [
                "ttnn.from_torch",
                "ttnn.to_torch",
                "ttnn.copy_host_to_device_tensor",
                "ttnn.synchronize_device",
            ],
            "guarded_steps": guarded_steps,
            "counter_deltas": deltas,
            "status": "passed",
        },
    )
