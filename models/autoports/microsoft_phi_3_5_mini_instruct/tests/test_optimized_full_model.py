# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

import os
import inspect
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch
import ttnn

from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator import Phi35MiniGenerator
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.generator import build_generator
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.model import Phi35MiniForCausalLMTT
from models.autoports.microsoft_phi_3_5_mini_instruct.tt.model import full_model_strategy_summary
from models.common.readiness_check.mesh_device import close_readiness_mesh_device, open_readiness_mesh_device
from models.common.sampling import SamplingGenerator
from models.common.sampling.tt_sampling import TTSampling

try:
    from tracy import signpost
except ImportError:

    def signpost(*_args, **_kwargs):
        return None


MODEL_DIR = Path(__file__).resolve().parents[1]
REFERENCE_PATH = MODEL_DIR / "readiness_aime24_chat_100.refpt"


@contextmanager
def _open_t3k_ring_mesh():
    mesh_device = open_readiness_mesh_device("T3K", fabric_config="FABRIC_1D_RING")
    try:
        yield mesh_device
    finally:
        close_readiness_mesh_device(mesh_device, fabric_config="FABRIC_1D_RING")


def _reference_prompt_tokens(limit: int = 128) -> list[int]:
    ref = torch.load(REFERENCE_PATH, map_location="cpu", weights_only=False)
    return ref["entries"][0]["prompt_tokens"][0].tolist()[:limit]


def test_full_model_strategy_static():
    summary = full_model_strategy_summary()
    assert summary["decoder"] == "MultichipDecoder"
    assert summary["residual_layout"] == "replicated BF16 between decoder layers"
    assert "multicore split sampling" in summary["lm_head"]
    assert summary["force_argmax_default"] is False


def test_token_out_no_readback_runtime_fallback_audit_static():
    runtime_callables = [
        Phi35MiniForCausalLMTT.prefill_forward_ttnn,
        Phi35MiniForCausalLMTT.decode_forward_from_ttnn_inputs,
        Phi35MiniForCausalLMTT.embed_tokens,
        Phi35MiniForCausalLMTT._lm_head,
        Phi35MiniGenerator._decode_next_token_traced,
        SamplingGenerator.sample,
        SamplingGenerator._execute_trace,
        SamplingGenerator._run_sampling,
        TTSampling.forward,
    ]
    forbidden = ("ttnn.from_torch", "ttnn.to_torch", "from_torch(", "to_torch(", "from_device(", ".cpu(")
    for callable_obj in runtime_callables:
        source = inspect.getsource(callable_obj)
        hits = [token for token in forbidden if token in source]
        assert not hits, f"{callable_obj.__name__} contains forbidden runtime fallback tokens: {hits}"


@pytest.mark.skipif(
    os.getenv("PHI35_RUN_OPTIMIZED_FULL_MODEL_SMOKE") != "1",
    reason="set PHI35_RUN_OPTIMIZED_FULL_MODEL_SMOKE=1 for hardware smoke",
)
@pytest.mark.timeout(700)
def test_token_out_no_readback_smoke_1x8_ring():
    prompt = _reference_prompt_tokens(limit=128)
    with _open_t3k_ring_mesh() as mesh_device:
        generator = build_generator(model_dir=MODEL_DIR, mesh_device=mesh_device, num_layers=1)
        try:
            result = generator.benchmark_token_out_decode(prompt, max_new_tokens=16, warmup_decode_steps=2)
        finally:
            generator.teardown()

    counters = result["counters"]
    assert result["readback"] is False
    assert counters["model_trace_captures"] == 1
    assert counters["sampling_trace_captures"] == 1
    assert counters["sampled_token_readbacks"] == 0
    assert counters["full_logits_decode_readbacks"] == 0
    assert counters["position_steady_state_refreshes"] == 0
    assert counters["page_table_changed_refreshes"] == 0
    assert counters["no_readback_decode_steps"] == counters["device_token_feedbacks"]


@pytest.mark.skipif(
    os.getenv("PHI35_RUN_OPTIMIZED_FULL_MODEL_PERF") != "1",
    reason="set PHI35_RUN_OPTIMIZED_FULL_MODEL_PERF=1 for reduced full-model profiling",
)
@pytest.mark.timeout(700)
def test_reduced_full_model_token_out_profile_1x8_ring():
    prompt = _reference_prompt_tokens(limit=128)
    max_new_tokens = int(os.getenv("PHI35_PROFILE_MAX_NEW_TOKENS", "16"))
    warmup_steps = int(os.getenv("PHI35_PROFILE_WARMUP_DECODE_STEPS", "8"))
    measured_steps = int(os.getenv("PHI35_PROFILE_MEASURED_DECODE_STEPS", "1"))

    with _open_t3k_ring_mesh() as mesh_device:
        generator = build_generator(model_dir=MODEL_DIR, mesh_device=mesh_device, num_layers=1)
        try:
            prompt_tensor = torch.tensor([prompt], dtype=torch.long)
            signpost("PERF_FULL_PREFILL")
            prefill_logits = generator.prefill_forward(
                prompt_tensor,
                page_table=generator.page_table,
                kv_cache=generator.kv_cache,
                prompt_lens=[len(prompt)],
                return_all_logits=False,
            )
            signpost("PERF_FULL_PREFILL_END")
            feed_token = int(torch.argmax(prefill_logits[0, -1]).item())
            generator._ensure_decode_trace(
                token_id=feed_token,
                start_pos=len(prompt),
                page_table=generator.page_table,
                kv_cache=generator.kv_cache,
                rope_sequence_length=max(generator.model.full_config.max_seq_len, len(prompt) + max_new_tokens),
            )
            for _ in range(warmup_steps):
                generator._decode_next_token_traced(readback=False)
            ttnn.synchronize_device(mesh_device)

            signpost("PERF_FULL_TOKEN_OUT")
            for _ in range(measured_steps):
                generator._decode_next_token_traced(readback=False)
            ttnn.synchronize_device(mesh_device)
            signpost("PERF_FULL_TOKEN_OUT_END")
        finally:
            generator.teardown()

    counters = generator.trace_counters()
    assert counters["sampled_token_readbacks"] == 0
    assert counters["full_logits_decode_readbacks"] == 0
    assert counters["position_steady_state_refreshes"] == 0
