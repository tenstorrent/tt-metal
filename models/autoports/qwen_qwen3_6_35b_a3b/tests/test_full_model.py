# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import inspect
import os

import pytest
import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_functional_decoder import (
    _synthetic_layer_state,
    _target_text_config,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tests.test_multichip_decoder import _run_with_target_mesh
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenReadinessGenerator, build_generator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenFullModel, iter_runtime_fallback_audit


def _synthetic_full_model(mesh_device, *, max_batch_size: int = 1, layer_types: list[str] | None = None):
    torch.manual_seed(0)
    cfg = copy.deepcopy(_target_text_config())
    cfg.vocab_size = 128
    cfg.max_position_embeddings = 64
    cfg.layer_types = layer_types or ["linear_attention", "full_attention"]
    cfg.num_hidden_layers = len(cfg.layer_types)

    state = {}
    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"model.language_model.layers.{layer_idx}."
        state.update({prefix + key: value for key, value in _synthetic_layer_state(cfg, layer_idx).items()})
    state["model.language_model.embed_tokens.weight"] = torch.randn(
        (cfg.vocab_size, cfg.hidden_size), dtype=torch.bfloat16
    )
    state["model.language_model.norm.weight"] = torch.zeros((cfg.hidden_size,), dtype=torch.bfloat16)
    state["lm_head.weight"] = torch.randn((cfg.vocab_size, cfg.hidden_size), dtype=torch.bfloat16)

    model = QwenFullModel.from_state_dict(
        mesh_device=mesh_device,
        hf_config=cfg,
        state_dict=state,
        max_batch_size=max_batch_size,
        max_seq_len=64,
        load_rope_tables=True,
    )
    return model, cfg


def test_readiness_generator_contract_exports():
    assert callable(build_generator)
    signature = inspect.signature(QwenReadinessGenerator.generate)
    assert "enable_trace" in signature.parameters
    assert signature.parameters["enable_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert hasattr(QwenReadinessGenerator, "teardown")


def test_runtime_fallback_audit_declares_full_multichip_path():
    audit = tuple(iter_runtime_fallback_audit())
    assert "decoder_stack=MultichipDecoder" in audit
    assert "lm_head=vocab_sharded_flat_4way" in audit
    assert "sampling=common_sampling_generator_flat_4way_topk1_composite_gather" in audit
    assert "no_single_chip_or_host_decoder_fallback" in audit


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FULL_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN36_FULL_MODEL_SMOKE=1 to run the 2x2 synthetic full-model smoke",
)
def test_full_model_non_aligned_prompt_smoke():
    def run(mesh_device):
        model, cfg = _synthetic_full_model(mesh_device)
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        cache = model.allocate_cache(max_batch_size=1, max_seq_len=64)
        prefill = model.prefill_forward(
            tokens,
            page_table=cache.page_table,
            kv_cache=cache,
            prompt_lens=[5],
            return_all_logits=True,
        )
        assert prefill.shape == (1, 5, cfg.vocab_size)
        decode = model.decode_forward(
            torch.tensor([[6]], dtype=torch.long),
            torch.tensor([5], dtype=torch.int32),
            page_table=cache.page_table,
            kv_cache=cache,
        )
        assert decode.shape == (1, cfg.vocab_size)

    _run_with_target_mesh(run, trace_region_size=64_000_000)


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FULL_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN36_FULL_MODEL_SMOKE=1 to run the 2x2 synthetic full-model smoke",
)
def test_full_model_mixed_prompt_inactive_row_smoke():
    def run(mesh_device):
        model, cfg = _synthetic_full_model(mesh_device, max_batch_size=2)
        page_table = torch.tensor([[1, 0], [3, 2]], dtype=torch.int32)
        cache = model.allocate_cache(max_batch_size=2, max_seq_len=64, page_table=page_table)
        tokens = torch.tensor([[1, 2, 3, 4, 5], [9, 8, 7, 6, 5]], dtype=torch.long)
        prefill = model.prefill_forward(
            tokens,
            page_table=cache.page_table,
            kv_cache=cache,
            prompt_lens=[5, 0],
            return_all_logits=True,
        )
        assert prefill.shape == (2, 5, cfg.vocab_size)
        assert torch.count_nonzero(prefill[1]).item() == 0

    _run_with_target_mesh(run, trace_region_size=64_000_000)


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FULL_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN36_FULL_MODEL_SMOKE=1 to run the 2x2 synthetic full-model smoke",
)
def test_full_model_decode_page_table_override_smoke():
    def run(mesh_device):
        model, cfg = _synthetic_full_model(mesh_device, layer_types=["full_attention"])
        fill_page_table = torch.tensor([[1, 0]], dtype=torch.int32)
        changed_page_table = torch.tensor([[0, 1]], dtype=torch.int32)
        tokens = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        start_pos = torch.tensor([5], dtype=torch.int32)

        same_cache = model.allocate_cache(max_batch_size=1, max_seq_len=64, page_table=fill_page_table)
        changed_cache = model.allocate_cache(max_batch_size=1, max_seq_len=64, page_table=fill_page_table)
        model.prefill_forward(tokens, page_table=fill_page_table, kv_cache=same_cache, prompt_lens=[5])
        model.prefill_forward(tokens, page_table=fill_page_table, kv_cache=changed_cache, prompt_lens=[5])

        same_logits = model.decode_forward(
            torch.tensor([[6]], dtype=torch.long),
            start_pos,
            page_table=fill_page_table,
            kv_cache=same_cache,
        )
        changed_logits = model.decode_forward(
            torch.tensor([[6]], dtype=torch.long),
            start_pos,
            page_table=changed_page_table,
            kv_cache=changed_cache,
        )
        assert same_logits.shape == (1, cfg.vocab_size)
        assert changed_logits.shape == (1, cfg.vocab_size)
        assert not torch.allclose(same_logits, changed_logits, atol=1e-4, rtol=1e-4)

        trace_cache = model.allocate_cache(max_batch_size=1, max_seq_len=64, page_table=fill_page_table)
        model.prefill_forward(tokens, page_table=fill_page_table, kv_cache=trace_cache, prompt_lens=[5])
        changed_page_table_cache = model.allocate_cache(
            max_batch_size=1,
            max_seq_len=64,
            page_table=changed_page_table,
        )
        changed_page_table_tt = changed_page_table_cache.page_table
        tt_tokens = model._tokens_to_tt_decode(torch.tensor([[6]], dtype=torch.long))
        tt_pos = model._positions_to_tt(start_pos)
        trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        try:
            model.decode_forward(
                tt_tokens,
                tt_pos,
                page_table=changed_page_table_tt,
                kv_cache=trace_cache,
                return_tt_logits=True,
            )
            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=0)
            ttnn.synchronize_device(mesh_device)
            ttnn.execute_trace(mesh_device, trace_id, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh_device)
        finally:
            ttnn.release_trace(mesh_device, trace_id)
            ttnn.synchronize_device(mesh_device)
            del changed_page_table_cache

    _run_with_target_mesh(run, trace_region_size=64_000_000)


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FULL_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN36_FULL_MODEL_SMOKE=1 to run the 2x2 synthetic full-model smoke",
)
def test_full_model_traced_token_out_smoke():
    def run(mesh_device):
        model, _ = _synthetic_full_model(mesh_device)
        generator = QwenReadinessGenerator(
            model_dir="models/autoports/qwen_qwen3_6_35b_a3b",
            mesh_device=mesh_device,
            model=model,
            max_batch_size=1,
            max_seq_len=64,
        )
        traced_tokens = generator.generate([1, 2, 3, 4, 5], 4, enable_trace=True)
        assert len(traced_tokens) == 4
        assert generator._trace is not None
        assert generator.last_timings["decode_tokens"] == 3
        generator.host_sampling_compat = True
        host_tokens = generator.generate([1, 2, 3, 4, 5], 4, enable_trace=True)
        assert traced_tokens == host_tokens

        forced_inputs = [6, 7, 8, 9]
        traced_seen = []

        def traced_next_input(step: int, predicted: int) -> int:
            traced_seen.append((step, predicted))
            return forced_inputs[step]

        generator.host_sampling_compat = False
        traced_teacher = generator.generate([1, 2, 3, 4, 5], 4, next_input=traced_next_input, enable_trace=True)
        assert len(traced_teacher) == 4
        assert len(traced_seen) == 4
        assert generator._trace is not None
        assert generator.last_timings["decode_tokens"] == 3

        host_seen = []

        def host_next_input(step: int, predicted: int) -> int:
            host_seen.append((step, predicted))
            return forced_inputs[step]

        generator.host_sampling_compat = True
        host_teacher = generator.generate([1, 2, 3, 4, 5], 4, next_input=host_next_input, enable_trace=True)
        assert traced_teacher == host_teacher
        assert traced_seen == host_seen
        generator.teardown()
        assert generator._trace is None

    _run_with_target_mesh(run, trace_region_size=64_000_000)


@pytest.mark.skipif(
    os.environ.get("RUN_QWEN36_FULL_MODEL_SMOKE") != "1",
    reason="set RUN_QWEN36_FULL_MODEL_SMOKE=1 to run the 2x2 synthetic full-model smoke",
)
def test_full_model_token_out_no_readback_measurement_smoke():
    def run(mesh_device):
        baseline_model, _ = _synthetic_full_model(mesh_device)
        baseline_generator = QwenReadinessGenerator(
            model_dir="models/autoports/qwen_qwen3_6_35b_a3b",
            mesh_device=mesh_device,
            model=baseline_model,
            max_batch_size=1,
            max_seq_len=64,
        )
        readback_tokens = baseline_generator.generate([1, 2, 3, 4, 5], 4, enable_trace=True)
        baseline_generator.teardown()

        model, _ = _synthetic_full_model(mesh_device)
        generator = QwenReadinessGenerator(
            model_dir="models/autoports/qwen_qwen3_6_35b_a3b",
            mesh_device=mesh_device,
            model=model,
            max_batch_size=1,
            max_seq_len=64,
        )
        metrics = generator.measure_token_out_no_readback([1, 2, 3, 4, 5], 4)
        counters = metrics["host_boundary_counters"]
        assert metrics["trace_present"] is True
        assert metrics["trace_generated_steps"] == 3
        assert metrics["position_end_expected_exclusive"] == 9
        assert metrics["final_token"] is not None
        assert counters["trace_replays"] == 3
        assert counters["trace_decode_steps"] == 3
        assert counters["execute_trace_blocking"] is False
        assert counters["steady_state_token_refreshes"] == 0
        assert counters["steady_state_position_refreshes"] == 0
        assert counters["steady_state_page_table_refreshes"] == 0
        assert counters["steady_state_synchronizations"] == 0
        assert counters["steady_state_token_readbacks"] == 0
        assert counters["terminal_validation_synchronizations"] == 1
        assert counters["terminal_validation_token_readbacks"] == 1

        assert metrics["final_token"] == readback_tokens[-1], {
            "metrics": metrics,
            "readback_tokens": readback_tokens,
        }
        generator.teardown()

    _run_with_target_mesh(run, trace_region_size=64_000_000)
