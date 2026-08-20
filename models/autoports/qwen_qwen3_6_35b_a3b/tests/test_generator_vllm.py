# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenReadinessGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm import Qwen3_5MoeForConditionalGeneration
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenFullModel

MODEL_DIR = Path(__file__).resolve().parents[1]


def test_vllm_adapter_capabilities_and_context_contract():
    caps = Qwen3_5MoeForConditionalGeneration.model_capabilities
    assert caps["supports_async_decode"] is True
    assert caps["supports_sample_on_device"] is True
    assert caps["supports_slot_independent_device_seeds"] is False
    assert caps["supports_on_device_penalties"] is False
    assert caps["supports_mixed_greedy_random_device_sampling"] is False
    assert caps["supports_prefix_caching"] is False

    contract = json.loads((MODEL_DIR / "doc" / "context_contract.json").read_text())
    assert (
        Qwen3_5MoeForConditionalGeneration.get_max_tokens_all_users(
            model_name="Qwen/Qwen3.6-35B-A3B",
            num_devices=4,
            tt_data_parallel=1,
            max_model_len=contract["supported_context"],
            max_num_seqs=32,
        )
        == contract["supported_context"]
    )


def test_vllm_adapter_uses_token_output_sampling_contract():
    source = inspect.getsource(Qwen3_5MoeForConditionalGeneration)
    forbidden = (
        "torch.argmax",
        ".argmax(",
        "decode_logits_to_torch",
        "logits_to_torch",
        "full_logits",
    )
    for token in forbidden:
        assert token not in source

    assert "vllm_prefill_sample_on_device" in source
    assert "execute_trace" in source
    assert "_release_decode_trace" in source
    assert "_capture_decode_trace_for_current_step" in source
    assert "read_decode_output" in source
    assert "process_decode_output_host" in source


def test_vllm_adapter_resets_serving_slot_state_and_defers_trace_capture():
    assert "reset_linear_attention_state" in inspect.getsource(QwenFullModel.reset_linear_attention_state)
    assert "reset_linear_attention_state" in inspect.getsource(QwenReadinessGenerator.vllm_prefill_sample_on_device)
    prefill_source = inspect.getsource(QwenFullModel.prefill_user)
    assert "chunk_start_idx=start if start > 0 else None" in prefill_source

    warmup_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration.warmup_model_decode)
    assert "_decode_sample_body" not in warmup_source
    assert "begin_trace_capture" not in warmup_source
    assert "_release_decode_trace" in warmup_source

    decode_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration.decode_forward)
    replay_branch = decode_source.split(
        "elif enable_trace and self._trace is not None and self._trace.cache is cache:", 1
    )[1]
    replay_branch = replay_branch.split("elif enable_trace:", 1)[0]
    assert "_reset_decode_trace_inputs(tokens=tokens, start_pos=start_pos, max_batch=max_batch)" in replay_branch
    assert replay_branch.index("_reset_decode_trace_inputs") < replay_branch.index("execute_trace")

    reset_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration._reset_decode_trace_inputs)
    assert "flat_tokens = tokens.reshape(-1).to(torch.int32)" in reset_source
    assert "flat_pos = torch.as_tensor(start_pos, dtype=torch.int32).reshape(-1)" in reset_source
    assert "self._trace.token_input" in reset_source
    assert "self._trace.current_pos" in reset_source

    capture_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration._capture_decode_trace_for_current_step)
    assert capture_source.index("end_trace_capture") < capture_source.index("self._trace =")
    assert capture_source.index("self._trace =") < capture_source.index("_reset_decode_trace_inputs")
    assert capture_source.index("_reset_decode_trace_inputs") < capture_source.index("execute_trace")
    assert capture_source.index("execute_trace") < capture_source.index("return token_output")


def test_vllm_adapter_normalizes_one_based_vllm_page_tables():
    cache = SimpleNamespace(page_table_host=torch.zeros((4, 5), dtype=torch.int32))
    compact_page_table = torch.tensor(
        [
            [1, 0, 7],
            [2, 3, 0],
        ],
        dtype=torch.int32,
    )

    normalized = Qwen3_5MoeForConditionalGeneration._normalize_page_table(
        object(),
        compact_page_table,
        cache,
    )

    assert normalized.tolist() == [
        [0, 0, 6, 0, 0],
        [1, 2, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ]


def test_vllm_adapter_scatters_prefill_page_tables_to_serving_slots():
    cache = SimpleNamespace(
        page_table_host=torch.tensor(
            [
                [90, 91, 92, 93],
                [80, 81, 82, 83],
                [70, 71, 72, 73],
                [60, 61, 62, 63],
            ],
            dtype=torch.int32,
        )
    )
    compact_page_table = torch.tensor(
        [
            [1, 2],
            [3, 4],
        ],
        dtype=torch.int32,
    )

    scattered = Qwen3_5MoeForConditionalGeneration._scatter_page_table_to_slots(
        object(),
        compact_page_table,
        cache,
        [2, 0],
    )

    assert scattered.tolist() == [
        [3, 4, 0, 0],
        [80, 81, 82, 83],
        [1, 2, 0, 0],
        [60, 61, 62, 63],
    ]
