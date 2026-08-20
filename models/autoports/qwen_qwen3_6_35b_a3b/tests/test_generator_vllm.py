# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tt import generator_vllm as generator_vllm_module
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator import QwenReadinessGenerator
from models.autoports.qwen_qwen3_6_35b_a3b.tt.generator_vllm import Qwen3_5MoeForConditionalGeneration
from models.autoports.qwen_qwen3_6_35b_a3b.tt.model import QwenFullModel

MODEL_DIR = Path(__file__).resolve().parents[1]


def test_vllm_adapter_capabilities_and_context_contract():
    caps = Qwen3_5MoeForConditionalGeneration.model_capabilities
    assert caps["supports_async_decode"] is True
    assert caps["supports_async_decode_overlap"] is True
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
    adapter_prefill_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration.prefill_forward)
    assert "_prefill_should_commit_active_decode_cache(empty_slots)" in adapter_prefill_source
    assert "commit_active_cache=commit_active_cache" in adapter_prefill_source
    assert "cleanup_active_cache_refs=force_full_decode_width or len(empty_slots) > 1" in adapter_prefill_source

    warmup_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration.warmup_model_decode)
    assert "_decode_sample_body" not in warmup_source
    assert "begin_trace_capture" not in warmup_source
    assert "_release_decode_trace(commit_active_cache=False)" in warmup_source

    decode_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration.decode_forward)
    assert (
        "formatted_params = format_sampling_params(sampling_params, max_batch)"
        in decode_source.split("if reset_inputs:", 1)[1].split("self.model.sampling.seed_manager.get_new_values", 1)[0]
    )
    replay_branch = decode_source.split(
        "elif enable_trace and self._trace is not None and self._trace.cache is cache:", 1
    )[1]
    replay_branch = replay_branch.split("elif enable_trace:", 1)[0]
    assert "if reset_inputs:" in replay_branch
    assert "_reset_decode_trace_inputs(tokens=tokens, start_pos=start_pos, max_batch=trace_width)" in replay_branch
    assert "steady_device_feedback_replays" in replay_branch
    assert replay_branch.index("_reset_decode_trace_inputs") < replay_branch.index("execute_trace")
    assert "blocking=False" in replay_branch

    assert "trace_width = self._trace_decode_width(active_slots, max_batch=max_batch)" in decode_source
    assert "force_full_decode_width" in decode_source
    assert "decode_trace_width_forced_full" in decode_source
    assert "_inactive_page_table_rows_present" in decode_source
    assert "decode_trace_width_full_for_inactive_pages" in decode_source
    assert "decode_trace_width_changes" in decode_source
    assert (
        "_release_decode_trace("
        in decode_source.split("decode_trace_width_changes", 1)[1].split(
            "reset_inputs =",
            1,
        )[0]
    )
    assert "cleanup_active_cache_refs=force_full_decode_width or trace_width == max_batch" in decode_source
    assert "_decode_cache_for_width(cache, trace_width)" in inspect.getsource(
        Qwen3_5MoeForConditionalGeneration._capture_decode_trace_for_current_step
    )
    release_source = inspect.getsource(Qwen3_5MoeForConditionalGeneration._release_decode_trace)
    assert "_commit_active_decode_cache_if_needed" in release_source
    assert release_source.index("super()._release_decode_trace()") < release_source.index(
        "_commit_active_decode_cache_if_needed"
    )
    assert "_discard_active_decode_cache" in release_source
    assert "active_decode_cache_cleanup_syncs" in release_source
    assert "gc.collect()" in release_source
    assert "ttnn.synchronize_device(self.mesh_device)" in release_source

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


def test_vllm_adapter_prefill_commits_only_live_active_decode_cache():
    adapter = object.__new__(Qwen3_5MoeForConditionalGeneration)

    adapter._active_decode_cache = object()
    adapter._active_decode_width = 1
    assert adapter._prefill_should_commit_active_decode_cache([1]) is True
    assert adapter._prefill_should_commit_active_decode_cache([0]) is False

    adapter._active_decode_width = 2
    assert adapter._prefill_should_commit_active_decode_cache([0]) is True
    assert adapter._prefill_should_commit_active_decode_cache([0, 1]) is False

    adapter._active_decode_cache = None
    assert adapter._prefill_should_commit_active_decode_cache([1]) is False


def test_vllm_adapter_trace_input_refresh_copies_changed_tokens_and_positions(monkeypatch):
    copies = []

    def fake_copy(host_tensor, device_tensor):
        copies.append((host_tensor, device_tensor))

    monkeypatch.setattr(generator_vllm_module.ttnn, "copy_host_to_device_tensor", fake_copy)

    fake = SimpleNamespace(
        _trace=SimpleNamespace(token_input="token-device", current_pos="position-device"),
        _vllm_audit_path=None,
        _audit_inc=lambda *args, **kwargs: None,
        _active_token_buffer=lambda values, on_host=False: ("token-host", values.clone(), on_host),
        _positions_host=lambda values: ("position-host", values.clone()),
    )

    Qwen3_5MoeForConditionalGeneration._reset_decode_trace_inputs(
        fake,
        tokens=torch.tensor([[11], [22]], dtype=torch.int64),
        start_pos=torch.tensor([5, 7], dtype=torch.int64),
        max_batch=4,
    )
    assert copies[0][0][0] == "token-host"
    assert copies[0][0][1].tolist() == [11, 22, 0, 0]
    assert copies[0][0][2] is True
    assert copies[0][1] == "token-device"
    assert copies[1][0][0] == "position-host"
    assert copies[1][0][1].tolist() == [5, 7, -1, -1]
    assert copies[1][1] == "position-device"

    copies.clear()
    Qwen3_5MoeForConditionalGeneration._reset_decode_trace_inputs(
        fake,
        tokens=torch.tensor([[33]], dtype=torch.int64),
        start_pos=torch.tensor([8], dtype=torch.int64),
        max_batch=4,
    )
    assert copies[0][0][1].tolist() == [33, 0, 0, 0]
    assert copies[1][0][1].tolist() == [8, -1, -1, -1]


def test_vllm_adapter_detects_inactive_live_page_table_rows():
    page_table = torch.tensor(
        [
            [1, 0, 0],
            [2, 0, 0],
            [0, 0, 0],
            [4, 0, 0],
        ],
        dtype=torch.int32,
    )

    assert Qwen3_5MoeForConditionalGeneration._inactive_page_table_rows_present(
        page_table,
        active_slots=[0],
        max_batch=4,
    )
    assert not Qwen3_5MoeForConditionalGeneration._inactive_page_table_rows_present(
        page_table,
        active_slots=[0, 1, 3],
        max_batch=4,
    )
    assert not Qwen3_5MoeForConditionalGeneration._inactive_page_table_rows_present(
        torch.zeros((4, 3), dtype=torch.int32),
        active_slots=[0],
        max_batch=4,
    )


def test_vllm_adapter_page_table_refreshes_only_on_changed_or_forced_inputs(monkeypatch):
    release_calls = []
    copies = []
    mesh = object()
    cache = SimpleNamespace(
        page_table_host=torch.tensor([[0, 1, 0], [2, 0, 0]], dtype=torch.int32),
        page_table="page-table-device",
    )
    fake = SimpleNamespace(
        cache=cache,
        mesh_device=mesh,
        _last_page_table_host=cache.page_table_host.clone(),
        _vllm_trace_warmed_cache_ids=set(),
        _vllm_audit_path=None,
        _audit_inc=lambda *args, **kwargs: None,
        _release_decode_trace=lambda: release_calls.append("release"),
    )
    fake._normalize_page_table = lambda page_table, cache: Qwen3_5MoeForConditionalGeneration._normalize_page_table(
        fake,
        page_table,
        cache,
    )
    fake._scatter_page_table_to_slots = (
        lambda page_table_host, cache, slots: Qwen3_5MoeForConditionalGeneration._scatter_page_table_to_slots(
            fake,
            page_table_host,
            cache,
            slots,
        )
    )

    monkeypatch.setattr(generator_vllm_module.ttnn, "ReplicateTensorToMesh", lambda device: ("replicate", device))
    monkeypatch.setattr(generator_vllm_module.ttnn, "from_torch", lambda host, **kwargs: ("host-tt", host.clone()))
    monkeypatch.setattr(
        generator_vllm_module.ttnn,
        "copy_host_to_device_tensor",
        lambda host_tensor, device_tensor: copies.append((host_tensor, device_tensor)),
    )

    same_one_based = torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.int32)
    returned = Qwen3_5MoeForConditionalGeneration._prepare_serving_cache(
        fake,
        cache,
        same_one_based,
        force=False,
    )
    assert returned is cache
    assert release_calls == []
    assert copies == []

    changed_one_based = torch.tensor([[1, 4, 0], [3, 0, 0]], dtype=torch.int32)
    Qwen3_5MoeForConditionalGeneration._prepare_serving_cache(
        fake,
        cache,
        changed_one_based,
        force=False,
    )
    assert release_calls == ["release"]
    assert copies[0][0][0] == "host-tt"
    assert copies[0][0][1].tolist() == [[0, 3, 0], [2, 0, 0]]
    assert copies[0][1] == "page-table-device"

    release_calls.clear()
    copies.clear()
    Qwen3_5MoeForConditionalGeneration._prepare_serving_cache(
        fake,
        cache,
        changed_one_based,
        force=True,
    )
    assert release_calls == ["release"]
    assert len(copies) == 1


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
