# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import urllib.error
import urllib.request

import torch

from models.demos.gemma4.demo import prefill_harness, prefill_runtime
from models.demos.gemma4.demo.prefill_harness import submit_prefill
from models.demos.gemma4.demo.prefill_runtime import (
    CacheSlotState,
    TracedPrefillRuntime,
    _chunk_page_table_row,
    _cp_chunk_valid_lengths,
    _fixed_cache_slot_blocks,
    _page_table_row,
)
from models.demos.gemma4.demo.prefill_service import _create_parser, create_server


class _FakeRuntime:
    def __init__(self):
        self.calls = []

    def info(self):
        return {"status": "ready", "cache_slots": 1, "next_token_enabled": False}

    def prefill(self, prompt, request_id):
        self.calls.append((prompt, request_id))
        return {
            "request_id": request_id,
            "status": "prefilled",
            "prompt_tokens": len(prompt.split()),
            "cache_slot": 0,
            "cache_generation": len(self.calls),
            "cache_resident": True,
            "next_token": None,
        }


class _BatchEncodingLike:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    chat_template = "present"

    def apply_chat_template(self, *_args, **_kwargs):
        return _BatchEncodingLike(torch.tensor([[1, 2, 3]], dtype=torch.int64))

    def decode(self, token_ids):
        return f"token-{token_ids[0]}"


def _get_json(url):
    with urllib.request.urlopen(url, timeout=2) as response:
        return response.status, json.loads(response.read())


def _post_json(url, payload):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def test_prefill_service_contract():
    runtime = _FakeRuntime()
    server = create_server(runtime, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        status, health = _get_json(f"{base_url}/health")
        assert status == 200
        assert health == {"status": "ready", "cache_slots": 1, "next_token_enabled": False}

        result = submit_prefill(base_url, "hello prefill", "req-1", timeout=2)
        assert result["status"] == "prefilled"
        assert result["request_id"] == "req-1"
        assert result["cache_resident"] is True
        assert result["next_token"] is None
        assert runtime.calls == [("hello prefill", "req-1")]

        status, error = _post_json(f"{base_url}/prefill", {"request_id": "req-2", "prompt": ""})
        assert status == 400
        assert error["status"] == "error"
        assert error["request_id"] == "req-2"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_prefill_tokenizer_accepts_batch_encoding():
    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.tokenizer = _FakeTokenizer()

    tokens = runtime._tokenize("hello")

    assert tokens.dtype == torch.int32
    assert tokens.tolist() == [[1, 2, 3]]


def test_prefill_returns_next_token_and_excludes_readback_from_latency(monkeypatch):
    events = []

    class FakeLogits:
        def deallocate(self, force):
            events.append(("deallocate", force))

    class FakeModel:
        def process_logits_after_prefill_trace(self, trace_output, last_token_idx):
            events.append(("lm_head", trace_output, last_token_idx))
            return FakeLogits()

        def process_output_prefill(self, _logits, last_token_idx):
            events.append(("readback", last_token_idx))
            return torch.tensor([[[0.0, 1.0, 4.0, 2.0]]])

    clock = iter((10.0, 12.0))

    def perf_counter():
        events.append(("clock",))
        return next(clock)

    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.max_context_len = 4
    runtime.chunk_size = 4
    runtime.pad_token_id = 0
    runtime._trace_id = 7
    runtime._trace_output = "hidden-states"
    runtime._generation = 0
    runtime._slot_states = [CacheSlotState(slot=0)]
    runtime.tokenizer = _FakeTokenizer()
    runtime._tokenize = lambda _prompt: torch.tensor([[11, 12, 13]], dtype=torch.int32)
    runtime._reserve_cache_slot = lambda _padded_tokens, _request_id: runtime._slot_states[0]
    runtime._stage_page_table = lambda _cache_slot: None
    runtime._stage = lambda *_args: None
    runtime.model = FakeModel()

    monkeypatch.setattr(prefill_runtime.time, "perf_counter", perf_counter)
    monkeypatch.setattr(prefill_runtime.ttnn, "execute_trace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(prefill_runtime.ttnn, "synchronize_device", lambda _device: None)
    runtime.mesh_device = object()

    result = runtime._prefill_serial("hello", "req-1")

    assert result["next_token"] == 2
    assert result["next_token_decoded"] == "token-2"
    assert result["prefill_time_ms"] == 2000.0
    assert events.index(("clock",)) < events.index(("readback", 2))
    assert events[events.index(("lm_head", "hidden-states", 2)) + 1] == ("clock",)
    assert events[-1] == ("deallocate", True)


def test_page_table_row_maps_owned_blocks_and_pads_safely():
    page_table = _page_table_row(width=8, cache_blocks=(3, 7, 11))

    assert page_table.tolist() == [[3, 7, 11, 3, 3, 3, 3, 3]]


def test_chunk_page_table_advances_through_slot_blocks():
    cache_blocks = tuple(range(100, 148))

    assert _chunk_page_table_row(cache_blocks, chunk_idx=0, chunk_page_table_width=16).tolist() == [
        list(range(100, 116))
    ]
    assert _chunk_page_table_row(cache_blocks, chunk_idx=2, chunk_page_table_width=16).tolist() == [
        list(range(132, 148))
    ]


def test_fixed_cache_slots_own_disjoint_full_context_ranges():
    assert _fixed_cache_slot_blocks(slot=0, blocks_per_slot=4) == (0, 1, 2, 3)
    assert _fixed_cache_slot_blocks(slot=1, blocks_per_slot=4) == (4, 5, 6, 7)
    assert _fixed_cache_slot_blocks(slot=7, blocks_per_slot=4) == (28, 29, 30, 31)


def test_cp_chunk_valid_lengths_are_specific_to_each_cp_rank():
    assert _cp_chunk_valid_lengths(valid_tokens=8192, chunk_size=8192, cp=8, tp=4) == (1024,) * 32
    assert _cp_chunk_valid_lengths(valid_tokens=492, chunk_size=8192, cp=8, tp=4) == (492,) * 4 + (0,) * 28


def test_hybrid_page_tables_keep_full_and_sliding_slots_disjoint():
    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime._page_table_width = 4
    runtime._sliding_blocks_per_slot = 1
    runtime._sliding_layers = (True, False)
    runtime._slot_states = [
        CacheSlotState(slot=slot, cache_blocks=_fixed_cache_slot_blocks(slot, 4)) for slot in range(8)
    ]

    sliding, full = runtime._layer_page_tables(cache_slot=2)

    assert sliding.tolist() == [[2, 2, 2, 2]]
    assert full.tolist() == [[8, 9, 10, 11]]


def test_prefill_rotates_eight_cache_slots(monkeypatch):
    class FakeLogits:
        def deallocate(self, _force):
            pass

    class FakeModel:
        def process_logits_after_prefill_trace(self, _trace_output, _last_token_idx):
            return FakeLogits()

        def process_output_prefill(self, _logits, _last_token_idx):
            return torch.tensor([[[1.0]]])

    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.tokenizer = _FakeTokenizer()
    runtime.pad_token_id = 0
    runtime.chunk_size = 8192
    runtime.max_context_len = 65536
    runtime.cache_slots = 8
    runtime.cp = 8
    runtime._page_table_width = 128
    runtime._generation = 0
    runtime._next_cache_slot = 0
    runtime._slot_states = [
        CacheSlotState(slot=slot, cache_blocks=_fixed_cache_slot_blocks(slot, runtime._page_table_width))
        for slot in range(8)
    ]
    runtime.mesh_device = object()
    runtime.model = FakeModel()
    runtime._trace_id = 1
    runtime._trace_output = object()
    staged_page_tables = []
    staged_chunks = []
    runtime._stage_page_table = staged_page_tables.append
    runtime._stage = (
        lambda _tokens, chunk_idx, cache_slot, _prompt_tokens: staged_chunks.append((chunk_idx, cache_slot)) or 0
    )
    monkeypatch.setattr("models.demos.gemma4.demo.prefill_runtime.ttnn.execute_trace", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("models.demos.gemma4.demo.prefill_runtime.ttnn.synchronize_device", lambda *_args: None)

    results = [runtime._prefill_serial("hello", f"req-{index}") for index in range(10)]

    assert [result["cache_slot"] for result in results] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert staged_page_tables == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert staged_chunks == [(0, slot) for slot in staged_page_tables]
    assert runtime._next_cache_slot == 2
    assert runtime._slot_states[0].request_id == "req-8"
    assert runtime._slot_states[1].request_id == "req-9"
    assert runtime._slot_states[2].request_id == "req-2"
    assert runtime._generation == 10


def test_full_context_slots_do_not_evict_other_residents():
    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.max_context_len = 32768
    runtime.cache_slots = 8
    runtime.cp = 8
    runtime._page_table_width = 64
    runtime._next_cache_slot = 0
    runtime._slot_states = [
        CacheSlotState(slot=slot, cache_blocks=_fixed_cache_slot_blocks(slot, runtime._page_table_width))
        for slot in range(8)
    ]
    for generation in range(1, 5):
        slot = runtime._reserve_cache_slot(8192, f"req-{generation}")
        slot.request_id = f"req-{generation}"
        slot.generation = generation

    large_slot = runtime._reserve_cache_slot(16384, "large-request")

    assert large_slot.slot == 4
    assert len(large_slot.cache_blocks) == 64
    assert large_slot.cache_blocks == tuple(range(256, 320))
    assert runtime._slot_states[0].resident is True
    assert runtime._slot_states[1].resident is True
    assert runtime._slot_states[2].resident is True
    assert runtime._slot_states[3].resident is True


def test_retried_request_reuses_its_cache_slot():
    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.max_context_len = 32768
    runtime.cache_slots = 8
    runtime.cp = 8
    runtime._page_table_width = 64
    runtime._next_cache_slot = 0
    runtime._slot_states = [
        CacheSlotState(slot=slot, cache_blocks=_fixed_cache_slot_blocks(slot, runtime._page_table_width))
        for slot in range(8)
    ]
    first = runtime._reserve_cache_slot(8192, "same-request")
    first.request_id = "same-request"
    first.generation = 1

    retry = runtime._reserve_cache_slot(16384, "same-request")

    assert retry.slot == 0
    assert len(retry.cache_blocks) == 64
    assert runtime._next_cache_slot == 1
    assert sum(slot.resident for slot in runtime._slot_states) == 0


def test_cache_slots_report_residents_and_next_slot():
    runtime = TracedPrefillRuntime.__new__(TracedPrefillRuntime)
    runtime.model_path = "test-model"
    runtime.mesh_device = type("Mesh", (), {"shape": (8, 4)})()
    runtime.chunk_size = 8192
    runtime.max_context_len = 262144
    runtime.cache_slots = 8
    runtime._page_table_width = 512
    runtime._sliding_blocks_per_slot = 16
    runtime._generation = 9
    runtime._next_cache_slot = 2
    runtime._slot_states = [CacheSlotState(slot=slot) for slot in range(8)]
    runtime._slot_states[0] = CacheSlotState(
        slot=0, request_id="req-1", prompt_tokens=10, generation=1, cache_blocks=tuple(range(512))
    )
    runtime._slot_states[1] = CacheSlotState(
        slot=1, request_id="req-2", prompt_tokens=20, generation=2, cache_blocks=tuple(range(512, 1024))
    )

    info = runtime.info()

    assert info["cache_slots"] == 8
    assert info["cache_slot_capacity_tokens"] == 262144
    assert info["cache_capacity_tokens"] == 8 * 262144
    assert info["cache_blocks_per_slot"] == 512
    assert info["sliding_cache_blocks_per_slot"] == 16
    assert info["free_cache_slots"] == 6
    assert info["resident_cache_slots"] == 2
    assert info["next_cache_slot"] == 2
    assert [slot["request_id"] for slot in info["resident_slots"]] == ["req-1", "req-2"]
    assert info["cache_resident"] is True
    assert info["cache_generation"] == 9


def test_service_defaults_to_eight_cache_slots():
    args = _create_parser().parse_args([])

    assert args.cache_slots == 8


def test_preset_prompt_prefix_uses_supplied_token_estimate():
    preset = prefill_harness.PromptPreset("test", 1000, "https://example.com/book.txt")
    text = "x" * 10_000

    prefix = prefill_harness.preset_prompt_prefix(text, preset, context_len=164)

    # 164 context - 64 chat-token reserve = 100/1000 of the source.
    assert prefix == text[:1000]


def test_preset_submit_shrinks_to_server_token_count(monkeypatch):
    calls = []

    def fake_submit(_service_url, prompt, request_id, _timeout):
        calls.append(prompt)
        if len(prompt) > 100:
            return {
                "request_id": request_id,
                "status": "error",
                "http_status": 400,
                "error": f"prompt has {len(prompt)} tokens, exceeding max_context_len=100",
            }
        return {"request_id": request_id, "status": "prefilled", "prompt_tokens": len(prompt)}

    monkeypatch.setattr(prefill_harness, "submit_prefill", fake_submit)
    preset = prefill_harness.PromptPreset("test", 200, "https://example.com/book.txt")

    result = prefill_harness.submit_preset_prefill(
        "http://service",
        "x" * 200,
        "req-1",
        preset=preset,
        context_len=100,
        source_characters=400,
        timeout=2,
    )

    assert result["status"] == "prefilled"
    assert result["prompt_tokens"] <= 100
    assert result["fit_attempts"] == 2
    assert result["prefix_fraction"] == len(calls[-1]) / 400


def test_preset_submit_keeps_successful_token_overshoot(monkeypatch):
    calls = []

    def fake_submit(_service_url, prompt, request_id, _timeout):
        calls.append(prompt)
        return {"request_id": request_id, "status": "prefilled", "prompt_tokens": 104}

    monkeypatch.setattr(prefill_harness, "submit_prefill", fake_submit)
    preset = prefill_harness.PromptPreset("test", 200, "https://example.com/book.txt")

    result = prefill_harness.submit_preset_prefill(
        "http://service",
        "x" * 100,
        "req-1",
        preset=preset,
        context_len=100,
        source_characters=400,
        timeout=2,
    )

    assert result["status"] == "prefilled"
    assert result["prompt_tokens"] == 104
    assert result["context_overflow_tokens"] == 4
    assert result["fit_attempts"] == 1
    assert len(calls) == 1
