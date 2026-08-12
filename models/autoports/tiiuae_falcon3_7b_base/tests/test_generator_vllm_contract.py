# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import torch

from models.autoports.tiiuae_falcon3_7b_base.tt import generator_vllm as generator_vllm_module
from models.autoports.tiiuae_falcon3_7b_base.tt.generator import Falcon3Generator
from models.autoports.tiiuae_falcon3_7b_base.tt.generator_vllm import Falcon3ForCausalLM


def _adapter_without_hardware():
    adapter = object.__new__(Falcon3ForCausalLM)
    adapter._trace_model_id = object()
    adapter._vllm_active_batch = 0
    return adapter


def _sampling_params(batch_size):
    return SimpleNamespace(
        temperature=torch.zeros(batch_size),
        top_k=torch.zeros(batch_size, dtype=torch.int32),
        top_p=torch.ones(batch_size),
    )


def test_steady_async_decode_ignores_stale_tokens_and_positions(monkeypatch):
    adapter = _adapter_without_hardware()
    captured = {}

    monkeypatch.setattr(Falcon3ForCausalLM, "set_sampling_params", lambda self, **kwargs: None)

    def fake_decode(self, tokens, start_pos, **kwargs):
        captured.update(tokens=tokens, start_pos=start_pos, **kwargs)
        return object()

    monkeypatch.setattr(Falcon3Generator, "decode_forward", fake_decode)
    page_table = torch.tensor([[3, 1, -1], [2, 0, -1]], dtype=torch.int32)
    kv_cache = object()
    tokens = torch.tensor([[111], [222]], dtype=torch.int32)
    positions = torch.tensor([19, 37], dtype=torch.int32)

    adapter.decode_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=positions,
        sampling_params=_sampling_params(2),
        reset_batch=False,
        slot_remap=torch.arange(2, dtype=torch.int32),
    )

    assert captured["tokens"] is None
    assert captured["start_pos"] is None
    assert captured["page_table"] is None
    assert captured["kv_cache"] is kv_cache
    assert captured["sampling_mode"] == "device"
    assert captured["enable_trace"] is True
    assert captured["active_batch"] == 2


def test_new_batch_forwards_current_tokens_positions_and_page_table(monkeypatch):
    adapter = _adapter_without_hardware()
    captured = {}

    monkeypatch.setattr(Falcon3ForCausalLM, "set_sampling_params", lambda self, **kwargs: None)

    def fake_decode(self, tokens, start_pos, **kwargs):
        captured.update(tokens=tokens, start_pos=start_pos, **kwargs)
        return object()

    monkeypatch.setattr(Falcon3Generator, "decode_forward", fake_decode)
    page_table = torch.tensor([[7, 5, -1]], dtype=torch.int32)
    kv_cache = object()
    tokens = torch.tensor([[42]], dtype=torch.int32)
    positions = torch.tensor([31], dtype=torch.int32)

    adapter.decode_forward(
        tokens,
        page_table=page_table,
        kv_cache=kv_cache,
        start_pos=positions,
        sampling_params=_sampling_params(1),
        reset_batch=True,
    )

    assert captured["tokens"] is tokens
    assert captured["start_pos"] is positions
    assert captured["page_table"] is page_table
    assert captured["kv_cache"] is kv_cache


def test_changed_page_table_is_forwarded_when_scheduler_resets_batch(monkeypatch):
    adapter = _adapter_without_hardware()
    captured = {}

    monkeypatch.setattr(Falcon3ForCausalLM, "set_sampling_params", lambda self, **kwargs: None)

    def fake_decode(self, tokens, start_pos, **kwargs):
        captured.update(tokens=tokens, start_pos=start_pos, **kwargs)
        return object()

    monkeypatch.setattr(Falcon3Generator, "decode_forward", fake_decode)
    changed_page_table = torch.tensor([[9, 11, 13, -1]], dtype=torch.int32)

    adapter.decode_forward(
        torch.tensor([[77]], dtype=torch.int32),
        page_table=changed_page_table,
        kv_cache=object(),
        start_pos=torch.tensor([65], dtype=torch.int32),
        sampling_params=_sampling_params(1),
        reset_batch=True,
    )

    assert captured["page_table"] is changed_page_table
    assert captured["tokens"].item() == 77
    assert captured["start_pos"].item() == 65


def test_async_decode_read_submits_only_one_replicated_token_shard(monkeypatch):
    adapter = _adapter_without_hardware()
    adapter.mesh_device = object()
    calls = []

    class FakeShard:
        def cpu(self, *, blocking):
            calls.append(("cpu", blocking))
            return "pending-host-token"

    class DistributedOutput:
        def cpu(self, **kwargs):
            raise AssertionError("distributed output must not be copied")

    shards = [FakeShard(), FakeShard(), FakeShard(), FakeShard()]
    monkeypatch.setattr(generator_vllm_module.ttnn, "get_device_tensors", lambda output: shards)
    monkeypatch.setattr(
        generator_vllm_module.ttnn,
        "record_event",
        lambda mesh, queue: calls.append(("event", mesh, queue)) or "ready-event",
    )

    host, events = adapter.read_decode_output(DistributedOutput(), async_read=True)

    assert host == "pending-host-token"
    assert events == ["ready-event"]
    assert calls == [("cpu", False), ("event", adapter.mesh_device, 0)]


def test_steady_async_decode_rejects_live_slot_remap(monkeypatch):
    adapter = _adapter_without_hardware()
    monkeypatch.setattr(Falcon3ForCausalLM, "set_sampling_params", lambda self, **kwargs: None)

    with torch.no_grad():
        try:
            adapter.decode_forward(
                torch.tensor([[1], [2]], dtype=torch.int32),
                page_table=torch.zeros((2, 2), dtype=torch.int32),
                kv_cache=object(),
                start_pos=torch.tensor([4, 9], dtype=torch.int32),
                sampling_params=_sampling_params(2),
                reset_batch=False,
                slot_remap=torch.tensor([1, 0], dtype=torch.int32),
            )
        except ValueError as exc:
            assert "cannot remap live slots" in str(exc)
        else:
            raise AssertionError("non-identity live-slot remap must be rejected")


def test_vllm_zero_padded_sdpa_tail_is_not_treated_as_live_cache_ownership():
    generator = object.__new__(Falcon3Generator)
    generator.model = SimpleNamespace(page_block_size=32, max_cache_len=32768)
    generator.num_blocks = 128
    # At position 32 each request owns two logical pages, while SDPA reads a
    # four-page rounded window. vLLM pads both masked tails with block zero.
    page_table = torch.tensor([[1, 2, 0, 0], [3, 4, 0, 0]], dtype=torch.int32)
    generator._validate_page_coverage(page_table, torch.tensor([32, 32]), active_batch=2)


def test_live_vllm_cache_pages_must_remain_disjoint():
    generator = object.__new__(Falcon3Generator)
    generator.model = SimpleNamespace(page_block_size=32, max_cache_len=32768)
    generator.num_blocks = 128
    page_table = torch.tensor([[1, 2, 0, 0], [2, 4, 0, 0]], dtype=torch.int32)
    try:
        generator._validate_page_coverage(page_table, torch.tensor([32, 32]), active_batch=2)
    except ValueError as exc:
        assert "disjoint physical cache pages" in str(exc)
    else:
        raise AssertionError("live cache-page alias must be rejected")


def test_prefill_grows_rope_to_declared_generation_horizon_before_forward(monkeypatch):
    adapter = _adapter_without_hardware()
    calls = []
    adapter.model = SimpleNamespace(
        max_cache_len=32768,
        ensure_rope_capacity=lambda horizon: calls.append(("grow", horizon)),
    )
    monkeypatch.setattr(
        Falcon3ForCausalLM,
        "_release_decode_traces_before_allocating_prefill",
        lambda self: calls.append(("release", None)),
    )

    def fake_prefill(self, tokens, **kwargs):
        calls.append(("forward", kwargs["prompt_lens"]))
        return object()

    monkeypatch.setattr(Falcon3Generator, "prefill_forward", fake_prefill)
    adapter.prefill_forward(
        torch.ones((1, 17), dtype=torch.int32),
        page_table=torch.zeros((1, 10), dtype=torch.int32),
        kv_cache=object(),
        prompt_lens=[17],
        generation_horizon=273,
    )

    assert calls == [("release", None), ("grow", 273), ("forward", [17])]
