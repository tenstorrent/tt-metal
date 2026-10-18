# SPDX-License-Identifier: Apache-2.0
"""Execute real generator methods with CPU recording stubs, without importing TTNN.

These tests establish host update/ownership and sampler-call contracts. They do
not establish device trace safety, token equivalence, or serving performance.
"""

import ast
import inspect
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_class(filename, class_name, **globals_):
    tree = ast.parse((ROOT / "tt" / filename).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    node.bases = []
    namespace = dict(torch=torch, **globals_)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "tt" / filename), "exec"), namespace)
    return namespace[class_name]


@pytest.fixture
def generator_factory():
    def make(batch):
        cls = load_class("generator.py", "QwenFlashNextGenerator")
        obj = cls.__new__(cls)
        obj.batch_size = batch
        obj.model = SimpleNamespace(context=1024, max_requests=batch, scratch_page_base=4096)
        obj.sampling_mode = "device"
        obj.kv_cache = object()
        obj.page_table = torch.arange(batch * 32, dtype=torch.int32).reshape(batch, 32)
        obj._request_page_host = obj.page_table.clone()
        obj._page_host = obj.page_table.clone()
        obj._request_slot_host = list(range(batch))
        obj._slot_host = list(range(batch))
        obj._active = [True] * batch
        obj._record_history = True
        obj.counters = Counter()
        obj.writes = []
        obj.replays = []
        for field in ("tt_pos", "tt_slots", "tt_page", "tt_token", "tt_active", "tt_logit_mask"):
            setattr(obj, field, field)
        obj._copy = lambda value, tensor: obj.writes.append((tensor, value.clone()))
        obj._bind_cache = lambda cache: None
        obj.prepare = lambda: None
        obj._replay = obj.replays.append
        return obj

    return make


def decode(obj, **kwargs):
    args = dict(tokens=None, start_pos=None, page_table=None, kv_cache=obj.kv_cache, read_tokens=False)
    args.update(kwargs)
    return obj.decode_forward(**args)


@pytest.mark.parametrize("batch", [1, 3, 32])
def test_page_growth_only_writes_pages_even_with_inactive_rows(generator_factory, batch):
    obj = generator_factory(batch)
    active = [True] + [False] * (batch - 1)
    decode(obj, start_pos=[31] * batch, active_rows=active, slots=list(range(batch)))
    saved_slots = obj._slot_host.copy()
    saved_public = obj._request_page_host.clone()
    obj.writes.clear()
    obj.replays.clear()
    pages = saved_public.clone()
    pages[0, 1] = 301  # Newly consumed mapping at the next 32-token boundary.
    result = decode(obj, page_table=pages)
    assert result is obj.tt_token
    assert [name for name, _ in obj.writes] == ["tt_page"]
    assert obj.replays == ["decode", "greedy"]
    assert obj._slot_host == saved_slots
    assert obj._request_slot_host == list(range(batch))
    assert obj._page_host[0, 1] == 301
    for row in range(1, batch):
        assert obj._slot_host[row] == batch + row
        assert torch.equal(obj._page_host[row, :16], torch.arange(16) + 4096 + row * 16)
        assert torch.equal(obj._request_page_host[row], pages[row])
    obj.writes.clear()
    decode(obj, page_table=pages)
    assert obj.writes == []


@pytest.mark.parametrize("batch", [3, 32])
def test_reactivation_and_slot_permutation_refresh_only_changed_metadata(generator_factory, batch):
    obj = generator_factory(batch)
    decode(obj, start_pos=[0] * batch, active_rows=[True] + [False] * (batch - 1))
    pages = obj._request_page_host.clone()
    permuted = list(reversed(range(batch)))
    obj.writes.clear()
    decode(obj, start_pos=[32] * batch, active_rows=[True] * batch, slots=permuted)
    assert [name for name, _ in obj.writes] == ["tt_active", "tt_logit_mask", "tt_page", "tt_slots", "tt_pos"]
    assert obj._slot_host == permuted
    assert obj._request_slot_host == permuted
    assert torch.equal(obj._page_host, pages)
    assert obj._active == [True] * batch
    obj.writes.clear()
    decode(obj, start_pos=[33] * batch, active_rows=[True] * batch, slots=permuted)
    assert [name for name, _ in obj.writes] == ["tt_pos"]


def test_metadata_snapshots_are_owned_and_validation_preserved(generator_factory):
    obj = generator_factory(3)
    slots = [2, 0, 1]
    obj._slots(slots)
    slots[0] = 99
    assert obj._slot_host == [2, 0, 1]
    public = obj.page_table.clone()
    working = obj._request_pages(public)
    owner = obj._request_page_host
    working.fill_(17)
    public.fill_(99)
    assert torch.equal(owner, obj.page_table)
    second_working = obj._request_pages(None)
    assert obj._request_page_host is owner
    second_working.fill_(23)
    assert torch.equal(owner, obj.page_table)
    with pytest.raises(ValueError, match="private scratch"):  # allow-pytest.raises: CPU-only AST harness.
        obj._request_pages(torch.full_like(obj.page_table, 4096))
    with pytest.raises(ValueError, match="unique"):  # allow-pytest.raises: CPU-only AST harness.
        obj._request_slots([0, 0, 2])
    with pytest.raises(ValueError, match="one flag"):  # allow-pytest.raises: CPU-only AST harness.
        obj._active_rows([True])
    with pytest.raises(ValueError, match="explicit positions"):  # allow-pytest.raises: CPU-only AST harness.
        decode(obj, active_rows=[True] * 3)


@pytest.mark.parametrize("mode", ["greedy", "sample"])
def test_history_flag_changes_only_recording_and_is_read_only(mode):
    calls = []
    cls = load_class(
        "generator.py",
        "QwenFlashNextGenerator",
        prepare_sample_seeds=lambda *args: calls.append(("seeds", args)),
        record_tokens=lambda *args: calls.append(("record", args)),
    )
    assert inspect.signature(cls.__init__).parameters["record_history"].default is True
    obj = cls.__new__(cls)
    for field in ("tt_logits", "seeds", "seed_init", "tt_pos", "sample_seeds", "tt_token", "tt_history"):
        setattr(obj, field, object())
    obj.sample_params = {mode: (object(), object(), object())}
    output = object()

    def sample(*args, **kwargs):
        calls.append(("sample", args, kwargs))
        return (output,)

    obj.sampler = SimpleNamespace(decode_forward=sample)
    obj._record_history = True
    assert obj._sample(mode) is output
    recorded = calls.copy()
    calls.clear()
    obj._record_history = False
    assert obj._sample(mode) is output
    assert calls == [call for call in recorded if call[0] != "record"]
    assert recorded[-1] == ("record", (obj.tt_token, obj.tt_pos, obj.tt_history))
    assert next(call for call in calls if call[0] == "sample")[2]["tt_out_tok"] is obj.tt_token
    with pytest.raises(AttributeError):  # allow-pytest.raises: CPU-only AST harness.
        obj.record_history = True
    with pytest.raises(ValueError, match="record_history=True"):  # allow-pytest.raises: CPU-only AST harness.
        obj.generate([1], 4)


def test_adapter_exact_shape_alias_is_cloned_by_generator(generator_factory):
    cls = load_class("generator_vllm.py", "QwenFlashNextForCausalLM")
    adapter = cls.__new__(cls)
    adapter.batch_size, adapter.context = 3, 1024
    table = torch.arange(96, dtype=torch.int32).reshape(3, 32)
    assert adapter._pages(table) is table
    obj = generator_factory(3)
    working = obj._request_pages(adapter._pages(table))
    working[1, :16] = 4096
    assert torch.equal(table, torch.arange(96, dtype=torch.int32).reshape(3, 32))
    assert torch.equal(obj._request_page_host, table)
    padded = adapter._pages(torch.tensor([[4, 9]], dtype=torch.int64))
    assert padded.shape == (3, 32) and padded.dtype == torch.int32
    assert padded[0, :2].tolist() == [4, 9]
    assert torch.count_nonzero(padded) == 2
    with pytest.raises(ValueError, match="Invalid serving page table"):  # allow-pytest.raises: CPU-only AST harness.
        adapter._pages(torch.zeros(3, 33))


def test_non_aligned_prefill_tail_restores_active_rows_and_public_slots(generator_factory):
    obj = generator_factory(3)
    obj.buckets = obj.prefill_trace_buckets = [1, 2, 4, 8]
    obj.prefill_inputs = {n: f"prefill_{n}" for n in obj.buckets}
    events = []
    obj.event = lambda kind, **fields: events.append((kind, fields))
    result = obj.prefill_forward(
        torch.arange(15).reshape(3, 5),
        page_table=obj.page_table,
        kv_cache=obj.kv_cache,
        prompt_lens=[5, 3, 0],
        read_tokens=False,
    )
    assert result is obj.tt_token
    assert obj.replays == ["prefill_2", "prefill_1", "prefill_2", "greedy"]
    assert [fields["active"] for _, fields in events] == [
        [True, True, False],
        [True, True, False],
        [True, False, False],
    ]
    assert [values[:, 0].tolist() for name, values in obj.writes if name == "tt_slots"] == [
        [0, 1, 5],
        [0, 4, 5],
        [0, 1, 5],
    ]
    assert obj._active == [True, True, False]
    assert obj._request_slot_host == [0, 1, 2]
    assert obj.counters["activity_refreshes"] == 3
    assert torch.equal(obj._request_page_host, obj.page_table)
