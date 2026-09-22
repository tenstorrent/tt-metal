# SPDX-FileCopyrightText: Copyright 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host coverage for externally scheduled Gemma4 dFlash prefill chunks.

The plugin supplies every token through each granted chunk's end position.
The CT adapter recomputes resumed prefixes eagerly so sliding attention does
not depend on a tail published by an earlier trace or a short external chunk.
These tests inspect production dispatch and request ownership with target
execution stubbed. Device numerical behavior requires hardware validation.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import _prefill, _tensor
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import adapter as _adapter_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import decoder_width_for as _decoder_width_for_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import expect_error as _expect_error_fixture
from models.demos.gemma4.tests.unit.test_dflash_contract_adapter import model as _model_fixture

adapter = _adapter_fixture
decoder_width_for = _decoder_width_for_fixture
expect_error = _expect_error_fixture
model = _model_fixture


def _forwarded(model):
    return [event[1] for event in model.events if event[0] == "prefill"][-1]


def test_external_chunks_recompute_the_complete_prefix_after_peer_prefill(model):
    prefix = list(range(193))
    table = _tensor([[10, 11, 12, 13, 10]])

    def submit(end, start):
        tokens = _tensor([prefix[:end]])
        result = model.prefill_forward(
            tokens=tokens,
            prompt_lens=np.array([end], dtype=np.int64),
            start_pos=np.array([start], dtype=np.int64),
            empty_slots=[7],
            page_table=table,
            kv_cache=model.kv_cache,
            enable_trace=True,
            warmup_prefill=False,
        )
        forwarded = _forwarded(model)
        consumed = forwarded["tokens"][0, int(forwarded["start_pos"][0]) : int(forwarded["prompt_lens"][0])]
        assert consumed.tolist() == prefix[:end]
        assert forwarded["tokens"] is tokens
        assert forwarded["kv_cache"] is model.kv_cache
        assert forwarded["empty_slots"] == [7]
        assert result.tolist() == [[0]]
        return forwarded

    assert submit(128, 0)["enable_trace"] is True
    first_owner = model._ct_requests[10]
    _prefill(model, keys=(20,), slots=[2], prompts=[[801, 802]])
    peer = model._ct_requests[20]
    model._ct_proposal = SimpleNamespace(owner=first_owner)

    for end, start in ((157, 128), (193, 157)):
        old = model._ct_requests[10]
        forwarded = submit(end, start)
        assert forwarded["start_pos"] == [0]
        assert forwarded["enable_trace"] is False
        assert old.live is False
        assert model._ct_requests[10].tokens == prefix[:end]
        assert model._ct_requests[10].slot == 7
        assert model._ct_requests[20] is peer
        assert peer.live is True
        assert peer.tokens == [801, 802]
        assert model._ct_proposal is None
    assert table.tolist() == [[10, 11, 12, 13, 10]]


def test_mixed_fresh_and_resumed_rows_preserve_absolute_lengths_and_layer_tables(model):
    tokens = _tensor([list(range(132)), [801, 802, 803] + [0] * 129])
    lengths = np.array([132, 3], dtype=np.int64)
    starts = np.array([128, 0], dtype=np.int64)
    table = _tensor([[10, 11, 12, 10], [20, 21, 22, 20]])
    second_layer = _tensor([[30, 31, 32, 30], [40, 41, 42, 40]])
    sampling = object()
    model.prefill_forward(
        tokens,
        prompt_lens=lengths,
        start_pos=starts,
        empty_slots=[7, 2],
        page_table=table,
        page_tables_per_layer=[table, second_layer],
        kv_cache=model.kv_cache,
        enable_trace=True,
        warmup_prefill=False,
        sampling_params=sampling,
    )
    forwarded = _forwarded(model)
    assert forwarded["start_pos"] == [0, 0]
    assert forwarded["prompt_lens"] is lengths
    assert forwarded["enable_trace"] is False
    assert forwarded["empty_slots"] == [7, 2]
    assert forwarded["sampling_params"] is sampling
    assert forwarded["page_table"].tolist() == [[10, 11, 12, 0], [20, 0, 0, 0]]
    assert [item.tolist() for item in forwarded["page_tables_per_layer"]] == [
        [[10, 11, 12, 0], [20, 0, 0, 0]],
        [[30, 31, 32, 0], [40, 0, 0, 0]],
    ]
    assert starts.tolist() == [128, 0]
    assert lengths.tolist() == [132, 3]
    assert table.tolist() == [[10, 11, 12, 10], [20, 21, 22, 20]]
    assert second_layer.tolist() == [[30, 31, 32, 30], [40, 41, 42, 40]]
    assert model._ct_requests[10].tokens == list(range(132))
    assert model._ct_requests[20].tokens == [801, 802, 803]
    assert [model._ct_requests[key].slot for key in (10, 20)] == [7, 2]


@pytest.mark.parametrize("starts", [None, [0], np.array([0], dtype=np.int64)])
def test_initial_prefill_preserves_trace_dispatch(model, starts):
    kwargs = {} if starts is None else {"start_pos": starts}
    model.prefill_forward(
        tokens=_tensor([[1, 2, 3]]),
        prompt_lens=[3],
        page_table=_tensor([[10, 11]]),
        kv_cache=model.kv_cache,
        enable_trace=True,
        warmup_prefill=False,
        **kwargs,
    )
    forwarded = _forwarded(model)
    assert forwarded["enable_trace"] is True
    assert forwarded.get("start_pos") is starts


@pytest.mark.parametrize("warmup_depth,warmup_prefill", [(1, False), (0, True)])
def test_warmup_preserves_original_prefill_dispatch(model, warmup_depth, warmup_prefill):
    model._ct_warmup_depth = warmup_depth
    starts = np.array([128], dtype=np.int64)
    model.prefill_forward(
        tokens=_tensor([list(range(157))]),
        prompt_lens=[157],
        start_pos=starts,
        page_table=_tensor([[10, 11, 12]]),
        kv_cache=model.kv_cache,
        enable_trace=True,
        warmup_prefill=warmup_prefill,
    )
    assert _forwarded(model)["start_pos"] is starts
    assert _forwarded(model)["enable_trace"] is True
    assert model._ct_requests == {}


@pytest.mark.parametrize(
    "starts,lengths",
    [([2, 2], [4]), ([2], [4, 4]), ([2], [5]), ([5], [4])],
)
def test_incomplete_prefix_or_misaligned_rows_fail_before_target_work(model, expect_error, starts, lengths):
    _prefill(model, keys=(10,), prompts=[[1, 2]])
    owner = model._ct_requests[10]
    model.events.clear()
    with expect_error(ValueError, match="Gemma4 dFlash resumed prefill requires"):
        model.prefill_forward(
            tokens=_tensor([[1, 2, 3, 4]]),
            prompt_lens=lengths,
            start_pos=starts,
            page_table=_tensor([[10, 11]]),
            kv_cache=model.kv_cache,
            enable_trace=True,
            warmup_prefill=False,
        )
    assert model.events == []
    assert model._ct_requests[10] is owner
    assert owner.live is True
    assert owner.tokens == [1, 2]


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("original", [None, False, True])
def test_continuation_restores_internal_chunk_dispatch_after_return_or_failure(
    model, monkeypatch, expect_error, fail, original
):
    if original is not None:
        model._ct_eager_prefill = original
    prefill = model._contract_target_prefill

    def target(**kwargs):
        assert model._ct_eager_prefill is True
        if fail:
            raise ValueError("target failure")
        return prefill(**kwargs)

    monkeypatch.setattr(model, "_contract_target_prefill", target)

    def submit():
        return model.prefill_forward(
            tokens=_tensor([[1, 2, 3, 4]]),
            prompt_lens=[4],
            start_pos=[2],
            page_table=_tensor([[10, 11]]),
            kv_cache=model.kv_cache,
            enable_trace=True,
            warmup_prefill=False,
        )

    if fail:
        with expect_error(ValueError, match="target failure"):
            submit()
        assert model._ct_requests == {}
    else:
        submit()
        assert model._ct_requests[10].tokens == [1, 2, 3, 4]
    if original is None:
        assert not hasattr(model, "_ct_eager_prefill")
    else:
        assert model._ct_eager_prefill is original


@pytest.mark.parametrize("force_eager", [None, False, True])
def test_production_internal_chunk_dispatch_obeys_ct_context(expect_error, force_eager):
    # Extract only the production dispatcher so host coverage cannot import
    # TT device libraries or replace the branch under test with a stub.
    source = Path(__file__).parents[2] / "tt" / "generator.py"
    module = ast.parse(source.read_text())
    generator = next(
        item
        for item in module.body
        if isinstance(item, ast.ClassDef) and item.name == "ChunkedPrefillPageTableGuardMixin"
    )
    dispatch = next(
        item
        for item in generator.body
        if isinstance(item, ast.FunctionDef) and item.name == "prefill_forward_single_user_text"
    )

    def trace_selected(*args):
        raise RuntimeError("internal trace path selected")

    namespace = {
        "torch": torch,
        "chunked_prefill_trace_enabled": lambda: True,
        "num_blocks_in_seq": lambda length, size: (length + size - 1) // size,
        "get_max_prefill_chunk_size": trace_selected,
    }
    exec(compile(ast.Module(body=[dispatch], type_ignores=[]), str(source), "exec"), namespace)
    forwarded = []
    sentinel = object()

    def eager(tokens, **kwargs):
        forwarded.append((tokens, kwargs))
        return sentinel

    instance = SimpleNamespace(
        model=[SimpleNamespace(layers=[], hidden_size_per_layer_input=0)],
        model_args=[SimpleNamespace(max_prefill_chunk_size=128)],
        _activate_sequential_per_layer_row=lambda page_table: None,
        _effective_paged_block_size=lambda kv_cache: 64,
        _refresh_prefill_valid_seq_len=lambda **kwargs: None,
        _uses_bounded_sliding_kv=lambda model_id: False,
        _prefill_forward_single_user_text_eager=eager,
    )
    if force_eager is not None:
        instance._ct_eager_prefill = force_eager
    tokens = _tensor([list(range(256))])
    table = _tensor([[10, 11, 12, 13]])
    cache = object()

    def submit():
        return namespace["prefill_forward_single_user_text"](
            instance, tokens, page_table=table, kv_cache=cache, num_cached_tokens=0, last_token_idx=192
        )

    if force_eager:
        assert submit() is sentinel
        assert forwarded[0][0] is tokens
        assert forwarded[0][1]["kv_cache"] is cache
        assert forwarded[0][1]["last_token_idx"] == 192
        assert forwarded[0][1]["num_cached_tokens"] == 0
    else:
        with expect_error(RuntimeError, match="internal trace path selected"):
            submit()
        assert forwarded == []
