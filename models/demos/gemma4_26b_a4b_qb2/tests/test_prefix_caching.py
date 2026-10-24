# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts for scheduler-owned cached prefixes."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import ttnn
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm import Gemma4ForCausalLM


@pytest.mark.parametrize("start,end", [(0, 31), (32, 33), (32, 63), (96, 129), (992, 1025), (1024, 1057)])
def test_continuation_preserves_absolute_positions(monkeypatch, start, end):
    monkeypatch.delenv("GEMMA4_PREFIX_TRACE_SUFFIX", raising=False)
    calls = []
    cache, tables = object(), object()
    generator = object.__new__(Gemma4Generator)
    generator.prefill_prepared = None

    def forward(ids, **kwargs):
        calls.append((ids.clone(), kwargs))
        return torch.zeros(1, 1, 1, 4)

    generator.model = SimpleNamespace(
        max_seq_len=2048,
        config=SimpleNamespace(vocab_size=16),
        upload=lambda value, *args: value,
        prefill_forward=forward,
    )
    generator._upload_page_tables = lambda value: value
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    tokens = torch.arange(end).reshape(1, end)
    generator.prefill_forward(tokens, page_table=tables, kv_cache=cache, prompt_lens=[end], start_pos=[start])
    ids, kwargs = calls[0]
    assert ids.flatten().tolist() == list(range(start, end))
    assert kwargs["start_pos"] == start
    assert kwargs["page_table"] is tables and kwargs["kv_cache"] is cache
    assert kwargs["user_id"] == 0


@pytest.mark.parametrize("starts,ends", [([-1], [33]), ([33], [33]), ([34], [33]), ([0, 32], [33])])
def test_invalid_continuation_fails_before_upload(starts, ends, expect_error):
    generator = object.__new__(Gemma4Generator)
    generator.prefill_prepared = None
    generator.model = SimpleNamespace(max_seq_len=2048)
    generator._upload_page_tables = Mock(side_effect=AssertionError("Invalid interval reached upload"))
    with expect_error(ValueError, "Invalid logical prefill interval|Start-position and batch dimensions differ"):
        generator.prefill_forward(
            torch.zeros(1, 64), page_table=None, kv_cache=None, prompt_lens=ends, start_pos=starts
        )


def test_host_sampling_forwards_cached_prefix():
    generator = Mock()
    adapter = Gemma4ForCausalLM(generator, 1)
    adapter._select_sampling_mode = Mock(return_value=True)
    adapter._read_host_logits = Mock(return_value="host logits")
    tables, cache = [torch.tensor([[7, 9]])], object()
    generator.model.layer_indices = [0]
    result = adapter.prefill_forward(
        torch.arange(33).reshape(1, 33), None, cache, [33], start_pos=[32], page_tables_per_layer=tables
    )
    assert result == "host logits"
    assert generator.prefill_forward.call_args.kwargs["start_pos"] == [32]
    assert generator.prefill_forward.call_args.kwargs["kv_cache"] is cache
    generator.serving_prefill_tokens.assert_not_called()
    generator.configure_sampling.assert_not_called()


def test_traced_suffix_uses_explicit_tokens_without_sampling(monkeypatch):
    monkeypatch.setenv("GEMMA4_PREFIX_TRACE_SUFFIX", "1")
    monkeypatch.setattr(ttnn, "reshape", torch.reshape)
    monkeypatch.setattr(ttnn, "clone", torch.clone)
    generator = object.__new__(Gemma4Generator)
    generator.prefill_prepared = None
    generator.model = SimpleNamespace(max_seq_len=2048, config=SimpleNamespace(vocab_size=16))
    generator._upload_page_tables = lambda value: value
    generator.decode_forward = Mock(return_value=torch.zeros(1, 1, 32, 4))
    cache, table = object(), torch.tensor([[7, 9]])
    generator.prefill_forward(
        torch.arange(63).reshape(1, 63), page_table=table, kv_cache=cache, prompt_lens=[63], start_pos=[32]
    )
    assert generator.decode_forward.call_count == 31
    for position, call in zip(range(32, 63), generator.decode_forward.call_args_list):
        assert call.args[0].item() == position
        assert call.args[1].item() == position
        assert call.kwargs["return_logits"] is True
        assert call.kwargs["page_table"] is table and call.kwargs["kv_cache"] is cache
