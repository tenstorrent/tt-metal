# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host generation control-flow checks with deterministic device boundaries."""

from collections import Counter
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from models.demos.llama31_8b_qb2.tt.generator import LlamaGenerator


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("record_token_history", [False, True])
@pytest.mark.parametrize("readback_each_token", [False, True])
def test_generate_returns_decode_tokens_with_or_without_history(batch, record_token_history, readback_each_token):
    current = torch.arange(batch) * 100 + 10
    # A disabled history buffer must never replace the actual sampled tokens.
    history = [current.clone()] if record_token_history else [torch.full((batch,), 999)] * 3

    def replay_decode(*, sample):
        assert sample
        current.add_(1)
        if record_token_history:
            history.append(current.clone())

    generator = SimpleNamespace(
        stats=Counter(),
        max_batch_size=batch,
        model=SimpleNamespace(supported_context=131072, config=SimpleNamespace(eos_token_id=0)),
        host_sampling=False,
        record_token_history=record_token_history,
        history_read_lengths={3},
        clear_cache_on_generate=False,
        kv_cache=(),
        _allocate_pages=Mock(return_value=torch.ones((batch, 1024), dtype=torch.int32)),
        reset=Mock(),
        set_sampling=Mock(),
        refresh_decode_inputs=Mock(),
        prefill_forward=Mock(side_effect=lambda *args, **kwargs: current.clone()),
        replay_decode=Mock(side_effect=replay_decode),
        read_tokens=Mock(side_effect=lambda: current.clone()),
        read_token_history=Mock(side_effect=lambda count: torch.stack(history[:count])),
        _counter_delta=Mock(return_value={}),
    )
    result = LlamaGenerator.generate(generator, [[1, 2]] * batch, 3, readback_each_token=readback_each_token)

    expected = [[10 + row * 100 + step for step in range(3)] for row in range(batch)]
    assert result == (expected[0] if batch == 1 else expected)
    assert generator.replay_decode.call_count == 2
    deferred = record_token_history and not readback_each_token
    assert generator.read_token_history.call_count == int(deferred)
    assert generator.read_tokens.call_count == (0 if deferred else 2)
