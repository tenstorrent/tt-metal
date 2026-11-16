# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType, SimpleNamespace

import pytest
import torch

import ttnn
from models.demos.gpt_oss.tt.model import Model


@pytest.mark.parametrize(
    "slots,padded_len,route_slots,lengths",
    [
        ([0, 1], 256, True, None),
        ([32, 33], 256, True, None),
        ([0, 32, 33], 256, True, None),
        ([96, 0, 64, 32], 256, True, None),
        (list(range(20)), 128, True, None),
        ([0, 32], 256, False, None),
        ([0, 32, 64, 96], 256, True, [65, 129, 100, 256]),
        ([0, 32], 256, True, [65, 129]),
    ],
)
def test_batched_prefill_preserves_unscheduled_lane_kv(monkeypatch, slots, padded_len, route_slots, lengths):
    """A lane's block IDs may also belong to unrelated live requests in other lanes."""
    n = len(slots)
    lengths = lengths or [padded_len] * n
    num_blocks = padded_len // 64
    tokens = torch.arange(1, n + 1).reshape(n, 1).expand(n, padded_len).clone()
    pages = torch.tensor(
        [[(slot % 32) * num_blocks + b for b in range(num_blocks)] for slot in slots], dtype=torch.int32
    )
    for index, length in enumerate(lengths):
        valid_blocks = (length + 63) // 64
        pages[index, valid_blocks:] = 127
    original_pages = pages.clone()
    sentinel = -1000
    actual_cache = torch.full((4, 128), sentinel)
    expected_cache = actual_cache.clone()
    for index, slot in enumerate(slots):
        expected_cache[slot // 32, pages[index, : (lengths[index] + 63) // 64].long()] = tokens[index, 0]

    def forward(tokens_iter, page_table, kv_cache, fixed_glt, skip_lm_head, batch_size):
        per_row_pages = page_table.reshape(4, batch_size, -1)
        per_row_tokens = tokens_iter.reshape(4, batch_size, padded_len)
        for row in range(4):
            for user in range(batch_size):
                valid = per_row_pages[row, user]
                valid = valid[valid >= 0].long()
                actual_cache[row, valid] = per_row_tokens[row, user, 0]

    model = SimpleNamespace(mesh_device=SimpleNamespace(shape=(4, 8)), vocab_size=16)
    model.prepare_row_sharded_prefill_iter = MethodType(Model.prepare_row_sharded_prefill_iter, model)
    model.run_row_sharded_prefill_forward = forward
    monkeypatch.setattr(ttnn, "synchronize_device", lambda _: None)
    args = SimpleNamespace(max_local_batch_size=32, vocab_size=16)
    fake_kv = [[SimpleNamespace(shape=(128, 1, 64, 32))]]
    result, _ = Model.row_sharded_batched_prefill(
        model,
        tokens,
        pages,
        fake_kv,
        prompt_lens=lengths,
        prefill_seq_lens=[padded_len] * n,
        enable_trace=False,
        sampling_params=object(),
        model_args=args,
        trace_cache={"ids": {}, "inputs": {}, "outputs": {}},
        empty_slots=slots if route_slots else None,
    )
    torch.testing.assert_close(pages, original_pages)
    assert result.shape == (n, 1)
    torch.testing.assert_close(actual_cache, expected_cache)
