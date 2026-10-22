# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Host-only checks for the shared generator's prefill cache ownership."""

from types import SimpleNamespace

import pytest
import torch

from models.tt_transformers.tt.generator import Generator


@pytest.mark.parametrize("trace_enabled", [False, True])
@pytest.mark.parametrize(
    "prompt_len,padded_len,block_size,stale_tail",
    [
        (32, 128, 32, True),
        (32, 128, 64, True),
        (65, 128, 64, True),
        (128, 128, 64, True),
        (129, 256, 64, True),
        (255, 256, 64, True),
        (1025, 2048, 128, True),
        (65, 128, 64, False),
        (129, 256, 64, False),
    ],
)
def test_padded_prefill_preserves_unowned_cache(prompt_len, padded_len, block_size, stale_tail, trace_enabled):
    """Kernel padding cannot write stale block IDs left by a previous request."""
    owned_blocks = (prompt_len + block_size - 1) // block_size
    kernel_blocks = padded_len // block_size
    table_width = kernel_blocks + 2 if stale_tail else owned_blocks
    pages = torch.full((1, table_width), 127, dtype=torch.int32)
    pages[0, :owned_blocks] = torch.arange(1, owned_blocks + 1)
    original_pages = pages.clone()
    fake_kv = [[SimpleNamespace(shape=(128, 1, block_size, 32))]]

    prepared = Generator._get_prefill_user_page_table(
        None, pages, fake_kv, prompt_len, trace_enabled=trace_enabled, prefill_seq_len=padded_len
    )

    cache = torch.full((128,), -1000)
    writable_pages = prepared[prepared >= 0].long()
    cache[writable_pages] = 42
    expected_cache = torch.full_like(cache, -1000)
    expected_cache[1 : owned_blocks + 1] = 42
    torch.testing.assert_close(cache, expected_cache)
    torch.testing.assert_close(prepared[0, :owned_blocks], pages[0, :owned_blocks])
    assert prepared.shape == (1, kernel_blocks)
    assert torch.all(prepared[0, owned_blocks:] == -1)
    torch.testing.assert_close(pages, original_pages)


@pytest.mark.parametrize("trace_enabled", [False, True])
@pytest.mark.parametrize("cached_tokens,new_tokens", [(128, 65), (2048, 129)])
def test_resumed_prefill_retains_cached_prefix(cached_tokens, new_tokens, trace_enabled):
    """Both traced and eager continuations need the cumulative prompt mapping."""
    block_size = 64
    prompt_len = cached_tokens + new_tokens
    owned_blocks = (prompt_len + block_size - 1) // block_size
    pages = torch.arange(10, 10 + owned_blocks + 8, dtype=torch.int32).reshape(1, -1)
    original_pages = pages.clone()
    fake_kv = [[SimpleNamespace(shape=(128, 1, block_size, 32))]]

    prepared = Generator._get_prefill_user_page_table(
        None,
        pages,
        fake_kv,
        prompt_len,
        trace_enabled=trace_enabled,
        prefill_seq_len=256,
        use_full_prompt_len=True,
    )

    torch.testing.assert_close(prepared, original_pages[:, :owned_blocks])
    cached_blocks = cached_tokens // block_size
    torch.testing.assert_close(prepared[:, :cached_blocks], original_pages[:, :cached_blocks])
    torch.testing.assert_close(pages, original_pages)
