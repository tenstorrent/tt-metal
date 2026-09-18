"""Regression: the per-user prefill page table must not expose stale block ids beyond the prompt's own blocks.

vLLM pads a 7151-token prompt to an 8192-token prefill (128 blocks of 64) while the request owns only 112 blocks; the
block-table row keeps the ids a previous, longer request left behind. ``paged_fill_cache`` writes the padded positions,
so those ids must become -1 (the kernel's skip sentinel) or live KV of this / another request gets overwritten.
"""
import torch

from models.tt_transformers.tt.generator import Generator


def _kv_cache_stub(block_size):
    return [[torch.empty(1, 1, block_size, 1)]]


def _table(width, fill_from, real):
    # real ids for the owned blocks, then "stale" ids left by an earlier occupant of the row
    return torch.cat([torch.arange(1, real + 1), torch.arange(fill_from, fill_from + width - real)]).to(torch.int32)[
        None
    ]


def test_padded_prefill_masks_stale_blocks_untraced():
    block_size = 64
    table = _table(width=256, fill_from=500, real=112)  # 7151 tokens -> 112 blocks; row width max_num_blocks_per_req
    out = Generator._get_prefill_user_page_table(
        None, table, _kv_cache_stub(block_size), prefill_len=7151, trace_enabled=False, prefill_seq_len=8192
    )
    assert out.shape == (1, 128)
    assert torch.equal(out[0, :112], torch.arange(1, 113, dtype=torch.int32))
    assert (out[0, 112:] == -1).all(), "pad blocks must be skipped, not stale ids"
    assert not (table[0, 112:128] == -1).any(), "the caller's table is not modified"


def test_padded_prefill_masks_stale_blocks_traced():
    block_size = 64
    table = _table(width=256, fill_from=900, real=9)  # 557 tokens -> 9 blocks, traced at 1024 -> 16 blocks
    out = Generator._get_prefill_user_page_table(
        None, table, _kv_cache_stub(block_size), prefill_len=557, trace_enabled=True, prefill_seq_len=1024
    )
    assert out.shape == (1, 16)
    assert torch.equal(out[0, :9], torch.arange(1, 10, dtype=torch.int32))
    assert (out[0, 9:] == -1).all()


def test_exact_fit_and_short_table_unchanged():
    block_size = 64
    table = _table(width=16, fill_from=700, real=16)
    out = Generator._get_prefill_user_page_table(
        None, table, _kv_cache_stub(block_size), prefill_len=1024, prefill_seq_len=1024
    )
    assert torch.equal(out, table)  # 1024 tokens own all 16 blocks: nothing to mask
    short = torch.arange(1, 5, dtype=torch.int32)[
        None
    ]  # table narrower than the padded width: padded with -1 as before
    out = Generator._get_prefill_user_page_table(
        None, short, _kv_cache_stub(block_size), prefill_len=200, prefill_seq_len=1024
    )
    assert out.shape == (1, 16) and torch.equal(out[0, :4], short[0]) and (out[0, 4:] == -1).all()


def test_resumed_chunk_uses_full_prompt_len():
    block_size = 64
    table = _table(width=64, fill_from=800, real=40)
    out = Generator._get_prefill_user_page_table(
        None, table, _kv_cache_stub(block_size), prefill_len=2500, prefill_seq_len=1024, use_full_prompt_len=True
    )
    assert out.shape == (1, 40) and torch.equal(out, table[:, :40])
