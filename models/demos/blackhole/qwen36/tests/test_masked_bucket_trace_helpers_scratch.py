# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host-only unit tests for the traced masked-bucket prefill's value builders.

These are the pieces that are easy to get subtly wrong (off-by-one in the FIR decode-window
one-hot, a pad page-table entry aliasing block 0, a mask that is one token short) and that a
device test would only surface as a PCC drop. Pure torch, no ttnn, no device:

    python -m pytest models/demos/blackhole/qwen36/tests/test_masked_bucket_trace_helpers_scratch.py \
        -p no:cacheprovider -q
"""
import pytest
import torch

from models.demos.blackhole.qwen36.tt.masked_bucket_trace import (
    fill_pt_row,
    host_conv_sel,
    host_logit_sel,
    host_masks,
    parse_bucket_trace_gate,
)

BUCKETS = (128, 256, 512, 1024, 2048)
BLOCK = 64


# --------------------------------------------------------------------------- masks
@pytest.mark.parametrize("bucket", [128, 256])
@pytest.mark.parametrize("actual_len", [1, 33, 100, 127, 128])
def test_host_masks_ones_below_actual_len(bucket, actual_len):
    actual_len = min(actual_len, bucket)
    m = host_masks(actual_len, bucket)
    assert m.shape == (1, bucket, 1) and m.dtype == torch.float32
    assert torch.equal(m[0, :actual_len, 0], torch.ones(actual_len))
    assert torch.equal(m[0, actual_len:, 0], torch.zeros(bucket - actual_len))


def test_host_masks_all_ones_at_full_bucket():
    """actual_len == bucket must be all ones, i.e. the always-on multiplies are the identity —
    this is what makes the traced (unconditionally masked) body match the unmasked numerics."""
    m = host_masks(128, 128)
    assert float(m.sum()) == 128.0


def test_host_masks_rejects_out_of_range():
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        host_masks(0, 128)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        host_masks(129, 128)


# --------------------------------------------------------------------------- conv one-hot
@pytest.mark.parametrize("bucket", [128, 256])
@pytest.mark.parametrize("actual_len", [1, 3, 33, 127, 128])
@pytest.mark.parametrize("K", [4])
def test_host_conv_sel_one_hot_positions(bucket, actual_len, K):
    actual_len = min(actual_len, bucket)
    sel = host_conv_sel(actual_len, bucket, K)
    assert sel.shape == (1, K - 1, bucket + K - 1)
    for j in range(K - 1):
        row = sel[0, j]
        assert float(row.sum()) == 1.0, "exactly one selected column per output row"
        assert int(torch.argmax(row)) == actual_len + j


def test_host_conv_sel_selects_the_real_tail():
    """x_padded = [conv_state (K-1 rows) | x], so x[i] sits at x_padded index (K-1)+i. Selecting
    x_padded rows actual_len..actual_len+K-2 therefore picks x[actual_len-(K-1)..actual_len-1] —
    the last K-1 REAL tokens, which is the decode conv window."""
    K, bucket, actual_len = 4, 128, 100
    sel = host_conv_sel(actual_len, bucket, K)
    x = torch.arange(bucket, dtype=torch.float32).reshape(1, bucket, 1)
    conv_state = torch.full((1, K - 1, 1), -1.0)
    x_padded = torch.cat([conv_state, x], dim=1)  # [1, bucket+K-1, 1]
    picked = torch.matmul(sel, x_padded).reshape(-1)
    assert torch.equal(picked, torch.tensor([97.0, 98.0, 99.0]))


def test_host_conv_sel_full_bucket_matches_static_slice():
    """At actual_len == bucket the one-hot must select exactly the rows the valid_len-None static
    slice takes (x_padded[-(K-1):]), i.e. the traced body is bit-parity with a full chunk."""
    K, bucket = 4, 128
    sel = host_conv_sel(bucket, bucket, K)
    x_padded = torch.randn(1, bucket + K - 1, 5)
    assert torch.allclose(torch.matmul(sel, x_padded), x_padded[:, bucket:, :])


# --------------------------------------------------------------------------- logit one-hot
@pytest.mark.parametrize("bucket", BUCKETS)
@pytest.mark.parametrize("actual_len", [1, 7, 128])
def test_host_logit_sel(bucket, actual_len):
    actual_len = min(actual_len, bucket)
    sel = host_logit_sel(actual_len, bucket)
    assert sel.shape == (1, 1, 1, bucket)
    assert float(sel.sum()) == 1.0
    assert int(torch.argmax(sel.reshape(-1))) == actual_len - 1


# --------------------------------------------------------------------------- fill page table
@pytest.mark.parametrize("bucket", BUCKETS)
def test_fill_pt_row_width_is_fixed_per_bucket(bucket):
    """The whole point: the width must depend ONLY on the bucket, never on actual_len."""
    pt = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    widths = {tuple(fill_pt_row(pt, 0, n, bucket, 63, BLOCK).shape) for n in (1, 33, bucket // 2, bucket - 1, bucket)}
    assert widths == {(1, bucket // BLOCK)}


def test_fill_pt_row_copies_real_blocks_at_chunk_start_0():
    pt = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    row = fill_pt_row(pt, 0, 100, 128, 63, BLOCK)  # 100 tokens -> 2 real blocks, width 2
    assert torch.equal(row, torch.tensor([[0, 1]], dtype=torch.int32))


def test_fill_pt_row_copies_real_blocks_at_a_tail_offset():
    """A long-prompt tail starts at chunk_start > 0, so the row starts at block chunk_start/64
    (T=4352 = two 2048 chunks + a 256 tail -> chunk_start 4096, block 64)."""
    pt = torch.arange(96, dtype=torch.int32).reshape(1, 96)
    row = fill_pt_row(pt, 4096, 33, 128, 95, BLOCK)  # blk0 = 64, 1 real block, width 2 -> pad with scratch
    assert torch.equal(row, torch.tensor([[64, 95]], dtype=torch.int32))
    row = fill_pt_row(pt, 4096, 33, 128, 95, BLOCK, trust_tail=True)  # rollback form: the row's own next block
    assert torch.equal(row, torch.tensor([[64, 65]], dtype=torch.int32))
    row = fill_pt_row(pt, 2048, 33, 128, 95, BLOCK)  # blk0 = 32
    assert int(row[0, 0]) == 32


def test_fill_pt_row_uses_scratch_block_when_the_row_is_zero_padded():
    """vLLM-shaped row: real blocks then zero padding. Pad entries must NOT alias block 0."""
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :2] = torch.tensor([7, 9], dtype=torch.int32)
    row = fill_pt_row(pt, 0, 65, 256, 63, BLOCK)  # 65 tokens -> 2 real blocks, width 4
    assert torch.equal(row, torch.tensor([[7, 9, 63, 63]], dtype=torch.int32))
    assert 0 not in row.tolist()[0]


def test_fill_pt_row_default_sends_pad_rows_to_the_scratch_block_even_on_a_fully_mapped_row():
    """The demo/test arange page table maps the whole bucket, but the default still pads with the
    scratch block: the row's tail is not trusted for ANY caller (vLLM's tail is stale, see below)."""
    pt = torch.arange(1, 65, dtype=torch.int32).reshape(1, 64)  # all non-zero
    row = fill_pt_row(pt, 0, 65, 256, 63, BLOCK)
    assert torch.equal(row, torch.tensor([[1, 2, 63, 63]], dtype=torch.int32))


def test_fill_pt_row_trust_tail_uses_the_requests_own_mapped_blocks():
    """Rollback form (QWEN36_PREFILL_TRUST_PT_TAIL=1): a fully mapped row fills its own blocks."""
    pt = torch.arange(1, 65, dtype=torch.int32).reshape(1, 64)
    row = fill_pt_row(pt, 0, 65, 256, 63, BLOCK, trust_tail=True)
    assert torch.equal(row, torch.tensor([[1, 2, 3, 4]], dtype=torch.int32))


def test_fill_pt_row_falls_back_to_scratch_past_the_end_of_the_row():
    pt = torch.arange(1, 4, dtype=torch.int32).reshape(1, 3)  # only 3 blocks mapped
    row = fill_pt_row(pt, 0, 129, 256, 63, BLOCK)  # width 4, 3 real blocks
    assert torch.equal(row, torch.tensor([[1, 2, 3, 63]], dtype=torch.int32))


# The row vLLM hands a 1-block prompt right after a [1, 2] request has freed: [2, 2, 0, ...] -- the stale tail entry
# aliases the request's OWN block (chip-1 server dump, profiles/pd/p1b_serve_chip1.log). The old trusted-tail rule
# turned it into the fill table [2, 2]: one paged_fill_cache naming block 2 twice -> the 63/64-token nondeterminism.
STALE_OWN_BLOCK_ROW = torch.tensor([[2, 2] + [0] * 62], dtype=torch.int32)


def test_fill_pt_row_default_never_reads_the_row_past_the_real_blocks():
    """Every pad entry is the scratch block, whatever the stale tail holds (own block, another request's block,
    or zero); the real blocks are copied exactly as before. Same rule for every device count."""
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :3] = torch.tensor([2, 2, 5], dtype=torch.int32)  # 1 real block [2]; stale tail [2, 5]
    assert torch.equal(fill_pt_row(pt, 0, 64, 128, 63, BLOCK), torch.tensor([[2, 63]], dtype=torch.int32))
    assert torch.equal(fill_pt_row(pt, 0, 63, 128, 63, BLOCK), torch.tensor([[2, 63]], dtype=torch.int32))
    assert torch.equal(fill_pt_row(pt, 0, 1, 128, 63, BLOCK), torch.tensor([[2, 63]], dtype=torch.int32))
    # 65..128 tokens: the two real blocks fill the whole width -> nothing to pad, never affected
    pt[0, :3] = torch.tensor([2, 5, 2], dtype=torch.int32)  # 2 real blocks [2, 5]; stale tail [2]
    assert torch.equal(fill_pt_row(pt, 0, 65, 128, 63, BLOCK), torch.tensor([[2, 5]], dtype=torch.int32))
    assert torch.equal(fill_pt_row(pt, 0, 65, 256, 63, BLOCK), torch.tensor([[2, 5, 63, 63]], dtype=torch.int32))
    # long-prompt tail (chunk_start > 0): the stale entries past the tail's real block are ignored too
    pt = torch.arange(1, 97, dtype=torch.int32).reshape(1, 96)
    pt[0, 65] = 65  # stale: the tail's own block repeated at the next index
    assert torch.equal(fill_pt_row(pt, 4096, 33, 128, 95, BLOCK), torch.tensor([[65, 95]], dtype=torch.int32))


def test_fill_pt_row_default_is_deterministic_in_the_stale_own_block_row():
    """The exact vLLM row from the server dump: the fill names block 2 ONCE and pads with the scratch block."""
    for T in (1, 2, 63, 64):
        assert torch.equal(
            fill_pt_row(STALE_OWN_BLOCK_ROW, 0, T, 128, 63, BLOCK), torch.tensor([[2, 63]], dtype=torch.int32)
        )


def test_fill_pt_row_trust_tail_rejects_the_stale_own_block_row():
    """Rollback form on the stale vLLM row: the alias guard refuses to name block 2 twice instead of racing."""
    with pytest.raises(AssertionError, match="pad rows into real block"):  # allow-pytest.raises: host-only guard
        fill_pt_row(STALE_OWN_BLOCK_ROW, 0, 64, 128, 63, BLOCK, trust_tail=True)


def test_fill_pt_row_trust_tail_rejects_a_stale_tail_aliasing_another_requests_block():
    """A stale tail id that is a real block of THIS request elsewhere in the row (or of another request) must never
    receive pad rows; with the guard the rollback form fails loudly."""
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :4] = torch.tensor([7, 9, 9, 7], dtype=torch.int32)  # real [7, 9], stale tail [9, 7]
    with pytest.raises(AssertionError, match="pad rows into real block"):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 65, 256, 63, BLOCK, trust_tail=True)
    # default: the same row is fine
    assert torch.equal(fill_pt_row(pt, 0, 65, 256, 63, BLOCK), torch.tensor([[7, 9, 63, 63]], dtype=torch.int32))


PAD_IN_REAL = "pad block 63 is one of the request's real blocks"


def test_fill_pt_row_rejects_the_pad_block_among_the_real_blocks():
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :2] = torch.tensor([7, 63], dtype=torch.int32)
    with pytest.raises(AssertionError, match=PAD_IN_REAL):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 65, 256, 63, BLOCK)
    with pytest.raises(AssertionError, match=PAD_IN_REAL):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 65, 256, 63, BLOCK, trust_tail=True)


def test_fill_pt_row_allows_the_pad_block_among_the_real_blocks_when_there_are_no_pad_entries():
    """nreal == width: no pad row is written, so the pad block's identity is irrelevant (the batched text_demo maps
    the whole pool incl. the last block -- the default pad block -- to its last user and must keep working)."""
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :2] = torch.tensor([7, 63], dtype=torch.int32)
    for T in (65, 128):
        assert torch.equal(fill_pt_row(pt, 0, T, 128, 63, BLOCK), torch.tensor([[7, 63]], dtype=torch.int32))
    pt[0, :8] = torch.arange(56, 64, dtype=torch.int32)
    assert fill_pt_row(pt, 0, 512, 512, 63, BLOCK).tolist()[0] == list(range(56, 64))
    with pytest.raises(AssertionError, match=PAD_IN_REAL):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 511, 1024, 63, BLOCK)  # 8 real + 8 pad entries in the 1024 bucket -> guard fires


def test_fill_pt_row_rejects_a_real_block_named_twice():
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, :2] = torch.tensor([5, 5], dtype=torch.int32)
    with pytest.raises(AssertionError, match="names a real block twice"):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 65, 256, 63, BLOCK)


def test_fill_pt_row_pad_block_may_repeat():
    """Only the scratch block may appear more than once (pad rows racing pad rows is harmless garbage)."""
    pt = torch.zeros(1, 64, dtype=torch.int32)
    pt[0, 0] = 7
    row = fill_pt_row(pt, 0, 1, 2048, 63, BLOCK)
    assert row.tolist()[0] == [7] + [63] * 31


def test_fill_pt_row_rejects_block_zero_as_scratch():
    pt = torch.arange(64, dtype=torch.int32).reshape(1, 64)
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 33, 128, 0, BLOCK)


def test_fill_pt_row_rejects_an_unmapped_real_block():
    pt = torch.arange(1, 3, dtype=torch.int32).reshape(1, 2)  # 2 blocks
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        fill_pt_row(pt, 0, 200, 256, 63, BLOCK)  # needs 4 real blocks


# --------------------------------------------------------------------------- env gate
def test_parse_bucket_trace_gate_defaults_off():
    for v in (None, "", "0", "off", " 0 "):
        assert parse_bucket_trace_gate(v, BUCKETS) == ()


def test_parse_bucket_trace_gate_all():
    assert parse_bucket_trace_gate("1", BUCKETS) == tuple(sorted(BUCKETS))


def test_parse_bucket_trace_gate_list():
    assert parse_bucket_trace_gate("128", BUCKETS) == (128,)
    assert parse_bucket_trace_gate("256,128", BUCKETS) == (128, 256)
    assert parse_bucket_trace_gate(" 128 , 128 ", BUCKETS) == (128,)


def test_parse_bucket_trace_gate_rejects_unknown_bucket():
    with pytest.raises(AssertionError):  # allow-pytest.raises: host-only guard
        parse_bucket_trace_gate("192", BUCKETS)
