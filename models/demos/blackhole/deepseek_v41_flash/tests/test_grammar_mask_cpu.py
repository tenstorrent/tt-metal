# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of tt/grammar_mask.py: packed-bit unpack (incl. tail bits), the runner row permutation, adaptive-k slicing, empty-row refusal, and the mask
application against an independent per-token host reference."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import grammar_mask as gm

V = gm.VOCAB
WORDS = gm.packed_words(V)


def pack(allowed: torch.Tensor, words: int = WORDS) -> torch.Tensor:
    """Independent packer: bool [..., n_tokens] -> int32 [..., words] (bit i of word w = token 32 w + i), padding bits 0."""
    n = allowed.shape[-1]
    out = torch.zeros(*allowed.shape[:-1], words, dtype=torch.int64)
    for t in range(n):
        out[..., t // 32] |= allowed[..., t].long() << (t % 32)
    out = torch.where(out >= 2**31, out - 2**32, out)
    return out.to(torch.int32)


def test_words_are_exact_for_the_vocabulary_and_shards():
    assert WORDS == 4040
    assert V % (32 * 8) == 0 and (V // 8) // 32 == 505  # every vocab column shard holds whole words


def test_unpack_matches_per_token_reference():
    g = torch.Generator().manual_seed(0)
    allowed = torch.rand(5, V, generator=g) < 0.3
    got = gm.unpack_mask(pack(allowed), V)
    assert got.shape == (5, V) and torch.equal(got, allowed)


def test_unpack_all_ones_and_sign_bit():
    ones = torch.full((2, WORDS), -1, dtype=torch.int32)
    assert gm.unpack_mask(ones, V).all()
    one = torch.zeros(1, WORDS, dtype=torch.int32)
    one[0, 7] = -(2**31)  # only bit 31 of word 7: token 7 * 32 + 31
    got = gm.unpack_mask(one, V)
    assert got.sum() == 1 and bool(got[0, 7 * 32 + 31])


def test_tail_bits_above_vocab_are_dropped(expect_error):
    vocab = 100  # 4 words = 128 bits, 28 tail bits
    packed = torch.full((1, 4), -1, dtype=torch.int32)
    got = gm.unpack_mask(packed, vocab)
    assert got.shape == (1, vocab) and got.all()
    only_tail = torch.zeros(1, 4, dtype=torch.int32)
    only_tail[0, 3] = -1 << 4  # bits 4..31 of word 3 = tokens 100..127
    assert not gm.unpack_mask(only_tail, vocab).any()
    with expect_error(ValueError, "allow no token"):
        gm.check_rows_allow_something(only_tail, vocab)


def test_unpack_rejects_wrong_dtype_or_width(expect_error):
    with expect_error(TypeError, "int32"):
        gm.unpack_mask(torch.zeros(1, WORDS, dtype=torch.int64), V)
    with expect_error(ValueError, "holds"):
        gm.unpack_mask(torch.zeros(1, WORDS - 1, dtype=torch.int32), V)


def test_apply_matches_host_reference_and_argmax():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(6, V, generator=g)
    allowed = torch.rand(6, V, generator=g) < 0.01
    allowed[:, 5] = True
    masked = gm.apply_mask_reference(logits, pack(allowed))
    ref = logits.clone()
    ref[~allowed] = float("-inf")
    assert torch.equal(masked, ref)
    # all-ones mask: identical argmax and values
    same = gm.apply_mask_reference(logits, torch.full((6, WORDS), -1, dtype=torch.int32))
    assert torch.equal(same, logits) and torch.equal(same.argmax(-1), logits.argmax(-1))
    # an allowed token outside the unmasked top-32 wins
    order = logits[0].argsort(descending=True)
    far = int(order[5000])
    only = torch.zeros(1, V, dtype=torch.bool)
    only[0, far] = True
    assert int(gm.apply_mask_reference(logits[:1], pack(only)).argmax(-1)) == far


def _block(W, width, structured):
    """int32 [W, width, WORDS], all ones except column 0 of the rows in ``structured`` {row: allowed token list}."""
    b = torch.full((W, width, WORDS), -1, dtype=torch.int32)
    for i, toks in structured.items():
        a = torch.zeros(V, dtype=torch.bool)
        a[toks] = True
        b[i, 0] = pack(a)
    return b


def test_permute_follows_spec_buckets_row_map():
    # logical rows 0, 2, 5 live; the bucket runner holds users in phys order [9, 4, 7, 0]: logical row i is user p, runner row = index of p in phys
    W, n = 8, 4
    block = _block(W, n, {2: [11, 12], 5: [7]})
    phys = [9, 4, 7, 0]
    users = {0: 4, 2: 7, 5: 0}
    rows = [(i, phys.index(p)) for i, p in users.items()]
    out = gm.permute_to_runner(block, rows, runner_rows=4, n=n)
    assert out.shape == (4 * n, WORDS)
    unp = gm.unpack_mask(out, V).reshape(4, n, V)
    assert unp[phys.index(7), 0].nonzero().flatten().tolist() == [11, 12]
    assert unp[phys.index(0), 0].nonzero().flatten().tolist() == [7]
    assert (
        unp[phys.index(4), 0].all() and unp[0].all() and unp[:, 1:].all()
    )  # unstructured rows, idle user 9, columns >= 1: all ones
    assert gm.has_structured_row(block, [0, 2, 5]) and not gm.has_structured_row(block, [0])


def test_permute_full_width_identity_rows_and_user_major_flatten():
    W, n = 4, 3
    block = _block(W, n, {3: [100]})
    out = gm.permute_to_runner(block, [(i, i) for i in range(W)], runner_rows=W, n=n)
    assert out[3 * n].ne(-1).any() and out[3 * n + 1].eq(-1).all() and out[2 * n].eq(-1).all()


def test_column0_only_vs_all_columns():
    W, n = 2, 3
    block = _block(W, n, {0: [1]})
    a = torch.zeros(V, dtype=torch.bool)
    a[[2, 3]] = True
    block[0, 1] = pack(a)
    only0 = gm.permute_to_runner(block, [(0, 0)], 2, n)
    assert only0[1].eq(-1).all()
    every = gm.permute_to_runner(block, [(0, 0)], 2, n, column0_only=False)
    assert gm.unpack_mask(every[1], V).nonzero().flatten().tolist() == [2, 3]


def test_adaptive_k_slices_to_the_verified_width(expect_error):
    W, K = 3, 5
    block = _block(W, K + 1, {1: [4]})
    for kv in (0, 2, 5):
        out = gm.permute_to_runner(block, [(0, 0), (1, 1)], runner_rows=3, n=kv + 1, kv=kv)
        assert out.shape == ((3) * (kv + 1), WORDS)
        assert gm.unpack_mask(out[1 * (kv + 1)], V).nonzero().flatten().tolist() == [4]
    with expect_error(ValueError, "kv \\+ 1"):
        gm.permute_to_runner(block, [(0, 0)], 3, n=4, kv=5)
    with expect_error(ValueError, "columns"):
        gm.permute_to_runner(block[:, :2], [(0, 0)], 3, n=4)


def test_empty_row_and_bad_mapping_raise(expect_error):
    block = _block(2, 2, {1: [3]})
    block[1, 0] = 0  # grammar that allows nothing
    with expect_error(ValueError, "allow no token"):
        gm.permute_to_runner(block, [(0, 0), (1, 1)], 2, 2)
    ok = _block(2, 2, {})
    with expect_error(ValueError, "twice"):
        gm.permute_to_runner(ok, [(0, 0), (1, 0)], 2, 2)
    with expect_error(ValueError, "outside"):
        gm.permute_to_runner(ok, [(0, 5)], 2, 2)
    with expect_error(ValueError, "outside"):
        gm.permute_to_runner(ok, [(7, 0)], 2, 2)
