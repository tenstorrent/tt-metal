# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""CPU tests of the model side of the grammar mask (MODEL_MASK_NOTES.md): the mesh sharding of the packed buffer (dims=(0, 1)) against the logits' (row, vocab column) shards, the
bias arithmetic (allowed tokens untouched bit for bit, banned tokens at a huge negative), masked argmax / top-32 candidates against a host reference, and the row layout of the
runner buffer (row r * n + j = mesh row (r * n + j) // T). The packing / permutation helpers themselves are tested in test_grammar_mask_cpu.py."""

import torch

from models.demos.blackhole.deepseek_v41_flash.tt import grammar_mask as gm

ROWS, COLS, V = 4, 8, gm.VOCAB
WORDS = gm.packed_words(V)
NEG = -1.0e30


def pack(allowed):
    n = allowed.shape[-1]
    a = torch.zeros(*allowed.shape[:-1], WORDS * 32, dtype=torch.int64)
    a[..., :n] = allowed.long()
    a = a.reshape(*allowed.shape[:-1], WORDS, 32)
    out = (a << torch.arange(32, dtype=torch.int64)).sum(-1)
    out = torch.where(out >= 2**31, out - 2**32, out)
    return out.to(torch.int32)


def device_bias(shard_words):
    """What ``SpecDecoder.mask_bias`` computes on one device: int32 [T, w] -> fp32 [T, 32 w] (shift by arange(32), and 1, * 1e30 - 1e30)."""
    bits = (shard_words.unsqueeze(-1) >> torch.arange(32, dtype=torch.int32)) & 1
    allowed = bits.reshape(shard_words.shape[0], -1).float()
    return allowed * (-NEG) + NEG


def mesh_shards(buf, T):
    """ShardTensor2dMesh dims=(0, 1): device (r, c) holds buf[r*T:(r+1)*T, c*w:(c+1)*w]."""
    w = buf.shape[1] // COLS
    return {(r, c): buf[r * T : (r + 1) * T, c * w : (c + 1) * w] for r in range(ROWS) for c in range(COLS)}


def test_shards_give_the_vocab_column_of_the_logits():
    T = 6
    g = torch.Generator().manual_seed(1)
    allowed = torch.rand(ROWS * T, V, generator=g) < 0.01
    allowed[:, 0] = True
    buf = pack(allowed)
    sh = mesh_shards(buf, T)
    per = V // COLS
    for (r, c), w in sh.items():
        b = device_bias(w)
        assert b.shape == (T, per)
        ref = torch.where(allowed[r * T : (r + 1) * T, c * per : (c + 1) * per], 0.0, NEG)
        assert torch.equal(b, ref)


def test_bias_leaves_allowed_logits_bit_identical_and_bans_the_rest():
    T = 5
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(T, V // COLS, generator=g) * 30
    allowed = torch.rand(T, V // COLS, generator=g) < 0.05
    allowed[:, 7] = True
    w = pack(allowed.repeat(1, 1))[:, : (V // COLS + 31) // 32]
    b = device_bias(w)[:, : V // COLS]
    out = logits + b
    assert torch.equal(out[allowed], logits[allowed])
    assert bool((out[~allowed] <= -1e29).all())


def test_masked_argmax_and_top32_match_the_host_reference():
    T = 4
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(T, V, generator=g) * 10
    allowed = torch.rand(T, V, generator=g) < 0.002
    allowed[torch.arange(T), torch.tensor([5, 50000, 129279, 77777])] = True
    buf = pack(allowed)
    full = torch.cat([device_bias(buf[:, c * (WORDS // COLS) : (c + 1) * (WORDS // COLS)]) for c in range(COLS)], dim=1)
    masked = logits + full
    ref = torch.where(allowed, logits, torch.full_like(logits, float("-inf")))
    assert torch.equal(masked.argmax(-1), ref.argmax(-1))
    # an allowed token far outside the unmasked top-32 is still the winner when it is the only allowed one
    only = torch.zeros(1, V, dtype=torch.bool)
    only[0, int(logits[0].argmin())] = True
    m1 = logits[:1] + torch.cat([device_bias(pack(only)[:, c * 505 : (c + 1) * 505]) for c in range(COLS)], dim=1)
    assert int(m1.argmax()) == int(logits[0].argmin())


def test_runner_rows_land_on_the_mesh_row_of_their_logits():
    """Runner row r (user), block column j -> buffer row r * n + j, held by mesh row (r * n + j) // T with T = U * n: the same rows as the logits / a_in of that user."""
    U, n = 4, 4
    T = U * n
    B = ROWS * U
    block = torch.full((B, n, WORDS), -1, dtype=torch.int32)
    allowed = torch.zeros(V, dtype=torch.bool)
    allowed[1234] = True
    r_struct = 9  # mesh row 2, local user 1
    block[r_struct, 0] = pack(allowed)
    buf = gm.permute_to_runner(block, [(i, i) for i in range(B)], B, n)
    assert tuple(buf.shape) == (B * n, WORDS)
    rows_not_all_ones = [i for i in range(B * n) if bool((buf[i] != -1).any())]
    assert rows_not_all_ones == [r_struct * n]
    assert rows_not_all_ones[0] // T == r_struct // U == 2
    assert (rows_not_all_ones[0] % T) // n == r_struct % U == 1


def test_adaptive_k_slice_and_bucket_permutation():
    n_plugin, kv = 6, 3
    W = 8
    block = torch.full((W, n_plugin, WORDS), -1, dtype=torch.int32)
    allowed = torch.zeros(V, dtype=torch.bool)
    allowed[[3, 4]] = True
    block[5, 0] = pack(allowed)
    block[5, 4] = pack(allowed)  # beyond the adaptive width: dropped
    # bucket runner of 2 users per mesh row: logical row 5 -> runner row 2
    buf = gm.permute_to_runner(block, [(5, 2), (0, 0)], 8, kv + 1, kv=kv, column0_only=False)
    assert tuple(buf.shape) == (8 * (kv + 1), WORDS)
    changed = [i for i in range(buf.shape[0]) if bool((buf[i] != -1).any())]
    assert changed == [2 * (kv + 1)]
