# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for the batched spec-decode selector helpers in gdn/tp.py.

``spec_state_blk_idx`` and ``spec_conv_sel`` are the whole "commit" of the multi-user speculative
decoder: after a verify the host knows mi[u] (the last ACCEPTED candidate row of user u) and,
instead of running a commit phase on device, it hands the next verify replay these two tensors.
Get them wrong and the model silently continues from the wrong state, so they are pinned here
against an independent naive construction. No device, no ttnn ops — pure torch.

Run:
    pytest models/demos/blackhole/qwen36/tests/test_spec_batch_helpers.py -q
"""

import itertools
import types

import pytest
import torch

from models.demos.blackhole.qwen36.demo.text_demo import _spec_batch_decision, _spec_batch_draft_len
from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_state_blk_idx
from models.demos.blackhole.qwen36.tt.model import spec_kv_write_conflicts

# (B, T) pairs _spec_batch_draft_len packs into the 32-row decode tile (B*T <= 32, T = K+1):
# K = 11 for B <= 2, 7 for B <= 4, 3 for B <= 8.
SHAPES = [(1, 12), (2, 12), (4, 8), (8, 4)]
KC = 4  # conv kernel size (args.gdn_conv_kernel_size)


def _mi_cases(B, T):
    """mi vectors worth checking: all-zero (the seed), all-last (full acceptance), mixed, and a Latin
    square (rotations of range(T)) so every (user, mi) pair occurs."""
    cases = [[0] * B, [T - 1] * B, *([(u + s) % T for u in range(B)] for s in range(T))]
    if T > 2:
        cases.append([(u * 3 + 1) % T for u in range(B)])
        cases.append([(T - 1 - u) % T for u in range(B)])
    if B > 1:
        cases.append([0 if u % 2 else T - 1 for u in range(B)])
    return cases


# --------------------------------------------------------------------------- #
# spec_state_blk_idx
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B,T", SHAPES)
@pytest.mark.parametrize("nv", [1, 8, 12])
def test_state_blk_idx_matches_naive(B, T, nv):
    """Naive: walk the ring the way the kernel lays it out and find each (user, head)'s block.

    The fused op also requires idx[h] % (B*nv) == h (head h may only re-target its OWN lane).
    """
    bh = B * nv
    for mi in _mi_cases(B, T):
        got = spec_state_blk_idx(mi, B, nv)

        # Independent construction: enumerate the ring in (token slot, user, head) order and pick
        # the block whose token slot is mi[u] and whose (user, head) is (u, h).
        want = torch.full((B * nv,), -1, dtype=torch.int32)
        blk = 0
        for t in range(T):
            for u in range(B):
                for h in range(nv):
                    if t == mi[u]:
                        want[u * nv + h] = blk
                    blk += 1
        assert (want >= 0).all()

        assert got.dtype == torch.int32
        assert got.shape == (B * nv,)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

        for h in range(bh):
            assert int(got[h]) % bh == h, f"idx[{h}]={int(got[h])} is not in lane {h} (BH={bh})"
            assert 0 <= int(got[h]) < T * bh, f"idx[{h}]={int(got[h])} outside the ring"
        assert len(set(got.tolist())) == bh


def test_state_blk_idx_rejects_bad_mi_length(expect_error):
    with expect_error(AssertionError, "need one mi per user"):
        spec_state_blk_idx([0, 1], 4, 8)


# --------------------------------------------------------------------------- #
# spec_conv_sel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B,T", SHAPES)
@pytest.mark.parametrize("kc", [2, KC])
def test_conv_sel_matmul_rebuilds_the_window(B, T, kc):
    """The load-bearing property: conv_sel @ concat == [E_prev[u, mi+1 : mi+kc] ; qkv_new[u]].

    That is the shift register T sequential decode steps would have left after accepting through
    row mi[u], followed by this iteration's T new conv inputs — i.e. exactly the window the next
    depthwise conv1d must see.
    """
    C = 7  # a stand-in for qkv_dim_tp; the selector is channel-agnostic
    gen = torch.Generator().manual_seed(1234 + B * 100 + T * 10 + kc)
    E_prev = torch.randn(B, kc - 1 + T, C, generator=gen)
    qkv_new = torch.randn(B, T, C, generator=gen)
    cat = torch.cat([E_prev, qkv_new], dim=1)  # [B, kc-1+2T, C]

    for mi in _mi_cases(B, T):
        sel = spec_conv_sel(mi, B, T, kc)
        assert sel.dtype == torch.bfloat16
        assert sel.shape == (B, kc - 1 + T, kc - 1 + 2 * T)
        got = torch.bmm(sel.float(), cat)  # [B, kc-1+T, C]
        for u in range(B):
            want = torch.cat([E_prev[u, mi[u] + 1 : mi[u] + kc], qkv_new[u]], dim=0)
            assert want.shape == (kc - 1 + T, C)
            torch.testing.assert_close(got[u], want, rtol=0, atol=0)


def test_conv_sel_rejects_out_of_range_mi(expect_error):
    with expect_error(AssertionError, "out of range"):
        spec_conv_sel([4], 1, 4, KC)
    with expect_error(AssertionError, "out of range"):
        spec_conv_sel([-1], 1, 4, KC)
    with expect_error(AssertionError, "need one mi per user"):
        spec_conv_sel([0, 0], 1, 4, KC)


# --------------------------------------------------------------------------- #
# auto spec policy (demo/text_demo.py): K >= 3, so B <= 8 in the 32-row verify tile
# --------------------------------------------------------------------------- #
def test_spec_batch_draft_len_floor(monkeypatch):
    monkeypatch.delenv("QWEN36_SPEC_DRAFT_LEN", raising=False)
    for batch, T, sampling in itertools.product(range(1, 9), [128, 4096, 8192], [None, object()]):
        K, _ = _spec_batch_draft_len(batch, T, sampling)
        assert K >= 3 and batch * (K + 1) <= 32 and K in (11, 7, 3), (batch, T, sampling, K)
        if T == 128 and sampling is None:
            assert K == {1: 11, 2: 11, 3: 7, 4: 7, 5: 3, 6: 3, 7: 3, 8: 3}[batch], (batch, K)


def test_spec_batch_decision_max_batch(monkeypatch):
    for name in (
        "QWEN36_SPEC",
        "QWEN35_TEMP",
        "QWEN35_REP_PENALTY",
        "QWEN35_NO_REPEAT_NGRAM",
        "QWEN35_PRESENCE_PENALTY",
    ):
        monkeypatch.delenv(name, raising=False)
    model = types.SimpleNamespace(
        args=types.SimpleNamespace(gdn_nv_tp=12),
        mtp=object(),
        mesh_device=types.SimpleNamespace(compute_with_storage_grid_size=lambda: types.SimpleNamespace(x=11, y=10)),
    )
    for batch in range(1, 9):
        assert _spec_batch_decision(model, batch)[0] is True, batch
    for batch in (9, 12, 16, 32):
        assert _spec_batch_decision(model, batch)[0] is False, batch


# --------------------------------------------------------------------------- #
# grouped verify KV write: which cross-user (block, tile) collisions are real
# --------------------------------------------------------------------------- #
def test_spec_kv_write_conflicts(expect_error):
    BS, T = 64, 2
    # (a) vLLM-like: B=4, nb=8, user u owns blocks [1+2u, 2+2u], every other column is the null
    # block 0. Whole rows overlap (all share 0), but no call writes block 0 -> safe.
    vllm = torch.zeros(4, 8, dtype=torch.int32)
    for u in range(4):
        vllm[u, :2] = torch.tensor([1 + 2 * u, 2 + 2 * u])
    assert set(vllm[0].tolist()) & set(vllm[1].tolist()) == {0}  # the old whole-row check would reject it
    assert spec_kv_write_conflicts(vllm, [3, 60, 64 + 3, 100], 8, BS) == []

    # (b) same block AND same 32-row tile in call j=0 only: user 0 writes 31, 32; user 1 writes 20, 21.
    shared = [[5, 6], [5, 7]]
    assert spec_kv_write_conflicts(shared, [31, 20], T, BS) == [(0, 0, 1, 5, 0)]
    # a third user on the same tile is reported against the first writer in that call
    third = spec_kv_write_conflicts(shared + [[5, 8]], [31, 20, 1], T, BS)
    assert third == [(0, 0, 1, 5, 0), (0, 0, 2, 5, 0), (1, 1, 2, 5, 0)]

    # (c) same block, different tiles -> no race
    assert spec_kv_write_conflicts(shared, [3, 40], T, BS) == []

    # (d) negative position = user not writing this replay
    assert spec_kv_write_conflicts(shared, [3, -1], T, BS) == []
    assert spec_kv_write_conflicts(shared + [[5, 8]], [-1, 3, 7], T, BS) == [(0, 1, 2, 5, 0), (1, 1, 2, 5, 0)]

    # (e) the SAME tile reached in DIFFERENT calls is not a race: user 0 writes 31 (tile 0), 32
    # (tile 1); user 1 writes 63 (tile 1), 64 (block 6). Tile (5, 1) is hit by user 0 in call 1 and
    # by user 1 in call 0, but never by two users in one call.
    assert spec_kv_write_conflicts([[5, 6], [5, 6]], [31, 63], T, BS) == []

    # a write past the page table is an error, not a silent skip
    with expect_error(IndexError, "past its 2-block page table"):
        spec_kv_write_conflicts(shared, [3, 127], T, BS)
