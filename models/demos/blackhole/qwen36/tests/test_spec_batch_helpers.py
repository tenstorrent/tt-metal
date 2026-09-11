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

import pytest
import torch

from models.demos.blackhole.qwen36.tt.gdn.tp import spec_conv_sel, spec_state_blk_idx

# (B, T) pairs the demo actually packs into the 32-row decode tile: B*T <= 32, T = K+1.
SHAPES = [(1, 12), (1, 8), (2, 12), (2, 8), (4, 8), (4, 4), (8, 4), (16, 2), (3, 5)]
KC = 4  # conv kernel size (args.gdn_conv_kernel_size)


def _mi_cases(B, T):
    """mi vectors worth checking: all-zero (the seed), all-last (full acceptance), and mixed."""
    cases = [[0] * B, [T - 1] * B]
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
    """Naive: walk the ring the way the kernel lays it out and find each (user, head)'s block."""
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


@pytest.mark.parametrize("B,T", SHAPES)
@pytest.mark.parametrize("nv", [1, 8])
def test_state_blk_idx_kernel_contract(B, T, nv):
    """The fused op requires idx[h] % (B*nv) == h (head h may only re-target its OWN lane)."""
    bh = B * nv
    for mi in _mi_cases(B, T):
        idx = spec_state_blk_idx(mi, B, nv)
        for h in range(bh):
            assert int(idx[h]) % bh == h, f"idx[{h}]={int(idx[h])} is not in lane {h} (BH={bh})"
            assert 0 <= int(idx[h]) < T * bh, f"idx[{h}]={int(idx[h])} outside the ring"
        # Distinct lanes never collide on a block.
        assert len(set(idx.tolist())) == bh


def test_state_blk_idx_rejects_bad_mi_length(expect_error):
    with expect_error(AssertionError, "need one mi per user"):
        spec_state_blk_idx([0, 1], 4, 8)


# --------------------------------------------------------------------------- #
# spec_conv_sel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("B,T", SHAPES)
@pytest.mark.parametrize("kc", [2, KC, 5])
def test_conv_sel_matches_naive(B, T, kc):
    """Naive: place one 1.0 per output row at the concat column the design names."""
    rows, cols = kc - 1 + T, kc - 1 + 2 * T
    for mi in _mi_cases(B, T):
        got = spec_conv_sel(mi, B, T, kc)
        assert got.dtype == torch.bfloat16
        assert got.shape == (B, rows, cols)

        want = torch.zeros(B, rows, cols, dtype=torch.bfloat16)
        for u in range(B):
            cols_for_u = [mi[u] + 1 + r for r in range(kc - 1)] + [(kc - 1 + T) + j for j in range(T)]
            for r, c in enumerate(cols_for_u):
                want[u, r, c] = 1.0
        torch.testing.assert_close(got.float(), want.float(), rtol=0, atol=0)


@pytest.mark.parametrize("B,T", SHAPES)
@pytest.mark.parametrize("kc", [2, KC, 5])
def test_conv_sel_is_one_hot_and_in_bounds(B, T, kc):
    rows, cols = kc - 1 + T, kc - 1 + 2 * T
    for mi in _mi_cases(B, T):
        sel = spec_conv_sel(mi, B, T, kc).float()
        assert torch.equal(sel.sum(dim=-1), torch.ones(B, rows)), "every output row selects exactly one input row"
        assert set(sel.unique().tolist()) <= {0.0, 1.0}
        # The E_prev half only ever reads rows the last commit kept; the new-qkv half only reads
        # the appended rows. Nothing crosses.
        assert sel[:, : kc - 1, kc - 1 + T :].sum() == 0, "tap rows must come from E_prev"
        assert sel[:, kc - 1 :, : kc - 1 + T].sum() == 0, "new rows must come from the appended qkv"


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
        sel = spec_conv_sel(mi, B, T, kc).float()
        got = torch.bmm(sel, cat)  # [B, kc-1+T, C]
        for u in range(B):
            want = torch.cat([E_prev[u, mi[u] + 1 : mi[u] + kc], qkv_new[u]], dim=0)
            assert want.shape == (kc - 1 + T, C)
            torch.testing.assert_close(got[u], want, rtol=0, atol=0)


@pytest.mark.parametrize("B,T", [(2, 8), (4, 4)])
def test_conv_sel_chained_over_iterations(B, T):
    """Two iterations in a row: feeding E_new back as E_prev keeps every user's register exact.

    This is how the device path runs (the matmul output IS the next _verify_win_buf), so an
    off-by-one in the tap columns would only show up on the SECOND iteration.
    """
    kc, C = KC, 5
    gen = torch.Generator().manual_seed(99)
    # Ground truth: a plain per-user shift register of the last kc-1 accepted inputs.
    reg = [torch.randn(kc - 1, C, generator=gen) for _ in range(B)]
    E_prev = torch.stack([torch.cat([torch.zeros(1, C), reg[u], torch.zeros(T - 1, C)]) for u in range(B)])
    # Seed convention: _verify_win_buf rows [0, kc) hold the live register, and the first replay
    # runs with mi = 0, so it reads rows 1 .. kc-1 — the register.
    mi = [0] * B

    for it in range(2):
        qkv_new = torch.randn(B, T, C, generator=gen)
        sel = spec_conv_sel(mi, B, T, kc).float()
        E_new = torch.bmm(sel, torch.cat([E_prev, qkv_new], dim=1))
        for u in range(B):
            want = torch.cat([reg[u], qkv_new[u]], dim=0)
            torch.testing.assert_close(E_new[u], want, rtol=0, atol=0), f"iteration {it}, user {u}"
        # Host commits: user u accepted through row mi_next[u], so its register becomes the kc-1
        # inputs ending at that row.
        mi_next = [(u * 3 + it) % T for u in range(B)]
        for u in range(B):
            hist = torch.cat([reg[u], qkv_new[u]], dim=0)  # kc-1 old inputs then T new ones
            reg[u] = hist[mi_next[u] + 1 : mi_next[u] + kc]
        E_prev, mi = E_new, mi_next


@pytest.mark.parametrize("kc", [2, KC])
def test_conv_sel_rejects_out_of_range_mi(kc, expect_error):
    with expect_error(AssertionError, "out of range"):
        spec_conv_sel([4], 1, 4, kc)
    with expect_error(AssertionError, "out of range"):
        spec_conv_sel([-1], 1, 4, kc)
    with expect_error(AssertionError, "need one mi per user"):
        spec_conv_sel([0, 0], 1, 4, kc)


def test_helpers_agree_on_the_same_mi():
    """Both selectors must describe the SAME commit: same mi, same user ordering (user-major)."""
    B, T, nv, kc = 4, 4, 8, KC
    mi = [0, 1, 2, 3]
    idx = spec_state_blk_idx(mi, B, nv)
    sel = spec_conv_sel(mi, B, T, kc).float()
    for u in range(B):
        # state: block token-slot recovered from the index == mi[u]
        assert int(idx[u * nv]) // (B * nv) == mi[u]
        # conv: first tap row reads concat row mi[u] + 1
        assert int(sel[u, 0].argmax()) == mi[u] + 1


@pytest.mark.parametrize("B,T", list(itertools.product([1, 2, 4], [2, 4]))[:6])
def test_full_acceptance_is_the_identity_tail(B, T):
    """mi = T-1 (every candidate accepted) must take the LAST kc-1 new rows as the next taps."""
    kc, C = KC, 3
    gen = torch.Generator().manual_seed(7)
    E_prev = torch.randn(B, kc - 1 + T, C, generator=gen)
    qkv_new = torch.randn(B, T, C, generator=gen)
    sel = spec_conv_sel([T - 1] * B, B, T, kc).float()
    E_new = torch.bmm(sel, torch.cat([E_prev, qkv_new], dim=1))
    for u in range(B):
        # rows [0, kc-1) of E_new are E_prev rows [T, T+kc-1) == the last kc-1 rows of E_prev,
        # which the previous iteration filled with its own last kc-1 new inputs.
        torch.testing.assert_close(E_new[u, : kc - 1], E_prev[u, T : T + kc - 1], rtol=0, atol=0)
        torch.testing.assert_close(E_new[u, kc - 1 :], qkv_new[u], rtol=0, atol=0)
