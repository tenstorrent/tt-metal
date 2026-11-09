# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""On-device: gdn_spec_step K4 (lazy) with QWEN36_SPEC_ADAPTIVE_K's alternating T.

One ring / window pair sized for T_alloc = 6 serves verifies of T = 6 (the wide family, mi_max 0) and T = 4 (the base
family, mi_max = 5). After a switch the ctrl page's word 0 carries the previous verify's T in bits 8..15, so the op
replays the stash rows that verify wrote at (u * T_prev) % 32, and a carried mi up to T_alloc - 1 is a real prefix
(FRESH = T_alloc). The sequence is checked against the fp64 serial recurrence from the true committed state, like
test_lazy_sequence: outputs PCC > 0.999, ring slot 0 PCC > 0.9999, the stash written at this verify's rows.

Needs the r12h build (gdn_spec_step mi_max + the GDN_MI_MAX lazy reader / writer). Single chip.
Run: pytest tests/ttnn/unit_tests/operations/experimental/kda/test_gdn_spec_step_adaptive_lazy.py -s
"""

import pytest
import torch

import ttnn
from tests.ttnn.unit_tests.operations.experimental.kda.test_gdn_spec_step import (
    GEOS,
    K,
    LAZY_OPTS,
    SCALE,
    Dk,
    Dv,
    LazyCase,
    _ctrl_lazy,
    _stash_host,
    make_rows,
    pcc,
    reference_lazy,
    round_up32,
)

T_ALLOC = 6


def _run(c, rows, mi_ctrl, T, prev_T, vb):
    g, B = c.g, c.B
    dev = lambda t, dt, lay=ttnn.TILE_LAYOUT: ttnn.from_torch(t, dtype=dt, layout=lay, device=c.device)  # noqa: E731
    page = _ctrl_lazy(g, B, mi_ctrl, c.par).clone()
    if prev_T and prev_T != T:
        page.view(-1)[0] = int(page.view(-1)[0]) | (int(prev_T) << 8)
    ctrl = dev(page, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
    qkv = dev(rows.reshape(1, rows.shape[0], g.W), ttnn.bfloat16)
    extra = {"mi_max": T_ALLOC - 1} if T < T_ALLOC else {}
    return ttnn.experimental.kda.gdn_spec_step(
        qkv,
        c.win_a,
        c.win_b,
        c.ring,
        ctrl,
        c.taps_d,
        c.dtb_d,
        c.nea_d,
        c.w_d,
        g.Nv,
        g.Nk,
        Dk,
        Dv,
        T,
        B,
        g.QKVZ,
        conv_kernel=K,
        scale=SCALE,
        hnew_depth=2,
        memory_config=ttnn.L1_MEMORY_CONFIG,
        output_dtype=c.out_dtype,
        fast_mode=True,
        fp32_state=True,
        value_blocks=vb,
        kernel_opts=LAZY_OPTS,
        **extra,
    )


@pytest.mark.parametrize("vb", [1, 4])
@pytest.mark.parametrize("sched", [(6, 4, 4, 6, 4, 6, 6, 4), (4, 6, 4, 6, 4, 4, 6, 6)])
def test_lazy_alternating_T(device, request, sched, vb):
    g = GEOS[0]
    B = 4  # the wide family's width (B * T_alloc = 24 rows); the base verifies of the same rows use T = 4
    c = LazyCase(device, g, B, T_ALLOC, seed=4242)  # ring [T_alloc * B * Nv], window sized for T_alloc
    S_true = c.S0.double()
    E_prev = c.E_prev
    mi_ctrl = [T_ALLOC] * B  # FRESH under adaptive K
    gen = torch.Generator().manual_seed(9)
    prev_T = None
    Nv = g.Nv
    for step, T in enumerate(sched):
        rows = make_rows(g, round_up32(B * T))
        mi_win = [0 if m >= T_ALLOC else m for m in mi_ctrl]
        out_ref, states, deltas, kns, decs, Wn_ref = reference_lazy(
            g, rows, E_prev, mi_win, S_true, c.taps, c.dtb, c.nea, c.w, B, T
        )
        o = _run(c, rows, mi_ctrl, T, prev_T, vb)
        out = ttnn.to_torch(o).reshape(-1, g.VD).float()[: B * T]
        ttnn.deallocate(o)
        ring = ttnn.to_torch(c.ring).reshape(T_ALLOC, B, Nv, Dk, Dv).float()
        win_next = ttnn.to_torch([c.win_a, c.win_b][1 - c.par]).float()
        p_out, p_s0 = pcc(out, out_ref), pcc(ring[0], S_true)
        kt_d, u_d, dec_d = _stash_host(ring[1 + (1 - c.par)], B, Nv, T)
        p_u, p_k = pcc(u_d, deltas), pcc(kt_d, kns)
        Lw = K - 1 + T
        for u in range(B):
            assert torch.equal(win_next[u, :Lw], Wn_ref[u]), f"step {step} (T={T}, prev {prev_T}) user {u}: window rows"
        print(
            f"adaptive lazy step {step} T={T} prev_T={prev_T} vb={vb} mi={mi_ctrl}: out pcc={p_out:.7f} slot0 pcc={p_s0:.8f}"
            f" stash U pcc={p_u:.8f} K pcc={p_k:.8f}"
        )
        assert torch.isfinite(out).all()
        assert p_out > 0.999 and p_s0 > 0.9999 and p_u > 0.9999 and p_k > 0.9999, (step, T, prev_T)
        mi_next = [int(x) for x in torch.randint(0, T, (B,), generator=gen)]
        S_true = torch.stack([states[mi_next[u], u] for u in range(B)])
        E_prev = win_next
        mi_ctrl = mi_next
        c.par = 1 - c.par
        prev_T = T


def test_lazy_flag_off_word0_unchanged(device):
    """mi_max 0 and a parity-only word 0 (no prev-T bits): the r12h op is prod's op -- same output bits as a second run of
    the same inputs through the plain K4 call (LazyCase.run)."""
    g = GEOS[0]
    B, T = 8, 4
    torch.manual_seed(0)
    rows = make_rows(g, round_up32(B * T))
    outs = []
    for _ in range(2):
        c = LazyCase(device, g, B, T, seed=77)
        o = c.run(rows, [T] * B, value_blocks=4)
        outs.append((ttnn.to_torch(o).float(), ttnn.to_torch(c.ring).float()))
        ttnn.deallocate(o)
    assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1][: B * g.Nv], outs[1][1][: B * g.Nv])
