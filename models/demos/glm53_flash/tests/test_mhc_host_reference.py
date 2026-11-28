# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Host (torch-only, no device) check of the batched mHC parametrization math in ``tt/mhc.py``:
``parametrize_graph`` processes every user of a token tile in one op per step (the matmul-form Sinkhorn on the
packed ``[T, n*n]`` rows with the RB / CB selection tiles of ``build_consts``), and must equal the per-user loop
(one ``n x n`` matrix at a time, ``m / (m.sum(...) + eps)`` as HF / the DeepSeek reference write it) up to fp32
rounding: same iteration count, same eps, same first-column-then-(row, column) order, ``pre = sigmoid + eps``,
``post = 2 sigmoid``. The torch backend runs the very op order the ``GLM53_MHC_SINKHORN=batched`` device path
issues, so this also pins that path's op count (131 at iters=20).

Run without the repo conftest (its helpers query the cluster): no fixtures, no ttnn device.

    python_env/bin/python -m pytest --noconftest -p no:cacheprovider \\
        models/demos/glm53_flash/tests/test_mhc_host_reference.py -q
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from models.demos.glm53_flash.tt.mhc import MHCKernelConfig, W, build_consts, parametrize_torch

N = 4
ITERS = 20
EPS = 1e-6
TOL_FP32 = 1e-6  # the two formulations differ by reciprocal-vs-divide rounding only (values in [0, 2])
TOL_FP64 = 1e-12  # in float64 the same math agrees to ~1e-15: the difference IS fp32 rounding


def sinkhorn_loop(logits: torch.Tensor, iters: int, eps: float) -> torch.Tensor:
    """The reference Sinkhorn on ONE ``[n, n]`` matrix (HF ``Glm5NextTextHyperConnection`` / DeepSeek
    ``sinkhorn_knopp``): row softmax + eps, one column normalisation, ``iters - 1`` (row, column) pairs."""
    m = torch.softmax(logits, dim=-1) + eps
    m = m / (m.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
    return m


def loop_reference(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, n: int, iters: int, eps: float):
    """Per-user loop: (pre [T,n], post [T,n], comb [T,n,n]) one user at a time."""
    T = mixes.shape[0]
    pre = torch.sigmoid(mixes[:, :n] * scale[0] + base[:n]) + eps
    post = 2.0 * torch.sigmoid(mixes[:, n : 2 * n] * scale[1] + base[n : 2 * n])
    comb = torch.stack(
        [sinkhorn_loop((mixes[t, 2 * n :] * scale[2] + base[2 * n :]).reshape(n, n), iters, eps) for t in range(T)]
    )
    return pre, post, comb


def _params(T: int, scale_val: float, seed: int, dtype):
    g = torch.Generator().manual_seed(seed)
    mix_hc = (2 + N) * N
    # fp32-exact values (build_consts is fp32 by construction), so the float64 run isolates the formulation
    mixes = (torch.randn(T, mix_hc, generator=g) * 3.0).float()  # wide logits: eps matters
    scale = torch.full((3,), scale_val).float()
    base = torch.randn(mix_hc, generator=g).float()
    return mixes.to(dtype), scale.to(dtype), base.to(dtype)


def _batched(mixes, scale, base, dtype):
    cfg = MHCKernelConfig(dim=64, n=N, sinkhorn_iters=ITERS, eps=EPS, norm_eps=1e-5)
    consts = build_consts(cfg, scale.float(), base.float()).to(dtype)  # [8, 32, 32]
    pre_t, post_t, comb_t, count = parametrize_torch(mixes, consts, ITERS, EPS)
    T = mixes.shape[0]
    return pre_t[:, :N], post_t[:, :N], comb_t[:, : N * N].reshape(T, N, N), (pre_t, post_t, comb_t), count


@pytest.mark.parametrize("T", [1, 4, 32, 640])
@pytest.mark.parametrize("scale_val", [0.01, 1.0])
@pytest.mark.parametrize("dtype,tol", [(torch.float32, TOL_FP32), (torch.float64, TOL_FP64)])
def test_batched_equals_loop(T, scale_val, dtype, tol):
    mixes, scale, base = _params(T, scale_val, seed=T * 7 + int(scale_val * 100), dtype=dtype)
    pre_l, post_l, comb_l = loop_reference(mixes, scale, base, N, ITERS, EPS)
    pre_b, post_b, comb_b, tiles, count = _batched(mixes, scale, base, dtype)
    for name, ref, got in (("pre", pre_l, pre_b), ("post", post_l, post_b), ("comb", comb_l, comb_b)):
        d = (ref.double() - got.double()).abs().max().item()
        assert d <= tol, f"{name}: max |batched - loop| {d:.3e} > {tol} (T={T}, scale={scale_val}, {dtype})"
    # the kernel's tile facts: finite pad columns (identity block of RB / CB), comb's last step is a column
    # normalisation (columns sum to 1 within the eps terms), rows to 1 within ~eps
    for t in tiles:
        assert torch.isfinite(t).all()
    assert torch.allclose(comb_b.sum(dim=-2), torch.ones(T, N, dtype=dtype), atol=1e-4)
    # rows converge only as far as 20 iterations take them at these (wide) logits: a Sinkhorn property, logged
    row_dev = (comb_b.sum(dim=-1) - 1.0).abs().max().item()
    assert row_dev < 0.5, row_dev
    assert count == 6 * (ITERS - 1) + 17 == 131, count


def test_batched_tile_equals_per_user_rows():
    """All users in one tile == each user through the graph alone (the batching is row-local)."""
    T = 32
    mixes, scale, base = _params(T, 1.0, seed=3, dtype=torch.float32)
    pre_b, post_b, comb_b, _, _ = _batched(mixes, scale, base, torch.float32)
    for t in range(T):
        p1, q1, c1, _, _ = _batched(mixes[t : t + 1], scale, base, torch.float32)
        assert (p1 - pre_b[t : t + 1]).abs().max() <= TOL_FP32
        assert (q1 - post_b[t : t + 1]).abs().max() <= TOL_FP32
        assert (c1 - comb_b[t : t + 1]).abs().max() <= TOL_FP32


def test_exp_cap_is_a_no_op_in_range():
    """The kernel caps logits at 80 before exp: identical to the uncapped softmax for every realistic logit."""
    mixes, scale, base = _params(8, 1.0, seed=11, dtype=torch.float64)
    assert (mixes[:, 2 * N :] * scale[2] + base[2 * N :]).abs().max() < 80.0
    _, _, comb_b, _, _ = _batched(mixes, scale, base, torch.float64)
    _, _, comb_l = loop_reference(mixes, scale, base, N, ITERS, EPS)
    assert (comb_b - comb_l).abs().max() <= TOL_FP64


def test_consts_layout_contract():
    """``build_consts`` tile order as the graph reads it (0 SEL_pre .. 5 base_comb, 6 RB, 7 CB)."""
    cfg = MHCKernelConfig(dim=64, n=N, sinkhorn_iters=ITERS, eps=EPS, norm_eps=1e-5)
    scale = torch.tensor([0.5, 0.25, 2.0])
    base = torch.arange(24, dtype=torch.float32)
    k = build_consts(cfg, scale, base)
    assert tuple(k.shape) == (8, W, W)
    assert k[0][:N, :N].diag().tolist() == [0.5] * N and k[0].sum() == 0.5 * N
    assert k[1][N : 2 * N, :N].diag().tolist() == [0.25] * N
    assert k[2][2 * N : 2 * N + N * N, : N * N].diag().tolist() == [2.0] * (N * N)
    assert torch.equal(k[5][0, : N * N], base[2 * N :]) and k[5][:, N * N :].sum() == 0
    rb, cb = k[6], k[7]
    # row-sum broadcast: entry (i, j) sums entries (i, j') ; col-sum broadcast: (i', j)
    for p in range(N * N):
        pi, pj = divmod(p, N)
        assert rb[:, p].nonzero().flatten().tolist() == [pi * N + j for j in range(N)]
        assert cb[:, p].nonzero().flatten().tolist() == [i * N + pj for i in range(N)]
    assert torch.equal(rb[N * N :, N * N :], torch.eye(W - N * N)) and torch.equal(
        cb[N * N :, N * N :], torch.eye(W - N * N)
    )


@pytest.mark.parametrize("T", [1, 4, 8, 16])
def test_mhc_pre_ms1_combined_round1_is_bitwise(T):
    """tt/mhc_pre.py GLM53_MHC_MS1 (kernels/mhc_pre MHC_MS1): round 1 gathers ONE fp32 tile per partial -- M in rows
    0..15, S's rows t at rows 16 + t -- instead of the (M, S) pair. The K core's compute moves S's faces 0 / 1 into faces
    2 / 3 of the M tile in DST (mhc_pre_sfpu.hpp mhc_ms1_merge), leaders / root sum the combined pages in slot order, the
    root's compute moves faces 2 / 3 back and zeroes them (mhc_ms1_split). Emulated on the tiles' face-major memory
    (tile_off), with garbage in the rows nothing reads: rows < T of M / S equal the pair's sums bitwise."""

    def tile_off(r, c):  # kernels/mhc_pre/mhc_pre_common.hpp tile_off
        return (((r >> 4) << 1) + (c >> 4)) * 256 + (r & 15) * 16 + (c & 15)

    def rows(mem, r0, n):  # rows r0 .. r0 + n - 1 of a face-major tile as [n, 32]
        return torch.stack([torch.stack([mem[tile_off(r0 + t, c)] for c in range(32)]) for t in range(n)])

    def chain_sum(tiles):  # sum_tiles: acc = tile 0; acc += tile k in slot order (fp32, elementwise)
        acc = tiles[0].clone()
        for x in tiles[1:]:
            acc = acc + x
        return acc

    g = torch.Generator().manual_seed(1234 + T)
    k_cores, group = 64, 8
    m_parts, s_parts = [], []
    for _ in range(k_cores):
        m, s = torch.randn(1024, generator=g), torch.rand(1024, generator=g) * 7.0
        live = torch.zeros(1024, dtype=torch.bool)
        for t in range(T):
            for c in range(32):
                live[tile_off(t, c)] = True
        # rows >= T: M zero up to row 15 (the row reads' zero fill), anything above; S anything (never read)
        m = torch.where(live, m, torch.zeros_like(m))
        for r in range(16, 32):
            for c in range(32):
                m[tile_off(r, c)] = torch.randn(1, generator=g).item()
        m_parts.append(m)
        s_parts.append(s)
    # the pair: leaders sum their 8 members' M and S, the root sums the 8 group sums
    m_tot = chain_sum([chain_sum(m_parts[group * q : group * q + group]) for q in range(k_cores // group)])
    s_tot = chain_sum([chain_sum(s_parts[group * q : group * q + group]) for q in range(k_cores // group)])
    # MS1: combine on the K core (faces 2 / 3 of M <- faces 0 / 1 of S: elements 512.. <- ..512), the same two-level
    # sums, split on the root (faces 0 / 1 of S <- faces 2 / 3 of the total, faces 2 / 3 of both <- 0)
    comb = []
    for m, s in zip(m_parts, s_parts):
        cmem = m.clone()
        cmem[512:] = s[:512]
        comb.append(cmem)
    c_tot = chain_sum([chain_sum(comb[group * q : group * q + group]) for q in range(k_cores // group)])
    s_split, m_split = torch.zeros(1024), c_tot.clone()
    s_split[:512] = c_tot[512:]
    m_split[512:] = 0.0
    assert torch.equal(rows(m_split, 0, T).view(torch.int32), rows(m_tot, 0, T).view(torch.int32))
    assert torch.equal(rows(s_split, 0, T).view(torch.int32), rows(s_tot, 0, T).view(torch.int32))
    # the face move is the row move of the layout: S's row r < 16 sits at row 16 + r of the combined tile
    assert all(tile_off(16 + r, c) == tile_off(r, c) + 512 for r in range(16) for c in range(32))

    def plane_off(p, t):  # kernels/mhc_pre/mhc_pre_common.hpp plane_off: the SFPU's view of a DST tile
        return (p >> 3) * 256 + (((p & 7) >> 1) * 4 + (t >> 3)) * 16 + 2 * (t & 7) + (p & 1)

    # the kernels move DST planes (mhc_ms1_merge / mhc_ms1_split: plane p + 16 <-> plane p, p < 16): planes 0..15 are
    # exactly elements 0..511 (faces 0 / 1), and plane p + 16 is plane p's elements + 512 -- the element move above
    assert sorted(plane_off(p, t) for p in range(16) for t in range(32)) == list(range(512))
    assert all(plane_off(p + 16, t) == plane_off(p, t) + 512 for p in range(16) for t in range(32))
    # S rows t < T live in faces 2 / 3 only: no row of M below 16 is overwritten
    assert all(tile_off(16 + t, c) >= 512 for t in range(T) for c in range(32))
    # GLM53_MHC_MSG_TRIM (sync.cpp MHC_MSG_TRIM): the root packs pre_s(t) at msg[T s + t] into a mailbox holding stale
    # words and sends only the first 4 T words; a member whose mailbox holds other stale words unpacks msg[T s + t]
    # into CB_PRE (column 0 of row t; the FPU collapse's diagonal) -- the 32-stride layout's values, bitwise
    coef = torch.randn(4, 32, generator=g)
    got = {}
    for trim in (False, True):
        row, sent = (T, 4 * T) if trim else (32, 128)
        root = torch.zeros(128) if not trim else torch.randn(128, generator=g)  # (knob off: the zero-filled page)
        for s_ in range(4):
            root[row * s_ : row * s_ + T] = coef[s_, :T]
        recv = torch.randn(128, generator=g)
        recv[:sent] = root[:sent]
        pre = torch.randn(4, 1024, generator=g)
        for s_ in range(4):
            for t in range(T):
                pre[s_, tile_off(t, 0)] = recv[row * s_ + t]
        got[trim] = torch.stack([pre[s_, tile_off(t, 0)] for s_ in range(4) for t in range(T)])
        assert 4 * T <= sent <= 128
    assert torch.equal(got[True].view(torch.int32), got[False].view(torch.int32))
    assert torch.equal(got[True].view(torch.int32), coef[:, :T].reshape(-1).view(torch.int32))
    repo_root = Path(__file__).resolve().parents[4]
    with open(repo_root / "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/sync.cpp") as f:
        sync_src = f.read()
    # every round-1 writer and reader addresses the message through R1_MSG_ROW (no 32-stride index left)
    assert sync_src.count("msg[R1_MSG_ROW * s + t]") >= 4 and "msg[32 * s + t]" not in sync_src
    assert "s * TILE + t]" not in sync_src
    # and R1_MSG_ROW / R1_MSG_BYTES are the layout modelled above, as the kernel defines them: MHC_MSG_TRIM rows of T
    # and 4 T words sent (N_STREAMS = 4), else rows of 32 and the 128-word mailbox
    assert sync_src.count("#if defined(MHC_MSG_TRIM)\n") == 1
    trim = sync_src[sync_src.index("#if defined(MHC_MSG_TRIM)\n") :]
    trim, full = trim[: trim.index("#else\n")], trim[trim.index("#else\n") : trim.index("#endif\n")]
    assert "    const uint32_t R1_MSG_ROW = T;\n" in trim, trim
    assert "    const uint32_t R1_MSG_BYTES = N_STREAMS * T * 4;\n" in trim, trim
    assert "    constexpr uint32_t R1_MSG_ROW = TILE;\n" in full, full
    assert "    constexpr uint32_t R1_MSG_BYTES = MSG_FLOATS * 4;\n" in full, full
    with open(repo_root / "models/autoports/zai_org_glm_5_3_flash/tt/kernels/mhc_pre/mhc_pre_common.hpp") as f:
        common = f.read()
    assert "constexpr uint32_t N_STREAMS = 4;" in common and "constexpr uint32_t MSG_FLOATS = 128;" in common
