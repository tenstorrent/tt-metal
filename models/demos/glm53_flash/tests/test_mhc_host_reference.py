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
