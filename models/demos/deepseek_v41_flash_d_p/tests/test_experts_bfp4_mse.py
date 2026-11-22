# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""EXPERTS_IMPL bfp4_mse / bfp8_down4 (P.27, owner 2026-11-02): tt/experts.py:_bfp4_mse_round picks each 16-value
group's shared exponent on the host (the group's largest binary exponent or one below, least squared error, ties the
largest) and hands ttnn values that its own bfloat4_b packing must hold exactly. Host only (ttnn's host packing, no
device):

- test_mse_packs_losslessly: ttnn.from_torch(bfloat4_b) -> to_torch of the rounded tensor is bit-exact, on random
  groups, crafted ties / saturation, groups where the lower exponent wins, and real expert matrices (2 experts of the
  spec's first layer, as stored [N, K]); the load-time check `_bf4_mse_lossless` agrees.
- test_mse_rule: per group the chosen error is the smaller of the two candidates' (never worse than ttnn's own
  `_bfp4_round`), ties keep the largest exponent, and the lower exponent is chosen on some real groups (16.5 % of e2m1
  values move under bfp4, DS41-196: 0.5 -> 0 and 1.5 -> 2 next to a 4 or 6).
- test_expert_weight_round: the CPU emulation per EXPERTS_IMPL value (bfp8 exact, bfp4 ttnn's, bfp4_mse, bfp8_down4 w2).
"""

import pytest
import torch

import ttnn
from models.demos.deepseek_v41_flash_d_p.tt import experts as E


def _bfp4_roundtrip(t: torch.Tensor) -> torch.Tensor:
    """t through ttnn's host bfloat4_b packing and back (no device)."""
    return ttnn.to_torch(ttnn.from_torch(t, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT)).float()


def _sse(q, x):
    return (q - x).square().reshape(-1, 16).sum(-1)


def _cases():
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(64, 256, generator=g) * 2.0 ** torch.randint(-8, 8, (64, 1), generator=g)
    # e2m1 x 2^s groups: one 6 (or 4) among halves - the bfp4 case DS41-196 names
    e2m1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    idx = torch.randint(0, 8, (64, 256), generator=g)
    sign = torch.where(torch.rand(64, 256, generator=g) < 0.5, -1.0, 1.0)
    fp4 = e2m1[idx] * sign * 2.0 ** torch.randint(-6, 2, (64, 1), generator=g)
    ties = torch.tensor([6.0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 7.5, 0.25, 0.75, -0.5, -1.5, -2.5, 3.0, 1.0, 0.0])
    sat = torch.tensor([15.0, 14.0, 13.0, 1.0] + [0.0] * 12)
    low = torch.tensor([4.0] + [0.5, 1.5, -0.5, 1.5, 0.5, -1.5, 0.5] * 2 + [1.5])  # the lower exponent wins
    return [rnd, fp4, ties.repeat(32, 16), sat.repeat(32, 16), low.repeat(32, 16), torch.zeros(32, 32)]


def _expert_mats():
    """2 experts' stored [N, K] matrices of the spec's first layer (exact in bf16), or [] without a checkpoint."""
    try:
        from models.demos.common.bringup.testing.harness import spec
        from models.demos.deepseek_v41_flash_d_p.bringup import hooks

        S = spec()
        ck = hooks._checkpoint(S)
        layer = int(S.layers()[0])
    except Exception as exc:  # noqa: BLE001 (host test: the random cases still run)
        print(f"no checkpoint ({exc}): random cases only")
        return []
    out = []
    for e in range(2):
        p = ck.expert(layer, e)
        out += [(f"L{layer} e{e} {nm}", E.fp4_dequant(*p[nm])) for nm in E.NAMES]
    return out


@pytest.mark.timeout(1800)
def test_mse_packs_losslessly():
    for i, t in enumerate(_cases()):
        t = t.to(torch.bfloat16).float()
        q = E._bfp4_mse_round(t)
        assert torch.equal(q.to(torch.bfloat16).float(), q), f"case {i}: not bf16-exact"
        assert torch.equal(_bfp4_roundtrip(q), q), f"case {i}: ttnn's bfloat4_b packing re-rounds the bfp4_mse values"
        assert E._bf4_mse_lossless(q), f"case {i}: _bfp4_round(q) != q"
    for name, w in _expert_mats():
        q = E._bfp4_mse_round(w)
        assert torch.equal(_bfp4_roundtrip(q), q), f"{name}: ttnn re-rounds"
        assert E._bf4_mse_lossless(q), name
        frac = float((q != w).float().mean())
        rel = float((q - w).norm() / w.norm())
        rne = E._bfp4_round(w)
        print(
            f"BFP4_MSE {name}: changed {100 * frac:.2f} % (rne {100 * float((rne != w).float().mean()):.2f} %), "
            f"||dW||/||W|| {rel:.4g} (rne {float((rne - w).norm() / w.norm()):.4g})"
        )
        assert rel <= float((rne - w).norm() / w.norm()) + 1e-12, name


@pytest.mark.timeout(1800)
def test_mse_rule():
    lower = groups = 0
    for t in _cases() + [w for _, w in _expert_mats()]:
        x = t.to(torch.bfloat16).float()
        q, rne = E._bfp4_mse_round(x), E._bfp4_round(x)
        g = x.reshape(-1, 16)
        amax = g.abs().amax(-1)
        _, e = torch.frexp(amax)
        u1 = torch.ldexp(torch.ones_like(amax), e - 4)  # the lower candidate's unit
        q1 = ((g / u1[:, None]).round().clamp(-7, 7) * u1[:, None]).reshape(x.shape)
        s, s0, s1 = _sse(q, x), _sse(rne, x), _sse(q1, x)
        assert bool((s <= s0).all()), "bfp4_mse worse than ttnn's own exponent on a group"
        assert torch.equal(s, torch.minimum(s0, s1)), "not the least-squares candidate"
        took_low = (q.reshape(-1, 16) != rne.reshape(-1, 16)).any(-1)
        assert not bool((took_low & (s1 >= s0)).any()), "a tie or a worse lower exponent was taken (ties: the largest)"
        lower += int(took_low.sum())
        groups += took_low.numel()
    print(f"BFP4_MSE lower exponent on {lower} of {groups} groups")
    assert lower > 0


def test_expert_weight_round():
    g = torch.Generator().manual_seed(1)
    w = (torch.randn(64, 256, generator=g) * 0.01).to(torch.bfloat16).float()
    r = {impl: E.expert_weight_round(impl) for impl in E.IMPLS}
    assert all(torch.equal(r["bfp8"][nm](w), w) for nm in E.NAMES)
    assert all(torch.equal(r["bfp4"][nm](w), E._bfp4_round(w)) for nm in E.NAMES)
    assert all(torch.equal(r["bfp4_mse"][nm](w), E._bfp4_mse_round(w)) for nm in E.NAMES)
    assert torch.equal(r["bfp8_down4"]["w1"](w), w) and torch.equal(r["bfp8_down4"]["w3"](w), w)
    assert torch.equal(r["bfp8_down4"]["w2"](w), E._bfp4_mse_round(w))
    assert E.impl_dtypes("bfp8_down4") == (ttnn.bfloat8_b, ttnn.bfloat4_b)
    assert E.impl_dtypes("bfp4_mse") == (ttnn.bfloat4_b, ttnn.bfloat4_b)
