# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""CPU reference for the V2A split-K cross-attention in test_transformer_ltx._v2a_split_k.

Replays the device op sequence per SP device (gather audio Q, attend to local video keys, merge
the partial softmax sums by a shared row max and a reduce-scatter) and checks it against dense
attention. No device needed.
"""

import pytest
import torch

SP = 8
AUDIO_N = 256  # 151 real audio tokens at 145 frames, padded to 32 * SP
AUDIO_HEAD_DIM = 64
STAGE_N = {"stage_1": (9690, 9728), "stage_2": (38760, 38912)}  # (real, SP-padded) video tokens


def _f32(x):
    return x


def _bf16(x):
    return x.bfloat16().float()


def v2a_split_k_reference(q, k, v, *, n_real, sp=SP, rnd=_f32):
    """Split-K V2A attention, one op per line in the order ``_v2a_split_k`` runs them on device.

    q is (1, H, Nq, d), k/v are (1, H, Nk, d) with Nk SP-padded; rows past ``n_real`` are masked
    by the -1e9 key bias. ``rnd`` rounds every op output (``_bf16`` emulates bf16 tensors with
    fp32 accumulation inside each op).
    """
    nk_local = k.shape[2] // sp
    nq_local = q.shape[2] // sp
    bias = torch.zeros(k.shape[2])
    bias[n_real:] = -1e9
    q_all = rnd(q * q.shape[3] ** -0.5)
    parts = []
    for dev in range(sp):
        ks = slice(dev * nk_local, (dev + 1) * nk_local)
        s = rnd(rnd(q_all @ k[:, :, ks].transpose(-1, -2)) + rnd(bias[ks]))
        m = s.amax(dim=3, keepdim=True)
        p = rnd(torch.exp(rnd(s - m)))
        l = rnd(p.sum(dim=3, keepdim=True))
        o = rnd(p @ v[:, :, ks])
        parts.append((m, l, o))
    m_glob = torch.stack([m for m, _, _ in parts]).amax(dim=0)
    packed = []
    for m, l, o in parts:
        rescale = rnd(torch.exp(rnd(m - m_glob)))
        packed.append(torch.cat([rnd(o * rescale), rnd(l * rescale)], dim=3))
    # reduce-scatter along Q rows: device i keeps rows [i * nq_local, (i + 1) * nq_local) of the sum.
    summed = torch.stack(packed).sum(dim=0)
    d = q.shape[3]
    out = torch.cat([rnd(summed[:, :, i * nq_local : (i + 1) * nq_local, :d]) for i in range(sp)], dim=2) / torch.cat(
        [rnd(summed[:, :, i * nq_local : (i + 1) * nq_local, d:]) for i in range(sp)], dim=2
    )
    return rnd(out)


def _dense(q, k, v, n_real):
    return torch.nn.functional.scaled_dot_product_attention(
        q.double(), k[:, :, :n_real].double(), v[:, :, :n_real].double()
    ).float()


def _rel_l2(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm()).item()


def _inputs(n_pad, heads, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(1, heads, AUDIO_N, AUDIO_HEAD_DIM, generator=g)
    k = torch.randn(1, heads, n_pad, AUDIO_HEAD_DIM, generator=g)
    v = torch.randn(1, heads, n_pad, AUDIO_HEAD_DIM, generator=g)
    return q, k, v


@pytest.mark.parametrize("stage", list(STAGE_N))
def test_v2a_split_k_matches_dense(stage):
    n_real, n_pad = STAGE_N[stage]
    q, k, v = _inputs(n_pad, heads=2)
    out = v2a_split_k_reference(q, k, v, n_real=n_real)
    assert _rel_l2(out, _dense(q, k, v, n_real)) < 1e-5


def test_v2a_split_k_all_padding_shard():
    """A device holding only padded keys must contribute nothing to the merged output."""
    n_pad = 32 * SP * 4
    n_real = n_pad - n_pad // SP - 32  # the last device's keys are all padding, the one before partly
    q, k, v = _inputs(n_pad, heads=2, seed=1)
    v[:, :, n_real:] = 1e3  # padding V would dominate the output if it leaked in
    out = v2a_split_k_reference(q, k, v, n_real=n_real)
    assert _rel_l2(out, _dense(q, k, v, n_real)) < 1e-5


def test_v2a_split_k_large_logits():
    """Logits far apart across devices: the shared-max rescale must keep exp() in range."""
    n_real, n_pad = STAGE_N["stage_1"]
    q, k, v = _inputs(n_pad, heads=2, seed=2)
    k[:, :, : n_pad // SP] *= 30  # device 0's logits dwarf the others', exp() would overflow unscaled
    out = v2a_split_k_reference(q, k, v, n_real=n_real)
    assert torch.isfinite(out).all()
    assert _rel_l2(out, _dense(q, k, v, n_real)) < 1e-5


@pytest.mark.parametrize("stage", list(STAGE_N))
def test_v2a_split_k_bf16_error(stage):
    """bf16 split-K error against exact attention is within 2x of bf16 dense attention's own error."""
    n_real, n_pad = STAGE_N[stage]
    q, k, v = (_bf16(t) for t in _inputs(n_pad, heads=2, seed=3))
    exact = _dense(q, k, v, n_real)
    split_err = _rel_l2(v2a_split_k_reference(q, k, v, n_real=n_real, rnd=_bf16), exact)
    dense_err = _rel_l2(v2a_split_k_reference(q, k, v, n_real=n_real, sp=1, rnd=_bf16), exact)
    print(f"{stage}: bf16 rel_l2 vs exact: split-K sp={SP} {split_err:.3g}, single-device {dense_err:.3g}")
    assert split_err < 2 * dense_err and split_err < 2e-2
