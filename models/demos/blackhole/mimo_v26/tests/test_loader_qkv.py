"""Checks of the qkv layout decision against the study's fp8 qkv samples (no full checkpoint needed)."""

import pytest
import torch

from models.demos.blackhole.mimo_v26.reference.loader import deinterleave_qkv, dequant_fp8_qkv_ranked, qkv_dims

SAMPLES = "/mnt/tt-data/ssinghal/bringup_study/mimo"


@pytest.mark.parametrize("layer", [0, 1])
def test_ranked_scale_layout_is_consistent(layer):
    w, s = torch.load(f"{SAMPLES}/qkv_L{layer}.pt")
    nq, nk, nv = qkv_dims(layer)
    d = dequant_fp8_qkv_ranked(w, s, nq, nk, nv)
    per = d.shape[0] // 4
    rms = d.pow(2).mean(1).sqrt()
    qs, ks = nq // 4, nk // 4
    # every rank shows the same q/k/v magnitude signature (k > q, v smallest or comparable) -> sections are where the rank-blocked layout says
    k_over_q = [
        (rms[r * per + qs : r * per + qs + ks].mean() / rms[r * per : r * per + qs].mean()).item() for r in range(4)
    ]
    assert max(k_over_q) / min(k_over_q) < 1.6, k_over_q
    hf = deinterleave_qkv(d, nq, nk, nv)
    assert hf.shape == d.shape
    # rank 0 q rows land first, rank 1 q rows follow
    assert torch.equal(hf[: qs // 1], d[:qs]) and torch.equal(hf[qs : 2 * qs], d[per : per + qs])
