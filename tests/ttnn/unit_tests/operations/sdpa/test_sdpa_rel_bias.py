# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0
"""SDPA prefill with a banded relative bias (rel_logits) and a per-query sink, against torch."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import comp_pcc


def reference(q, k, v, rel, sink, scale, window):
    """q [b,h,sq,d], k/v [b,kh,sk,d], rel [b,h,sq,w], sink [b,h,sq,1] or None. Query i is at position i + sk - sq."""
    b, h, sq, _ = q.shape
    sk, w = k.shape[2], rel.shape[-1]
    g = h // k.shape[1]
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    s = (q @ k.transpose(-1, -2)) * scale
    d = (torch.arange(sq).view(sq, 1) + sk - sq) - torch.arange(sk).view(1, sk)  # [sq, sk]
    bias = torch.gather(rel, -1, d.clamp(0, w - 1).expand(b, h, sq, sk))
    s = s + torch.where((d >= 0) & (d < w), bias, torch.zeros(()))
    visible = d >= 0
    if window:
        visible &= d < window
    s = s.masked_fill(~visible, float("-inf"))
    if sink is not None:
        s = torch.cat([s, sink], dim=-1)
    p = torch.softmax(s, dim=-1)
    return p[..., :sk] @ v


@pytest.mark.parametrize(
    "sq, sk, w, window, nh, nkh, qc, kc, scale",
    [
        (256, 256, 128, 128, 4, 2, 128, 128, None),  # local, first chunk, window = band
        (256, 768, 256, 256, 4, 2, 128, 128, 1.0),  # local with history (queries are the last sq keys)
        (256, 1024, 256, None, 4, 2, 128, 128, None),  # global: no window, bias only in the band
        (224, 480, 128, 128, 3, 1, 128, 64, None),  # padded Q and K chunks
        (512, 1536, 1024, 1024, 2, 1, 128, 128, 1.0),  # Inkling local shape (W = 1024), smaller head count
    ],
)
@pytest.mark.parametrize("use_sink", [True, False], ids=["sink", "nosink"])
def test_sdpa_rel_bias(device, sq, sk, w, window, nh, nkh, qc, kc, scale, use_sink):
    torch.manual_seed(0)
    dh = 128
    q = torch.randn(1, nh, sq, dh)
    k = torch.randn(1, nkh, sk, dh)
    v = torch.randn(1, nkh, sk, dh)
    rel = torch.randn(1, nh, sq, w) * 2
    sink = torch.randn(1, nh, sq, 1) * 2 if use_sink else None
    q, k, v, rel = (t.bfloat16().float() for t in (q, k, v, rel))
    if sink is not None:
        sink = sink.bfloat16().float()
    eff_scale = scale if scale is not None else dh**-0.5
    # The kernel scales the sink logit like q.k; give it 1/scale so the reference sink is the raw logit.
    ref = reference(q, k, v, rel, sink, eff_scale, window)

    tt = lambda t, layout=ttnn.TILE_LAYOUT: ttnn.from_torch(  # noqa: E731
        t, dtype=ttnn.bfloat16, layout=layout, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.transformer.scaled_dot_product_attention(
        tt(q),
        tt(k),
        tt(v),
        is_causal=True,
        scale=scale,
        sliding_window_size=window,
        rel_logits=tt(rel, ttnn.ROW_MAJOR_LAYOUT),
        attention_sink=tt(sink / eff_scale) if sink is not None else None,
        program_config=ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
            q_chunk_size=qc,
            k_chunk_size=kc,
            exp_approx_mode=False,
        ),
        compute_kernel_config=ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        ),
    )
    got = ttnn.to_torch(out).float()
    ok, pcc = comp_pcc(ref, got, 0.9999)
    assert ok, f"PCC {pcc}"
