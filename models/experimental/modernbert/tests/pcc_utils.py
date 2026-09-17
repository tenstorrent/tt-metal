# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""PCC helpers. Accumulates in float64: float32 on near-identical tensors returns
values slightly above 1.0, which makes thresholds ambiguous."""

import torch


def pcc(a, b):
    """Pearson correlation between two tensors, flattened."""
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt((a * a).sum()) * torch.sqrt((b * b).sum())
    if denom == 0:
        return float("nan")
    return float((a * b).sum() / denom)


def max_abs_err(a, b):
    return float((a.detach().float() - b.detach().float()).abs().max())


def outlier_report(x, top_k=5):
    """Per-channel max |x|. ModernBERT develops extreme channel-localised outliers
    from layer 16 onward (max|x| ~34000 vs a median channel max of ~34), which
    dominates whole-tensor PCC. Reporting this alongside PCC keeps the metric
    interpretable instead of mysterious.
    """
    v = x.detach().float().reshape(-1, x.shape[-1])
    per_ch = v.abs().max(dim=0).values
    top = torch.topk(per_ch, min(top_k, per_ch.numel()))
    return {
        "max_abs": float(per_ch.max()),
        "median_channel_max": float(per_ch.median()),
        "top_channels": top.indices.tolist(),
        "top_values": [round(v, 2) for v in top.values.tolist()],
    }
