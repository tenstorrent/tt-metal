# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vendored rope helpers from the WAN 2.2 reference repo.

Ported verbatim (with light cleanup) from
``wan/modules/model.py:rope_params`` and
``wan/modules/s2v/s2v_utils.py:rope_precompute``. The reference repo's
top-level ``wan/__init__.py`` eagerly evaluates ``flash_attn``,
``decord``, and ``torch.cuda.current_device()`` at import time; vendoring
keeps tt_dit free of those runtime dependencies for CPU-only rope
construction.
"""

from __future__ import annotations

import numpy as np
import torch


def rope_params(max_seq_len: int, dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Per-axis rope frequency table.

    Returns a complex tensor of shape ``[max_seq_len, dim // 2]``.
    """
    assert dim % 2 == 0, f"dim must be even, got {dim}"
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    return torch.polar(torch.ones_like(freqs), freqs)


def rope_precompute(
    x: torch.Tensor,
    grid_sizes: list,
    freqs: torch.Tensor | list,
    start: torch.Tensor | None = None,
) -> torch.Tensor:
    """Per-token rope embedding for an extended ``noisy + ref + motion`` sequence.

    Args:
        x: Placeholder of shape ``[B, N_total, num_heads, head_dim]``. Only its
            shape is read; the actual values are discarded.
        grid_sizes: List of ``[start_xyz, end_xyz, range_xyz]`` triples (each a
            torch.long ``[B, 3]`` tensor) describing each contiguous segment's
            time/H/W extent and the rope-frequency range it occupies. See
            :class:`WanModel_S2V.forward` for the canonical construction.
        freqs: Per-axis rope frequency table as built by :func:`rope_params`,
            concatenated as ``[max_seq_len, head_dim // 2]``. Or a 2-element
            list ``[freqs, trainable_freqs]`` for the trainable-token path.
        start: Optional per-batch starting offset; ``None`` uses each segment's
            ``start_xyz``.

    Returns:
        Complex tensor of shape ``[B, N_total, num_heads, head_dim // 2]``.
    """
    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    trainable_freqs = None
    if isinstance(freqs, list):
        trainable_freqs = freqs[1]
        freqs = freqs[0]
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    output = torch.view_as_complex(x.detach().reshape(b, s, n, -1, 2).to(torch.float64))
    seq_bucket = [0]
    if not isinstance(grid_sizes, list):
        grid_sizes = [grid_sizes]
    for g in grid_sizes:
        if not isinstance(g, list):
            g = [torch.zeros_like(g), g]
        batch_size = g[0].shape[0]
        for i in range(batch_size):
            f_o, h_o, w_o = start[i] if start is not None else g[0][i]
            f, h, w = g[1][i]
            t_f, t_h, t_w = g[2][i]
            seq_f, seq_h, seq_w = f - f_o, h - h_o, w - w_o
            seq_len = int(seq_f * seq_h * seq_w)
            if seq_len == 0:
                continue
            if t_f > 0:
                if f_o >= 0:
                    f_sam = np.linspace(f_o.item(), (t_f + f_o).item() - 1, seq_f).astype(int).tolist()
                else:
                    f_sam = np.linspace(-f_o.item(), (-t_f - f_o).item() + 1, seq_f).astype(int).tolist()
                h_sam = np.linspace(h_o.item(), (t_h + h_o).item() - 1, seq_h).astype(int).tolist()
                w_sam = np.linspace(w_o.item(), (t_w + w_o).item() - 1, seq_w).astype(int).tolist()

                assert f_o * f >= 0 and h_o * h >= 0 and w_o * w >= 0
                freqs_0 = freqs[0][f_sam] if f_o >= 0 else freqs[0][f_sam].conj()
                freqs_0 = freqs_0.view(seq_f, 1, 1, -1)

                freqs_i = torch.cat(
                    [
                        freqs_0.expand(seq_f, seq_h, seq_w, -1),
                        freqs[1][h_sam].view(1, seq_h, 1, -1).expand(seq_f, seq_h, seq_w, -1),
                        freqs[2][w_sam].view(1, 1, seq_w, -1).expand(seq_f, seq_h, seq_w, -1),
                    ],
                    dim=-1,
                ).reshape(seq_len, 1, -1)
            elif t_f < 0:
                assert trainable_freqs is not None, "trainable_freqs required for negative t_f"
                freqs_i = trainable_freqs.unsqueeze(1)
            else:
                continue
            output[i, seq_bucket[-1] : seq_bucket[-1] + seq_len] = freqs_i
        seq_bucket.append(seq_bucket[-1] + seq_len)
    return output
