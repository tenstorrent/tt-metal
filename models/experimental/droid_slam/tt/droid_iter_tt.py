# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Iterative DROID-SLAM NN forward wrapper around the on-device port.

Mirrors :class:`models.experimental.droid_slam.reference.droid_iter_ref.DroidNetIter`
but runs the convolutional work on a p150a via
:class:`models.experimental.droid_slam.tt.droid_net_tt.TtDroidNet`.
Correlation (all-pairs + pyramid + bilinear lookup) stays on CPU —
the op set is cheap (small spatial) and there's no tt-nn equivalent
of ``F.grid_sample`` for the radius-3 window.

Each outer call does:
    1. fnet + cnet on device via ``TtDroidNet.extract_features``. The
       returned ``fmaps`` is a ``_LazyTensor`` — we materialize it
       once per call for the correlation volume build.
    2. Build a 4-level correlation pyramid on host.
    3. Loop ``num_iters`` times:
         * sample corr pyramid at current ``coords1``
         * assemble motion tensor
         * call ``TtDroidNet.update`` → ``delta`` (materialized into
           torch to advance ``coords1`` for the next iter)
    4. Return the final-iter tensors (delta/weight/eta/upmask) as
       torch tensors plus fmaps.

Unlike the single-step benchmark the per-iteration correlation
tensor changes every iter, so the ttnn trace-capture path inside
``TtDroidNet.update`` will always miss — that's fine, the iterative
path is latency-limited by device compute, not by Python dispatch.
"""

from __future__ import annotations

import torch

from models.experimental.droid_slam.reference.corr_ref import CorrBlock, coords_grid
from models.experimental.droid_slam.tt.droid_net_tt import TtDroidNet


def _materialize(t):
    """_LazyTensor returns a torch.Tensor from its deferred ttnn.to_torch."""
    return t._materialize() if hasattr(t, "_materialize") else t


@torch.no_grad()
def iter_forward(
    tt_model: TtDroidNet,
    images: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    num_iters: int = 12,
    num_levels: int = 4,
    radius: int = 3,
) -> dict:
    """Run extract_features + ``num_iters`` UpdateModule steps on device.

    Args:
        tt_model: Pre-built :class:`TtDroidNet`.
        images: ``(B, NK, 3, H, W)`` BGR 0..255 torch tensor.
        ii, jj: edge source/destination keyframe indices (long, shape ``(E,)``).
        num_iters: GRU iterations per forward (upstream default 12).
        num_levels: corr pyramid levels (upstream default 4).
        radius: corr lookup window half-extent (upstream default 3).

    Returns:
        Dict with materialized torch tensors for ``fmaps``, ``net``,
        ``delta``, ``weight``, ``eta``, ``upmask``. Shapes match
        the pure-torch reference.
    """
    b, nk, _, h, w = images.shape
    device = images.device
    ht8, wd8 = h // 8, w // 8

    fmaps_lazy, _, _ = tt_model.extract_features(images)
    fmaps = _materialize(fmaps_lazy)  # (B, NK, 128, ht8, wd8) torch

    # Correlation pyramid on host — cheap at 30x40 spatial.
    corr_fn = CorrBlock(fmaps[:, ii], fmaps[:, jj], num_levels=num_levels, radius=radius)

    # net/inp into UpdateModule: the tt_model reuses its on-device
    # cnet cache, so the torch tensors we hand in only need to match
    # shapes (see TtDroidNet.update's fast-path). Using empty tensors
    # is intentional and safe.
    n_edges = ii.shape[0]
    net_shell = torch.empty((b, nk, 128, ht8, wd8), dtype=torch.float32)
    inp_shell = torch.empty((b, nk, 128, ht8, wd8), dtype=torch.float32)

    coords0 = coords_grid(ht8, wd8, device=device, dtype=fmaps.dtype)
    coords1 = coords0[None, None].expand(b, n_edges, ht8, wd8, 2).contiguous()
    target = coords1.clone()

    last = None
    for _ in range(num_iters):
        corr = corr_fn(coords1)
        resd = target - coords1
        flow = coords1 - coords0[None, None]
        motion = torch.cat([flow, resd], dim=-1).permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

        net_ii = net_shell[:, ii]
        inp_ii = inp_shell[:, ii]
        net_out_l, delta_l, weight_l, eta_l, upmask_l = tt_model.update(
            net_ii, inp_ii, corr, motion, ii
        )
        # Force download of every output before the next inner iteration.
        # If we leave them as _LazyTensor, subsequent tt.update calls that
        # trigger trace capture or replay will overwrite the backing L1
        # buffers — materializing here copies the data out while it is
        # still valid.
        net_t = _materialize(net_out_l)
        delta_t = _materialize(delta_l)
        weight_t = _materialize(weight_l)
        eta_t = _materialize(eta_l)
        upmask_t = _materialize(upmask_l)
        last = (net_t, delta_t, weight_t, eta_t, upmask_t)

        # advance coords for the next iteration (pure-torch side)
        target = coords1 + delta_t
        coords1 = target

    net_t, delta_t, weight_t, eta_t, upmask_t = last
    return {
        "fmaps": fmaps,
        "net": net_t,
        "delta": delta_t,
        "weight": weight_t,
        "eta": eta_t,
        "upmask": upmask_t,
    }
