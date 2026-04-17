# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Iterative DROID-SLAM NN forward (pure torch, no SLAM geometry).

Upstream's ``DroidNet.forward`` runs the UpdateModule inside an outer
loop that interleaves 2 iterations of dense Bundle Adjustment per
UpdateModule call. BA needs ``lietorch`` (SE3 group ops) and the
``droid_backends`` CUDA extension; neither can be built on this
Tenstorrent-only machine.

For the NN pipeline the BA step is *not* in scope — what we care
about is the repeated correlation lookup + UpdateModule call. This
module provides a stripped-down iterative forward that:

  1. Extracts feature maps once per keyframe (fnet + cnet).
  2. Builds a correlation pyramid for each edge (ii -> jj).
  3. Repeats ``num_iters`` times:
       * sample corr pyramid at current ``coords1``
       * assemble ``motion = cat([coords1 - coords0, target - coords1])``
       * run UpdateModule -> ``delta``, ``weight``, ``eta``, ``upmask``
       * ``target = coords1 + delta``
       * absent BA, we propagate ``coords1 = target`` so subsequent
         lookups see the NN's predicted flow.

This is not a SLAM trajectory — there are no SE3 poses and no
disparity refinement from a BA solve — but it exercises every NN
component the paper defines: fnet, cnet, correlation pyramid, corr
lookup, flow encoder, corr encoder, GRU, delta head, weight head,
GraphAgg (eta + upmask). That is the scope the p150a chip can run
and that we can validate with PCC.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.experimental.droid_slam.reference.corr_ref import CorrBlock, coords_grid
from models.experimental.droid_slam.reference.droid_net_ref import DroidNet


class DroidNetIter(nn.Module):
    """Wrap a :class:`DroidNet` with an iterative forward.

    Attributes:
        net: the underlying DROID-SLAM NN (fnet, cnet, UpdateModule).
        num_iters: GRU iterations per forward.
        num_levels: correlation pyramid levels (matches upstream default).
        radius: correlation lookup window half-extent.
    """

    def __init__(self, net: DroidNet, num_iters: int = 12, num_levels: int = 4, radius: int = 3):
        super().__init__()
        self.net = net
        self.num_iters = num_iters
        self.num_levels = num_levels
        self.radius = radius

    @torch.no_grad()
    def forward(
        self,
        images: torch.Tensor,
        ii: torch.Tensor,
        jj: torch.Tensor,
    ) -> dict:
        """Run extract_features + iterative update over a keyframe window.

        Args:
            images: ``(B, NK, 3, H, W)`` torch tensor, BGR 0..255.
            ii, jj: ``(E,)`` long tensors — edge source/destination
                keyframe indices. Every value must be in ``[0, NK)``.

        Returns:
            Dict with keys ``fmaps``, ``net`` (final hidden state),
            ``delta`` (last iteration's flow delta), ``weight``, ``eta``,
            ``upmask`` — the same tensors ``UpdateModule`` returns on
            the final step.
        """
        b, nk, _, h, w = images.shape
        device = images.device
        ht8, wd8 = h // 8, w // 8

        fmaps, net_all, inp_all = self.net.extract_features(images)
        # fmaps, net_all, inp_all: (B, NK, 128, ht8, wd8)

        # Pick out per-edge feature maps — ii is (E,), so net[:, ii]
        # gives (B, E, C, H, W).
        net = net_all[:, ii]
        inp = inp_all[:, ii]

        corr_fn = CorrBlock(
            fmaps[:, ii], fmaps[:, jj], num_levels=self.num_levels, radius=self.radius
        )

        # Identity projection as the iteration's starting estimate. With
        # no BA, `coords1` drifts under the NN's own flow deltas.
        coords0 = coords_grid(ht8, wd8, device=device, dtype=images.dtype)
        coords1 = coords0[None, None].expand(b, ii.shape[0], ht8, wd8, 2).contiguous()
        target = coords1.clone()

        delta = weight = eta = upmask = None
        for _ in range(self.num_iters):
            corr = corr_fn(coords1)
            resd = target - coords1
            flow = coords1 - coords0[None, None]
            motion = torch.cat([flow, resd], dim=-1)
            # (B, E, H, W, 4) -> (B, E, 4, H, W) and clamp to match upstream.
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            net, delta, weight, eta, upmask = self.net.update(
                net, inp, corr, motion, ii.to(net.device)
            )
            target = coords1 + delta
            coords1 = target

        return {
            "fmaps": fmaps,
            "net": net,
            "delta": delta,
            "weight": weight,
            "eta": eta,
            "upmask": upmask,
        }


def make_default_graph(num_keyframes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a simple bi-directional forward/backward edge graph.

    Each consecutive keyframe pair (i, i+1) contributes two edges:
    ``i -> i+1`` and ``i+1 -> i``. Good enough for a benchmark window
    without requiring the full factor-graph construction from upstream.
    """
    src, dst = [], []
    for k in range(num_keyframes - 1):
        src.append(k)
        dst.append(k + 1)
        src.append(k + 1)
        dst.append(k)
    return torch.as_tensor(src, dtype=torch.long), torch.as_tensor(dst, dtype=torch.long)
