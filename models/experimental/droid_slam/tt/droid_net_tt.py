# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Tenstorrent tt-nn port of DROID-SLAM.

Porting is done progressively. On iteration 0 this module is a pure
CPU passthrough of the reference implementation — it exists only to
establish the benchmark harness. Subsequent iterations replace parts
of the forward path with tt-nn primitives while the benchmark tracks
frames/sec and PCC against the torch reference.
"""

from __future__ import annotations

import torch

from models.experimental.droid_slam.reference.droid_net_ref import DroidNet as ReferenceDroidNet


class TtDroidNet:
    """tt-nn / torch hybrid DROID-SLAM forward path."""

    def __init__(self, device, reference: ReferenceDroidNet):
        self.device = device
        self.reference = reference
        self.reference.eval()

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        return self.reference.extract_features(images)

    @torch.no_grad()
    def update(self, net, inp, corr, flow, ii):
        return self.reference.update(net, inp, corr, flow, ii)
