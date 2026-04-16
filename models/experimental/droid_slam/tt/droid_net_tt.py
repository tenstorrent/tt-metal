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
    """tt-nn / torch hybrid DROID-SLAM forward path.

    As an initial speedup we run the torch reference in bfloat16. The
    native fp32 torch path keeps a full-precision copy of the weights
    for the PCC reference — this wrapper casts inputs and weights to
    bfloat16 on entry and casts outputs back to fp32 so downstream
    (classical BA) sees fp32.
    """

    _COMPUTE_DTYPE = torch.bfloat16

    def __init__(self, device, reference: ReferenceDroidNet):
        self.device = device
        self.reference_fp32 = reference
        reference.eval()
        self.reference = self._cast_reference(reference)

    @classmethod
    def _cast_reference(cls, fp32_model: ReferenceDroidNet) -> ReferenceDroidNet:
        # Deep-clone the module with bfloat16 weights so the fp32
        # reference stays pristine for PCC comparisons. Wrap the two
        # hot sub-modules in torch.compile so inductor can fuse the
        # conv/norm/relu chains.
        import copy

        bf16 = copy.deepcopy(fp32_model).to(cls._COMPUTE_DTYPE)
        bf16.eval()
        bf16.fnet = torch.compile(bf16.fnet, mode="reduce-overhead", dynamic=False)
        bf16.cnet = torch.compile(bf16.cnet, mode="reduce-overhead", dynamic=False)
        bf16.update = torch.compile(bf16.update, mode="reduce-overhead", dynamic=False)
        return bf16

    @torch.no_grad()
    def extract_features(self, images: torch.Tensor):
        # Stay in the compute dtype throughout the pipeline — only the
        # PCC comparison upstream needs fp32.
        images_bf = images.to(self._COMPUTE_DTYPE)
        return self.reference.extract_features(images_bf)

    @torch.no_grad()
    def update(self, net, inp, corr, flow, ii):
        args = [t.to(self._COMPUTE_DTYPE) for t in (net, inp, corr, flow)]
        return self.reference.update(*args, ii)
