# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native ttnn port of `Qwen3NextRMSNorm`.

Reference: `transformers/models/qwen3_next/modeling_qwen3_next.py::Qwen3NextRMSNorm`:

    out = x * rsqrt(mean(x^2, -1) + eps) * (1.0 + weight)

Two details that a generic RMSNorm port gets wrong here:

  * Qwen3-Next stores the *residual* of the scale -- `weight` is initialised to ZEROS and the
    forward applies `(1.0 + weight)`. Feeding `weight` straight into `ttnn.rms_norm` would scale
    the normalised activations by ~0 instead of ~1. We fold the `1.0 +` into the device tensor at
    build time, so the kernel stays a single fused `rms_norm` dispatch.
  * The scale multiply happens in fp32 before the cast back (`(x * w).to(dtype)`, not
    `x.to(dtype) * w`), so the weight is materialised from the fp32 state dict.

Tensor-parallel: RMSNorm reduces over the FULL model dim, so there is nothing to split -- the
weight is REPLICATED on every chip and the op needs no collective. Each chip normalises its own
copy of the activations and produces the identical golden result.
"""
from __future__ import annotations

import ttnn

from models.demos.qwen3_coder_next._stubs.gated_delta_net import (
    TILE,
    num_devices,
    replicate_mapper,
    to_device,
)


class TtQwen3NextRMSNorm:
    """Native ttnn Qwen3-Next RMSNorm (weight stored as a residual around 1.0)."""

    def __init__(self, device, *, weight, hidden_size, eps) -> None:
        self.device = device
        self.weight = weight
        self.hidden_size = hidden_size
        self.eps = eps
        self.num_devices = num_devices(device)
        self.compute_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm stub needs the torch reference module for its weights")

        w = torch_module.state_dict()["weight"].detach().float()
        hidden = int(w.shape[-1])
        eps = float(getattr(torch_module, "eps", None) or getattr(torch_module, "variance_epsilon", 1e-6))

        # Fold HF's `(1.0 + weight)` into the device-side scale.
        scale = w + 1.0
        if hidden % TILE:
            raise NotImplementedError(f"r_m_s_norm expects a tile-aligned model dim; got {hidden}")

        replicate = replicate_mapper(device, num_devices(device))
        return cls(
            device,
            weight=to_device(
                scale.view(1, 1, hidden // TILE, TILE),
                device,
                mesh_mapper=replicate,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            hidden_size=hidden,
            eps=eps,
        )

    def __call__(self, x, *args, **kwargs):
        seq = int(x.shape[-2])
        h = ttnn.reshape(x, (1, 1, seq, self.hidden_size))
        h = ttnn.rms_norm(
            h,
            weight=self.weight,
            epsilon=self.eps,
            compute_kernel_config=self.compute_config,
        )
        return ttnn.reshape(h, (1, seq, self.hidden_size))


def build(device, torch_module=None):
    return TtQwen3NextRMSNorm.build(device, torch_module)


def r_m_s_norm(device, torch_module=None):
    return TtQwen3NextRMSNorm.build(device, torch_module)
