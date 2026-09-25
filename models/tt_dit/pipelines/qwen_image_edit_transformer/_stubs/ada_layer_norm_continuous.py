# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of diffusers `AdaLayerNormContinuous` (QwenImage `norm_out`).

emb          = linear(silu(conditioning_embedding))
scale, shift = chunk(emb, 2, dim=-1)
out          = norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
"""

from __future__ import annotations

import torch

import ttnn
from models.tt_dit.pipelines.qwen_image_edit_transformer._stubs import _precise


def _to_tt(t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    if t is None:
        return None
    if isinstance(t, ttnn.Tensor):
        return t
    kwargs = {}
    if isinstance(device, ttnn.MeshDevice):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(t.detach().to(torch.float32), dtype=dtype, layout=layout, device=device, **kwargs)


class AdaLayerNormContinuous:
    def __init__(self, device, torch_module):
        self.device = device
        lin = torch_module.linear
        w = lin.weight.detach().to(torch.float32)  # [2*D, C]
        self.dim = w.shape[0] // 2
        # Split into scale / shift halves so no on-device chunk is needed.
        self.w_scale = _to_tt(w[: self.dim].t().contiguous(), device)
        self.w_shift = _to_tt(w[self.dim :].t().contiguous(), device)
        if lin.bias is not None:
            b = lin.bias.detach().to(torch.float32)
            # Fold the "+1" of (1 + scale) into the scale bias.
            self.b_scale = _to_tt((b[: self.dim] + 1.0).reshape(1, -1), device)
            self.b_shift = _to_tt(b[self.dim :].reshape(1, -1), device)
            # float32 copies for the precise path (b + 1 does not survive a bf16 round trip)
            self.b_scale32 = _to_tt((b[: self.dim] + 1.0).reshape(1, -1), device, dtype=ttnn.float32)
            self.b_shift32 = _to_tt(b[self.dim :].reshape(1, -1), device, dtype=ttnn.float32)
        else:
            self.b_scale = _to_tt(torch.ones(1, self.dim), device)
            self.b_shift = None
            self.b_scale32 = _to_tt(torch.ones(1, self.dim), device, dtype=ttnn.float32)
            self.b_shift32 = None

        norm = torch_module.norm
        self.is_rms = type(norm).__name__ == "RMSNorm"
        eps = getattr(norm, "eps", 1e-6)
        self.eps = float(eps[0] if isinstance(eps, tuple) else eps)
        nw = getattr(norm, "weight", None)
        nb = getattr(norm, "bias", None)
        self.norm_w = _to_tt(nw.detach().reshape(1, -1), device) if nw is not None else None
        self.norm_b = _to_tt(nb.detach().reshape(1, -1), device) if nb is not None else None

        self.compute_cfg = ttnn.init_device_compute_kernel_config(
            device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def _precise_forward(self, x, c):
        """float32 throughout: modulation from silu(c) as hi + lo products, manual LayerNorm, float32
        affine (bf16 scale/shift would round the float32 product to bf16)."""
        x = x if x.dtype == ttnn.float32 else ttnn.typecast(x, ttnn.float32)
        c = ttnn.silu(c if c.dtype == ttnn.float32 else ttnn.typecast(c, ttnn.float32))
        scale = _precise.linear(c, self.w_scale, bias=self.b_scale32)
        shift = _precise.linear(c, self.w_shift, bias=self.b_shift32)
        B = scale.shape[0]
        scale = ttnn.reshape(scale, (B, 1, self.dim))
        shift = ttnn.reshape(shift, (B, 1, self.dim))
        mu = ttnn.mean(x, dim=-1, keepdim=True, compute_kernel_config=self.compute_cfg)
        xc = ttnn.subtract(x, mu)
        if self.is_rms:
            ms = ttnn.mean(ttnn.multiply(x, x), dim=-1, keepdim=True, compute_kernel_config=self.compute_cfg)
            xn = ttnn.multiply(x, ttnn.rsqrt(ttnn.add(ms, self.eps)))
        else:
            var = ttnn.mean(ttnn.multiply(xc, xc), dim=-1, keepdim=True, compute_kernel_config=self.compute_cfg)
            xn = ttnn.multiply(xc, ttnn.rsqrt(ttnn.add(var, self.eps)))
        if self.norm_w is not None:
            xn = ttnn.multiply(xn, ttnn.typecast(self.norm_w, ttnn.float32))
        if self.norm_b is not None:
            xn = ttnn.add(xn, ttnn.typecast(self.norm_b, ttnn.float32))
        return ttnn.add(ttnn.multiply(xn, scale), shift)

    def __call__(self, x, conditioning_embedding):
        x = _to_tt(x, self.device)
        c = _to_tt(conditioning_embedding, self.device)  # [B, C]
        if _precise.ENABLED:
            return self._precise_forward(x, c)
        c = ttnn.silu(c)
        scale = ttnn.linear(c, self.w_scale, bias=self.b_scale, compute_kernel_config=self.compute_cfg)  # [B, D]
        shift = ttnn.linear(c, self.w_shift, bias=self.b_shift, compute_kernel_config=self.compute_cfg)
        B = scale.shape[0]
        scale = ttnn.reshape(scale, (B, 1, self.dim))
        shift = ttnn.reshape(shift, (B, 1, self.dim))

        if self.is_rms:
            xn = ttnn.rms_norm(x, epsilon=self.eps, weight=self.norm_w, compute_kernel_config=self.compute_cfg)
        else:
            xn = ttnn.layer_norm(
                x, epsilon=self.eps, weight=self.norm_w, bias=self.norm_b, compute_kernel_config=self.compute_cfg
            )
        return ttnn.add(ttnn.multiply(xn, scale), shift)


def build(device, torch_module):
    return AdaLayerNormContinuous(device, torch_module)
