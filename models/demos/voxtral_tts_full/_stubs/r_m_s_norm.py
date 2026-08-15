# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TT-NN port of `VoxtralRMSNorm` (Voxtral-TTS backbone, Block 1).

Reference: `modeling_layers.VoxtralRMSNorm.forward` -> `voxtral_common_ref.rms_norm`:

    x * rsqrt(mean(x^2) + eps) * weight,  reduced over the last dim

WHY THIS IS NOT `models/common/rmsnorm.py::RMSNorm`, THE BRING-UP PLAN'S REUSE TARGET. That
class wraps `ttnn.rms_norm`, and on this model the FUSED kernel is the single largest error
term in the backbone. Measured on this Blackhole against a float64 reference, fp32 in / fp32
out, on a [1, 224, 3072] activation:

    ttnn.rms_norm, RMSNorm's own HiFi2 config          4.5e-3
    ttnn.rms_norm, this model's HiFi4 + fp32_dest_acc  1.56e-3
    mean -> rsqrt -> mul, composed (what runs here)    6.7e-8

No compute-kernel config reaches the composition: the loss is inside the fused kernel's
reduction, not in the operand precision. That distinction matters here because this norm is
applied to the RESIDUAL STREAM itself -- twice per layer, 26 layers deep -- and the hidden
state it produces is quantised onto 21 FSQ levels downstream, where 1e-3 is the difference
between a code and its neighbour. The composition is four ttnn ops and is shared with every
other norm in this model as `_stubs/attention.py::rms_norm`, so there is still exactly one
copy of the definition; it just is not the fused one.

EPS IS READ OFF THE MODULE, NOT ASSUMED. The backbone norms use 1e-5, but the codec's use
1e-2 -- the same class, three orders apart -- so taking the module's own value is what keeps
this port correct if it is ever pointed at the other one.
"""
from __future__ import annotations

import torch
import ttnn

from models.demos.voxtral_tts_full._stubs.attention import rms_norm

_NORM_EPS = 1e-5


class TtVoxtralRMSNorm:
    def __init__(self, weight, eps):
        self.weight = weight
        self.eps = eps

    @classmethod
    def build(cls, device, torch_module, dtype=ttnn.bfloat16):
        weight = torch_module.weight.detach().float().contiguous()
        torch_dtype = torch.float32 if dtype == ttnn.float32 else torch.bfloat16
        staged = ttnn.from_torch(
            weight.reshape(1, -1).to(torch_dtype), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device
        )
        return cls(staged, float(getattr(torch_module, "eps", _NORM_EPS)))

    def __call__(self, x, *args, **kwargs):
        return rms_norm(x, self.weight, self.eps)


def build(device, torch_module=None, **kwargs):
    return TtVoxtralRMSNorm.build(device, torch_module, **kwargs)


def r_m_s_norm(device, torch_module=None, **kwargs):
    return TtVoxtralRMSNorm.build(device, torch_module, **kwargs)
