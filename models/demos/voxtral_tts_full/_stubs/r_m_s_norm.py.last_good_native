# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `backbone.layers.0.input_layernorm` (`VoxtralRMSNorm`).

    x * rsqrt(mean(x^2) + eps) * weight        (eps = config.rms_norm_eps = 1e-5)

The reuse target was `models/common/rmsnorm.py::RMSNorm`, which wraps the FUSED `ttnn.rms_norm`.
That op is built on the same loose reduction as `ttnn.softmax`: measured on this board against a
float64 reference at [1, 64, 3072], the fused call carries **1.5e-3** relative error where the
composition above carries **8.8e-8**.  PCC is scale-invariant, so a single normalisation hides it
at 0.9999 either way -- it only surfaces once norms are stacked, where the fused op leaves every
layer's residual delta uniformly short (a 26-layer stack of this same model sat at 0.9886 for
exactly this reason).

So this uses the composed form, which is also what `decoder_layer` and `tts_backbone` run: a
component that graduated on a different implementation than the one the model actually executes
would be measuring the wrong thing.  The helper is shared from
`models/demos/voxtral_tts_full/tt_backbone.py`.
"""

from __future__ import annotations

from models.demos.voxtral_tts_full.tt_backbone import TtRMSNorm


class TtVoxtralRMSNorm:
    def __init__(self, norm):
        self.norm = norm

    @classmethod
    def build(cls, device, torch_module):
        return cls(TtRMSNorm.from_module(device, torch_module))

    def __call__(self, x):
        return self.norm(x)


def build(device, torch_module=None):
    return TtVoxtralRMSNorm.build(device, torch_module)


def r_m_s_norm(device, torch_module=None):
    return TtVoxtralRMSNorm.build(device, torch_module)
