# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm with the all-core backward for layers whose gamma is frozen (LoRA).

``ttml.ops.rmsnorm.rmsnorm`` runs its backward (``metal/ops/rmsnorm_bw``) with one core per tile-row, so a
``[1, 1, 1024, 5120]`` activation uses 32 of 110 cores (0.37 ms per call, 30 ms per 40-layer step, 6x its roofline;
``doc/perf_analysis.md``). ``ttml.ops.metal.rmsnorm_bw_allcore`` (ported from ``umales/qwen38`` as its own op)
splits every tile-row over several cores in two phases: per-slice partial sums of ``x * gamma * dL_dout``, then the
row scale and the gradients. The forward is the unchanged ``rmsnorm_fw`` metal op. Gamma gets no gradient here
(``compute_dgamma=False``), which is what a frozen gamma needs and also skips the dgamma pass.
"""

from __future__ import annotations

import ttml
from ttml.autograd import Function
from ttml.models.llama.transformer import RMSNormLayer

__all__ = ["RMSNormFrozenGamma", "use_allcore_rmsnorm_backward"]


class RMSNormFrozenGamma(Function):
    @staticmethod
    def forward(ctx, x, gamma, epsilon):
        x_val, g_val = x.get_value(), gamma.get_value()
        out, rms = ttml.ops.metal.rmsnorm_fw(x_val, g_val, True, epsilon)
        ctx.save_for_backward(x_val, g_val, rms)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x_val, g_val, rms = ctx.saved_tensors
        ctx.save_for_backward()
        dx = ttml.ops.metal.rmsnorm_bw_allcore(x_val, g_val, rms, grad_output, False)[0]
        return dx, None


def _rmsnorm_frozen_gamma(x, gamma, epsilon):
    if not x.get_requires_grad():
        # Nothing upstream needs a gradient (the first norm reads the frozen embedding output): the C++ op builds
        # no backward node here, while a Python Function always would (review round 3, finding 1).
        return ttml.ops.rmsnorm.rmsnorm(x, gamma, epsilon)
    return RMSNormFrozenGamma.apply(x, gamma, epsilon)


def use_allcore_rmsnorm_backward(model) -> list[str]:
    """Route every :class:`RMSNormLayer` of ``model`` whose gamma is frozen through :class:`RMSNormFrozenGamma`.
    Returns the converted module names."""
    converted: list[str] = []
    for name, module in model.named_modules():
        if isinstance(module, RMSNormLayer) and not module.gamma.tensor.get_requires_grad():
            module.rmsnorm_op = _rmsnorm_frozen_gamma
            converted.append(name)
    return converted
