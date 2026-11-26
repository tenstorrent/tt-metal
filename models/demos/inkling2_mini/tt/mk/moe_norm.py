# SPDX-FileCopyrightText: © 2026 Olof Johansson <olof@lixom.net>
# SPDX-License-Identifier: Apache-2.0

"""MoE input norm as one program (FusedRowNorm with an L1 output) instead of reshard + sharded rms_norm +
to L1 bf16 (3 programs per app). NOT bit-equal: fp32 SFPU math instead of the HiFi2 FPU path (closer to fp32, see
tests/test_fused_moe_norm.py), so it is opt-in and goes through the model gate before use.

    blocks = install(model)   # every layer's MoE; uninstall(model) restores
"""

import ttnn
from models.demos.inkling2_mini.tt.fused import common as F
from models.demos.inkling2_mini.tt.fused.norm import FusedRowNorm


class _NormMoE:
    """Inkling2MoE.__call__ with the fused norm; everything else is the wrapped Inkling2MoE's."""

    def __init__(self, moe, out_cache):
        self.moe = moe
        w = ttnn.to_torch(moe.norm_w).float().reshape(-1)
        self.fnorm = FusedRowNorm(moe.device, [w], out_l1=True)
        self.fnorm.out = out_cache  # one L1 output row for all layers (x lives only until the experts ran)

    def __getattr__(self, name):
        return getattr(self.moe, name)

    def __call__(self, h):
        x = self.fnorm(h)[0]  # cached L1 row (do not deallocate: FusedRowNorm reuses it)
        weights, indices = self.moe.route(x)
        out = self.moe.experts(x, weights, indices)
        ttnn.deallocate(weights)
        return out, indices


def install(model):
    cache = F.OutCache(model.device)
    for lay in model.layers:
        if not isinstance(lay.moe, _NormMoE):
            lay.moe = _NormMoE(lay.moe, cache)


def uninstall(model):
    for lay in model.layers:
        if isinstance(lay.moe, _NormMoE):
            lay.moe = lay.moe.moe
