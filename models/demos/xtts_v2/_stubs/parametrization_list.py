# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native TTNN port of `parametrization_list` of coqui/XTTS-v2.

Reference submodule: `hifigan_decoder.waveform_decoder.ups.0.parametrizations.weight`,
a `torch.nn.utils.parametrize.ParametrizationList` holding a single `_WeightNorm`
parametrization (`torch.nn.utils.parametrizations.weight_norm`, `dim=0`).

Called with NO inputs, it reconstructs the effective weight from the two stored
parameters (`original0` = magnitude `g`, `original1` = direction `v`):

    weight = g * v / ||v||   with the L2 norm taken over every dim EXCEPT `dim`

For this ConvTranspose1d weight: `v` is `[512, 256, 16]`, `g` is `[512, 1, 1]`, and
`dim=0`, so the norm is one scalar per output channel (`||v[i]||` over the 256*16
remaining elements) and `weight[i] = g[i] * v[i] / ||v[i]||`. Output `[512, 256, 16]`.

Computed natively in ttnn (float32): flatten the per-channel elements, reduce the
sum-of-squares, and rescale. There is no activation input — this is a pure
parameter reconstruction (the values, not the torch reference module).
"""

from __future__ import annotations

import ttnn

from models.demos.xtts_v2._stubs.weight_norm import build as _build_weight_norm


def build(device, torch_module):
    """Reconstruct the weight-norm weight natively in ttnn and return a forward closure."""
    pl = torch_module
    g = pl.original0                            # magnitude `g` (weight_g), [C, 1, 1] (dim=0)
    v = pl.original1                            # direction `v` (weight_v), [C, *rest]

    # Delegate the reparametrization to the weight_norm leaf, built from the
    # `_WeightNorm` parametrization (`pl[0]`). Its forward(weight_g, weight_v)
    # returns the same reconstructed ttnn weight this stub computed inline.
    _wn_fwd = _build_weight_norm(device, pl[0])
    weight = _wn_fwd(g, v)                       # [C, ...] reconstructed weight

    def forward(*args, **kwargs):
        return weight

    return forward


def parametrization_list(*args, **kwargs):
    raise RuntimeError(
        "parametrization_list requires build(device, torch_module) to bind the "
        "stored weight-norm parameters; the bare callable has no parameters."
    )
