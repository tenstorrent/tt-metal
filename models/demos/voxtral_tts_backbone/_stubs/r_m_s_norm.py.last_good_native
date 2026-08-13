# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""`r_m_s_norm` (`MistralRMSNorm`) for
`/localdev/lserbedzija/hf_models/voxtral-tts-backbone`.

This is the plan's REUSE target used as-is: the canonical
`models/common/rmsnorm.py::RMSNorm`. Only the construction needed adapting —
the scaffold called it with `mesh_device=`/`args=`/`layer_num=`, but the canonical
signature is `RMSNorm(device, dim, state_dict, weight_key, ...)` and it looks the
gain up as `f"{weight_key}.weight"`, so the HF module's `weight` is re-keyed here.

The library module owns the compute (`ttnn.rms_norm`, ROW_MAJOR gain reshaped to
`[1, 1, dim/32, 32]`, HiFi2 + fp32 accumulate) and normalizes over the last dim,
which is what HF's RMSNorm does. `_stubs/decoder_layer.py` builds its two norms
through this same adapter.

`build` touches torch only to re-key/cast the checkpoint gain; the forward is the
library's ttnn dispatch — `models/common/native_probe.py` sees it as native.
"""
from __future__ import annotations

import torch

from models.common.rmsnorm import RMSNorm


class TtRMSNorm:
    """Adapter around the canonical `models/common/rmsnorm.py::RMSNorm`."""

    _WEIGHT_KEY = "norm"

    def __init__(self, canonical_instance) -> None:
        self._impl = canonical_instance

    @classmethod
    def build(cls, device, torch_module):
        if torch_module is None:
            raise RuntimeError("r_m_s_norm build needs the HF MistralRMSNorm module to read its gain from")
        weight = torch_module.weight.detach().to(torch.float32)
        dim = int(weight.shape[-1])
        eps = getattr(torch_module, "variance_epsilon", None)
        if eps is None:
            eps = getattr(torch_module, "eps", 1e-5)
        canonical = RMSNorm(
            device=device,
            dim=dim,
            state_dict={f"{cls._WEIGHT_KEY}.weight": weight},
            weight_key=cls._WEIGHT_KEY,
            eps=float(eps),
        )
        return cls(canonical)

    def __call__(self, x, *_args, **_ignored):
        # The canonical forward is `forward(x, mode)`; this harness exercises the
        # non-sharded prefill path (interleaved in, interleaved out).
        return self._impl(x, mode="prefill")


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)


# Module-level shim with the component's lowercase slug name. Kept for
# backward compatibility with legacy SMOKE/PCC tests that import the
# slug directly.
def r_m_s_norm(device, torch_module=None):
    return TtRMSNorm.build(device, torch_module)
