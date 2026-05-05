# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""
TT RMSNorm for Pixtral vision (HF ``PixtralRMSNorm`` / ``LlamaRMSNorm`` math).

Loads the affine ``weight`` vector from an explicit checkpoint key (no Llama-style
``base_url`` / ``layer_num`` layout). The forward uses ``ttnn.rms_norm`` with
``pad_by_zero``, matching the path used by ``TtLlamaRMSNorm`` in
``models/experimental/llama/tt/llama_layer_norm.py``.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import pad_by_zero


def _resolve_weight_state_dict_key(weight_key: str | None, state_dict_prefix: str | None) -> str:
    if weight_key is not None:
        return weight_key
    if state_dict_prefix is None:
        raise ValueError(
            "Provide exactly one of `weight_key` (full key) or `state_dict_prefix` (path without `.weight`)."
        )
    p = state_dict_prefix
    if p.endswith(".weight"):
        return p
    if p.endswith("."):
        return f"{p}weight"
    return f"{p}.weight"


class TtPixtralRMSNorm(LightweightModule):
    """
    Thin Pixtral/Llama-style RMSNorm on TTNN: ``output = rms_norm(x, eps) * gamma``.

    HF reference: ``transformers.models.pixtral.modeling_pixtral.PixtralRMSNorm``
    (default ``eps=1e-6`` in the class; Pixtral blocks use ``eps=1e-5`` — match the
    checkpoint you load).

    Args:
        mesh_device: Mesh or device hosting the norm weight and compute.
        state_dict: Torch state dict (e.g. meta-format keys under ``vision_tower...``).
        eps: Epsilon inside RMS (use ``1e-5`` for Pixtral transformer norms).
        weight_key: Full key for the 1D gamma tensor, e.g.
            ``"vision_tower.transformer.layers.0.attention_norm.weight"``.
        state_dict_prefix: Alternative to ``weight_key``: prefix without the final
            ``weight`` field, e.g. ``"vision_tower.transformer.layers.0.attention_norm."``
            or ``"...attention_norm"`` (``.weight`` is appended).
        dtype: TT dtype for the tilized weight (``pad_by_zero`` / ``torch2tt_tensor``).
    """

    def __init__(
        self,
        mesh_device,
        state_dict: dict,
        *,
        eps: float = 1e-5,
        weight_key: str | None = None,
        state_dict_prefix: str | None = None,
        dtype=ttnn.bfloat16,
    ):
        super().__init__()
        self.mesh_device = mesh_device
        self.variance_epsilon = eps

        key = _resolve_weight_state_dict_key(weight_key, state_dict_prefix)
        if key not in state_dict:
            raise KeyError(f"TtPixtralRMSNorm: missing weight key {key!r} in state_dict.")
        pytorch_gamma = state_dict[key]
        if pytorch_gamma.dim() != 1:
            raise ValueError(f"Expected 1D gamma at {key!r}, got shape {tuple(pytorch_gamma.shape)}.")

        self.weight = pad_by_zero(
            pytorch_gamma,
            mesh_device,
            ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED),
            dtype,
        )[0]

    def forward(self, hidden_states: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.rms_norm(hidden_states, epsilon=self.variance_epsilon, weight=self.weight)


__all__ = ["TtPixtralRMSNorm"]
