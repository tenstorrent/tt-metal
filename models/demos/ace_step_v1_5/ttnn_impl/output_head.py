# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the AceStep DiT decoder output head (HF `AceStepDiTModel` tail)."""

from __future__ import annotations

from typing import Any, Optional

import ttnn
from models.demos.ace_step_v1_5.ttnn_impl.patchify import (
    PatchifyMetadata,
    TtAceStepDePatchify1D,
    _maybe_get_state_dict_key,
)


def _state_key(base_address: str, suffix: str) -> str:
    return f"{base_address}.{suffix}" if base_address else suffix


class TtAceStepDiTOutputHead:
    """
    TTNN equivalent of the final block in HF ``AceStepDiTModel`` (see
    ``acestep/models/*/modeling_acestep_v15_*.py``):

    - ``norm_out``: ``Qwen3RMSNorm`` over the last dimension (patch tokens, ``hidden_size``).
    - Adaptive scale/shift from ``scale_shift_table`` and diffusion timestep embedding ``temb``.
    - ``proj_out``: same de-patchify path as :class:`TtAceStepDePatchify1D` (``ConvTranspose1d`` stack).

    Expected ``temb`` matches HF: the ``[B, hidden_size]`` tensor returned by ``TimestepEmbedding``
    (the ``temb`` branch, not ``timestep_proj``), i.e. the tensor summed for ``t`` and ``t - r`` before
    the output head.
    """

    def __init__(
        self,
        *,
        config,
        state_dict: dict,
        base_address: str,
        device: ttnn.Device,
        activation_dtype: ttnn.DataType | None = None,
        weights_dtype: ttnn.DataType | None = None,
    ) -> None:
        self.device = device
        self.config = config
        self.hidden_size = int(getattr(config, "hidden_size"))
        self.eps = float(getattr(config, "rms_norm_eps", 1e-6))
        if activation_dtype is None:
            activation_dtype = getattr(ttnn, "bfloat16", None)
        if weights_dtype is None:
            weights_dtype = getattr(ttnn, "bfloat16", None)
        if activation_dtype is None or weights_dtype is None:
            raise RuntimeError("TTNN build missing bfloat16 dtype; pass activation_dtype/weights_dtype explicitly.")
        self.activation_dtype = activation_dtype

        norm_w_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "norm_out.weight"),
                "norm_out.weight",
            ),
        )
        sst_key = _maybe_get_state_dict_key(
            state_dict,
            (
                _state_key(base_address, "scale_shift_table"),
                "scale_shift_table",
            ),
        )

        norm_w_host = state_dict[norm_w_key]
        if int(norm_w_host.shape[0]) != self.hidden_size:
            raise ValueError(
                f"norm_out.weight length mismatch: got {norm_w_host.shape[0]}, expected {self.hidden_size}"
            )

        sst_host = state_dict[sst_key]
        if tuple(sst_host.shape) != (1, 2, self.hidden_size):
            raise ValueError(
                f"scale_shift_table must be [1, 2, hidden_size], got {tuple(sst_host.shape)} vs expected (1, 2, {self.hidden_size})"
            )

        self.norm_weight = ttnn.as_tensor(
            norm_w_host,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # Split so timestep broadcast matches HF: shift = sst[:,0:1], scale = sst[:,1:2], each + temb[:,None,:]
        self.shift_table = ttnn.as_tensor(
            sst_host[:, 0:1, :],
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.scale_table = ttnn.as_tensor(
            sst_host[:, 1:2, :],
            dtype=weights_dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        proj_prefix = _state_key(base_address, "proj_out")
        self.depatchify = TtAceStepDePatchify1D(
            config=config,
            state_dict=state_dict,
            base_address=proj_prefix,
            device=device,
            activation_dtype=activation_dtype,
            weights_dtype=weights_dtype,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        temb: ttnn.Tensor,
        meta: PatchifyMetadata,
        *,
        debug: Optional[dict[str, Any]] = None,
    ) -> ttnn.Tensor:
        if len(hidden_states.shape) != 3:
            raise ValueError(f"Expected hidden_states [B, T_p, hidden_size], got shape={hidden_states.shape}")
        if len(temb.shape) != 2:
            raise ValueError(f"Expected temb [B, hidden_size], got shape={temb.shape}")
        if not ttnn.is_tensor_storage_on_device(hidden_states):
            raise AssertionError("Expected hidden_states on device.")
        if not ttnn.is_tensor_storage_on_device(temb):
            raise AssertionError("Expected temb on device.")

        b = int(hidden_states.shape[0])
        t_p = int(hidden_states.shape[1])
        d = int(hidden_states.shape[2])
        if d != self.hidden_size:
            raise ValueError(f"Expected hidden_size={self.hidden_size} on last dim, got {d}")
        if int(temb.shape[0]) != b or int(temb.shape[1]) != self.hidden_size:
            raise ValueError(f"Expected temb shape ({b}, {self.hidden_size}), got {tuple(temb.shape)}")

        x_tile = ttnn.to_layout(hidden_states, ttnn.TILE_LAYOUT)
        normed = ttnn.rms_norm(
            x_tile,
            weight=self.norm_weight,
            epsilon=self.eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        temb_u = ttnn.reshape(temb, (b, 1, self.hidden_size))
        temb_u = ttnn.to_layout(temb_u, ttnn.ROW_MAJOR_LAYOUT)

        shift = ttnn.add(self.shift_table, temb_u)
        scale = ttnn.add(self.scale_table, temb_u)

        shift_t = ttnn.to_layout(shift, ttnn.TILE_LAYOUT)
        scale_t = ttnn.to_layout(scale, ttnn.TILE_LAYOUT)
        ones = ttnn.ones_like(scale_t)
        one_plus_scale = ttnn.add(scale_t, ones)

        modulated = ttnn.add(ttnn.multiply(normed, one_plus_scale), shift_t)
        modulated_rm = ttnn.to_layout(modulated, ttnn.ROW_MAJOR_LAYOUT)
        if debug is not None and debug.get("enabled", False):
            debug["head.after_norm"] = normed
            debug["head.modulated_patches"] = modulated_rm

        out = self.depatchify.forward(modulated_rm, meta)
        if debug is not None and debug.get("enabled", False):
            debug["pipe.acoustic"] = out
        return out

    def __call__(
        self,
        hidden_states: ttnn.Tensor,
        temb: ttnn.Tensor,
        meta: PatchifyMetadata,
        *,
        debug: Optional[dict[str, Any]] = None,
    ) -> ttnn.Tensor:
        return self.forward(hidden_states, temb, meta, debug=debug)
