# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""TTNN implementation of the AceStep DiT decoder output head (HF `AceStepDiTModel` tail)."""

from __future__ import annotations

from typing import Any, Optional

import ttnn
from models.experimental.ace_step_v1_5.ttnn_impl.math_perf_env import (
    ace_step_add_one,
    ace_step_binary_kwargs,
    ace_step_dit_rms_norm_kwargs,
    ace_step_dit_weight_dtype,
    ace_step_ensure_dram_activation,
    ace_step_ensure_l1_activation,
    ace_step_ensure_tile_layout,
    ace_step_linear_l1_memory_config,
    ace_step_reshape_kwargs,
)
from models.experimental.ace_step_v1_5.ttnn_impl.patchify import (
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
        weights_dtype = ace_step_dit_weight_dtype(ttnn, weights_dtype)

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

        _tbl_mc = ace_step_linear_l1_memory_config(ttnn)
        if _tbl_mc is None:
            _tbl_mc = ttnn.DRAM_MEMORY_CONFIG
        self.norm_weight = ttnn.as_tensor(
            norm_w_host,
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_tbl_mc,
        )
        # Tiny tables (~4 KiB raw each): L1-interleaved TILE avoids DRAM ``in0`` on output-head BinaryNg.
        self.shift_table = ttnn.as_tensor(
            sst_host[:, 0:1, :],
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_tbl_mc,
        )
        self.scale_table = ttnn.as_tensor(
            sst_host[:, 1:2, :],
            dtype=weights_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.device,
            memory_config=_tbl_mc,
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
        use_dram_activations: bool = False,
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

        x_tile = ace_step_ensure_tile_layout(ttnn, hidden_states)
        _sr = ace_step_reshape_kwargs(ttnn)
        _dram_mc = getattr(ttnn, "DRAM_MEMORY_CONFIG", None)
        if use_dram_activations:
            _el_mc = _dram_mc
            _bin_kw = ace_step_binary_kwargs(ttnn, _dram_mc)

            def _l1(t):
                return ace_step_ensure_dram_activation(ttnn, t, _dram_mc)

        else:
            _el_mc = ace_step_linear_l1_memory_config(ttnn) or _dram_mc
            _bin_kw = ace_step_binary_kwargs(ttnn, _el_mc)

            def _l1(t):
                return ace_step_ensure_l1_activation(ttnn, t, _el_mc)

        _rms_kw = ace_step_dit_rms_norm_kwargs(ttnn, _el_mc, device=self.device)
        normed = ttnn.rms_norm(
            x_tile,
            weight=self.norm_weight,
            epsilon=self.eps,
            **_rms_kw,
        )
        normed = _l1(normed)

        temb_u = _l1(ttnn.reshape(temb, (b, 1, self.hidden_size), **_sr))
        temb_u = ace_step_ensure_tile_layout(ttnn, temb_u)
        shift = _l1(ttnn.add(temb_u, self.shift_table, **_bin_kw))
        scale = _l1(ttnn.add(temb_u, self.scale_table, **_bin_kw))
        one_plus_scale = _l1(ace_step_add_one(ttnn, scale, **_bin_kw))
        scaled = _l1(ttnn.multiply(normed, one_plus_scale, **_bin_kw))
        modulated = _l1(ttnn.add(scaled, shift, **_bin_kw))
        modulated = _l1(modulated)
        if debug is not None and debug.get("enabled", False):
            debug["head.after_norm"] = normed
            debug["head.modulated_patches"] = modulated

        out = self.depatchify.forward(modulated, meta)
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
        use_dram_activations: bool = False,
    ) -> ttnn.Tensor:
        return self.forward(hidden_states, temb, meta, debug=debug, use_dram_activations=use_dram_activations)
