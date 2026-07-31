# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Functional TTNN decoder layer for ``google/gemma-4-12B``.

Contract
--------
This module implements one dense Gemma 4 text decoder layer on a single 1x1
mesh.  It is intentionally the functional-decoder stage only: it does not
perform optimized-decoder, multichip, full-model, or serving work.

``FunctionalDecoder.from_state_dict`` accepts either a full Hugging Face
checkpoint state dict or a layer-only state dict from
``Gemma4TextDecoderLayer.state_dict()``.  Weight conversion to TTNN tensors is
done during construction.  Runtime ``prefill_forward`` and ``decode_forward``
expect device-resident TTNN tensors and do not call host conversion APIs.

Forward shapes
--------------
``prefill_forward``:
    ``hidden_states`` has shape ``[1, 1, seq_len, 3840]`` in TILE layout.
    ``rope_mats`` is a pair of 4D TTNN cos/sin tables with shape
    ``[1, 1, max_seq_len, head_dim]`` for the target layer kind.
    ``page_table`` is an int32 paged-attention table.  ``kv_cache`` is
    ``[k_cache, v_cache]`` allocated for this layer kind.

``decode_forward``:
    ``hidden_states`` has shape ``[1, 1, batch, 3840]``; this bringup tests
    batch 1.  ``position_idx`` is a device tensor holding the absolute current
    position.  For traced decode, pass 2D RoPE tables of shape
    ``[max_seq_len, head_dim]`` and a tensor ``position_idx`` so RoPE uses an
    on-device embedding lookup.  ``position_idx_cache`` may be supplied as an
    int32 tensor when ``position_idx`` uses an embedding-friendly dtype.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.demos.gemma4.config import MeshConfig, ModeConfig
from models.demos.gemma4.tt.attention import Gemma4Attention, Gemma4AttentionConfig
from models.demos.gemma4.tt.attention.kv_cache import init_kv_cache
from models.demos.gemma4.tt.gemma4_attention_config import get_attention_program_config
from models.demos.gemma4.tt.model_config import Gemma4ModelArgs
from models.demos.gemma4.tt.rms_norm import RMSNorm
from models.demos.gemma4.tt.shared_mlp import SharedMLP
from models.demos.gemma4.utils.substate import substate
from models.tt_transformers.tt.common import PagedAttentionConfig


SUPPORTED_HF_MODEL_ID = "google/gemma-4-12B"
_SUPPORTED_LAYER_TYPES = ("sliding_attention", "full_attention")


def _as_text_config(hf_config):
    return getattr(hf_config, "text_config", hf_config)


def _normalize_layer_state_dict(state_dict, layer_idx: int) -> dict:
    """Return keys under ``model.layers.<idx>`` for the demo Gemma4 loaders.

    Accepted inputs:
    - full unified checkpoint keys such as
      ``model.language_model.layers.5.self_attn.q_proj.weight``;
    - text-model keys such as ``layers.5.self_attn.q_proj.weight``;
    - demo/test keys such as ``model.layers.5.self_attn.q_proj.weight``;
    - layer-only keys such as ``self_attn.q_proj.weight``.
    """

    if not state_dict:
        return {}

    demo_prefix = f"model.layers.{layer_idx}"
    accepted_prefixes = (
        demo_prefix,
        f"model.language_model.layers.{layer_idx}",
        f"language_model.layers.{layer_idx}",
        f"layers.{layer_idx}",
    )

    has_prefixed_layer_keys = any(
        key.startswith(f"{prefix}.") for key in state_dict for prefix in accepted_prefixes
    )

    normalized = {}
    for key, value in state_dict.items():
        matched = False
        for prefix in accepted_prefixes:
            if key == prefix:
                normalized[demo_prefix] = value
                matched = True
                break
            dotted = f"{prefix}."
            if key.startswith(dotted):
                normalized[f"{demo_prefix}.{key[len(dotted):]}"] = value
                matched = True
                break
        if not matched and not has_prefixed_layer_keys and "." in key:
            normalized[f"{demo_prefix}.{key}"] = value

    return normalized


def _require_target_config(hf_config, layer_idx: int) -> Gemma4ModelArgs:
    text_config = _as_text_config(hf_config)
    model_args = Gemma4ModelArgs.from_hf_config(hf_config)

    if model_args.hidden_size != 3840:
        raise ValueError(f"{SUPPORTED_HF_MODEL_ID} hidden_size must be 3840, got {model_args.hidden_size}")
    if model_args.num_hidden_layers != 48:
        raise ValueError(
            f"{SUPPORTED_HF_MODEL_ID} num_hidden_layers must be 48, got {model_args.num_hidden_layers}"
        )
    if model_args.enable_moe_block:
        raise NotImplementedError("google/gemma-4-12B is dense; MoE configs belong to a separate target bringup")
    if model_args.hidden_size_per_layer_input:
        raise NotImplementedError("google/gemma-4-12B does not use per-layer input embeddings")
    if model_args.num_kv_shared_layers:
        raise NotImplementedError("google/gemma-4-12B does not use shared-KV decoder layers")
    if layer_idx < 0 or layer_idx >= model_args.num_hidden_layers:
        raise ValueError(f"layer_idx must be in [0, {model_args.num_hidden_layers}), got {layer_idx}")
    layer_type = model_args.layer_types[layer_idx]
    if layer_type not in _SUPPORTED_LAYER_TYPES:
        raise ValueError(f"unsupported layer_type {layer_type!r} in {SUPPORTED_HF_MODEL_ID}")

    if getattr(text_config, "attention_bias", False):
        raise NotImplementedError("attention_bias=True is not part of the google/gemma-4-12B decoder contract")

    return model_args


class FunctionalDecoder(LightweightModule):
    """Dense Gemma 4 decoder layer with paged prefill and traceable decode."""

    def __init__(
        self,
        *,
        mesh_device,
        hf_config: Gemma4ModelArgs,
        layer_idx: int,
        layer_state: dict,
        mesh_config: MeshConfig,
        ccl_manager=None,
        dtype=ttnn.bfloat16,
        attention_dtype=ttnn.bfloat16,
        shared_mlp_dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        bounded_sliding_kv_cache: bool = False,
    ):
        self.mesh_device = mesh_device
        self.hf_config = hf_config
        self.layer_idx = layer_idx
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.layer_type = hf_config.layer_types[layer_idx]
        self.attention_config = Gemma4AttentionConfig(hf_config, layer_idx)

        cache_root = str(tensor_cache_path) if tensor_cache_path is not None else None

        def norm(name):
            return RMSNorm(
                mesh_device=mesh_device,
                hf_config=hf_config,
                state_dict=substate(layer_state, name) if layer_state else {},
                tensor_cache_path=f"{cache_root}/layer_{layer_idx}/{name}" if cache_root else None,
                mesh_config=mesh_config,
                with_scale=True,
            )

        self.input_layernorm = norm("input_layernorm")
        self.post_attention_layernorm = norm("post_attention_layernorm")
        self.pre_feedforward_layernorm = norm("pre_feedforward_layernorm")
        self.post_feedforward_layernorm = norm("post_feedforward_layernorm")

        if layer_state and "layer_scalar" in layer_state:
            self.layer_scalar = layer_state["layer_scalar"].item()
        else:
            self.layer_scalar = 1.0

        attention_program_config = get_attention_program_config(
            self.attention_config, mesh_config=mesh_config, is_decode=True
        )
        self.self_attn = Gemma4Attention(
            mesh_device=mesh_device,
            config=self.attention_config,
            state_dict=substate(layer_state, "self_attn") if layer_state else {},
            ccl_manager=ccl_manager,
            mesh_config=mesh_config,
            program_config=attention_program_config,
            layer_idx=layer_idx,
            tensor_cache_path=f"{cache_root}/layer_{layer_idx}/self_attn" if cache_root else None,
            weight_dtype=attention_dtype,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )
        self.shared_mlp = SharedMLP(
            mesh_device=mesh_device,
            hf_config=hf_config,
            state_dict=substate(layer_state, "mlp") if layer_state else {},
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=shared_mlp_dtype,
            tensor_cache_path=f"{cache_root}/layer_{layer_idx}/mlp" if cache_root else None,
        )

    @classmethod
    def from_state_dict(
        cls,
        state_dict,
        *,
        hf_config,
        layer_idx: int,
        mesh_device,
        mesh_config: MeshConfig | None = None,
        ccl_manager=None,
        dtype=ttnn.bfloat16,
        attention_dtype=ttnn.bfloat16,
        shared_mlp_dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
        bounded_sliding_kv_cache: bool = False,
        **kwargs,
    ):
        """Build a functional decoder layer and convert weights to TTNN tensors.

        ``state_dict`` may be a full Hugging Face checkpoint dict or a layer-only
        dict.  Extra ``kwargs`` are rejected so the setup/runtime contract stays
        explicit.
        """

        if kwargs:
            raise TypeError(f"unsupported FunctionalDecoder.from_state_dict kwargs: {sorted(kwargs)}")

        model_args = _require_target_config(hf_config, layer_idx)

        if mesh_config is None:
            mesh_shape = getattr(mesh_device, "shape", (1, 1))
            mesh_config = MeshConfig(mesh_shape, decode=ModeConfig(tp=1))
        if mesh_config.tp != 1:
            raise NotImplementedError("functional-decoder bringup is single-chip only; got mesh_config.tp != 1")

        normalized_state = _normalize_layer_state_dict(state_dict, layer_idx)
        layer_state = {}
        for prefix in (f"model.layers.{layer_idx}", f"model.language_model.layers.{layer_idx}"):
            layer_state = substate(normalized_state, prefix)
            if layer_state:
                break

        return cls(
            mesh_device=mesh_device,
            hf_config=model_args,
            layer_idx=layer_idx,
            layer_state=layer_state,
            mesh_config=mesh_config,
            ccl_manager=ccl_manager,
            dtype=dtype,
            attention_dtype=attention_dtype,
            shared_mlp_dtype=shared_mlp_dtype,
            tensor_cache_path=tensor_cache_path,
            bounded_sliding_kv_cache=bounded_sliding_kv_cache,
        )

    def create_paged_kv_cache(
        self,
        *,
        block_size: int,
        max_num_blocks: int,
        cache_dtype=ttnn.bfloat16,
        tensor_cache_path: str | Path | None = None,
    ):
        """Allocate this layer's paged KV cache for tests or callers."""

        return init_kv_cache(
            mesh_device=self.mesh_device,
            config=self.attention_config,
            paged_attention_config=PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks),
            cache_dtype=cache_dtype,
            tensor_cache_path=str(tensor_cache_path) if tensor_cache_path is not None else None,
        )

    def forward(self, *, mode: str, **kwargs):
        if mode == "prefill":
            return self.prefill_forward(**kwargs)
        if mode == "decode":
            return self.decode_forward(**kwargs)
        raise ValueError(f"mode must be 'prefill' or 'decode', got {mode!r}")

    def _feed_forward(self, hidden_states):
        residual = hidden_states
        normed = self.pre_feedforward_layernorm.forward(hidden_states)
        mlp_output = self.shared_mlp(normed)
        normed.deallocate(True)

        hidden_states = self.post_feedforward_layernorm.forward(mlp_output)
        mlp_output.deallocate(True)
        combined = ttnn.add(residual, hidden_states)
        residual.deallocate(True)
        hidden_states.deallocate(True)

        if self.layer_scalar != 1.0:
            combined = ttnn.mul(combined, self.layer_scalar)
        return combined

    def _attention_block(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        is_decode: bool,
        position_idx=None,
        token_index=None,
        position_idx_cache=None,
    ):
        residual = hidden_states
        normed = self.input_layernorm.forward(hidden_states)
        attn_output = self.self_attn(
            normed,
            rope_mats=rope_mats,
            position_idx=position_idx,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=is_decode,
            token_index=token_index,
            position_idx_cache=position_idx_cache,
        )
        normed.deallocate(True)

        attn_output = self.post_attention_layernorm.forward(attn_output)
        hidden_states = ttnn.add(residual, attn_output)
        attn_output.deallocate(True)
        return hidden_states

    def prefill_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
    ):
        """Run a paged prefill pass and return device-resident hidden states."""

        hidden_states = self._attention_block(
            hidden_states,
            rope_mats=rope_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=False,
        )
        return self._feed_forward(hidden_states)

    def decode_forward(
        self,
        hidden_states,
        *,
        rope_mats,
        page_table,
        kv_cache,
        position_idx,
        token_index: int | None = None,
        position_idx_cache=None,
    ):
        """Run one paged decode step.

        ``position_idx`` must be a TTNN tensor.  For traced execution, pass 2D
        RoPE tables and leave ``token_index`` as ``None``; the current position
        is then consumed by on-device embedding lookup and cache update.
        """

        cos_cache, _ = rope_mats
        if len(cos_cache.shape) != 2 and token_index is None:
            raise ValueError("token_index is required when decode rope_mats are 4D tables")

        hidden_states = self._attention_block(
            hidden_states,
            rope_mats=rope_mats,
            page_table=page_table,
            kv_cache=kv_cache,
            is_decode=True,
            position_idx=position_idx,
            token_index=token_index,
            position_idx_cache=position_idx_cache,
        )
        return self._feed_forward(hidden_states)


__all__ = ["FunctionalDecoder", "SUPPORTED_HF_MODEL_ID"]
