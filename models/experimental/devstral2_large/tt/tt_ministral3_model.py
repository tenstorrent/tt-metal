# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent stack for Hugging Face ``Ministral3Model`` (no ``lm_head``).

Mirrors ``transformers.models.ministral3.modeling_ministral3.Ministral3Model`` and
``models/experimental/devstral2_large/reference/model_structure.txt``:

- ``embed_tokens`` → :class:`~models.tt_transformers.tt.embedding.Embedding` (meta key ``tok_embeddings.weight``);
  on multi-device meshes the embedding output can be **width-sharded** on dim 3, so we **all-gather**
  concatenate to full ``args.dim`` before the first RMSNorm (same helper as ``TtMinistral3DecoderLayer``).
- ``layers`` → list of :class:`~models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer.TtMinistral3DecoderLayer`
- ``norm`` → :class:`~models.experimental.devstral2_large.tt.tt_ministralrmsnorm.TtDevstral2LargeRMSNorm`
  (``model_final_norm=True``, meta key ``norm.weight``; BH-wide prefill uses the same workaround as layer norms)
- ``rotary_emb`` → :class:`~models.experimental.devstral2_large.tt.tt_ministral_rotary_emb.TtDevstral2LargeRotaryEmbedding`
  (HF-aligned cos/sin on device via NumPy tables + slice, same as ``prepare_inputs_prefill`` in ``Transformer``).

``state_dict`` must follow meta naming from :func:`~models.tt_transformers.tt.load_checkpoints.map_hf_to_meta_keys`
(``layers.N.*``, ``tok_embeddings.weight``, ``norm.weight``). ``attention_mask`` is accepted for API parity with HF
but ignored (TT attention is causal on device). ``use_cache`` / ``past_key_values`` are not implemented yet.

Forward uses **ttnn only** for position indices and RoPE slicing (no PyTorch in the hot path).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.devstral2_large.tt.tt_ministral3_decoder_layer import (
    TtMinistral3DecoderLayer,
    _all_gather_concat_hidden_dim,
)
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtDevstral2LargeRMSNorm
from models.experimental.devstral2_large.tt.tt_ministral_rotary_emb import TtDevstral2LargeRotaryEmbedding
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.embedding import Embedding


@dataclass
class TtMinistral3ModelOutput:
    """TTNN analogue of ``transformers.modeling_outputs.BaseModelOutputWithPast`` (subset)."""

    last_hidden_state: ttnn.Tensor
    past_key_values: Optional[Any] = None


def _text_config_from_model_args(args) -> Any:
    cfg = getattr(args, "hf_config", None)
    if cfg is None:
        raise ValueError("ModelArgs.hf_config is required for TtDevstral2LargeRotaryEmbedding / Ministral3Config")
    return getattr(cfg, "text_config", None) or cfg


class TtMinistral3Model(LightweightModule):
    """
    TTNN ``Ministral3Model``: embedding, decoder stack, final RMSNorm, TT HF-format rotary tables.

    Forward order matches HF: build ``inputs_embeds`` if needed, compute ``position_embeddings`` once,
    run each ``Ministral3DecoderLayer``-equivalent block, apply final ``norm``.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
    ):
        super().__init__()

        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher

        text_cfg = _text_config_from_model_args(args)
        head_dim = int(getattr(text_cfg, "head_dim", None) or text_cfg.hidden_size // text_cfg.num_attention_heads)
        rope_params = getattr(text_cfg, "rope_parameters", None) or {}
        if not isinstance(rope_params, dict):
            rope_params = dict(rope_params)
        self._llama_4_scaling_beta = rope_params.get("llama_4_scaling_beta")
        self._original_max_position_embeddings = rope_params.get("original_max_position_embeddings")

        self.embed_tokens = Embedding(mesh_device, args, weight_cache_path, state_dict, dtype)
        self.layers = [
            TtMinistral3DecoderLayer(
                args,
                mesh_device,
                tt_ccl,
                dtype,
                state_dict,
                layer_num=i,
                weight_cache_path=weight_cache_path,
                transformation_mats=transformation_mats,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher=prefetcher,
                llama_4_scaling_beta=self._llama_4_scaling_beta,
                original_max_position_embeddings=self._original_max_position_embeddings,
            )
            for i in range(args.n_layers)
        ]
        self.norm = TtDevstral2LargeRMSNorm(
            mesh_device,
            args,
            state_dict,
            weight_cache_path,
            0,
            tt_ccl,
            model_final_norm=True,
        )
        self.rotary_emb = TtDevstral2LargeRotaryEmbedding(
            mesh_device,
            batch_size=args.max_batch_size,
            head_dim=head_dim,
            max_seq_len=args.max_seq_len,
            config=text_cfg,
            datatype=dtype,
            prefetcher=prefetcher,
        )

    def _embed_input_ids(self, input_ids: ttnn.Tensor, mode: Mode) -> ttnn.Tensor:
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        embd_out = self.embed_tokens(
            input_ids,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.prefetcher is None else skip_mem_cfg,
        )
        out = ttnn.unsqueeze_to_4D(embd_out)
        return ttnn.to_memory_config(out, skip_mem_cfg)

    def _default_position_ids_tt(self, batch_size: int, seq_len: int, *, past_seen_tokens: int = 0) -> ttnn.Tensor:
        """Contiguous indices ``past_seen_tokens .. past_seen_tokens+seq_len-1`` per row (uint32, ROW_MAJOR)."""
        row = ttnn.arange(
            past_seen_tokens,
            past_seen_tokens + seq_len,
            1,
            dtype=ttnn.uint32,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        )
        row = ttnn.reshape(row, (1, seq_len))
        if batch_size == 1:
            return row
        return ttnn.repeat(row, (batch_size, 1))

    def _ensure_position_ids_bs_row_major_uint32(
        self, pos_tt: ttnn.Tensor, batch_size: int, seq_len: int
    ) -> ttnn.Tensor:
        if tuple(int(x) for x in pos_tt.shape) != (batch_size, seq_len):
            raise ValueError(
                f"position_ids shape {tuple(pos_tt.shape)} does not match activations (batch={batch_size}, seq={seq_len})"
            )
        out = pos_tt
        if out.layout != ttnn.ROW_MAJOR_LAYOUT:
            out = ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)
        if out.dtype != ttnn.uint32:
            out = ttnn.typecast(out, ttnn.uint32)
        return out

    def _expand_rot_mats_batch_if_needed(self, rot_mats, batch_dim: int):
        if rot_mats is None or batch_dim <= 1:
            return rot_mats
        c0 = rot_mats[0]
        if int(c0.shape[1]) >= batch_dim:
            return rot_mats
        return [
            ttnn.repeat(c0, (1, batch_dim, 1, 1)),
            ttnn.repeat(rot_mats[1], (1, batch_dim, 1, 1)),
        ]

    def _prepare_rot_mats_prefill(
        self,
        rot_mats,
        rope_start_pos: int,
        batch_dim: int,
        seq_len: int,
    ):
        if rot_mats is not None:
            return self._expand_rot_mats_batch_if_needed(rot_mats, batch_dim)
        rot_mats = self.rotary_emb.slice_rot_mats_prefill(rope_start_pos, seq_len)
        return self._expand_rot_mats_batch_if_needed(rot_mats, batch_dim)

    def forward(
        self,
        input_ids: Optional[ttnn.Tensor] = None,
        attention_mask: Any = None,
        position_ids: Optional[ttnn.Tensor] = None,
        past_key_values: Any = None,
        inputs_embeds: Optional[ttnn.Tensor] = None,
        use_cache: Optional[bool] = None,
        *,
        mode: Mode | str = Mode.PREFILL,
        current_pos=None,
        rot_mats_global: Optional[list[ttnn.Tensor]] = None,
        rot_mats_local: Optional[list[ttnn.Tensor]] = None,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size: int = 1,
        rope_start_pos: int = 0,
        **_kwargs: Any,
    ) -> TtMinistral3ModelOutput:
        if isinstance(mode, str):
            mode = Mode(mode)

        if use_cache:
            raise NotImplementedError("TtMinistral3Model does not implement use_cache yet.")
        if past_key_values is not None:
            raise NotImplementedError("TtMinistral3Model does not implement past_key_values yet.")

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds (Hugging Face Ministral3Model rule).")

        if attention_mask is not None:
            pass  # API parity with HF; causal masking is implicit on device.

        if inputs_embeds is None:
            assert input_ids is not None
            hidden_states = self._embed_input_ids(input_ids, mode)
        else:
            hidden_states = inputs_embeds

        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        assert (
            hidden_states.memory_config() == skip_mem_cfg
        ), f"hidden_states memcfg {hidden_states.memory_config()} != {skip_mem_cfg}"

        if int(hidden_states.shape[-1]) != self.args.dim:
            hidden_states = _all_gather_concat_hidden_dim(
                self.mesh_device,
                self.tt_ccl,
                hidden_states,
                topology=self.args.ccl_topology(),
                dtype=self.args.ccl_dtype,
            )
            hidden_states = ttnn.to_memory_config(hidden_states, skip_mem_cfg)

        b = int(hidden_states.shape[0])
        s = int(hidden_states.shape[2])

        if position_ids is None:
            pos_tt = self._default_position_ids_tt(b, s, past_seen_tokens=0)
        else:
            pid = position_ids
            if len(pid.shape) == 1:
                if int(pid.shape[0]) != s:
                    raise ValueError(f"1D position_ids length {int(pid.shape[0])} does not match sequence length {s}")
                pid = ttnn.reshape(pid, (1, s))
                if b > 1:
                    pid = ttnn.repeat(pid, (b, 1))
            pos_tt = self._ensure_position_ids_bs_row_major_uint32(pid, b, s)

        rot_mats_global = self._prepare_rot_mats_prefill(rot_mats_global, rope_start_pos, b, s)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                current_pos=current_pos,
                rot_mats_global=rot_mats_global,
                rot_mats_local=rot_mats_local,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
                position_ids=pos_tt,
            )

        norm_cfg = self.args.get_norm_config("lm_head", mode, self.prefetcher)
        hidden_states = self.norm(hidden_states, mode=mode, norm_config=norm_cfg)

        return TtMinistral3ModelOutput(last_hidden_state=hidden_states, past_key_values=None)


__all__ = [
    "TtMinistral3Model",
    "TtMinistral3ModelOutput",
]
