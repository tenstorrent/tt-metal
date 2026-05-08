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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
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

    def _default_position_ids_torch(self, batch_size: int, seq_len: int, *, past_seen_tokens: int = 0) -> torch.Tensor:
        row = torch.arange(seq_len, dtype=torch.long) + past_seen_tokens
        return row.unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _contiguous_start_pos_from_position_ids(pos_torch: torch.Tensor) -> Optional[int]:
        """Return start ``p`` if each row equals ``[p, p+1, ..., p+S-1]`` (sliceable RoPE cache), else ``None``."""
        b, s = pos_torch.shape
        row0 = pos_torch[0]
        p = int(row0[0].item())
        expected = torch.arange(p, p + s, dtype=pos_torch.dtype, device=pos_torch.device)
        if not torch.equal(row0, expected):
            return None
        for i in range(1, b):
            if not torch.equal(pos_torch[i], row0):
                return None
        return p

    def _rot_mats_from_tt_rope(
        self,
        hidden_states: ttnn.Tensor,
        position_ids_torch: torch.Tensor,
    ) -> list[ttnn.Tensor]:
        """Slice pre-uploaded HF-format cos/sin (same idea as ``Transformer.prepare_inputs_prefill``)."""
        b = int(hidden_states.shape[0])
        s = int(hidden_states.shape[2])
        if tuple(int(x) for x in position_ids_torch.shape) != (b, s):
            raise ValueError(
                f"position_ids shape {tuple(position_ids_torch.shape)} does not match activations (batch={b}, seq={s})"
            )
        start_pos = self._contiguous_start_pos_from_position_ids(position_ids_torch.long())
        if start_pos is None:
            raise ValueError(
                "position_ids must be contiguous and identical across batch rows to slice "
                "TtDevstral2LargeRotaryEmbedding caches; otherwise pass rot_mats_global=[cos_tt, sin_tt]."
            )
        end = start_pos + s
        if end > self.rotary_emb.max_seq_len:
            raise ValueError(
                f"RoPE slice [{start_pos}, {end}) exceeds rotary_emb.max_seq_len={self.rotary_emb.max_seq_len}; "
                "increase ModelArgs.max_seq_len or pass rot_mats_global."
            )
        cos_slice = self.rotary_emb.cos_matrix_prefill[:, :, start_pos:end, :]
        sin_slice = self.rotary_emb.sin_matrix_prefill[:, :, start_pos:end, :]
        if b > 1:
            cos_slice = ttnn.repeat(cos_slice, (1, b, 1, 1))
            sin_slice = ttnn.repeat(sin_slice, (1, b, 1, 1))
        return [cos_slice, sin_slice]

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
            pos_torch = self._default_position_ids_torch(b, s, past_seen_tokens=0)
        else:
            pos_torch = ttnn.to_torch(position_ids).long()
            if pos_torch.ndim == 1:
                pos_torch = pos_torch.unsqueeze(0).expand(b, -1)

        pos_tt = ttnn.from_torch(
            pos_torch.to(torch.int32),
            device=self.mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        if rot_mats_global is None:
            rot_mats_global = self._rot_mats_from_tt_rope(hidden_states, pos_torch)

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
