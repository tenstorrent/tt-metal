# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""
Tenstorrent decoder layer for Hugging Face ``Ministral3DecoderLayer``.

Mirrors ``transformers.models.ministral3.modeling_ministral3.Ministral3DecoderLayer`` structure:

- ``self_attn`` → :class:`~models.experimental.devstral2_large.tt.tt_ministralattn.TtDevstral2LargeAttention`
- ``mlp`` → :class:`~models.experimental.devstral2_large.tt.tt_ministralmlp.TtDevstral2LargeMLP`
- ``input_layernorm`` / ``post_attention_layernorm`` →
  :class:`~models.experimental.devstral2_large.tt.tt_ministralrmsnorm.TtDevstral2LargeRMSNorm`
  (meta keys ``attention_norm`` / ``ffn_norm``)

Forward matches HF order (pre-norm attention block, residual, pre-norm MLP block, residual). Rotary
``position_embeddings`` from HF are supplied here as ``rot_mats_global`` / ``rot_mats_local`` (``[cos,
sin]`` device tensors), computed once per forward of :class:`Ministral3Model` and reused per layer.
"""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.tt_transformers.tt.ccl import tt_all_gather
from models.experimental.devstral2_large.tt.model_utils import (
    get_decode_mem_config_after_hidden_dim_concat,
)
from models.experimental.devstral2_large.tt.tt_ministralattn import TtDevstral2LargeAttention
from models.experimental.devstral2_large.tt.tt_ministralmlp import TtDevstral2LargeMLP
from models.experimental.devstral2_large.tt.tt_ministralrmsnorm import TtDevstral2LargeRMSNorm
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import TensorGroup


def _all_gather_concat_hidden_dim(mesh_device, tt_ccl, tensor, *, topology, dtype):
    """Concatenate width-sharded activations on dim 3 for residual add / norms.

    On a **line mesh** (``1`` in ``mesh_device.shape``, e.g. 1×4), ``cluster_axis=0`` only has one
    device so ``all_gather_async`` fails with *num_devices > 1, but has 1*. ``cluster_axis=1`` hits
    the early-return in ``tt_all_gather`` whenever ``1 in mesh_shape``. Use the no-axis gather path
    (same idea as fused attention ``all_gather_async`` without ``cluster_axis``).
    """
    mesh_shape = list(mesh_device.shape)
    cluster_axis = None if 1 in mesh_shape else 0
    return tt_all_gather(
        tensor,
        mesh_device,
        tt_ccl,
        cluster_axis,
        dim=3,
        topology=topology,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=dtype,
    )


def _resolve_llama4_rope_kwargs(args, llama_4_scaling_beta, original_max_position_embeddings):
    """Fill Llama-4 Q-scale kwargs from explicit args or ``rope_parameters``-like mapping on ``args``."""
    beta, orig_max = llama_4_scaling_beta, original_max_position_embeddings
    rope_params = getattr(args, "rope_parameters", None)
    if rope_params is None and hasattr(args, "hf_config"):
        hf_cfg = getattr(args.hf_config, "text_config", None) or args.hf_config
        rope_params = getattr(hf_cfg, "rope_parameters", None)
    if isinstance(rope_params, dict):
        if beta is None:
            beta = rope_params.get("llama_4_scaling_beta")
        if orig_max is None:
            orig_max = rope_params.get("original_max_position_embeddings")
    return beta, orig_max


class TtMinistral3DecoderLayer(LightweightModule):
    """
    TTNN equivalent of ``Ministral3DecoderLayer`` (same public attribute names as Hugging Face).

    Parameters mirror :class:`~models.tt_transformers.tt.decoder.TransformerBlock` where applicable;
    Ministral3 uses dense SwiGLU MLP and two RMSNorms without extra feed-forward norms.
    """

    def __init__(
        self,
        args,
        mesh_device,
        tt_ccl,
        dtype,
        state_dict,
        layer_num,
        weight_cache_path,
        transformation_mats,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
        llama_4_scaling_beta=None,
        original_max_position_embeddings=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.num_devices = args.num_devices
        self.args = args
        self.hidden_size = args.dim
        self.layer_num = layer_num
        self.layer_idx = layer_num  # same meaning as HF ``Ministral3DecoderLayer(layer_idx=...)``
        self.model_config = args.get_model_config()

        beta, orig_max = _resolve_llama4_rope_kwargs(
            args,
            llama_4_scaling_beta,
            original_max_position_embeddings,
        )

        self.self_attn = TtDevstral2LargeAttention(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher=prefetcher,
            llama_4_scaling_beta=beta,
            original_max_position_embeddings=orig_max,
        )

        self.mlp = TtDevstral2LargeMLP(
            mesh_device=mesh_device,
            tt_ccl=tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher=prefetcher,
        )

        self.input_layernorm = TtDevstral2LargeRMSNorm(
            mesh_device,
            args,
            state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=False,
        )
        self.post_attention_layernorm = TtDevstral2LargeRMSNorm(
            mesh_device,
            args,
            state_dict,
            weight_cache_path,
            layer_num,
            tt_ccl,
            post_attention=True,
        )

    def forward(
        self,
        hidden_states: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id: int = 0,
        mode: Mode | str = Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size: int = 1,
        position_ids: ttnn.Tensor | None = None,
    ) -> ttnn.Tensor:
        """
        Apply one decoder layer (HF ``Ministral3DecoderLayer.forward`` ordering on device).

        ``rot_mats_*`` correspond to HF ``position_embeddings`` (cos/sin) prepared outside the layer.
        ``position_ids`` is an optional device tensor for prefill Llama-4 query scaling inside attention.
        """
        if isinstance(mode, str):
            mode = Mode(mode)

        TG = self.args.is_galaxy
        x = hidden_states
        residual = x
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        decode_full_hidden_mem = None
        if mode == Mode.DECODE and self.prefetcher is None and not TG:
            decode_full_hidden_mem = get_decode_mem_config_after_hidden_dim_concat(self.args, mode, self.prefetcher)

        assert x.memory_config() == skip_mem_cfg or (
            decode_full_hidden_mem is not None and x.memory_config() == decode_full_hidden_mem
        ), (
            "decoder input memcfg mismatch: "
            f"{x.memory_config()} not in (residual={skip_mem_cfg}, full_hidden={decode_full_hidden_mem})"
        )

        rot_mats = (
            rot_mats_local if (hasattr(self.self_attn, "is_sliding") and self.self_attn.is_sliding) else rot_mats_global
        )

        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.input_layernorm(x, mode, norm_config=attn_norm_config)
        if mode == Mode.PREFILL:
            attn_in = ttnn.to_memory_config(attn_in, ttnn.L1_MEMORY_CONFIG)

        if batch_size > 1:
            attn_in = ttnn.reshape(attn_in, [batch_size, 1, attn_in.shape[-2] // batch_size, -1])

        if mode == Mode.PREFILL:
            attn_out = self.self_attn.forward_prefill(
                attn_in,
                rot_mats,
                user_id=user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                position_ids=position_ids,
            )
        else:
            attn_out = self.self_attn.forward(
                attn_in,
                current_pos,
                rot_mats,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
            )

        if mode == Mode.PREFILL and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])

        # TP ``wo`` / MLP outputs are width-sharded (disjoint feature slices per device) while tests and
        # local RMSNorm paths use a full ``hidden_size`` activation per device. ``ttnn.add`` needs
        # matching last dims. Do **not** use ``tt_all_reduce`` here: on line meshes (``1`` in mesh
        # shape) that helper runs ``reduce_scatter`` and *shrinks* dim 3 instead of replicating /
        # concatenating shards. Concatenate TP shards along the hidden axis like attention's fused
        # ``all_gather_async`` before ``wo``.
        if residual.shape[-1] != attn_out.shape[-1]:
            attn_out = _all_gather_concat_hidden_dim(
                self.mesh_device,
                self.tt_ccl,
                attn_out,
                topology=self.self_attn.ccl_topology,
                dtype=self.self_attn.ccl_dtype,
            )

        attn_residual_mem = skip_mem_cfg
        if mode == Mode.DECODE and self.prefetcher is None and not TG and int(attn_out.shape[-1]) == int(self.args.dim):
            attn_residual_mem = get_decode_mem_config_after_hidden_dim_concat(self.args, mode, self.prefetcher)
        attn_out = ttnn.to_memory_config(attn_out, attn_residual_mem)

        hidden_states = ttnn.add(
            residual,
            attn_out,
            memory_config=attn_residual_mem,
            dtype=ttnn.bfloat16 if TG else None,
        )
        residual = hidden_states
        if mode == Mode.PREFILL:
            x.deallocate(True)

        ttnn.deallocate(attn_out)

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.post_attention_layernorm(hidden_states, mode, norm_config=ff_norm_config)
        if mode == Mode.PREFILL:
            hidden_states = ttnn.to_memory_config(hidden_states, ttnn.L1_MEMORY_CONFIG)

        if TG and mode == Mode.DECODE:
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.args.get_mlp_act_mem_config(mode))

        hidden_states = self.mlp.forward(hidden_states, mode)

        if residual.shape[-1] != hidden_states.shape[-1]:
            hidden_states = _all_gather_concat_hidden_dim(
                self.mesh_device,
                self.tt_ccl,
                hidden_states,
                topology=self.args.ccl_topology(),
                dtype=self.args.ccl_dtype,
            )

        mlp_add_mem = skip_mem_cfg
        if (
            mode == Mode.DECODE
            and self.prefetcher is None
            and not TG
            and int(residual.shape[-1]) == int(self.args.dim)
            and int(hidden_states.shape[-1]) == int(self.args.dim)
        ):
            mlp_add_mem = get_decode_mem_config_after_hidden_dim_concat(self.args, mode, self.prefetcher)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=mlp_add_mem,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return out


__all__ = [
    "TtMinistral3DecoderLayer",
]
