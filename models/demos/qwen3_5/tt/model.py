# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5-27B full text transformer model wiring."""

from __future__ import annotations

from typing import Optional

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen3_5.tt.decoder import HybridTransformerBlock
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.embedding import Embedding
from models.tt_transformers.tt.lm_head import LMHead


class Qwen3_5Transformer(LightweightModule):
    """Qwen3.5-27B hybrid text transformer.

    Layer caches
    -------------
    For full_attention layers  : standard KV cache  (tuple[Tensor, Tensor] per layer)
    For linear_attention layers: conv_state + recurrent_state (Torch tensors, CPU)
    """

    def __init__(
        self,
        args,
        dtype,
        mesh_device,
        state_dict: dict,
        weight_cache_path,
        transformation_mats=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher=None,
        max_n_layers: int = None,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.dtype = dtype
        # Allow reducing number of layers for testing
        if max_n_layers is not None and max_n_layers < args.n_layers:
            args.n_layers = max_n_layers
            if hasattr(args, "linear_attention_pattern"):
                args.linear_attention_pattern = args.linear_attention_pattern[:max_n_layers]
            if hasattr(args, "sliding_window_pattern"):
                args.sliding_window_pattern = args.sliding_window_pattern[:max_n_layers]
        self.n_layers = args.n_layers

        self.tok_embeddings = Embedding(
            mesh_device=mesh_device,
            args=args,
            weight_cache_path=weight_cache_path,
            state_dict=state_dict,
            dtype=dtype,
        )

        self.layers = [
            HybridTransformerBlock(
                args=args,
                mesh_device=mesh_device,
                tt_ccl=None,
                dtype=dtype,
                state_dict=state_dict,
                layer_num=i,
                weight_cache_path=weight_cache_path,
                transformation_mats=transformation_mats,
                paged_attention_config=paged_attention_config,
                use_paged_kv_cache=use_paged_kv_cache,
                prefetcher=prefetcher,
            )
            for i in range(self.n_layers)
        ]

        _final_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix="",
            weight_cache_path=weight_cache_path if not args.dummy_weights else None,
            weight_dtype=ttnn.bfloat16,
            weight_key="norm",
            fp32_dest_acc_en=False,  # Keeps L1 CB within limits for dim=5120
        )
        self.norm = DistributedNorm(
            _final_norm,
            args,
            tt_ccl=None,
            prefetcher=None,
            TG=args.is_galaxy,
            ag_config_key=None,
        )

        # LM Head: DRAMSharded matmul (device-side) – splits vocab into chunks to
        # stay within L1 circular buffer limits (~26720 cols per split for dim=5120).
        self.lm_head = LMHead(
            args=args,
            mesh_device=mesh_device,
            tt_ccl=None,
            dtype=dtype,
            state_dict=state_dict,
            state_dict_prefix=args.get_state_dict_prefix("", None),
            weight_cache_path=weight_cache_path,
            max_columns_per_device=args.max_columns_per_device_lm_head,
        )

        self.linear_attention_pattern = getattr(args, "linear_attention_pattern", [False] * self.n_layers)

    # ------------------------------------------------------------------
    def _init_linear_states(self, batch: int):
        """Return lists of per-layer CPU state tensors for linear_attention layers."""
        num_v = self.args.linear_num_value_heads
        num_k = self.args.linear_num_key_heads
        k_dim = self.args.linear_key_head_dim
        v_dim = self.args.linear_value_head_dim
        conv_k = self.args.linear_conv_kernel_dim
        conv_dim = num_k * k_dim * 2 + num_v * v_dim

        conv_states, recurrent_states = [], []
        for is_linear in self.linear_attention_pattern:
            if is_linear:
                conv_states.append(torch.zeros(batch, conv_dim, conv_k - 1))
                recurrent_states.append(torch.zeros(batch, num_v, k_dim, v_dim))
            else:
                conv_states.append(None)
                recurrent_states.append(None)
        return conv_states, recurrent_states

    # ------------------------------------------------------------------
    def forward(
        self,
        tokens: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        kv_caches: Optional[list] = None,
        conv_states: Optional[list] = None,
        recurrent_states: Optional[list] = None,
    ):
        x = self.tok_embeddings(tokens)
        x = ttnn.unsqueeze_to_4D(x)

        new_kv_caches, new_conv_states, new_recurrent_states = [], [], []

        for i, layer in enumerate(self.layers):
            if mode == Mode.DECODE:
                x = ttnn.to_memory_config(x, self.args.get_residual_mem_config(mode, None))
            kv = kv_caches[i] if kv_caches is not None else None
            cv = conv_states[i] if conv_states is not None else None
            rv = recurrent_states[i] if recurrent_states is not None else None

            x, new_kv, new_cv, new_rv = layer.forward(
                x=x,
                current_pos=current_pos,
                rot_mats_global=rot_mats,
                user_id=user_id,
                mode=mode,
                page_table=page_table,
                kv_cache=kv,
                conv_state=cv,
                recurrent_state=rv,
            )

            new_kv_caches.append(new_kv)
            new_conv_states.append(new_cv)
            new_recurrent_states.append(new_rv)

        # Final norm: DistributedNorm reshards x to lm_head_core_grid before norm in decode.
        norm_cfg = self.args.get_norm_config("lm_head", mode)
        x = self.norm(x, mode, norm_config=norm_cfg)

        # For prefill the norm output is DRAM-interleaved; LMHead expects sharded input.
        if mode == Mode.PREFILL:
            lm_input_cfg = self.args.get_lm_head_input_mem_config(mode, None)
            if lm_input_cfg.is_sharded():
                x = ttnn.interleaved_to_sharded(x, lm_input_cfg)

        logits = self.lm_head(x)

        return logits, new_kv_caches, new_conv_states, new_recurrent_states
