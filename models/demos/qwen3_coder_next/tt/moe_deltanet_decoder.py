# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Decoder block with MoE FFN for Qwen3-Coder-Next (LightweightModule / ttnn).

Each decoder block contains:
1. Attention RMSNorm
2. Attention (GatedDeltaNet for 3/4 layers, GatedAttention for 1/4 layers)
3. Residual connection
4. FFN RMSNorm
5. MoE FFN (512 experts, top-10 routing + shared expert)
6. Residual connection

The attention type is determined by layer_idx and args.is_linear_attention_layer().

Has the same forward signature as DeltaNetDecoderBlock from qwen35_decoder.py
so the model forward loop doesn't need per-layer dispatch logic.
"""

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.demos.qwen3_coder_next.tt.moe import MoELayer
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet
from models.tt_transformers.tt.model_config import TensorGroup


class _IdentityAttention:
    """Placeholder for GQA layers — returns input unchanged.
    TODO: replace with proper replicated GQA attention."""

    def forward(self, x, **kwargs):
        return ttnn.mul(x, 0.0)  # Zero contribution (skip attention, only MoE contributes)


class MoEDeltaNetDecoderBlock(LightweightModule):
    """Decoder block wrapping attention (DeltaNet or GatedAttention) + MoE FFN.

    Has the same forward signature as DeltaNetDecoderBlock so the model forward
    loop can treat all layers uniformly.
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
        prefetcher=None,
        attention_class=None,
        device_id=0,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.prefetcher = prefetcher
        self.layer_num = layer_num

        # Attention: DeltaNet for linear attention layers, GQA for full attention layers
        is_deltanet = (
            args.is_linear_attention_layer(layer_num)
            if hasattr(args, "is_linear_attention_layer")
            else (attention_class is None)
        )

        if is_deltanet:
            self.attention = GatedDeltaNet(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
            )
            self.attention.initialize_states(batch_size=getattr(args, "max_batch_size", 1))
        elif attention_class is not None:
            self.attention = attention_class(
                mesh_device=mesh_device,
                tt_ccl=tt_ccl,
                args=args,
                state_dict=state_dict,
                weight_cache_path=weight_cache_path,
                layer_num=layer_num,
                dtype=dtype,
                transformation_mats=None,
                configuration=args,
            )
        else:
            # GQA layer — n_kv_heads=2 < num_devices=8, so use DeltaNet in force_replicated mode
            # DeltaNet ignores GQA-specific keys and loads its own weight format.
            # For GQA layers, use an identity-like passthrough until proper GQA is implemented.
            # TODO: implement proper replicated GQA attention for layers 3,7,11,...,47
            self.attention = _IdentityAttention()

        # MoE FFN (replaces the standard MLP)
        self.feed_forward = MoELayer(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
            device_id=device_id,
        )

        # Attention norm
        self.attention_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="attention_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )

        # FFN norm
        self.ff_norm = DistributedNorm(
            RMSNorm(
                device=mesh_device,
                dim=args.dim,
                eps=args.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                weight_cache_path=None if args.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key="ffn_norm",
                is_distributed=args.is_distributed_norm,
                add_unit_offset=args.rms_norm_add_unit_offset,
                ccl_topology=args.ccl_topology(),
                tt_ccl=tt_ccl,
            ),
            args,
            tt_ccl=tt_ccl,
            prefetcher=prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )

    def forward(
        self,
        x: ttnn.Tensor,
        current_pos,
        rot_mats_global=None,
        rot_mats_local=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ) -> ttnn.Tensor:
        """Forward pass with same signature as DeltaNetDecoderBlock.

        DeltaNet layers ignore: current_pos, rot_mats, page_table, kv_cache.
        GQA layers use current_pos and rot_mats_global for RoPE.
        """
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        x = ttnn.to_memory_config(x, skip_mem_cfg)
        residual = x

        # Attention norm
        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        # Attention forward (DeltaNet or GatedAttention)
        if hasattr(self.attention, "initialize_states"):
            attn_out = self.attention.forward(attn_in)
        else:
            attn_out = self.attention.forward(attn_in, current_pos=current_pos, rot_mats=rot_mats_global, mode=mode)

        # Residual add after attention
        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)
        hidden_states = ttnn.add(residual, attn_out, memory_config=skip_mem_cfg)
        residual = hidden_states
        if mode == Mode.PREFILL:
            x.deallocate(True)
        ttnn.deallocate(attn_out)

        # FFN norm + MoE FFN
        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)
        hidden_states = self.feed_forward(hidden_states)

        # Residual add after MoE
        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=activation_dtype or ttnn.bfloat16,
        )
        return out
