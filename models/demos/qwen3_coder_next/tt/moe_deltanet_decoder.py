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
from models.tt_transformers.tt.gated_deltanet import GatedDeltaNet
from models.tt_transformers.tt.model_config import TensorGroup


class ReplicatedGQAAttention(LightweightModule):
    """Replicated GQA attention for layers where n_kv_heads < num_devices.

    Each device computes all query/KV heads independently (replicated weights).
    Single-token decode: softmax of 1 element = 1.0, so output = V projected.
    """

    def __init__(self, mesh_device, args, state_dict, layer_num, dtype):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self._is_mesh = hasattr(mesh_device, "get_num_devices") and mesh_device.get_num_devices() > 1
        rep_kw = {"mesh_mapper": ttnn.ReplicateTensorToMesh(mesh_device)} if self._is_mesh else {}

        prefix = args.get_state_dict_prefix("", layer_num) + "self_attn."

        def load_w(name):
            w = state_dict[f"{prefix}{name}.weight"]
            return ttnn.as_tensor(
                w.transpose(-2, -1).contiguous().unsqueeze(0).unsqueeze(0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **rep_kw,
            )

        self.wq = load_w("q_proj")
        self.wk = load_w("k_proj")
        self.wv = load_w("v_proj")
        self.wo = load_w("o_proj")

    def forward(self, x, current_pos=None, rot_mats=None, mode="decode", **kwargs):
        q = ttnn.linear(x, self.wq)
        k = ttnn.linear(x, self.wk)
        v = ttnn.linear(x, self.wv)
        gqa_ratio = self.n_heads // self.n_kv_heads
        v_expanded = ttnn.repeat_interleave(v, gqa_ratio, dim=3)
        ttnn.deallocate(q)
        ttnn.deallocate(k)
        ttnn.deallocate(v)
        output = ttnn.linear(v_expanded, self.wo)
        ttnn.deallocate(v_expanded)
        return output


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
            # GQA layer — n_kv_heads=2 < num_devices=8, replicate across all devices
            self.attention = ReplicatedGQAAttention(
                mesh_device=mesh_device,
                args=args,
                state_dict=state_dict,
                layer_num=layer_num,
                dtype=dtype,
            )

        # MoE FFN (replaces the standard MLP)
        self.feed_forward = MoELayer(
            mesh_device=mesh_device,
            args=args,
            state_dict=state_dict,
            layer_num=layer_num,
            dtype=dtype,
            device_id=device_id,
        )

        # Plain RMSNorm (no DistributedNorm — hidden_size=2048 is too small for sharded norm)
        norm_prefix = args.get_state_dict_prefix("", layer_num)
        self.attention_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=norm_prefix,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="attention_norm",
            is_distributed=False,
            add_unit_offset=args.rms_norm_add_unit_offset,
        )
        self.ff_norm = RMSNorm(
            device=mesh_device,
            dim=args.dim,
            eps=args.norm_eps,
            state_dict=state_dict,
            state_dict_prefix=norm_prefix,
            weight_cache_path=None if args.dummy_weights else weight_cache_path,
            weight_dtype=ttnn.bfloat16,
            weight_key="ffn_norm",
            is_distributed=False,
            add_unit_offset=args.rms_norm_add_unit_offset,
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
        if getattr(self.args, "is_qwen3_next", False):
            skip_mem_cfg = ttnn.DRAM_MEMORY_CONFIG
        else:
            skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)
        x = ttnn.to_memory_config(x, skip_mem_cfg)
        residual = x

        # Attention norm
        attn_in = self.attention_norm(x, mode)

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
        hidden_states = self.ff_norm(hidden_states, mode)
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
