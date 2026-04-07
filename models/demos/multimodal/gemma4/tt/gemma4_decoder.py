# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Gemma 4 Transformer Decoder Block.

Extends the base TransformerBlock with:
- Per-layer config proxy for hybrid attention (sliding vs full)
- Layer scalar (trainable per-layer scaling factor)
- is_sliding attribute on attention for RoPE selection in the decoder forward
"""

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.common import Mode

import os

from .gemma4_attention import Gemma4Attention
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup

from .gemma4_layer_config import Gemma4LayerConfig


class Gemma4TransformerBlock(LightweightModule):
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
        transformation_mats_global=None,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        attention_class=None,
        prefetcher=None,
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.prefetcher = prefetcher
        self.num_devices = args.num_devices
        self.args = args
        self.hidden_size = args.dim
        self.n_heads = args.n_heads
        self.head_dim = args.get_layer_head_dim(layer_num)
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.get_layer_n_kv_heads(layer_num)
        self.current = 0
        self.model_config = args.get_model_config()
        self.is_mixture_of_experts = False
        self.layer_num = layer_num

        # Determine layer type
        self.is_sliding = args.is_layer_sliding(layer_num)

        # Create per-layer config proxy for attention
        layer_config = Gemma4LayerConfig(args, layer_num)

        # Gemma 4 requires v_norm (parameter-free RMSNorm on V) for correct output
        ActualAttentionClass = attention_class if attention_class is not None else Gemma4Attention

        # For attention, we pass both the original args (for matmul program configs)
        # and the per-layer proxy as configuration (for head_dim, n_kv_heads, etc.)
        # The proxy is also passed as args so that matmul configs use the correct qkv_size
        # We need to ensure the proxy also provides correct program configs
        # by overriding the cache-sensitive qkv_size
        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,  # Original args for matmul program configs
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=layer_config,  # Per-layer config proxy for dimensions
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
            prefetcher=prefetcher,
        )

        # Mark the attention with is_sliding so decoder forward can select RoPE
        self.attention.is_sliding = self.is_sliding

        # Set sliding_window on the attention for SDPA
        if self.is_sliding:
            self.attention.sliding_window = args.sliding_window
        else:
            self.attention.sliding_window = None  # Full attention, no sliding window

            # For full attention layers (head_dim=512), the standard RoPE kernel
            # only supports head_dim <= 256. Implement partial rotary embedding:
            # Apply RoPE to first rotary_dim dims, pass through the rest unchanged.
            rotary_dim = int(args.global_head_dim * args.partial_rotary_factor)  # 512 * 0.25 = 128
            full_head_dim = args.global_head_dim  # 512

            # Get global transformation matrices for partial RoPE (head_dim=128)
            global_trans_mats = transformation_mats_global if transformation_mats_global else transformation_mats

            # TEMPORARY: Skip RoPE for full attention layers to isolate the error source
            skip_full_attn_rope = os.environ.get("GEMMA4_SKIP_FULL_ROPE", "0") == "1"

            def _partial_rope_prefill(q, k, rot_mats, _rotary_dim=rotary_dim, _skip=skip_full_attn_rope):
                if _skip:
                    return ttnn.clone(q), ttnn.clone(k)
                # Q/K are in HF format (unpermuted). rot_mats are HF-format cos/sin.
                # Use ttnn.experimental.rotary_embedding (HF-style, no trans_mats).
                q_rot = q[:, :, :, :_rotary_dim]
                q_pass = q[:, :, :, _rotary_dim:]
                k_rot = k[:, :, :, :_rotary_dim]
                k_pass = k[:, :, :, _rotary_dim:]

                if q_rot.dtype != ttnn.bfloat16:
                    q_rot = ttnn.typecast(q_rot, dtype=ttnn.bfloat16)
                if k_rot.dtype != ttnn.bfloat16:
                    k_rot = ttnn.typecast(k_rot, dtype=ttnn.bfloat16)

                q_rot = ttnn.experimental.rotary_embedding(q_rot, rot_mats[0], rot_mats[1])
                k_rot = ttnn.experimental.rotary_embedding(k_rot, rot_mats[0], rot_mats[1])

                q_result = ttnn.concat([q_rot, q_pass], dim=-1)
                k_result = ttnn.concat([k_rot, k_pass], dim=-1)
                return q_result, k_result

            def _partial_rope_decode(q, k, rot_mats, current_pos, _rotary_dim=rotary_dim, _full_hd=full_head_dim):
                # Must clone because base attention deallocates the pre-rotation
                # tensors after this returns (lines 687-688 in attention.py).
                # If we return the same objects, they get deallocated then reused.
                # TODO: Apply proper partial RoPE once validated.
                return ttnn.clone(q), ttnn.clone(k)

            self.attention.rotary_embedding_prefill = _partial_rope_prefill
            self.attention.rotary_embedding_decode = _partial_rope_decode

        # MLP (same for all layers)
        self.feed_forward = MLP(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            args=args,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            model_config=self.model_config,
            prefetcher=prefetcher,
        )

        # Norms - same structure as Gemma 3 (4 norms per layer)
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
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=args.is_galaxy,
            ag_config_key="ATTN_LN_AG_CONFIG",
        )

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
                is_distributed=self.args.is_distributed_norm,
                add_unit_offset=self.args.rms_norm_add_unit_offset,
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )

        # Pre-feedforward norm
        prefix = args.get_state_dict_prefix("", layer_num)
        if f"layers.{layer_num}.pre_feedforward_layernorm.weight" in state_dict:
            self.pre_ff_norm = DistributedNorm(
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    state_dict=state_dict,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key="pre_feedforward_layernorm",
                    is_distributed=self.args.is_distributed_norm,
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                prefetcher=self.prefetcher,
                TG=args.is_galaxy,
            )
            self.ff_norm.enable_all_gather = False
        else:
            self.pre_ff_norm = None

        # Post-feedforward norm
        if f"layers.{layer_num}.post_feedforward_layernorm.weight" in state_dict:
            self.post_ff_norm = DistributedNorm(
                RMSNorm(
                    device=mesh_device,
                    dim=args.dim,
                    eps=args.norm_eps,
                    add_unit_offset=self.args.rms_norm_add_unit_offset,
                    state_dict=state_dict,
                    state_dict_prefix=args.get_state_dict_prefix("", layer_num),
                    weight_cache_path=None if args.dummy_weights else weight_cache_path,
                    weight_dtype=ttnn.bfloat16,
                    weight_key="post_feedforward_layernorm",
                    is_distributed=self.args.is_distributed_norm,
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                prefetcher=self.prefetcher,
                TG=args.is_galaxy,
                enable_all_gather=False,
            )
        else:
            self.post_ff_norm = None

        # Override norm compute kernel: Gemma4 benefits from HiFi4 distributed norm
        # (DeepSeek V3 style). This is Gemma4-specific and must NOT be applied globally
        # in rmsnorm.py as it would affect all other models.
        _gemma4_norm_cfg = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )
        for norm_wrapper in [self.attention_norm, self.ff_norm, self.pre_ff_norm, self.post_ff_norm]:
            if norm_wrapper is not None:
                norm_wrapper.norm.compute_kernel_config_hifi2 = _gemma4_norm_cfg

        # Layer scalar - Gemma 4 specific
        layer_scalar_key = f"layers.{layer_num}.layer_scalar"
        if layer_scalar_key in state_dict:
            self.layer_scalar_value = state_dict[layer_scalar_key].item()
        else:
            self.layer_scalar_value = 1.0

        # Pre-compute layer scalar as TT tensor if not 1.0
        if abs(self.layer_scalar_value - 1.0) > 1e-6:
            scalar_tensor = torch.full((1, 1, 1, 1), self.layer_scalar_value, dtype=torch.bfloat16)
            self.layer_scalar = ttnn.from_torch(
                scalar_tensor,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
            )
        else:
            self.layer_scalar = None

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
        TG = self.args.is_galaxy
        residual = x

        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)

        # Choose rotation matrices based on layer type
        rot_mats = rot_mats_local if self.is_sliding else rot_mats_global

        # Attention norm
        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        # Reshape for batched prefill
        if batch_size > 1:
            attn_in = ttnn.reshape(attn_in, [batch_size, 1, attn_in.shape[-2] // batch_size, -1])

        attn_out = self.attention.forward(
            attn_in,
            current_pos,
            rot_mats,
            user_id,
            mode,
            page_table=page_table,
            chunk_page_table=chunk_page_table,
            chunk_start_idx=chunk_start_idx,
            kv_cache=kv_cache,
        )

        if mode == Mode.PREFILL and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])

        attn_out = ttnn.to_memory_config(attn_out, skip_mem_cfg)

        if self.pre_ff_norm is None:
            hidden_states = ttnn.add(
                residual, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            residual = hidden_states
            if mode == "prefill":
                x.deallocate(True)
        else:
            hidden_states = attn_out

        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        hidden_states = self.ff_norm(hidden_states, mode, norm_config=ff_norm_config)

        if self.pre_ff_norm is not None:
            if self.num_devices > 1 and not self.args.is_distributed_norm(mode):
                hidden_states = ttnn.mesh_partition(
                    hidden_states,
                    memory_config=hidden_states.memory_config(),
                    dim=3,
                    cluster_axis=1,
                )
            hidden_states = ttnn.add(
                residual, hidden_states, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            residual = hidden_states
            pre_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            hidden_states = self.pre_ff_norm(hidden_states, mode, norm_config=pre_ff_norm_config)

        ttnn.deallocate(attn_out)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.args.get_mlp_act_mem_config(mode))

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            post_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            hidden_states = self.post_ff_norm(hidden_states, mode, norm_config=post_ff_norm_config)
            if self.num_devices > 1 and not self.args.is_distributed_norm(mode):
                hidden_states = ttnn.mesh_partition(
                    hidden_states,
                    memory_config=hidden_states.memory_config(),
                    dim=3,
                    cluster_axis=1,
                )

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        # Apply layer scalar if not 1.0
        if self.layer_scalar is not None:
            out = ttnn.multiply(out, self.layer_scalar_value, memory_config=skip_mem_cfg)

        return out
