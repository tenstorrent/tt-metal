# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.distributed_norm import DistributedNorm
from models.tt_transformers.tt.mixtral_mlp import TtMixtralMLP
from models.tt_transformers.tt.mixtral_moe import TtMoeLayer
from models.tt_transformers.tt.mlp import MLP
from models.tt_transformers.tt.model_config import TensorGroup


class TransformerBlock(LightweightModule):
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
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = args.max_seq_len
        self.dim = args.dim
        self.max_batch_size = args.max_batch_size
        self.n_kv_heads = args.n_kv_heads
        self.current = 0
        self.model_config = args.get_model_config()
        self.is_mixture_of_experts = False
        self.layer_num = layer_num
        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention

        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
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
        )

        if getattr(self.args, "is_mixture_of_experts", False):
            self.feed_forward = TtMoeLayer(
                mesh_device=mesh_device,
                state_dict=state_dict,
                experts=TtMixtralMLP(
                    mesh_device=mesh_device,
                    state_dict=state_dict,
                    args=args,
                    layer_num=layer_num,
                    dtypes={
                        "w1": dtype,
                        "w2": dtype,
                        "w3": dtype,
                    },
                ),
                args=args,
                layer_num=layer_num,
                dtype=dtype,
                tt_ccl=self.tt_ccl,
            )
        else:
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

        # TODO: remove after https://github.com/tenstorrent/tt-metal/issues/35650 is fixed
        extra_rmsnorm_kwargs = {}
        if args.base_model_name in ("Qwen2.5-7B", "Qwen2.5-VL-7B"):
            extra_rmsnorm_kwargs["fp32_dest_acc_en"] = False
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
                **extra_rmsnorm_kwargs,
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
                **extra_rmsnorm_kwargs,
            ),
            args,
            tt_ccl=self.tt_ccl,
            prefetcher=self.prefetcher,
            TG=args.is_galaxy,
            ag_config_key="FFN_LN_AG_CONFIG",
        )
        if f"layers.{layer_num}.pre_feedforward_layernorm.weight" in state_dict:
            self.pre_ff_norm = DistributedNorm(  # pre_feedforward_layernorm
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
            self.ff_norm.enable_all_gather = (
                False  # output of ff_norm should be sharded if model uses pre_ff_norm, so skip all_gather
            )
        else:
            # If pre_feedforward_layernorm is not in state_dict, we do not use it
            self.pre_ff_norm = None

        if f"layers.{layer_num}.post_feedforward_layernorm.weight" in state_dict:
            self.post_ff_norm = DistributedNorm(  # post_feedforward_layernorm
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
            # If post_feedforward_layernorm is not in state_dict, we do not use it
            self.post_ff_norm = None

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
        if self.args.is_post_norm:
            return self._forward_post_norm(
                x,
                current_pos,
                rot_mats_global,
                rot_mats_local,
                user_id,
                mode,
                page_table,
                chunk_page_table,
                chunk_start_idx,
                kv_cache,
                batch_size,
            )

        TG = self.args.is_galaxy
        residual = x

        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)

        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # Choose the correct rotation matrices based on the mode
        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # Norms take fractured inputs and output replicated across devices
        attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
        attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config)

        # Reshape to [B, 1, S_per_user, H] so attention infers batch_size from shape[0]
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
        # To match the batch-related reshape inside the attention module
        # Use the batch_size parameter instead of inferring from shape[-3]
        # because for [32, 1, S, H] tensors, shape[-3] is 1, not 32
        # This reshape is only applicable in prefill mode with batched prefill
        if mode == Mode.PREFILL and batch_size > 1:
            residual = ttnn.reshape(residual, [1, 1, residual.shape[-2] * residual.shape[-3] * residual.shape[0], -1])
        # TODO: create correct memory config in RopeSetup (issue is in ttnn.add op because of different shape in memory config for residual and rot_mats)
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
            # Mesh partition ff_norm output to match residual sharding, skip if using distributed norm, because output is already sharded
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
        # MLP takes replicated inputs and produces fractured outputs

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            post_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            hidden_states = self.post_ff_norm(hidden_states, mode, norm_config=post_ff_norm_config)  # Gathered
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

        return out  # fractured across devices

    def _forward_post_norm(
        self,
        x,
        current_pos,
        rot_mats_global,
        rot_mats_local,
        user_id,
        mode,
        page_table,
        chunk_page_table,
        chunk_start_idx,
        kv_cache,
        batch_size,
    ):
        """Post-norm forward pass for EXAONE 4.0.

        Post-norm applies norm AFTER attention/FFN (not before).
        Flow: attention(x) → norm → residual add → FFN(x) → norm → residual add
        """
        TG = self.args.is_galaxy
        residual = x
        skip_mem_cfg = self.args.get_residual_mem_config(mode, self.prefetcher)

        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # Post-norm: all-gather without normalization for attention input
        if self.args.is_multichip and not TG:
            if not self.args.is_distributed_norm(mode):
                # Decode: use attention norm's sharded config for all-gather
                attn_norm_config = self.args.get_norm_config("attn", mode, self.prefetcher)
                attn_in = self.attention_norm(x, mode, norm_config=attn_norm_config, gather_only=True)
            else:
                # Prefill: explicit all-gather to DRAM
                attn_in = ttnn.experimental.all_gather_async(
                    x,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=self.tt_ccl.get_num_links(1),
                    topology=self.args.ccl_topology(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
        else:
            attn_in = x

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

        # Post-attention norm: ff_norm has post_attention_layernorm weights
        ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
        attn_normed = self.ff_norm(attn_out, mode, norm_config=ff_norm_config)

        # Partition back to fractured for residual add (always needed in post-norm)
        if self.num_devices > 1:
            attn_normed = ttnn.mesh_partition(
                attn_normed,
                memory_config=attn_normed.memory_config(),
                dim=3,
                cluster_axis=1,
            )

        hidden_states = ttnn.add(residual, attn_normed, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None)
        residual = hidden_states

        ttnn.deallocate(attn_out)
        if mode == "prefill":
            x.deallocate(True)

        # FFN: all-gather without normalization
        if self.args.is_multichip and not TG:
            if not self.args.is_distributed_norm(mode):
                # Decode: use ff_norm's sharded config (produces correct format for MLP DRAM-sharded matmul)
                ffn_norm_config2 = self.args.get_norm_config("ff", mode, self.prefetcher)
                ffn_in = self.ff_norm(hidden_states, mode, norm_config=ffn_norm_config2, gather_only=True)
            else:
                # Prefill: explicit all-gather to DRAM interleaved
                ffn_in = ttnn.experimental.all_gather_async(
                    hidden_states,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=self.tt_ccl.get_num_links(1),
                    topology=self.args.ccl_topology(),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
        else:
            ffn_in = hidden_states

        if TG and mode == "decode":
            ffn_in = ttnn.to_memory_config(ffn_in, memory_config=self.args.get_mlp_act_mem_config(mode))

        ffn_out = self.feed_forward.forward(ffn_in, mode)

        # Post-FFN norm: post_ff_norm has post_feedforward_layernorm weights
        activation_dtype = self.args.decoders_optimizations.get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            post_ff_norm_config = self.args.get_norm_config("ff", mode, self.prefetcher)
            ffn_out = self.post_ff_norm(ffn_out, mode, norm_config=post_ff_norm_config)
            # post_ff_norm has enable_all_gather=False, so distributed norm (prefill) output
            # is already fractured. Only partition when non-distributed (decode) where output is replicated.
            if self.num_devices > 1 and not self.args.is_distributed_norm(mode):
                ffn_out = ttnn.mesh_partition(
                    ffn_out,
                    memory_config=ffn_out.memory_config(),
                    dim=3,
                    cluster_axis=1,
                )

        out = ttnn.add(
            residual,
            ffn_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return out
