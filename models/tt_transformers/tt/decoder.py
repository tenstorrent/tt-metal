# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
from models.tt_transformers.tt.ccl import tt_all_reduce
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
    ):
        super().__init__()

        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl

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
                sharded_program_config=self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_ATTN_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
                **extra_rmsnorm_kwargs,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
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
                sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
                **extra_rmsnorm_kwargs,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
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
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                TG=args.is_galaxy,
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
                    sharded_program_config=self.model_config["SHARDED_NORM_MLP_PRGM_CFG"],
                    sharded_output_config=self.model_config["SHARDED_MLP_INPUT_MEMCFG"],
                    ccl_topology=self.args.ccl_topology(),
                    tt_ccl=self.tt_ccl,
                ),
                args,
                tt_ccl=self.tt_ccl,
                TG=args.is_galaxy,
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
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
        residual = x
        # x is fractured across devices and interleaved in DRAM (for prefill) and sharded in L1 (for decode)
        skip_mem_cfg = self.model_config["DECODE_RESIDUAL_MEMCFG"] if mode == "decode" else ttnn.DRAM_MEMORY_CONFIG
        assert (
            x.memory_config() == skip_mem_cfg
        ), f"decoder input memcfg mismatch: {x.memory_config()} != {skip_mem_cfg}"

        # Choose the correct rotation matrices based on the mode
        rot_mats = (
            rot_mats_local if (hasattr(self.attention, "is_sliding") and self.attention.is_sliding) else rot_mats_global
        )

        # Norms take fractured inputs and output replicated across devices
        attn_in = self.attention_norm(x, mode)
        # Attention takes replicated inputs and produces fractured outputs
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
        hidden_states = self.ff_norm(hidden_states, mode)
        if self.pre_ff_norm is not None:
            # The output of the ff_norm is replicated across the device
            # but the residual is fractured across the devices
            if self.num_devices > 1:
                hidden_states = tt_all_reduce(
                    hidden_states,
                    self.mesh_device,
                    tt_ccl=self.tt_ccl,
                    cluster_axis=0,
                    dim=3,
                    num_reduce_scatter_links=self.args.num_reduce_scatter_links,
                    num_all_gather_links=self.args.num_all_gather_links,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=self.args.ccl_dtype,
                )

                hidden_states = ttnn.div(hidden_states, self.num_devices)
            hidden_states = ttnn.add(
                residual, hidden_states, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
            )
            residual = hidden_states
            hidden_states = self.pre_ff_norm(hidden_states, mode)

        ttnn.deallocate(attn_out)

        if TG and mode == "decode":
            hidden_states = ttnn.to_memory_config(hidden_states, memory_config=self.model_config["MLP_ACT_MEMCFG"])
        # MLP takes replicated inputs and produces fractured outputs

        hidden_states = self.feed_forward.forward(hidden_states, mode)

        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )

        if self.post_ff_norm is not None:
            hidden_states = self.post_ff_norm(hidden_states, mode)  # Gathered
            if self.num_devices > 1:
                hidden_states = tt_all_reduce(
                    hidden_states,
                    self.mesh_device,
                    tt_ccl=self.tt_ccl,
                    cluster_axis=0,
                    dim=3,
                    num_reduce_scatter_links=self.args.num_reduce_scatter_links,
                    num_all_gather_links=self.args.num_all_gather_links,
                    topology=ttnn.Topology.Ring,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=self.args.ccl_dtype,
                )

                hidden_states = ttnn.div(hidden_states, self.num_devices)

        out = ttnn.add(
            residual,
            hidden_states,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )

        return out  # fractured across devices
