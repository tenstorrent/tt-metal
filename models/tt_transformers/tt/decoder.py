# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.tt_transformers.tt.attention import Attention as DefaultAttention
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

        self.layer_num = layer_num

        ActualAttentionClass = attention_class if attention_class is not None else DefaultAttention

        self.attention = ActualAttentionClass(
            mesh_device=mesh_device,
            tt_ccl=self.tt_ccl,
            state_dict=state_dict,
            weight_cache_path=weight_cache_path,
            layer_num=layer_num,
            dtype=dtype,
            transformation_mats=transformation_mats,
            configuration=args,
            paged_attention_config=paged_attention_config,
            use_paged_kv_cache=use_paged_kv_cache,
        )
        if self.args.is_mixture_of_experts:
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
                sharded_program_config=self.model_config[
                    "SHARDED_NORM_ATTN_PRGM_CFG"
                ],  # LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(x=8,y=4),subblock_w=4,block_h=1,block_w=4,inplace=0)
                sharded_output_config=self.model_config[
                    "SHARDED_ATTN_INPUT_MEMCFG"
                ],  # MemoryConfig(memory_layout=TensorMemoryLayout::WIDTH_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=0) - (x=7,y=3)]},shape={32, 128},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
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
                sharded_program_config=self.model_config[
                    "SHARDED_NORM_MLP_PRGM_CFG"
                ],  # LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(x=8,y=1),subblock_w=4,block_h=1,block_w=16,inplace=0)
                sharded_output_config=self.model_config[
                    "SHARDED_MLP_INPUT_MEMCFG"
                ],  # MemoryConfig(memory_layout=TensorMemoryLayout::WIDTH_SHARDED,buffer_type=BufferType::L1,shard_spec=ShardSpec(grid={[(x=0,y=0) - (x=7,y=0)]},shape={32, 512},orientation=ShardOrientation::ROW_MAJOR,mode=ShardMode::PHYSICAL,physical_shard_shape=std::nullopt))
                ccl_topology=self.args.ccl_topology(),
                tt_ccl=self.tt_ccl,
            ),
            args,
            tt_ccl=self.tt_ccl,
            TG=args.is_galaxy,
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
    ) -> ttnn.Tensor:
        TG = self.args.is_galaxy
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
        )  # attn_out is sharded across devices,
        # Here x and attn_out are both fractured across devices
        h_val = ttnn.add(
            x, attn_out, memory_config=skip_mem_cfg, dtype=ttnn.bfloat16 if TG else None
        )  # h_val is fractured across devices
        ttnn.deallocate(attn_out)
        if mode == "prefill":
            x.deallocate(True)

        # Norms take fractured inputs and output replicated across devices
        ff_in = self.ff_norm(h_val, mode)
        if TG and mode == "decode":
            ff_in = ttnn.to_memory_config(ff_in, memory_config=self.model_config["MLP_ACT_MEMCFG"])
        # MLP takes replicated inputs and produces fractured outputs
        # Check the input sizes here and make sure they are what a MOE expects for Mixtral
        ff_out = self.feed_forward.forward(ff_in, mode)  # ff_out is replicated
        # ff_out = ff_out[:, :, :, 0:512]
        # ff_out = ff_out.to_memory_config(memory_config=ttnn.MemoryConfig(memory_config=ttnn.MemoryConfig(memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED,buffer_type=ttnn.BufferType.DRAM)))
        ff_out = ttnn.to_memory_config(
            ff_out,
            memory_config=ttnn.MemoryConfig(
                memory_layout=ttnn.TensorMemoryLayout.INTERLEAVED, buffer_type=ttnn.BufferType.DRAM
            ),
        )
        # ff_out and h are both fractured across devices
        activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=self.layer_num, tensor=TensorGroup.ACTIVATION
        )
        out = ttnn.add(
            h_val,
            ff_out,
            memory_config=skip_mem_cfg,
            dtype=self.args.ccl_dtype
            if TG and not self.args.is_distributed_norm(mode)
            else activation_dtype or ttnn.bfloat16,
        )
        return out  # fractured across devices
