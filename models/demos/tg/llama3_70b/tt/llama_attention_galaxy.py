# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

from loguru import logger
import math
import torch
import ttnn
from ttnn import ShardTensorToMesh, ReplicateTensorToMesh
from models.demos.t3000.llama2_70b.tt.llama_common import (
    ShardTensor2dMesh,
    ConcatMesh2DToTensor,
)
from models.demos.t3000.llama2_70b.tt.llama_common import (
    num_to_corerange,
)


class TtLlamaAttention_galaxy:
    def __init__(
        self,
        device_mesh,
        cluster_shape,
        state_dict,
        base_url,
        layer_num,
        model_config,
        configuration,
        transformation_mats,
        cache_path=None,
        read_cache=False,
    ):
        self.state_dict = state_dict
        self.device_mesh = device_mesh
        self.num_devices = device_mesh.get_num_devices()
        assert self.num_devices == 32, "Only 32 devices supported for TG"
        self.model_config = model_config
        self.read_cache = read_cache
        self.cluster_shape = cluster_shape

        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.n_kv_heads = configuration.n_kv_heads
        self.head_dim = self.hidden_size // self.n_heads
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = model_config["MAX_BATCH_SIZE"]
        self.llama3 = configuration.vocab_size == 128256
        self.scale = 1 / math.sqrt(self.head_dim)

        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads
        self.batch_size_per_device_group = self.max_batch_size // self.num_device_groups

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group
        self.padded_local_heads = 32

        self.layer_name = f"{base_url}.{layer_num}"
        self.cache_path = cache_path
        self.transformation_mats = transformation_mats

        self.get_attn_model_config()
        self.get_slice_mat()
        self.load_weights()
        self.init_kv_cache()

    def set_model_config(self, model_config):
        self.model_config = model_config

    def get_slice_mat(self):
        # Create the slice weight matrices
        weight = torch.zeros(1, 32, 8, 32)
        for i in range(32):
            col = i % 4  # This determines which group of 8 to select
            weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

        self.slice_mat = ttnn.from_torch(
            weight,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            mesh_mapper=ShardTensorToMesh(self.device_mesh, dim=1),
        )

    def get_attn_model_config(self):
        if self.model_config["LLM_MODE"] == "decode":
            self.FUSED_QKV_MM_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 5),
                in0_block_w=8,
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=32 // 32,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )
            self.COMPUTE_KERNEL_QKV = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi2,
                math_approx_mode=True,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )
            self.COMPUTE_KERNEL_SELFOUT = self.COMPUTE_KERNEL_QKV

            total_cores = (self.n_local_heads + self.n_local_kv_heads * 2) * self.head_dim // 32
            shard_spec_n_cores_grid = ttnn.CoreRangeSet({num_to_corerange(total_cores)})
            self.CREATE_HEAD_INPUT_MEMCFG = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    shard_spec_n_cores_grid,
                    [
                        32,
                        32,
                    ],
                    ttnn.ShardOrientation.ROW_MAJOR,
                    False,
                ),
            )
            self.COMPUTE_KERNEL_ROTARY = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=True,
            )

            self.ROTARY_PROGCFG = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=[8, 1],
                in0_block_w=4,
                out_subblock_h=1,
                out_subblock_w=4,
                per_core_M=1,
                per_core_N=4,
            )

            self.COMPUTE_KERNEL_SDPA = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )

            self.SELFOUT_PROGCFG = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                in0_block_w=8,  # (32 x 8k) x (8k x 1k) = (32 x 1k)
                out_subblock_h=1,
                out_subblock_w=1,
                per_core_M=32 // 32,
                per_core_N=1,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=True,
            )

            shard_grid = ttnn.CoreRangeSet({num_to_corerange(self.batch_size_per_device_group)})
            shard_spec = ttnn.ShardSpec(
                shard_grid, (self.padded_local_heads, self.head_dim), ttnn.ShardOrientation.ROW_MAJOR, False
            )

            self.SDPA_HEIGHT_SHARDED_MEMCFG = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1, shard_spec
            )

    def init_kv_cache(self):
        """
        Generates empty KV cache and pushed to device memory
        """

        cache_k = torch.zeros(
            (
                self.n_local_kv_heads,
                self.batch_size_per_device_group,
                self.model_config["MAX_CONTEXT_LEN"],
                self.head_dim,
            )
        )
        cache_v = torch.zeros(
            (
                self.n_local_kv_heads,
                self.batch_size_per_device_group,
                self.model_config["MAX_CONTEXT_LEN"],
                self.head_dim,
            )
        )
        layer_past = [cache_k, cache_v]
        self.layer_past = [
            ttnn.to_device(
                ttnn.as_tensor(
                    lp,
                    device=self.device_mesh,
                    mesh_mapper=ReplicateTensorToMesh(self.device_mesh),
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    dtype=ttnn.bfloat8_b,
                    cache_file_name=self.cache_path / f"empty_attn_cache_galaxy_{cache_k.shape}",
                ),
                self.device_mesh,
            )
            for lp in layer_past
        ]

    def load_weights(self):
        assert not hasattr(self, "qkv_list"), "qkv_list is already an attribute of this object"
        assert not hasattr(self, "wo_list"), "wo_list is already an attribute of this object"
        # Load weights
        wqkv_cache_str = f"{self.layer_name}.attention.wqkv_fused_galaxy_2d.weight"
        wq_str = f"{self.layer_name}.attention.wq.weight"
        wk_str = f"{self.layer_name}.attention.wk.weight"
        wv_str = f"{self.layer_name}.attention.wv.weight"
        wo_str = f"{self.layer_name}.attention.wo.weight"
        wo_cache_str = f"{self.layer_name}.attention.wo_galaxy_shard_by_4.weight"

        qkv_cat = None
        pt_wo = None
        if not self.read_cache:
            qkv_list = []
            for i in range(self.num_devices_per_group):
                ### Fused QKV Weights
                # Chunk weights
                wq_chunks = torch.chunk(self.state_dict[wq_str], self.n_heads, dim=0)
                wk_chunks = torch.chunk(self.state_dict[wk_str], self.n_kv_heads, dim=0)
                wv_chunks = torch.chunk(self.state_dict[wv_str], self.n_kv_heads, dim=0)

                # Select chunks for the current device
                wq_selected = torch.cat(wq_chunks[i * self.n_local_heads : (i + 1) * self.n_local_heads], dim=0)
                wk_selected = torch.cat(wk_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)
                wv_selected = torch.cat(wv_chunks[i * self.n_local_kv_heads : (i + 1) * self.n_local_kv_heads], dim=0)

                # Transpose the selected chunks
                wq = torch.transpose(wq_selected, -2, -1)
                wk = torch.transpose(wk_selected, -2, -1)
                wv = torch.transpose(wv_selected, -2, -1)

                # Create interleaved qkv list
                n_repeat = self.n_heads // self.n_kv_heads
                qkv_interleaved = [
                    [
                        wq[..., i * n_repeat * self.head_dim : (i + 1) * n_repeat * self.head_dim],
                        wk[..., i * self.head_dim : (i + 1) * self.head_dim],
                        wv[..., i * self.head_dim : (i + 1) * self.head_dim],
                    ]
                    for i in range(self.n_local_kv_heads)
                ]
                qkv_interleaved = [item for sublist in qkv_interleaved for item in sublist]

                # Concatenate Q, K, V for the current device
                qkv = torch.cat(qkv_interleaved, dim=-1)
                qkv_list.append(qkv)

            qkv_cat = torch.cat(qkv_list, dim=-1)
            qkv_cat = qkv_cat.unsqueeze(0).unsqueeze(0)

            pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        self.qkv = ttnn.as_tensor(
            qkv_cat,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / wqkv_cache_str,
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.device_mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=(3, None), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / wo_cache_str,
        )

    def __call__(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
        user_id: int = 0,
    ):
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if self.model_config["LLM_MODE"] == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, attn_masks)
        else:
            raise ValueError(f"Unknown llm_mode: {self.model_config['LLM_MODE']}")

    def decode_forward(
        self,
        xs,
        rot_mats,
        start_pos: int,
        attn_masks,
    ):
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(query_layer, key_layer, value_layer, start_pos, attn_masks)
        return self.attn_selfout(attn_outputs)

    def tt_all_reduce(self, tensors, cluster_axis, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG):
        """
        reduction of a multi-device tensor
        """
        concat_dim = (1, 3) if cluster_axis == 0 else (3, 1)

        out = ttnn.to_torch(
            tensors,
            mesh_composer=ConcatMesh2DToTensor(self.device_mesh, dims=concat_dim, cluster_shape=self.cluster_shape),
        )
        out = torch.sum(out, dim=1, keepdim=True)

        shape = (
            out.shape[2],
            out.shape[3] // 8 // (self.cluster_shape[1] if cluster_axis == 0 else self.cluster_shape[0]),
        )

        if memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
            act_mem_config = ttnn.create_sharded_memory_config(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=1, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            act_mem_config = None

        act_shard_dim = (None, 3) if cluster_axis == 0 else (3, None)
        out_tt = ttnn.from_torch(
            out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=act_mem_config,
            device=self.device_mesh,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=act_shard_dim, cluster_shape=self.cluster_shape),
        )

        return out_tt

    def attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        # Fused QKV
        fused_query_key_value = ttnn.matmul(
            xs,
            self.qkv,
            # program_config=self.model_config["FUSED_QKV_MM_PROGCFG"],
            core_grid=ttnn.CoreGrid(y=5, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.COMPUTE_KERNEL_QKV,
        )
        xs.deallocate(True)

        fused_query_key_value = self.tt_all_reduce(
            fused_query_key_value, cluster_axis=0, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # TODO: Slice the fused_query_key_value tensor get batch=8
        fused_query_key_value = ttnn.matmul(
            self.slice_mat,
            fused_query_key_value,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = fused_query_key_value.shape
        fused_query_key_value = ttnn.reshape(
            fused_query_key_value,
            ttnn.Shape(
                (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, self.max_batch_size, fqkv_shape[3])
            ),
        )

        fused_query_key_value = ttnn.to_memory_config(
            fused_query_key_value, memory_config=self.CREATE_HEAD_INPUT_MEMCFG
        )

        # Split QKV
        (
            query_layer,  # [seqlen, bsz, padded_n_local_heads, head_dim]
            key_layer,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
            value_layer,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
        ) = ttnn.experimental.tensor.nlp_create_qkv_heads_decode(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            output_mem_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer = ttnn.matmul(
            query_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.COMPUTE_KERNEL_ROTARY,
        )

        key_layer = ttnn.matmul(
            key_layer,
            rot_mats,
            program_config=self.model_config["ROT_MAT_MM_PROGCFG"],
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.COMPUTE_KERNEL_ROTARY,
        )

        return query_layer, key_layer, value_layer

    def attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        start_pos: int,
        attn_masks,
        batch_offset: int = 0,
    ):
        # K CACHE UPDATE
        keys = self.layer_past[0]
        ttnn.experimental.tensor.update_cache(keys, key_layer, start_pos, batch_offset=batch_offset)
        key_layer.deallocate(True)

        # V CACHE UPDATE
        values = self.layer_past[1]
        ttnn.experimental.tensor.update_cache(values, value_layer, start_pos, batch_offset=batch_offset)
        value_layer.deallocate(True)

        program_config = ttnn.experimental.operations.primary.transformers.SDPAMultiCoreProgramConfig(
            compute_with_storage_grid_size=self.device_mesh.get_device(0).compute_with_storage_grid_size(),
            q_chunk_size=0,  # unused
            k_chunk_size=0,  # unused
        )

        attn_output = ttnn.experimental.operations.primary.transformers.scaled_dot_product_attention_decode(
            query_layer,
            keys,
            values,
            [start_pos for _ in range(self.max_batch_size)],
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.COMPUTE_KERNEL_SDPA,
            output_mem_config=self.SDPA_HEIGHT_SHARDED_MEMCFG,
        )
        return attn_output

    def tt_all_gather(self, tensors, dim, cluster_axis, memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG):
        """
        gather of a multi-device tensor
        """
        concat_dim = (dim, 1) if cluster_axis == 0 else (1, dim)
        shard_dim = (None, 1) if cluster_axis == 0 else (1, None)

        out = ttnn.to_torch(
            tensors,
            mesh_composer=ConcatMesh2DToTensor(self.device_mesh, dims=concat_dim, cluster_shape=self.cluster_shape),
        )

        shape = (out.shape[2], out.shape[3] // 32)

        if memory_config == ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG:
            act_mem_config = ttnn.create_sharded_memory_config(
                shape=shape,
                core_grid=ttnn.CoreGrid(y=4, x=8),
                strategy=ttnn.ShardStrategy.WIDTH,
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
        else:
            act_mem_config = None

        out_tt = ttnn.from_torch(
            out,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=act_mem_config,
            device=self.device_mesh,
            mesh_mapper=ShardTensor2dMesh(self.device_mesh, dims=shard_dim, cluster_shape=self.cluster_shape),
        )

        return out_tt

    def attn_selfout(
        self,
        attn_output,
    ):
        # ATTENTION SELFOUT
        attn_output = ttnn.experimental.tensor.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,
        )

        attn_output = ttnn.reshape(
            attn_output,
            ttnn.Shape(
                (1, 1, self.batch_size_per_device_group, attn_output.shape[3]),
                (1, 1, self.max_batch_size, attn_output.shape[3]),
            ),
        )

        attn_output = self.tt_all_gather(
            attn_output,
            dim=2,
            cluster_axis=0,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        attn_output = self.tt_all_gather(
            attn_output,
            dim=3,
            cluster_axis=1,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            # memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.COMPUTE_KERNEL_SELFOUT,
        )

        return attn_output
