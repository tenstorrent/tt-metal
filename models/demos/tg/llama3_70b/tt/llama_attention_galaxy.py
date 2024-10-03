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
from models.demos.tg.llama3_70b.tt.llama_common import (
    tt_all_reduce,
    tt_all_gather,
    tt_sharded_all_reduce,
    tt_sharded_all_gather,
)
from models.demos.t3000.falcon40b.tt.model_utils import (
    matmul_2d_config_from_tensor_shapes as get_matmul_2d_config_from_tensor_shapes,
)


class TtLlamaAttention_galaxy:
    def __init__(
        self,
        mesh_device,
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
        self.mesh_device = mesh_device
        self.num_devices = mesh_device.get_num_devices()
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

        self.get_slice_mat()
        self.get_user_selection_mat()
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
            device=self.mesh_device,
            mesh_mapper=ShardTensorToMesh(self.mesh_device, dim=1),
        )

    def get_user_selection_mat(self):
        user_selection_matrix = torch.eye(8, 8)
        user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
        user_selection_matrix = [user_selection_matrix] * 4
        user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
        self.user_selection_matrix = ttnn.from_torch(
            user_selection_matrix,
            dtype=ttnn.bfloat4_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
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
            ttnn.as_tensor(
                lp,
                device=self.mesh_device,
                mesh_mapper=ReplicateTensorToMesh(self.mesh_device),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
                cache_file_name=self.cache_path / f"empty_attn_cache_galaxy_{cache_k.shape}",
            )
            for lp in layer_past
        ]
        # work around for CI error
        self.layer_past = [
            ttnn.reshape(lp, [self.batch_size_per_device_group, self.n_local_kv_heads, -1, self.head_dim])
            for lp in self.layer_past
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
        wo_cache_str = f"{self.layer_name}.attention.wo_galaxy_2d.weight"

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
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(2, 3), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / wqkv_cache_str,
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ShardTensor2dMesh(self.mesh_device, dims=(3, 2), cluster_shape=self.cluster_shape),
            cache_file_name=self.cache_path / wo_cache_str,
        )

    def __call__(self, xs, rot_mats, start_pos: int, cache_idxs, attn_masks, user_id: int = 0, mode="decode"):
        self.attention_config = self.model_config["attention"][mode]
        # Decode should have input tensor of shape (seqlen=1, 1, batch, hidden_size)
        if mode == "decode":
            return self.decode_forward(xs, rot_mats, start_pos, cache_idxs, attn_masks)
        elif mode == "prefill":
            return self.prefill_forward(xs, rot_mats, attn_masks, user_id)
        else:
            raise ValueError(f"Unknown llm_mode: {mode}")

    def decode_forward(
        self,
        xs,
        rot_mats,
        start_pos: int,
        cache_idxs,
        attn_masks,
    ):
        query_layer, key_layer, value_layer = self.attn_qkv(xs, rot_mats)
        attn_outputs = self.attn_mqa(query_layer, key_layer, value_layer, start_pos, cache_idxs, attn_masks)
        return self.attn_selfout(attn_outputs)

    def attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        batch_size = xs.shape[2]
        # Fused QKV
        fused_query_key_value = ttnn.matmul(
            xs,
            self.qkv,
            program_config=self.attention_config["FUSED_QKV_MM_PROGCFG"],
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_QKV"],
        )
        xs.deallocate(True)

        # TODO: Use sharded all_reduce when PCC issue is fixed in this particular configuration
        # fused_query_key_value = tt_sharded_all_reduce(
        #     fused_query_key_value, self.mesh_device, cluster_axis=1, num_links=2, memory_config=self.attention_config["QKV_OUT_GATHERED_MEMCFG"](self.cluster_shape[0])
        # )

        fused_query_key_value = tt_all_reduce(
            fused_query_key_value, self.mesh_device, cluster_axis=1, num_links=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        # TODO: Slice the fused_query_key_value tensor get batch=8
        fused_query_key_value = ttnn.matmul(
            self.slice_mat,
            fused_query_key_value,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_MEMORY_CONFIG,
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
            fused_query_key_value, memory_config=self.attention_config["CREATE_HEAD_INPUT_MEMCFG"]
        )

        # Split QKV
        (
            query_layer,  # [seqlen, bsz, padded_n_local_heads, head_dim]
            key_layer,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
            value_layer,  # [seqlen, bsz, padded_n_local_kv_heads, head_dim]
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
        )

        fused_query_key_value.deallocate(True)

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        query_layer = ttnn.matmul(
            query_layer,
            rot_mats,
            program_config=self.attention_config["ROT_MAT_MM_PROGCFG"](batch_size),
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_ROTARY"],
        )

        key_layer = ttnn.matmul(
            key_layer,
            rot_mats,
            program_config=self.attention_config["ROT_MAT_MM_PROGCFG"](batch_size),
            memory_config=ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_ROTARY"],
        )

        return query_layer, key_layer, value_layer

    def attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        start_pos: int,
        cache_idxs,
        attn_masks,
        batch_offset: int = 0,
    ):
        # K CACHE UPDATE
        keys = self.layer_past[0]
        ttnn.experimental.paged_update_cache(
            keys, key_layer, update_idxs_tensor=cache_idxs, batch_offset=batch_offset, page_table=None
        )  # TODO: do we need batch_offset here?
        key_layer.deallocate(True)

        # V CACHE UPDATE
        values = self.layer_past[1]
        ttnn.experimental.paged_update_cache(
            values, value_layer, update_idxs_tensor=cache_idxs, batch_offset=batch_offset, page_table=None
        )
        value_layer.deallocate(True)

        program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.mesh_device.compute_with_storage_grid_size(),
            q_chunk_size=0,  # unused
            k_chunk_size=0,  # unused
        )

        attn_output = ttnn.transformer.scaled_dot_product_attention_decode(
            query_layer,
            keys,
            values,
            cur_pos_tensor=cache_idxs,
            scale=self.scale,
            program_config=program_config,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_SDPA"],
            memory_config=self.attention_config["SDPA_HEIGHT_SHARDED_MEMCFG"](self.batch_size_per_device_group),
        )
        return attn_output

    def attn_selfout(
        self,
        attn_output,
    ):
        # ATTENTION SELFOUT
        # (1, 8, 8(32), 128) - > (1, 1, 8(32), 1024) ->(1, 1, 32, 1024)
        attn_output = ttnn.experimental.nlp_concat_heads_decode(
            attn_output,
            num_heads=self.n_local_heads,
        )

        attn_output = tt_sharded_all_gather(
            attn_output,
            self.mesh_device,
            dim=2,
            cluster_axis=1,
            num_links=2,
            memory_config=self.attention_config["GATHER_USERS_MEMCFG"](self.cluster_shape[0]),
        )
        attn_output = ttnn.to_memory_config(attn_output, ttnn.L1_MEMORY_CONFIG)
        # user_selection_matrix = [1, 1, 32, 128]
        # user_selection_matrix @ activation -> [1, 1, 32, 128] * [1, 1, 128, 2048] -> [1, 1, 32, 2048]
        attn_output = ttnn.matmul(
            self.user_selection_matrix,
            attn_output,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            dtype=ttnn.bfloat16,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
        )

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            core_grid=ttnn.CoreGrid(y=4, x=8),
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            dtype=ttnn.bfloat8_b,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_SELFOUT"],
        )

        attn_output = tt_sharded_all_reduce(
            attn_output,
            self.mesh_device,
            cluster_axis=0,
            num_links=2,
            memory_config=self.attention_config["SELF_OUT_GATHERED_MEMCFG"](self.cluster_shape[1]),
        )

        return attn_output

    def prefill_forward(
        self,
        xs,
        rot_mats,
        attn_masks,
        user_id: int = 0,
    ):
        query_layer, key_layer, value_layer = self.prefill_attn_qkv(xs, rot_mats)
        attn_outputs = self.prefill_attn_mqa(query_layer, key_layer, value_layer, attn_masks, user_id)
        return self.prefill_attn_selfout(attn_outputs)

    def prefill_attn_qkv(
        self,
        xs,
        rot_mats,
    ):
        assert xs.shape[1] == 1, "batch must be 1"
        assert xs.shape[2] % 32 == 0 and xs.shape[2] > 0, "Seqlen must be divisible by 32"
        _, _, seq_len, _ = xs.shape

        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"](seq_len)
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor

        xs = ttnn.reshape(xs, (1, batch_dim, seq_len // batch_dim, -1))

        # Fused QKV
        fused_query_key_value = ttnn.matmul(
            xs,
            self.qkv,
            dtype=ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_QKV"],
            program_config=self.attention_config["FUSED_QKV_MM_PROGCFG"](seq_len),
        )

        fused_query_key_value = tt_all_reduce(
            fused_query_key_value, self.mesh_device, cluster_axis=1, num_links=2, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )

        fused_query_key_value = ttnn.reshape(fused_query_key_value, (1, 1, seq_len, -1))

        (
            query_layer,  # [bsz, n_local_heads, seq_len, head_dim]
            key_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
            value_layer,  # [bsz, n_local_kv_heads, seq_len, head_dim]
        ) = ttnn.experimental.nlp_create_qkv_heads(
            fused_query_key_value,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ROTARY EMBEDDINGS
        # Q Rotary Embeddings
        # query_layer: ttnn.Shape([1, 8, seq_len, 128]) -> [bsz, n_local_heads, seq_len, head_dim]
        query_layer_ret = ttnn.experimental.rotary_embedding_llama(
            query_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )

        # K Rotary Embeddings
        # key_layer: ttnn.Shape([1, 1, seq_len, 128]) -> [bsz, n_local_kv_heads, seq_len, head_dim]
        key_layer_ret = ttnn.experimental.rotary_embedding_llama(
            key_layer, rot_mats[0], rot_mats[1], self.transformation_mats
        )

        return query_layer_ret, key_layer_ret, value_layer

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: self.cluster_shape[0]]
        # Create multi-device tensor
        multi_device_tensor = ttnn.aggregate_as_tensor(single_column_tensors)

        return multi_device_tensor

    def prefill_attn_mqa(
        self,
        query_layer,
        key_layer,
        value_layer,
        attn_masks,
        user_id: int = 0,
    ):
        # FILL K CACHE
        keys = self.layer_past[0]
        single_user_key_layer = self.prefill_prepare_tensor_for_kv_cache(key_layer, user_id)

        # Fill cache with multi-device tensor
        ttnn.fill_cache(
            keys,
            ttnn.experimental.typecast(single_user_key_layer, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            user_id % self.batch_size_per_device_group,
        )

        # FILL V CACHE
        values = self.layer_past[1]
        single_user_value_layer = self.prefill_prepare_tensor_for_kv_cache(value_layer, user_id)

        ttnn.fill_cache(
            values,
            ttnn.experimental.typecast(single_user_value_layer, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG),
            user_id % self.batch_size_per_device_group,
        )

        # SDPA
        attn_output = ttnn.transformer.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            is_causal=True,
            scale=self.scale,
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_SDPA"],
            program_config=self.attention_config["SDPA_PROG_CFG"](query_layer.shape[-2]),  # pass seq_len
        )

        return attn_output

    def prefill_attn_selfout(self, attn_output):
        # ATTENTION SELFOUT
        attn_output = ttnn.experimental.nlp_concat_heads(
            attn_output,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )  # bsz, 1, seqlen, hidden_size

        _, _, seq_len, _ = attn_output.shape

        max_mm_seq_len = self.model_config["MAX_MM_SEQ_LEN"](seq_len)
        batch_dim = 1 if seq_len < max_mm_seq_len else seq_len // max_mm_seq_len  # Find the division factor
        attn_output = ttnn.reshape(attn_output, (1, batch_dim, seq_len // batch_dim, -1))

        attn_output = ttnn.matmul(
            attn_output,
            self.wo,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            program_config=self.attention_config["SELFOUT_PROGCFG"](seq_len),
            compute_kernel_config=self.attention_config["COMPUTE_KERNEL_SELFOUT"],
        )  # bsz, 1, seqlen, hidden_size

        attn_output = ttnn.reshape(attn_output, (1, 1, seq_len, -1))

        # Call all reduce here
        attn_output = tt_all_reduce(
            attn_output,
            self.mesh_device,
            cluster_axis=0,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        return attn_output
