import math
import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.utility_functions import nearest_32
from models.demos.llama3_70b_galaxy.tt.llama_model import TtTransformer
from models.demos.llama3_70b_galaxy.tt.model_config import (
    LM_HEAD_16_GRID,
    LM_HEAD_32_GRID,
    LM_HEAD_INPUT_GRID,
    LM_HEAD_OUTPUT_GRID,
    PREFETCHER_NOC1_GRID,
    get_core_ranges,
    num_to_coregrid,
    set_tg_attention_config,
)
from models.tt_transformers.demo.simple_text_demo import prepare_generator_args
from models.tt_transformers.tt.common import (
    calculate_hidden_dim,
    create_tt_model,
    freqs_to_rotation_matrix,
    get_out_subblock_w,
    nearest_multiple,
    num_to_core_range_set,
    precompute_freqs,
)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.model_config import DecodersPrecision, ModelArgs, TensorGroup


class TtQwenAttention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
        prefetcher_setup=None,
        tt_ccl=None,
    ):
        super().__init__()
        self.state_dict = state_dict
        self.mesh_device = mesh_device
        self.num_devices = configuration.num_devices
        self.TG = self.num_devices == 32
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.min_kv_prefill_shard_seqlen = configuration.min_kv_prefill_shard_seqlen
        self.ccl_dtype = configuration.ccl_dtype
        self.num_reduce_scatter_links = configuration.num_reduce_scatter_links
        self.num_all_gather_links = configuration.num_all_gather_links

        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl

        # TODO: Fix this once all-gather supports < tile_size
        if self.TG:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            # Select batch_offset with create_qkv_heads_decode instead of selection matmul
            batch_offset = [
                0,
                8,
                16,
                24,
            ]  # TODO: batch offset is 8 for batch=32, this should be adjusted for variable batch_size
            self.batch_offset_tt_tensor = ttnn.as_tensor(
                torch.tensor(batch_offset, dtype=torch.int32).reshape(4, 1),
                dtype=ttnn.int32,
                device=mesh_device,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    mesh_device=mesh_device, dims=(None, 0), mesh_shape=list(mesh_device.shape)
                ),
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            self.slice_size = 8  # Slice size is 8 since we are consuming 8 users per chip

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip

        layer_name = configuration.get_state_dict_prefix("Attention", layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq.weight"
        wk_str = f"{layer_name}.wk.weight"
        wv_str = f"{layer_name}.wv.weight"
        wo_str = f"{layer_name}.wo.weight"

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

        # wqkv: 4096 x 3072 (2 devices): width-sharded on 12 banks, 3072 over 12 banks.
        wqkv_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim, configuration.qkv_size // configuration.num_devices
        )

        qkv_list = []
        for i in range(self.num_devices_per_group):
            # Chunk weights
            wq_selected = torch.chunk(self.state_dict[wq_str], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(self.state_dict[wk_str], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(self.state_dict[wv_str], self.num_devices_per_group, dim=0)[i]

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # Ring stuff
        # 9216, 12288

        # [1, 1, 8192, 10240] -> [2304, 1536]
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_QKV_RING_MEMCFG"] if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d_prefetcher"),  ## TODO: Fix caching
        )
        self.wqkv_interleaved = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d_dram"),  ## TODO: Fix caching
        )

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )
        self.wo_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO
        )
        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if (self.use_fused_all_gather_matmul or self.TG) else wo_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=(
                cache_name("wo_width_sharded_2d") if (self.use_fused_all_gather_matmul or self.TG) else cache_name("wo")
            ),
        )
        self.wo_interleaved = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_dram"),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5
        if tt_ccl.mode == "decode":
            self.prefetch(prefetcher_setup, tt_ccl)

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.wqkv)
            self.prefetcher_setup.insert_tensor(self.wo)
        self.tt_ccl = tt_ccl

    def init_kv_cache(self, configuration, weight_cache_path):
        """
        Generates empty KV cache and pushed to device memory
        """

        if self.paged_attention_config:
            cache_k = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.paged_attention_config.max_num_blocks,
                    self.n_local_kv_heads,
                    self.paged_attention_config.block_size,
                    self.head_dim,
                )
            )
        else:
            cache_k = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )
            cache_v = torch.zeros(
                (
                    self.batch_size_per_device_group,
                    self.n_local_kv_heads,
                    self.max_seq_len,
                    self.head_dim,
                )
            )

        self.layer_past = [
            ttnn.as_tensor(
                k_or_v,
                dtype=self.dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                if weight_cache_path and not configuration.dummy_weights
                else None,
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def forward_decode(
        self,
        x,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ):
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        xqkv_fused_sharded = ttnn.matmul(
            x,
            self.wqkv,
            program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat16,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        ttnn.deallocate(x)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = self.tt_ccl.llama_rs_create_heads(
            xqkv_fused_sharded,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            dim=3,
            qkv_memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )

        # print("done create qkv heads")
        ttnn.deallocate(xqkv_fused_sharded)
        # Q, K Rotary Embeddings
        q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
        )

        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

        # print("done rotary embeddings")

        ###
        # KV update
        ###
        if kv_cache:
            keys = kv_cache[0]
            values = kv_cache[1]
        else:
            keys = self.layer_past[0]
            values = self.layer_past[1]
        # k_heads, [seqlen, n_kv_heads, bsz, head_dim]
        # v_heads [seqlen, n_kv_heads, bsz, head_dim]
        # keys, [max_batch_size, n_kv_heads // configuration.num_devices, max_seq_len, head_dim]
        ttnn.experimental.paged_fused_update_cache(
            keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # print("done update cache")
        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        sdpa_out_mem_cfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group)
        if page_table:
            attn_output_1G4D_sharded = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                page_table_tensor=page_table,
                scale=self.scale,
                program_config=self.model_config["PAGED_SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,
            )
        else:
            attn_output_1G4D_sharded = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_cat = self.tt_ccl.all_gather_concat(
            attn_output_1G4D_sharded,
            dim=1,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_1G4D_sharded)
        # print("done concat heads")

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        dense_out_ttnn = ttnn.matmul(
            attn_output_cat,
            self.wo,
            program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat8_b,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id,
        )
        # [1, 1, 32, 2304]
        # print("done matmul")
        dense_out_reduced = self.tt_ccl.line_all_reduce(
            dense_out_ttnn,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
            use_optimal_ccl_for_llama=True,
        )
        ttnn.deallocate(dense_out_ttnn)

        # print("done all reduce")

        return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ):
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

        xqkv = ttnn.linear(
            x_11SH,
            self.wqkv_interleaved,
            dtype=self.ccl_dtype if self.TG else ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.compute_kernel_config_hifi2,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )

        ttnn.deallocate(x_11SH)

        xqkv_fused = self.tt_ccl.line_all_reduce(
            xqkv,
            cluster_axis=1,
            num_links=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="QKV",
        )
        ttnn.deallocate(xqkv)

        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

        # split qkv into heads
        (
            q_heads_1QSD_pre_rot,
            k_heads_1KSD_pre_rot,
            v_heads_1VSD,
        ) = ttnn.experimental.nlp_create_qkv_heads(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot_bf8 = q_heads_1QSD_pre_rot
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(q_heads_1QSD_pre_rot_bf8)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot_bf8 = k_heads_1KSD_pre_rot
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(k_heads_1KSD_pre_rot_bf8)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(k_heads_1KSD)

        k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=ttnn.bfloat8_b)

        ttnn.deallocate(v_heads_1VSD)

        v_fill = v_heads_1VSD_8b

        if batch_size > 1:
            k_fill = ttnn.reshape(k_fill, [1, 1, seq_len, -1])
            v_fill = ttnn.reshape(v_fill, [1, 1, seq_len, -1])

        if self.TG and not page_table:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table:
            if isinstance(user_id, int):
                user_id = ttnn.from_torch(
                    torch.tensor([user_id], dtype=torch.int32),
                    device=self.mesh_device,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, page_table, batch_idx_tensor=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, page_table, batch_idx_tensor=user_id)

        else:
            ttnn.fill_cache(
                keys_BKSD,
                k_fill,
                user_id % self.batch_size_per_device_group,
            )
            ttnn.fill_cache(
                values_BKSD,
                v_fill,
                user_id % self.batch_size_per_device_group,
            )

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        # Run ring_distributed_sdpa for > 1k seqlen because we are seeing worse perf for <=1k seqlen as compared to regular SDPA
        # ring_distributed_sdpa needs seqlen//8 to be atleast one tile (32)
        ring_distributed_sdpa = seq_len > 1024 and batch_size == 1
        if ring_distributed_sdpa:
            # Ring attention splits seqlen into 8 chunks and computes chunk i and chunk ring_size - i - 1 per device
            # where i (device id on a mesh column) ranges from 0 to ring_size-1 (0 to 3), so ring_size - i - 1 ranges from 3 to 0
            # This ensures each device processes two complementary chunks of the attention matrix
            attn_output_1QSD = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                ring_size=4,  # Number of devices in the ring topology (4 devices per row in 8x4 mesh)
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )
        else:
            attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        if ring_distributed_sdpa:
            # Split the attention output into two chunks along the sequence dimension for ring all-gather
            # 4 = ring_size (number of devices in ring), 2 = number of chunks per device
            attn_output_11SH_chunks = ttnn.split(attn_output_11SH, seq_len // 4 // 2, dim=2)
            attn_output_11SH.deallocate(True)

            # Perform ring all-gather on the first chunk (normal order)
            attn_output_11SH_chunk_0 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[0],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                buffer_key="SDPA",
            )
            attn_output_11SH_chunks[0].deallocate(True)

            # Perform ring all-gather on the second chunk (reverse order)
            attn_output_11SH_chunk_1 = self.tt_ccl.ring_all_gather(
                attn_output_11SH_chunks[1],
                dim=2,
                cluster_axis=1,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                reverse_order=True,
                buffer_key="SDPA_REVERSE",
            )
            attn_output_11SH_chunks[1].deallocate(True)

            # Concatenate the gathered chunks along the sequence dimension to form the final output
            attn_output_11SH = ttnn.concat([attn_output_11SH_chunk_0, attn_output_11SH_chunk_1], dim=2)
        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo_interleaved,
            compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        output_11SH = self.tt_ccl.line_all_reduce(
            output_11SH,
            cluster_axis=0,
            num_links=3,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO",
        )

        return output_11SH

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
        batch_size=1,
    ):
        if mode == "prefill":
            return self.forward_prefill(
                x,
                rot_mats,
                user_id,
                page_table=page_table,
                chunk_page_table=chunk_page_table,
                chunk_start_idx=chunk_start_idx,
                kv_cache=kv_cache,
                batch_size=batch_size,
            )
        else:
            return self.forward_decode(x, current_pos, rot_mats, page_table=page_table, kv_cache=kv_cache)

    def prefill_prepare_tensor_for_kv_cache(self, key_or_value_layer, user_id):
        tensor_copy = ttnn.clone(key_or_value_layer)
        # key_or_value_layer.deallocate(True)
        # Get all tensors from multi-device tensor
        tensors = ttnn.get_device_tensors(tensor_copy)
        # Get only tensors from specific column chips
        # Get every 4th tensor starting from user_id // 8
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: 4]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(single_column_tensors)

        return multi_device_tensor


def get_QwenEmbeddingArgs():
    class QwenEmbeddingArgs(ModelArgs):
        def __init__(self, *args, **kwargs):
            HF_MODEL = os.getenv("HF_MODEL")
            assert (
                HF_MODEL == "Qwen/Qwen3-Embedding-8B"
            ), f"When QwenEmbeddingArgs is used, HF_MODEL must be Qwen/Qwen3-Embedding-8B, but got {HF_MODEL}"
            super().__init__(*args, **kwargs)
            self.use_prefetcher = False
            self.max_top_k = 32
            if self.num_devices == 32:
                self.use_prefetcher = True

            # Set up prefetcher stuff
            _, _, _, self.pf_receiver_cores_list, _, _, _, _ = get_core_ranges(12, 2, False)

            self.sub_core_grids = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                    ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                ]
            )
            self.sub_core_grid_topk = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                ]
            )
            self.start_core = ttnn.CoreCoord(1, 0)

            self.tile_padded_batch_rows = self.tile_size * int(math.ceil(self.max_batch_size / self.tile_size))

            # Enable workarounds by default until di/dt issues are fixed
            self.di_dt_workaround = os.getenv("DISABLE_DI_DT_WORKAROUND") != "1"
            if not self.di_dt_workaround:
                logger.info("Disabling di/dt workaround, re-enable if you see hangs")

            self.TG = self.num_devices == 32
            self.num_device_groups = self.num_devices // self.n_kv_heads
            self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
            self.batch_size_per_device_group = (
                max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
            )

            DRAM_MEMCFG = ttnn.DRAM_MEMORY_CONFIG
            L1_MEMCFG = ttnn.L1_MEMORY_CONFIG
            self.model_config = {}
            # Update memory configs (weights->DRAM, activations->L1)
            self.model_config.update(
                {f"{key}_MEMCFG": DRAM_MEMCFG if "WEIGHTS" in key else L1_MEMCFG for key in self.OP_KEYS}
            )
            # Update memory layouts (Tile, except MLP)
            self.model_config.update({f"{key}_TILE": ttnn.TILE_LAYOUT for key in self.OP_KEYS if "LAYOUT" in key})

            self.cos, self.sin = precompute_freqs(
                self.head_dim, self.max_seq_len * 2, self.rope_theta, self.rope_scaling_factor, self.orig_context_len
            )  # for prefill
            self.rot_emb = freqs_to_rotation_matrix(self.cos, self.sin)  # for decode

            self.is_galaxy = self.num_devices == 32
            self.galaxy_type = None

            # self.model_config["GALAXY_NUM_LINKS"] = {"6U": 4, "4U": 3}.get(self.galaxy_type)
            # self.model_config["CCL_TOPOLOGY"] = {"6U": ttnn.Topology.Ring, "4U": ttnn.Topology.Linear}.get(self.galaxy_type)
            self.model_config["DECODERS_OPTIMIZATIONS"] = self.optimizations
            if self.mesh_device is not None:  # Avoid issue with test_llama_torch.py not having a device
                self.n_local_heads = self.n_heads // self.cluster_shape[1]

                grid = self.mesh_device.compute_with_storage_grid_size()
                self.max_grid_size = ttnn.CoreGrid(x=grid.x, y=grid.y)

                # DRAM weight grid specs for dram sharding matmuls
                self.dram_weight_grid = ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(
                                self.mesh_device.dram_grid_size().x - 1, self.mesh_device.dram_grid_size().y - 1
                            ),
                        )
                    }
                )

                # Compute kernels. FP32 acc does not appear to be needed for accuracy in model tests or demo runs.
                self.compute_kernel_config_lofi = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.LoFi,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                )
                self.compute_kernel_config_hifi2 = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=True,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                    dst_full_sync_en=True,
                )
                self.compute_kernel_config_hifi2_fp16 = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=True,
                )
                self.compute_kernel_config_hifi4 = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=True,
                )
                self.compute_kernel_config_sdpa = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi4,
                    math_approx_mode=False,
                    fp32_dest_acc_en=True,
                    packer_l1_acc=False,
                )

                self.model_config["COMPUTE_KERNEL_CONFIG_HIFI2"] = self.compute_kernel_config_hifi2
                core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(1, 0)
                core_range = ttnn.CoreRange(
                    grid_offset,
                    ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1),
                )
                num_cores_ln = core_grid_ln[0] * core_grid_ln[1]
                residual_grid = self.dram_shard_core_grid_for_k(self.dim // self.num_devices)
                self.model_config["DECODE_RESIDUAL_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(1, 1, 32, 2048 // num_cores_ln),
                        core_grid=ttnn.CoreRangeSet(
                            {
                                core_range,
                            }
                        ),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.is_galaxy
                    else ttnn.create_sharded_memory_config(
                        (
                            self.tile_padded_batch_rows,
                            self.dim // residual_grid.num_cores // self.num_devices,
                        ),
                        residual_grid,
                        ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )

                start_core = ttnn.CoreCoord(1, 0)
                core_grid = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 9)),
                        ttnn.CoreRange(ttnn.CoreCoord(5, 0), ttnn.CoreCoord(6, 9)),
                    ]
                )
                num_cores = self.cluster_shape[0]
                shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    start_core, num_cores, core_grid, row_wise=False
                )

                self.model_config["DECODE_SAMPLING_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(1, 1, max(self.max_batch_size, self.tile_size), self.max_top_k),
                    core_grid=shard_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                num_cores = 32
                shard_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    start_core, num_cores, core_grid, row_wise=False
                )
                self.model_config["DECODE_LOGITS_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(1, 1, max(self.max_batch_size, self.tile_size), self.padded_vocab_size // num_cores),
                    core_grid=shard_grid,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                # Chunk values based on what works best empirically
                self.model_config["SDPA_PROGCFG"] = lambda seqlen: ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(7, 10),
                    exp_approx_mode=False,
                    q_chunk_size=256 if seqlen >= 2048 else 64,
                    k_chunk_size=512 if seqlen >= 2048 else 64,
                )

                def find_largest_divisor(n, max_divisor=8):
                    for i in range(max_divisor, 0, -1):
                        if n % i == 0:
                            return i
                    return 1  # Fallback to 1 if no divisor found

                # nlp_concat_heads_decode will shard the data across this number of cores
                assert (
                    self.n_heads % self.cluster_shape[1] == 0
                ), f"n_heads must be divisible by num_devices: {self.n_heads} % {self.cluster_shape[1]}"

                self.model_config["ATTN_OUTPUT_PROGCFG"] = (
                    None
                    if self.is_galaxy
                    else self.dram_matmul_config(
                        m=self.tile_padded_batch_rows,
                        k=self.dim // self.num_devices,
                        n=self.dim,
                        num_cores=self.n_heads // self.num_devices,
                    )
                )

                # All Gather Matmul for Dense Out (DO)
                # TODO: Is there a better way to decide if fused all gather matmul should be used? And is there a better way to use the flag, instead of passing it into model_config?
                # NOTE: Fused all gather matmul only suppports a core grid of size num_devices x 1
                self.model_config["USE_FUSED_ALL_GATHER_MATMUL"] = (
                    self.ccl_topology() == ttnn.Topology.Ring
                    and (self.dim // self.tile_size // self.num_devices) % self.num_devices == 0
                    and self.num_devices > 1
                )

                if self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]:
                    do_core_grid_size = (8, 1)
                    do_per_core_N = (
                        self.dim // self.num_devices // self.tile_size // (do_core_grid_size[0] * do_core_grid_size[1])
                    )
                    self.model_config[
                        "ATTN_ALL_GATHER_MATMUL_PROGCFG"
                    ] = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=do_core_grid_size,
                        in0_block_w=self.dim
                        // self.tile_size
                        // (do_core_grid_size[0] * do_core_grid_size[1]),  # [32 x 8k] x [8k x 1k] = [32 x 1k]
                        out_subblock_h=1,
                        out_subblock_w=get_out_subblock_w(
                            do_per_core_N, out_subblock_h=1
                        ),  # Max out_subblock_w = 4, needs to be divisible by per_core_N
                        per_core_M=self.tile_padded_batch_rows // self.tile_size,
                        per_core_N=do_per_core_N,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                else:
                    self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"] = None

                def w1_w3_prg_config(seq_len, use_interleaved):
                    if seq_len == 128:
                        return self.matmul_1d_config(
                            128, 2048, 3584, grid=ttnn.CoreGrid(x=7, y=4), overwrite_per_core_k=16
                        )
                    if not use_interleaved:
                        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 10),
                            in0_block_w=8,
                            out_subblock_h=1,  # Must be divisible by per_core_M
                            out_subblock_w=4,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                            per_core_M=max(
                                1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8  # 8 rows
                            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                            per_core_N=math.ceil(28672 / 8 / 32 / 7),  # N / TILE_WIDTH / grid width
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=seq_len <= 2048,
                        )

                    if seq_len % 4096 == 0:
                        per_core_M = 20 * seq_len // 4096
                        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 7),
                            in0_block_w=4,
                            out_subblock_h=1,
                            out_subblock_w=8,
                            out_block_h=10,
                            out_block_w=16,
                            per_core_M=per_core_M,
                            per_core_N=16,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=False,
                        )
                    elif seq_len % 2048 == 0:
                        per_core_M = 10 * seq_len // 2048

                        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 7),
                            in0_block_w=4,
                            out_subblock_h=1,
                            out_subblock_w=8,
                            out_block_h=10,
                            out_block_w=16,
                            per_core_M=per_core_M,
                            per_core_N=16,
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=False,
                        )
                    else:
                        raise NotImplementedError(
                            f"W1 Program config generation for sequence length {seq_len} not implemented"
                        )

                self.model_config["PREFILL_MLP_W1_W3_PRG_CONFIG"] = w1_w3_prg_config

                #  Only used when seq_len >= 4096
                def prefill_ff1_ff3_minimal_matmul_config(seq_len):
                    """
                    Returns the best minimal matmul config for prefill FF1/FF3 based on sequence length.
                    Configurations are optimized based on sweep results.
                    """
                    # Best configurations from sweep results for each M value
                    if seq_len <= 4096:
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=4,
                            subblock_w=2,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                        )
                    elif seq_len <= 8192:
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=1,
                            subblock_w=8,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                        )
                    else:  # For seq_len >= 16384, use the best config from sweep results
                        # This covers 16384, 32768, 65536, 131072
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=4,
                            subblock_w=2,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                        )

                self.model_config["PREFILL_FF1_FF3_MINIMAL_MATMUL_CONFIG"] = prefill_ff1_ff3_minimal_matmul_config

                #  Only used when seq_len >= 4096
                def prefill_ff2_minimal_matmul_config(seq_len):
                    """
                    Returns the best minimal matmul config for prefill FF2 based on sequence length.
                    Configurations are optimized based on sweep results.
                    """
                    # Best configurations from sweep results for each M value
                    if seq_len <= 4096:
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=4,
                            subblock_w=2,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                        )
                    elif seq_len <= 16384:  # Both 8K and 16K share the same config
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=2,
                            subblock_w=4,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                        )
                    elif seq_len <= 32768:
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=4,
                            subblock_w=2,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                        )
                    elif seq_len <= 65536:
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=2,
                            subblock_w=4,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
                        )
                    else:  # For seq_len >= 131072
                        return ttnn.MinimalMatmulConfig(
                            M_block_size=8,
                            K_block_size=8,
                            N_block_size=8,
                            subblock_h=2,
                            subblock_w=4,
                            compute_with_storage_grid_size=ttnn.CoreCoord(7, 9),
                        )

                self.model_config["PREFILL_FF2_MINIMAL_MATMUL_CONFIG"] = prefill_ff2_minimal_matmul_config

                def w2_prg_config(seq_len):
                    if seq_len == 128:
                        return self.matmul_1d_config(
                            128, 3584, 2048, grid=ttnn.CoreGrid(x=7, y=10), overwrite_per_core_k=14
                        )
                    # For sequence lengths < 4096, we use this config as it performs better that what would be generated below
                    if seq_len < 4096:
                        return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 10),
                            in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                            out_subblock_h=1,  # Must be divisible by per_core_M
                            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                            per_core_M=max(1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8),  # 8~10 rows
                            per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=seq_len <= 2048,
                        )

                    # For very large activation heights (arbitrarily chosen to be > 320) we want the per_core_M to have many divisors
                    # so that there are many options for out_block_h and out_block_w. Padding to the next multiple of 8 ensures that
                    # per_core_M can at least be divisible by 2, 4, and 8 in addition to 1 and itself.
                    #
                    # If the number is less than or equal to 320 we still wouldn't want it to be prime so we'll add one if thats the case.
                    next_multiple_of_8 = lambda x: int(x + (8 - x % 8) % 8)
                    add_one_if_prime = (
                        lambda n: n + 1 if n > 1 and all(n % i != 0 for i in range(2, int(n**0.5) + 1)) else n
                    )
                    total_per_core_out_M = add_one_if_prime(math.ceil(seq_len / (7 * self.tile_size)))
                    per_core_M = (
                        next_multiple_of_8(total_per_core_out_M) if total_per_core_out_M > 320 else total_per_core_out_M
                    )
                    per_core_N = 10

                    # Want out_block_h and out_block_w such that:
                    # out_block_h * out block_w <= 320
                    # out_block_h % per_core_M == 0
                    # out_block_w % per_core_N == 0
                    # Since we're fixing per_core_N = 10, out_block_w can only be 5 or 10

                    def find_out_block_h(out_block_w):
                        max_out_block_h = -1
                        for i in range(1, per_core_M + 1):
                            if i * out_block_w > 320:
                                break
                            if per_core_M % i == 0:
                                if i > max_out_block_h:
                                    max_out_block_h = i
                        if max_out_block_h == -1:
                            return None
                        return max_out_block_h

                    out_block_h_if_w_5 = find_out_block_h(5)
                    out_block_h_if_w_10 = find_out_block_h(10)

                    if out_block_h_if_w_5 is None and out_block_h_if_w_10 is None:
                        assert False, "This should never happen"

                    # Pick the configuration that exists if one of them does not
                    if out_block_h_if_w_5 is None:
                        out_block_w = 10
                        out_block_h = out_block_h_if_w_10
                    elif out_block_h_if_w_10 is None:
                        out_block_w = 5
                        out_block_h = out_block_h_if_w_5
                    # If both exist, pick the one that is larger in volume
                    elif out_block_h_if_w_5 * 5 > out_block_h_if_w_10 * 10:
                        out_block_h = out_block_h_if_w_5
                        out_block_w = 5
                    elif out_block_h_if_w_10 * 10 > out_block_h_if_w_5 * 5:
                        out_block_h = out_block_h_if_w_10
                        out_block_w = 10
                    # If both have the same volume, pick the configuration that is more "square"
                    else:
                        # Want to use the out_block_h/w combination which is the most "square"
                        # This calculates the height/width ratio of the blocks and then gets their
                        # distance from 1 (1 is the ideal ratio) to determine which is more square
                        squareness_5 = abs(1 - (max(out_block_h_if_w_5, 5) / min(out_block_h_if_w_5, 5)))
                        squareness_10 = abs(1 - (max(out_block_h_if_w_10, 10) / min(out_block_h_if_w_10, 10)))

                        if squareness_5 < squareness_10:
                            out_block_w = 5
                            out_block_h = out_block_h_if_w_5
                        else:
                            out_block_w = 10
                            out_block_h = out_block_h_if_w_10

                    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(7, 7),
                        in0_block_w=2,  # seeing this to 2 because 4 gives oom for long seqlen continuous batching
                        out_subblock_h=1,
                        out_subblock_w=5,
                        out_block_h=out_block_h,
                        out_block_w=out_block_w,
                        per_core_M=per_core_M,
                        per_core_N=per_core_N,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=False,
                    )

                self.model_config["PREFILL_MLP_W2_PRG_CONFIG"] = w2_prg_config

                self.model_config["WO_PREFILL_PROGCFG"] = (
                    lambda seq_len: self.matmul_1d_config(
                        seq_len, 1024, 2048, grid=ttnn.CoreGrid(x=7, y=10), overwrite_per_core_k=16
                    )
                    if seq_len == 128
                    else (
                        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 10),
                            in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                            out_subblock_h=1,  # Must be divisible by per_core_M
                            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                            per_core_M=max(1, 4 if seq_len >= 1024 else seq_len // self.tile_size // 8),  # 8~10 rows
                            per_core_N=math.ceil(2048 / 32 / 7),  # N / TILE_WIDTH / grid width
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=seq_len <= 1024,
                        )
                    )
                )

                # Calculate largest number of lm_head_num_rows such that self.dim % (lm_head_num_rows * 8) == 0
                if self.num_devices == 32:
                    lm_head_num_rows = 4
                    while self.dim % (32 * 32 * lm_head_num_rows) != 0:
                        lm_head_num_rows -= 1
                else:
                    lm_head_num_rows = 8
                    while self.dim % (32 * lm_head_num_rows * 8) != 0:
                        lm_head_num_rows -= 1
                assert (
                    lm_head_num_rows > 0
                ), f"Could not find a lm_head_num_rows such that self.dim(={self.dim}) % (lm_head_num_rows * 4) == 0"
                self.lm_head_core_grid = ttnn.CoreGrid(y=lm_head_num_rows, x=8)

                self.model_config["LM_HEAD_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        nearest_32((self.dim // (4 if self.is_galaxy else 1)) // self.lm_head_core_grid.num_cores),
                    ),  # Shard shape: [32, 128] -> 1 shard per core
                    self.lm_head_core_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.qkv_size = self.head_dim * (2 * self.n_kv_heads + self.n_heads)
                self.min_kv_prefill_shard_seqlen = (self.tile_size * 8 * 8) / (self.n_kv_heads // self.cluster_shape[1])
                self.model_config["XQKV_PREFILL_PROGCFG"] = (
                    lambda seq_len: self.matmul_1d_config(
                        seq_len, 2048, 1280, grid=ttnn.CoreGrid(x=4, y=10), overwrite_per_core_k=16
                    )
                    if seq_len == 128
                    else (
                        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                            compute_with_storage_grid_size=(7, 10),
                            in0_block_w=8,  # FIXME: optimize this config for prefill, careful use DI_DT_WORKAROUND if necessary
                            out_subblock_h=1,  # Must be divisible by per_core_M
                            out_subblock_w=2,  # Must be divisible by per_core_N, out_subblock_w * out_subblock_h <= 4
                            per_core_M=max(
                                1, 8 if seq_len >= 2048 else seq_len // self.tile_size // 8  # 8 rows
                            ),  # M / TILE_HEIGHT / Grid_Size (dynamic based on seqlen)
                            per_core_N=math.ceil(1280 / 32 / 7),  # N / TILE_WIDTH / grid width
                            transpose_mcast=False,
                            fused_activation=None,
                            fuse_batch=seq_len <= 2048,
                        )
                    )
                )

                assert self.n_kv_heads % self.cluster_shape[1] == 0, "n_kv_heads must be divisible by num_devices"
                self.model_config["KV_PREFILL_MEM_CFG"] = lambda seq_len: ttnn.create_sharded_memory_config(
                    (((self.n_kv_heads // self.cluster_shape[1]) * seq_len // (8 * 8)), self.head_dim),
                    ttnn.CoreGrid(y=8, x=8),
                    ttnn.ShardStrategy.HEIGHT,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                self.model_config["PAGED_SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        self.start_core, 32, self.sub_core_grids, row_wise=True
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=0,
                    k_chunk_size=0,
                )

                # TODO: Need to uplift UpdateCache to support dynamic chunk sizes if non-paged
                self.model_config["SDPA_DECODE_PROGCFG"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 4),
                    sub_core_grids=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        self.start_core, 32, self.sub_core_grids, row_wise=True
                    ),
                    exp_approx_mode=False,
                    q_chunk_size=256,
                    k_chunk_size=256,
                )

                self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"] = ttnn.WormholeComputeKernelConfig(
                    math_fidelity=ttnn.MathFidelity.HiFi2,
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )

                # Useful core grid based on batch size
                if self.max_batch_size == 32:
                    grid_by_batch = (8, 4)
                elif self.max_batch_size == 16:
                    grid_by_batch = (8, 2)
                elif self.max_batch_size == 8:
                    grid_by_batch = (8, 1)
                elif self.max_batch_size == 4:
                    grid_by_batch = (4, 1)
                elif self.max_batch_size == 2:
                    grid_by_batch = (2, 1)
                elif self.max_batch_size == 1:
                    grid_by_batch = (1, 1)
                else:
                    raise ValueError(f"Batch size {self.max_batch_size} not supported")
                core_grid_by_batch = ttnn.CoreGrid(y=grid_by_batch[1], x=grid_by_batch[0])
                core_range_set_by_batch = ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(grid_by_batch[0] - 1, grid_by_batch[1] - 1),
                        ),
                    }
                )

                self.model_config[
                    "SCORES_BATCHED_MM_OUTPUT_MEMCFG"
                ] = lambda batch_size_per_device_group: ttnn.create_sharded_memory_config(
                    shape=(math.ceil(self.n_local_heads / 32) * 32, self.head_dim),  # self.n_heads padded to tile size
                    core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        self.start_core, batch_size_per_device_group, self.sub_core_grids, row_wise=True
                    ),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["ROT_MAT_MEMCONFIG"] = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        core_range_set_by_batch,
                        [
                            128,
                            128,
                        ],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )

                # MLP configs
                mlp_core_grid = (
                    self.dram_shard_core_grid_for_k(self.dim)
                    if self.is_galaxy
                    else self.dram_shard_core_grid_for_k_and_n(self.dim, self.hidden_dim // self.num_devices)
                )

                self.model_config["SHARDED_MLP_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                    (
                        self.tile_padded_batch_rows,
                        self.dim // mlp_core_grid.num_cores,
                    ),  # Shard shape: [32, 128] -> 1 shard per core
                    mlp_core_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["DECODE_MLP_W1_W3_PRG_CONFIG"] = self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=self.dim,
                    n=self.hidden_dim // self.cluster_shape[1],
                    num_cores=mlp_core_grid.num_cores,
                )

                mlp2_core_grid = (
                    ttnn.CoreGrid(y=1, x=8)
                    if self.is_galaxy
                    else self.dram_shard_core_grid_for_k_and_n(self.hidden_dim // self.num_devices, self.dim)
                )

                self.model_config["SHARDED_MLP2_INPUT_MEMCFG"] = ttnn.create_sharded_memory_config(
                    (
                        32 if self.is_galaxy else self.tile_padded_batch_rows,
                        self.hidden_dim // self.cluster_shape[1] // mlp2_core_grid.num_cores,
                    ),
                    mlp2_core_grid,
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["DECODE_MLP_W2_PRG_CONFIG"] = self.dram_matmul_config(
                    m=self.tile_padded_batch_rows,
                    k=self.hidden_dim // self.cluster_shape[1],
                    n=self.dim,
                    num_cores=mlp2_core_grid.num_cores,
                )

                ##### Prefetcher stuff #####
                self.model_config["USE_PREFETCHER"] = self.use_prefetcher
                RING_SIZE = 24
                ring_core_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in PREFETCHER_NOC1_GRID
                    ]
                )
                pf_mm_out_core_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in self.pf_receiver_cores_list
                    ]
                )
                attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
                # QKV
                self.model_config["SHARDED_ATTN_INPUT_RING_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(32, 2304 // RING_SIZE),  # Use padded K
                        core_grid=ring_core_range_set,
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.is_galaxy
                    else ttnn.create_sharded_memory_config(
                        (
                            self.tile_padded_batch_rows,
                            self.dim // attn_input_grid.num_cores,
                        ),  # Shard shape: [32, 128] -> 1 shard per core
                        attn_input_grid,
                        ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )
                qkv_shape_ring = (8192 // 4, 12288 // 8)  # Use padded K and N
                self.model_config["SHARDED_QKV_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                    k=qkv_shape_ring[0],
                    n=qkv_shape_ring[1],
                )

                qkv_out_shard_shape_ring = (32, 12288 // 8 // RING_SIZE)  # Use padded N
                self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=qkv_out_shard_shape_ring,
                    core_grid=pf_mm_out_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["XQKV_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
                    1,
                    32,
                    8192 // 4,
                    12288 // 8,  # Use padded N
                    RING_SIZE,
                    untilize_out=True,
                )
                RS_CREATE_HEADS_PACKET_WORKER_CRS = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(3, 0)),
                        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(2, 1)),
                    ]
                )
                self.model_config["RS_CREATE_HEADS_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 512),
                    core_grid=RS_CREATE_HEADS_PACKET_WORKER_CRS,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                # WO
                self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 12288 // 8 // RING_SIZE),  # Use padded K
                    core_grid=ring_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                wo_shape_ring = (8192 // 8, 9216 // 4)  # Use padded K and N
                self.model_config["SHARDED_WO_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                    k=wo_shape_ring[0],
                    n=wo_shape_ring[1],
                )

                wo_out_shard_shape_ring = (32, 9216 // 4 // RING_SIZE)  # Use padded N
                self.model_config["SHARDED_WO_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=wo_out_shard_shape_ring,
                    core_grid=pf_mm_out_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                self.model_config["WO_DECODE_RING_PROGCFG"] = self.matmul_1d_ring_config(
                    1,
                    32,
                    10240 // 8,
                    9216 // 4,  # Use padded N
                    RING_SIZE,
                )

                # Use padded K and N
                self.model_config["W1W3_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                    k=8192 // 4,
                    n=3840,
                )

                # Use padded K and N
                self.model_config["W2_RING_MEMCFG"] = self.create_dram_sharded_mem_config(
                    k=3584,
                    n=9216 // 4,
                )

                self.model_config["FF1_3_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
                    1,
                    32,
                    8192 // 4,
                    3840,  # Use padded N
                    RING_SIZE,
                )

                self.model_config["FF2_TG_RING_PROGCFG"] = self.matmul_1d_ring_config(
                    1,
                    32,
                    3584,
                    9216 // 4,  # Use padded N
                    RING_SIZE,
                )

                self.model_config["SHARDED_FF12_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 9216 // 4 // RING_SIZE),  # Use padded N
                    core_grid=ring_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                self.model_config["SHARDED_FF12_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 3840 // RING_SIZE),  # Use padded N
                    core_grid=pf_mm_out_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["SHARDED_FF12_PRE_MUL_RING_REDUCE_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 3840 // 30),  # Use padded N
                    core_grid=ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        self.start_core, 30, self.sub_core_grids, row_wise=True
                    ),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                mul_core_range_set = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, 28, self.sub_core_grids, row_wise=True
                )
                self.model_config["MUL_IN_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 3584 // 28),  # Use padded K
                    core_grid=mul_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                self.model_config["FF2_IN_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 3840 // RING_SIZE),  # Use padded K
                    core_grid=ring_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                self.model_config["FF2_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 9216 // 4 // RING_SIZE),  # Use padded N
                    core_grid=pf_mm_out_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                core_grid_ln, grid_offset = (8, 2), ttnn.CoreCoord(1, 0)
                core_range = ttnn.CoreRange(
                    grid_offset,
                    ttnn.CoreCoord(core_grid_ln[1] + grid_offset.x - 1, core_grid_ln[0] + grid_offset.y - 1),
                )
                LM_HEAD_RING_SIZE = 24
                self.lm_head_shape = (8192 // 4, 128 * 1024 // 8)

                lm_head_ring_core_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in LM_HEAD_32_GRID
                    ]
                )

                lm_head_ring_core_input_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in LM_HEAD_INPUT_GRID
                    ]
                )

                lm_head_ring_core_output_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in LM_HEAD_OUTPUT_GRID
                    ]
                )

                lm_head_ring_16_core_range_set = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(
                            ttnn.CoreCoord(x, y),
                            ttnn.CoreCoord(x, y),
                        )
                        for x, y in LM_HEAD_16_GRID
                    ]
                )
                self.model_config["SHARDED_LM_HEAD_INPUT_32_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 2304 // LM_HEAD_RING_SIZE),  # padded shape
                    core_grid=lm_head_ring_core_input_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["SHARDED_LM_HEAD_INPUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, self.lm_head_shape[0] // 16),
                    core_grid=lm_head_ring_16_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["LM_HEAD_OUT_RING_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 16896 // LM_HEAD_RING_SIZE),  # padded shape
                    core_grid=lm_head_ring_core_output_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["LM_HEAD_OUT_RING_RESHARD_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, self.lm_head_shape[1] // 32),
                    core_grid=lm_head_ring_core_range_set,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["LM_HEAD_TG_RING_PROGCFG"] = self.matmul_1d_ring_lm_head_config(
                    1,
                    32,
                    self.dim // 4,
                    16896,  # use padded shape
                    LM_HEAD_RING_SIZE,
                    prefetch=False,
                )

                self.model_config["LM_HEAD_PREFILL_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                    in0_shape=(1, 1, 32, 2048),
                    in1_shape=(1, 1, 2048, 16384),
                    grid=ttnn.CoreGrid(x=7, y=7),  # (7,10) leads to hangs
                    act=None,
                    is_fp32_accumulate=False,
                    # overwrite_subblock_w=1,
                    # overwrite_subblock_h=1,
                )

                attn_input_grid = self.dram_shard_core_grid_for_k(self.dim)
                attn_input_sub_core_grid = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    self.start_core, 32, self.sub_core_grids, row_wise=True
                )
                self.model_config["SHARDED_ATTN_INPUT_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(32, nearest_32(self.dim // (8 * lm_head_num_rows) // 4)),
                        core_grid=attn_input_sub_core_grid,
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.is_galaxy
                    else ttnn.create_sharded_memory_config(
                        (
                            self.tile_padded_batch_rows,
                            self.dim // attn_input_grid.num_cores,
                        ),  # Shard shape: [32, 128] -> 1 shard per core
                        attn_input_grid,
                        ttnn.ShardStrategy.WIDTH,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )

                # glx doesn't support DRAM sharded matmuls yet
                self.model_config["XQKV_DECODE_PROGCFG"] = (
                    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(8, 5 if self.is_70b else lm_head_num_rows),
                        in0_block_w=2 if self.is_70b else 1,
                        out_subblock_h=1,
                        out_subblock_w=1,
                        per_core_M=1,
                        per_core_N=1,
                        fuse_batch=True,
                        fused_activation=None,
                        mcast_in0=True,
                    )
                    if self.is_galaxy
                    else self.dram_matmul_config(
                        m=self.tile_padded_batch_rows,
                        k=self.dim,
                        n=self.qkv_size // self.num_devices,
                        num_cores=attn_input_grid.num_cores,
                    )
                )

                full_grid = ttnn.CoreRangeSet(
                    {
                        ttnn.CoreRange(
                            ttnn.CoreCoord(0, 0),
                            ttnn.CoreCoord(7, 7),
                        )
                    }
                )
                self.model_config["FULL_GRID_MEMCFG"] = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        full_grid,
                        [
                            32,
                            nearest_32(56),
                        ],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )

                self.model_config["MLP_ACT_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(32, self.dim // 4 // 16),  # dim / num devices / 16 cores
                        core_grid=ttnn.CoreGrid(x=8, y=2),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.dim >= 4096
                    else self.model_config["FULL_GRID_MEMCFG"]
                )

                self.model_config["FF1_3_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                    (
                        1,
                        1,
                        32,
                        self.dim // 4,
                    ),
                    (
                        1,
                        1,
                        self.dim // 4,
                        self.hidden_dim // 8,
                    ),
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )

                self.model_config["FF2_TG_PROGCFG"] = self.matmul_1d_config_from_tensor_shapes(
                    (
                        1,
                        1,
                        32,
                        self.hidden_dim // 8,
                    ),
                    (
                        1,
                        1,
                        self.hidden_dim // 8,
                        self.dim // 4,
                    ),
                    grid=ttnn.CoreGrid(x=8, y=2),
                    overwrite_subblock_h=1,
                    overwrite_subblock_w=1,
                )
                self.model_config["FF1_OUT_REDUCE_SCATTER_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, self.hidden_dim // 28 // 8),  # shard_grid_cores = 28, num_devices=8
                    core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(6, 3))}),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )  # if self.dim==8192 else ttnn.DRAM_MEMORY_CONFIG

                self.model_config["FF1_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32 * 4, self.hidden_dim // 8 // 8),
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.model_config["FF2_OUT_REDUCE_SCATTER_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(32, self.dim // 8 // 4),  # shard_grid_cores = 8, num_devices=4
                        core_grid=ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))}),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.dim == 8192
                    else ttnn.create_sharded_memory_config(
                        shape=(32 * 8, self.dim // 4 // 8),
                        core_grid=ttnn.CoreGrid(y=1, x=8),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )

                # Note PACKET_WORKER_CRS is 8 cores and it can NOT use any core in the following ranges:
                # {2,8}-{3,8},{5,3}-{6,3}  (CCL cores),
                # {1,0}-{2,0}, {1,4}-{2,5}, {1,9}-{2,9}, {5,0}-{6,2}, {5,4}-{6,7}, {5,9}-{6,9} (Matmul)
                # {0,0}-{0,9}, {4,0}-{4,9} (Prefetcher)
                # {3,6} (Matmul hop core)
                PACKET_WORKER_CRS = ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 1), ttnn.CoreCoord(3, 2)),
                        ttnn.CoreRange(ttnn.CoreCoord(1, 3), ttnn.CoreCoord(2, 3)),
                    ]
                )
                self.model_config["REDUCE_SCATTER_INTERIM_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32, 512),
                    core_grid=PACKET_WORKER_CRS,
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                FF1_CRS_RS_OUT = ttnn.num_cores_to_corerangeset_in_subcoregrids(
                    ttnn.CoreCoord(1, 0), 30, self.sub_core_grids, row_wise=True
                )
                self.model_config["REDUCE_SCATTER_OUT_MEMCFG"] = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        FF1_CRS_RS_OUT,
                        [32, 32],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )

                self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"] = (
                    ttnn.create_sharded_memory_config(
                        shape=(32, 2048 // 8 // 8),  # mesh_rows = 8, num_cores=8
                        core_grid=ttnn.CoreGrid(y=1, x=8),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                    if self.dim == 8192
                    else ttnn.create_sharded_memory_config(
                        shape=(32 * 8, nearest_32(self.dim // 4 // 32)),  # mesh_rows = 8
                        core_grid=ttnn.CoreGrid(y=4, x=8),
                        strategy=ttnn.ShardStrategy.WIDTH,
                        orientation=ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )
                )

                self.model_config["FF2_OUT_GATHERED_MEMCFG"] = ttnn.create_sharded_memory_config(
                    shape=(32 * 8, self.dim // 4 // 8),
                    core_grid=ttnn.CoreGrid(y=1, x=8),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )

                # Vision model configs
                self.model_config["IMAGE_MLP_FC_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=self.vision_dim,
                    n=self.vision_hidden_dim // self.num_devices,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )
                self.model_config["IMAGE_MLP_PROJ_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=self.vision_hidden_dim // self.num_devices,
                    n=self.vision_dim,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )
                self.model_config["IMAGE_ATTN_QKV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=self.vision_dim,
                    n=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3)
                    // self.num_devices,  # Head dim was padded to nearest 32
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )
                self.model_config["IMAGE_ATTN_OUT_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=(nearest_32(self.vision_head_dim) * self.vision_attn_n_heads * 3) // self.num_devices,
                    n=self.vision_dim,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )
                self.model_config["VISION_XATTN_Q_PROGCFG"] = lambda seq_len: self.matmul_config(
                    m=min(seq_len, 1024),
                    k=self.dim,
                    n=(self.head_dim * self.n_heads) // self.num_devices,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= 1024,
                )
                self.model_config["VISION_XATTN_KV_PROGCFG"] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=self.dim,
                    n=(self.head_dim * self.n_kv_heads) // self.num_devices,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )
                self.model_config["VISION_XATTN_SCORE_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                    m=seq_len,
                    k=self.head_dim,
                    n=cache_seq_len,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=False,
                )
                self.model_config["VISION_XATTN_OUTPUT_PROGCFG"] = lambda seq_len, cache_seq_len: self.matmul_config(
                    m=seq_len,
                    k=cache_seq_len,
                    n=self.head_dim,
                    grid_size=(8, 8),
                    # in0_block_w=1, # TODO: Remove this when we get non-causal FlashDecode
                    fuse_batch=False,
                )
                self.model_config["VISION_XATTN_DENSE_PROGCFG"] = lambda seq_len: self.matmul_config(
                    m=min(seq_len, 1024),
                    k=self.dim // self.num_devices,
                    n=self.dim,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=False,
                )

                self.model_config["VISION_PROJ_PROGCFG"] = lambda seq_len: self.matmul_config(
                    m=seq_len,
                    k=self.vision_dim * 6,
                    n=self.dim // self.num_devices,
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=False,
                )

                self.model_config[
                    "CROSS_TRANSFORMER_TEXT_OUTPUT_PROGCFG"
                ] = lambda seq_len, max_seq: self.matmul_config(
                    m=min(seq_len, max_seq),
                    k=self.dim,
                    n=self.vocab_size // 8,  # Magic number. LM Head always contains 8 splits
                    grid_size=(8, 8),
                    in0_block_w=1,
                    fuse_batch=seq_len <= max_seq,
                )

                def _get_xattn_kv_prefill_mem_cfg(seq_len):
                    M = (self.n_kv_heads // self.num_devices) * seq_len
                    cores_x, cores_y = self.find_grid(M // self.tile_size)
                    return ttnn.create_sharded_memory_config(
                        (
                            nearest_32(M // (cores_x * cores_y)),
                            self.head_dim,
                        ),
                        ttnn.CoreGrid(y=cores_y, x=cores_x),
                        ttnn.ShardStrategy.HEIGHT,
                        ttnn.ShardOrientation.ROW_MAJOR,
                        use_height_and_width_as_shard_shape=True,
                    )

                self.model_config["XATTN_KV_PREFILL_MEM_CFG"] = _get_xattn_kv_prefill_mem_cfg

                self.VISION_MAX_MM_SEQ = nearest_32(self.vision_chunk_ntok)
                # RMS NORM
                self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"] = self.create_sharded_norm_config(attn_input_grid)
                self.model_config["SHARDED_NORM_MLP_PRGM_CFG"] = self.create_sharded_norm_config(mlp_core_grid)
                self.model_config["SHARDED_NORM_LM_HEAD_PRGM_CFG"] = self.create_sharded_norm_config(
                    self.lm_head_core_grid
                )

                # All gather matmuls currently only supported on T3K
                # We need it sharded on num_cores = num_devices
                self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"] = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        num_to_core_range_set(self.num_devices),
                        [
                            self.tile_padded_batch_rows,
                            self.dim // self.num_devices,
                        ],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )

                self.model_config = set_tg_attention_config(self.model_config, self.dim)

                self.is_multichip = self.num_devices > 1
                self.num_reduce_scatter_links = 1
                self.num_all_gather_links = (
                    2 if self.is_galaxy else 1
                )  # TODO: try out 3 for short axis and 4 for long axis (TG only) <- should work but untested in model
                self.ccl_dtype = ttnn.bfloat8_b

        def _set_params_from_dict(self, params, is_hf=False):
            # Common params with different names between Meta and HF
            super()._set_params_from_dict(params, is_hf)
            self.dim = params.get("dim", params.get("hidden_size"))
            self.n_heads = params.get("n_heads", params.get("num_attention_heads"))
            self.n_kv_heads = params.get("n_kv_heads", params.get("num_key_value_heads"))
            self.n_layers = params.get("n_layers", params.get("num_hidden_layers"))
            self.full_model_n_layers = self.n_layers
            self.norm_eps = params.get("norm_eps", params.get("rms_norm_eps"))
            self.vocab_size = params["vocab_size"]
            self.padded_vocab_size = 128 * 1024
            self.head_dim = params.get("head_dim", self.dim // self.n_heads)
            if is_hf:
                self.max_context_len = params.get("max_position_embeddings")
            else:
                self.max_context_len = (
                    128 * 1024
                )  # For Llama3 Meta weights TODO: Remove this when we move to HF weights only

            # Handle different MLP dimension specifications
            if "intermediate_size" in params:
                self.hidden_dim = params["intermediate_size"]
                self.ffn_dim_multiplier = None
                self.multiple_of = None
            else:
                self.ffn_dim_multiplier = params["ffn_dim_multiplier"]
                self.multiple_of = params["multiple_of"]
                self.hidden_dim = calculate_hidden_dim(self.dim, self.ffn_dim_multiplier, self.multiple_of)

            if "_name_or_path" in params:
                if is_hf:
                    normalized_path = os.path.normpath(params["_name_or_path"])
                    # For HF paths, they might end with `<model_name>/snapshots/<snapshot_id>/`
                    if "snapshots" in normalized_path:
                        full_model_name = normalized_path.split(os.path.sep)[-3]
                        self.model_name = full_model_name.split("--")[-1]
                    else:
                        self.model_name = os.path.basename(normalized_path)
                else:
                    self.model_name = os.path.basename(params["_name_or_path"])
                logger.info(f"Model name from params: {self.model_name}")

            if self.base_model_name == "Qwen2.5-7B" and self.num_devices not in [0, 2, 4]:
                raise AssertionError(
                    "Qwen2.5-7B is only supported on 2 or 4 devices, run on an N300 or use MESH_DEVICE=N150x4"
                )

            self.unpadded_hidden_dim = self.hidden_dim
            # Don't need to pad for CPU runs
            if self.num_devices:
                # Default padding cores for each model, 0 if not set here
                default_padded_cores = {
                    "Qwen2.5-72B": 32,
                    "Qwen2.5-7B": 16,
                    "QwQ-32B": 16,
                }.get(self.base_model_name, 0)

                # Override MLP padding cores from env var
                mlp_padded_cores = int(os.environ.get("PAD_MLP_CORES", default_padded_cores))

                # Only pad if MLP_PADDED_CORES is non-zero
                if mlp_padded_cores > 0:
                    padded_hidden_dim = nearest_multiple(
                        self.hidden_dim, mlp_padded_cores * self.tile_size * self.num_devices
                    )
                    if padded_hidden_dim != self.hidden_dim:
                        logger.info(
                            f"PAD_MLP_CORES={mlp_padded_cores}, padding hidden dim from {self.hidden_dim} to {padded_hidden_dim}"
                        )
                        self.hidden_dim = padded_hidden_dim

            # RoPE params
            self.rope_theta = params.get("rope_theta")
            # If use_scaled_rope is not present, assume setting rope_scaling means use scaled rope
            # If it is present and is set to false, do not use scaled rope
            # Setting self.rope_scaling_factor to None is our way of saying do not use scaled rope
            rope_scaling_params = params.get("rope_scaling", None)
            if rope_scaling_params:
                self.rope_scaling_factor = rope_scaling_params.get("factor", None)
                self.orig_context_len = rope_scaling_params.get("original_max_position_embeddings", None)
            else:
                self.rope_scaling_factor = None
                self.orig_context_len = None

            # Vision params (Meta-specific)
            self.vision_chunk_size = params.get("vision_chunk_size", -1)
            self.vision_max_num_chunks = params.get("vision_max_num_chunks", 4)
            self.vision_num_cross_attention_layers = params.get("vision_num_cross_attention_layers", -1)

            # Vision constants
            self.vision_dim = 1280
            self.vision_mlp_ratio = 4
            self.vision_hidden_dim = int(self.vision_dim * self.vision_mlp_ratio)
            self.vision_act_layer = ttnn.UnaryOpType.GELU
            self.vision_dropout = 0.0
            self.vision_attn_n_heads = 16
            self.vision_head_dim = self.vision_dim // self.vision_attn_n_heads
            self.vision_n_layers = 32
            self.vision_n_global_layers = 8
            self.vision_max_num_tiles = 4
            self.vision_patch_size = 14
            self.vision_in_channels = 3

        def matmul_1d_ring_config(
            self,
            B,
            M,
            K,
            N,
            num_cores,
            prefetch=True,
            untilize_out=False,
        ):
            M *= B  # Fuse batch always enabled

            in0_block_h = M // ttnn.TILE_SIZE
            in0_block_w = K // num_cores // ttnn.TILE_SIZE
            out_block_h = M // ttnn.TILE_SIZE
            out_block_w = N // num_cores // ttnn.TILE_SIZE

            num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
            num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
            num_blocks_total = num_blocks_y * num_blocks_x

            if num_blocks_total != num_cores:
                assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

            out_subblock_h = 1
            out_subblock_w = 8
            while out_block_w % out_subblock_w != 0:
                out_subblock_w -= 1

            hop_grid = [(3, 6)] if prefetch else []  # FIXME: Make not hard coded
            hop_core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in hop_grid
                }
            )
            grid = num_to_coregrid(num_cores)

            program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
                gather_in0=True,
                hop_cores=hop_core_range_set,
                num_global_cb_receivers=2 if prefetch else 1,
                untilize_out=untilize_out,
            )

            return program_config

        def matmul_1d_ring_lm_head_config(
            self,
            B,
            M,
            K,
            N,
            num_cores,
            prefetch=True,
        ):
            M *= B  # Fuse batch always enabled

            in0_block_h = M // ttnn.TILE_SIZE
            in0_block_w = K // num_cores // ttnn.TILE_SIZE
            out_block_h = M // ttnn.TILE_SIZE
            out_block_w = N // num_cores // ttnn.TILE_SIZE

            num_blocks_y = (M // ttnn.TILE_SIZE - 1) // out_block_h + 1
            num_blocks_x = (N // ttnn.TILE_SIZE - 1) // out_block_w + 1
            num_blocks_total = num_blocks_y * num_blocks_x

            if num_blocks_total != num_cores:
                assert False, f"num_blocks_total {num_blocks_total} != num_cores {num_cores}"

            out_subblock_h = 1
            out_subblock_w = 8
            while out_block_w % out_subblock_w != 0:
                out_subblock_w -= 1

            hop_grid = [(3, 6)]
            hop_core_range_set = ttnn.CoreRangeSet(
                {
                    ttnn.CoreRange(
                        ttnn.CoreCoord(x, y),
                        ttnn.CoreCoord(x, y),
                    )
                    for x, y in hop_grid
                }
            )
            grid = num_to_coregrid(num_cores)

            program_config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=in0_block_w,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=out_block_h,
                per_core_N=out_block_w,
                fuse_batch=True,
                fused_activation=None,
                mcast_in0=False,
                gather_in0=True,
                hop_cores=hop_core_range_set,
            )

            return program_config

        def matmul_1d_config(
            self,
            m,
            k,
            n,
            grid=ttnn.CoreGrid(x=8, y=8),
            act=None,
            is_fp32_accumulate=False,
            overwrite_per_core_k=None,
            overwrite_subblock_w=None,
            overwrite_subblock_h=None,
        ):
            tile_width = 32
            tile_height = 32

            if (
                n // tile_width // grid.num_cores < 1
            ):  # use less number of cores in case we have more N num tiles than cores
                # assert (n // tile_width) % grid.x == 0
                grid_y = n // tile_width // grid.x
                grid = ttnn.CoreGrid(x=grid.x, y=grid_y)

            per_core_m = m // tile_height
            per_core_k = math.ceil(k / tile_width / grid.num_cores)
            per_core_n = math.ceil(n / tile_width / grid.num_cores)

            if is_fp32_accumulate:
                max_subblock_w_h = 4
            else:
                max_subblock_w_h = 8

            # find the largest value between 1 and 8 that is a factor of per_core_n
            # e.g. if per_core_n is 14, then out_subblock_w = 7
            out_subblock_w = max([i for i in range(1, max_subblock_w_h + 1) if per_core_n % i == 0])

            # find the largest value that is a factor of per_core_m such that
            # out_subblock_w * out_subblock_h <= 8
            out_subblock_h = max(
                [
                    i
                    for i in range(1, max_subblock_w_h + 1)
                    if per_core_m % i == 0 and i * out_subblock_w <= max_subblock_w_h
                ]
            )

            if overwrite_per_core_k is not None:
                per_core_k = overwrite_per_core_k

            if overwrite_subblock_w is not None:
                out_subblock_w = overwrite_subblock_w

            if overwrite_subblock_h is not None:
                out_subblock_h = overwrite_subblock_h

            return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(grid.x, grid.y),
                in0_block_w=per_core_k,
                out_subblock_h=out_subblock_h,
                out_subblock_w=out_subblock_w,
                per_core_M=per_core_m,
                per_core_N=per_core_n,
                fuse_batch=True,
                fused_activation=act,
                mcast_in0=True,
            )

        def matmul_1d_config_from_tensor_shapes(
            self,
            in0_shape,
            in1_shape,
            grid=ttnn.CoreGrid(x=8, y=8),
            act=None,
            is_fp32_accumulate=False,
            overwrite_subblock_w=None,
            overwrite_subblock_h=None,
        ):
            m, k, n = in0_shape[0] * in0_shape[1] * in0_shape[2], in0_shape[3], in1_shape[3]
            return self.matmul_1d_config(
                m,
                k,
                n,
                grid,
                act,
                is_fp32_accumulate,
                overwrite_subblock_w=overwrite_subblock_w,
                overwrite_subblock_h=overwrite_subblock_h,
            )

    return QwenEmbeddingArgs


class QwenEmbeddingModel:
    def __init__(self, device, data_parallel=1):
        self.generator_args_config = {
            "num_devices": device.get_num_devices() if isinstance(device, ttnn.MeshDevice) else 1,
            "data_parallel": data_parallel,
            "mesh_device": device,
            "instruct": False,
            "global_batch_size": data_parallel,
            "optimizations": lambda model_args: DecodersPrecision.performance(
                model_args.n_layers, model_args.model_name
            ),
            "max_seq_len": 1024,
            "page_params": {"page_block_size": 32, "page_max_num_blocks_per_dp": 1024},
            "paged_attention": True,
            "num_layers": 10,
        }
        (
            self.model_args,
            self.model,
            self.page_table,
            self.tt_kv_cache,
            self.tokenizer,
            processor,
        ) = prepare_generator_args(
            **self.generator_args_config,
            model_factory_fn=lambda *args, **kwargs: create_tt_model(
                *args,
                **kwargs,
                ModelArgsClass=get_QwenEmbeddingArgs(),
                TransformerClass=TtTransformer,
                attentionClass=TtQwenAttention,
            ),
        )
        self.generator = Generator(self.model, self.model_args, device, self.tokenizer)


@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_1D,
            "trace_region_size": 70000000,
            "num_command_queues": 1,
            "l1_small_size": 81920,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "mesh_device",
    [
        {
            "N150": (1, 1),
            "N300": (1, 2),
            "N150x4": (1, 4),
            "T3K": (1, 8),
            "TG": (8, 4),
            "P150": (1, 1),
            "P300": (1, 2),
            "P150x4": (1, 4),
            "P150x8": (1, 8),
        }.get(os.environ.get("MESH_DEVICE"), len(ttnn.get_device_ids()))
    ],
    indirect=True,
)
@pytest.mark.parametrize("data_parallel", [1])
def test_qwen_embedding_demo(
    device_params,
    mesh_device,
    data_parallel,
):
    model = QwenEmbeddingModel(mesh_device, data_parallel)

    # Run model to generate embeddings
    test_prompts = [
        "Embedding models convert text into vector representations.",
    ]

    logger.info(f"Testing Qwen3-Embedding-8B with {len(test_prompts)} prompts")

    embeddings = []
    for idx, prompt in enumerate(test_prompts):
        logger.info(f"Processing prompt {idx + 1}/{len(test_prompts)}: {prompt}")

        # Tokenize the prompt
        tokens = model.tokenizer.encode(prompt, add_special_tokens=True)
        tokens_tensor = torch.tensor([tokens], dtype=torch.long)

        logger.info(f"Prompt tokenized to {len(tokens)} tokens")

        # Run prefill to get embeddings (using the last hidden state)
        # For embedding models, we typically use the model output before the LM head
        logits = model.generator.prefill_forward_text(
            tokens_tensor,
            page_table=model.page_table,
            kv_cache=model.tt_kv_cache,
            prompt_lens=torch.tensor([len(tokens)], dtype=torch.long),
        )

        logger.info(f"Generated embedding with shape: {logits.shape}")
        embeddings.append(logits)

    logger.info(f"Successfully generated {len(embeddings)} embeddings")

    # Verify embeddings have expected dimensions
    for idx, embedding in enumerate(embeddings):
        logger.info(f"Embedding {idx + 1} shape: {embedding.shape}")
        assert embedding.shape[0] == 1, "Batch size should be 1"
        assert embedding.dim() >= 2, "Embedding should have at least 2 dimensions"

    logger.info("Qwen3-Embedding-8B demo completed successfully!")
