# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm


class TtLlamaAttention(LightweightModule):
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
        self.layer_num = layer_num
        self.num_devices = configuration.num_devices
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
        self.num_devices_per_group = self.n_kv_heads
        self.batch_size_per_device_group = max(self.max_batch_size // self.num_device_groups, 1)

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.prefetcher_setup = prefetcher_setup
        self.tt_ccl = tt_ccl
        self.cluster_shape = configuration.cluster_shape
        self.is_qwen = getattr(configuration, "is_qwen", False)
        self.is_blackhole = getattr(configuration, "is_blackhole", False)

        # TODO: Fix this once all-gather supports < tile_size
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

        # Column bounds for prefix caching: mask = (lower <= user_id < upper)
        # Column 0: [0, 8), Column 1: [8, 16), Column 2: [16, 24), Column 3: [24, 32)
        # Shape [8, 4, 1, 32]: last dim must be 32 for ttnn typecast compatibility (ROW_MAJOR requires %32)
        # Use uint32 to match user_id dtype;
        # Per-column user_id bounds for chunked SDPA mask: column col is active for user_id in [col*8, (col+1)*8).
        # Sharded over 8x4 mesh (dims 0,1); each device gets (1, 1, 1, 32). Column 0: [0,8), 1: [8,16), 2: [16,24), 3: [24,32).
        lower = torch.zeros(8, 4, 1, 32, dtype=torch.int32)
        upper = torch.zeros(8, 4, 1, 32, dtype=torch.int32)
        for col in range(4):
            lower[:, col, :, :] = col * 8
            upper[:, col, :, :] = (col + 1) * 8
        self.column_lower = ttnn.from_torch(
            lower,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        )
        self.column_upper = ttnn.from_torch(
            upper,
            device=mesh_device,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, dims=(0, 1), mesh_shape=[8, 4]),
        )

        self.dtype = dtype
        self.qk_norm = configuration.qk_norm

        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        # Single source of truth for the prefetcher gate (Wormhole/TG True, Blackhole bring-up False).
        # Also mirrored into model_config for any config-dict consumers.
        self.use_prefetcher = configuration.use_prefetcher
        self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
        self.sdpa_decode_compute_kernel_config = self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"]
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
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
        assert self.num_devices == 32

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

        self._qkv_n_local = self.n_local_heads * self.head_dim + 2 * self.n_local_kv_heads * self.head_dim
        # nlp_create_qkv_heads_decode only honors batch_offset/slice_size in its WIDTH_SHARDED
        # program factory. The model's CREATE_HEAD_INPUT_MEMCFG is sized for the llama_rs path
        # (8 cores / width 1024 on Blackhole), not our per-device nlp fused width
        # (_qkv_n_local = (n_q + 2*n_kv)*head_dim = 1280 -> 10 heads). Build a dedicated
        # width-sharded config with one head (head_dim) per core so batch_offset selects the
        # correct per-column users.
        self._create_head_input_memcfg_nlp = None
        sub_core_grids = getattr(configuration, "sub_core_grids", None)
        start_core = getattr(configuration, "start_core", None)
        if sub_core_grids is not None and start_core is not None and self._qkv_n_local % self.head_dim == 0:
            n_qkv_heads_local = self._qkv_n_local // self.head_dim
            self._create_head_input_memcfg_nlp = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.num_cores_to_corerangeset_in_subcoregrids(
                        start_core, n_qkv_heads_local, sub_core_grids, row_wise=False
                    ),
                    [32, self.head_dim],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        # Ring stuff
        # Llama3: 9216, 12288
        # Qwen3: 6144, 12288

        # Llama3: [1, 1, 8192, 10240] -> [2304, 1536]
        # Qwen3: [1, 1, 5120, 10240] -> [1280, 1536]
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_QKV_RING_MEMCFG"],
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name("wqkv_sharded_2d_prefetcher"),  ## TODO: Fix caching
        )
        self.wqkv_interleaved = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name("wqkv_sharded_2d_dram"),  ## TODO: Fix caching
        )
        # Blackhole no-prefetch per-device QKV weight: the interleaved-DRAM weight over the device
        # group (native qkv dims, no ring K/N padding), summed across mesh columns on device after
        # the matmul. Same tensor as wqkv_interleaved; Wormhole/prefetcher uses the ring self.wqkv
        # above and never touches this.
        self.wqkv_per_device = None if configuration.use_prefetcher else self.wqkv_interleaved

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_WO_RING_MEMCFG"],
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_prefetcher"),
        )
        self.wo_interleaved = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_dram"),
        )
        self.wo_interleaved_col_sharded = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(None, 3),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_col_sharded_dram"),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5
        if tt_ccl.mode == "decode" and self.use_prefetcher:
            self.prefetch(prefetcher_setup, tt_ccl)

        # If we are using qk_norm, we need to add a layer norm to the q and k
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize QK norm if weights are present in state_dict
        if f"{q_norm_str}.weight" in self.state_dict:
            self.qk_norm = True

            # Memory configurations for QK norm
            self.reshape_intermediate_q_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 128),  # [1, 8, 8 (32), 128] ==> *[1, 1, 64, 128]* ==> [1, 1, 64, 32 * 4 = 128]
                core_grid=ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))
                    ]  # This captures the fact that we are using 1 core (height sharded)
                ),  # resharding tensor to 1 core
                strategy=ttnn.ShardStrategy.HEIGHT,  # Literally stating to the device to perform height sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )
            self.reshape_intermediate_k_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 128),  # [1, 8, 8 (32), 128] ==> *[1, 1, 64, 128]* ==> [1, 1, 64, 32 * 4 = 128]
                core_grid=ttnn.CoreRangeSet(
                    [
                        ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0))
                    ]  # This captures the fact that we are using 1 core (height sharded)
                ),  # resharding tensor to 1 core
                strategy=ttnn.ShardStrategy.HEIGHT,  # Literally stating to the device to perform height sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.reshape_output_q_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
                core_grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]
                ),  # resharding tensor to cores
                strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            self.reshape_output_k_mem_cfg = ttnn.create_sharded_memory_config(
                shape=(64, 32),  # [1, 8, 8, 128] ==> [1, 1, 64, 128] ==> *[1, 1, 64, 32 * 4 = 128]*
                core_grid=ttnn.CoreRangeSet(
                    [ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 3))]
                ),  # resharding tensor to cores
                strategy=ttnn.ShardStrategy.WIDTH,  # Literally stating to the device to perform width sharding
                orientation=ttnn.ShardOrientation.ROW_MAJOR,
                use_height_and_width_as_shard_shape=True,
            )

            # Program configuration for norm
            block_w = 128 // 4 // 32
            # Find largest value <= 4 that evenly divides block_w
            subblock_w = 1
            while subblock_w > 0:
                if block_w % subblock_w == 0:
                    break
                subblock_w -= 1
            self.norm_program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                compute_with_storage_grid_size=[2, 2],
                subblock_w=subblock_w,
                block_h=2,  # 64 // 32
                block_w=block_w,
                inplace=False,
            )

            # Create Q norm
            self.q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                state_dict=self.state_dict,
                state_dict_prefix=None,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                sharded_program_config=self.norm_program_cfg,
                sharded_output_config=self.reshape_output_q_mem_cfg,
            )

            # Create K norm
            self.k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                state_dict=self.state_dict,
                state_dict_prefix=None,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                sharded_program_config=self.norm_program_cfg,
                sharded_output_config=self.reshape_output_k_mem_cfg,
            )

            self.q_norm_weight = self.state_dict[q_norm_str + ".weight"]
            self.k_norm_weight = self.state_dict[k_norm_str + ".weight"]

        else:
            self.qk_norm = False

    def prefetch(self, prefetcher_setup, tt_ccl):
        self.prefetcher_setup = prefetcher_setup
        if tt_ccl.mode == "decode" and self.prefetcher_setup is not None:
            self.prefetcher_setup.insert_tensor(self.wqkv)
            self.prefetcher_setup.insert_tensor(self.wo)
        self.tt_ccl = tt_ccl

    def _apply_decode_qk_norm(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD):
        if self.use_prefetcher:
            return self._apply_decode_qk_norm_sharded(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)
        return self._apply_decode_qk_norm_flat(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)

    def _apply_decode_qk_norm_flat(self, q_heads, k_heads):
        """DRAM interleaved TILE RMSNorm; use reshape (not view) for TILE [1,H,B,D] -> [1,1,H*B,D]."""
        rm_q = q_heads.memory_config()
        rm_k = k_heads.memory_config()
        q_shape = list(q_heads.shape)
        k_shape = list(k_heads.shape)
        q_rows = q_shape[1] * q_shape[2]
        k_rows = k_shape[1] * k_shape[2]

        q_heads = ttnn.to_memory_config(q_heads, ttnn.DRAM_MEMORY_CONFIG)
        k_heads = ttnn.to_memory_config(k_heads, ttnn.DRAM_MEMORY_CONFIG)
        if q_heads.layout != ttnn.TILE_LAYOUT:
            q_heads = ttnn.to_layout(q_heads, ttnn.TILE_LAYOUT)
        if k_heads.layout != ttnn.TILE_LAYOUT:
            k_heads = ttnn.to_layout(k_heads, ttnn.TILE_LAYOUT)
        q_heads = ttnn.reshape(q_heads, (1, 1, q_rows, self.head_dim))
        k_heads = ttnn.reshape(k_heads, (1, 1, k_rows, self.head_dim))
        q_heads = self.q_norm(q_heads, mode="decode", in_sharded=False, out_sharded=False)
        k_heads = self.k_norm(k_heads, mode="decode", in_sharded=False, out_sharded=False)
        q_heads = ttnn.reshape(q_heads, tuple(q_shape))
        k_heads = ttnn.reshape(k_heads, tuple(k_shape))
        q_heads = ttnn.to_memory_config(q_heads, rm_q)
        k_heads = ttnn.to_memory_config(k_heads, rm_k)
        return q_heads, k_heads

    def _apply_decode_qk_norm_sharded(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD):
        """Prefetcher decode: sharded RMSNorm on [1,1,64,128]."""
        rm_mem_cfg_q = q_heads_pre_rot_1BQD.memory_config()
        rm_mem_cfg_k = k_heads_pre_rot_1BKD.memory_config()
        k_shape = list(k_heads_pre_rot_1BKD.shape)
        q_shape = list(q_heads_pre_rot_1BQD.shape)

        q_heads_pre_rot_1BQD = ttnn.to_memory_config(
            q_heads_pre_rot_1BQD, memory_config=self.reshape_intermediate_q_mem_cfg
        )
        k_heads_pre_rot_1BKD = ttnn.to_memory_config(
            k_heads_pre_rot_1BKD, memory_config=self.reshape_intermediate_k_mem_cfg
        )

        norm_rows = 64
        q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 1, norm_rows, self.head_dim])
        k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 1, norm_rows, self.head_dim])

        q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.TILE_LAYOUT)
        k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)

        q_heads_intermediate_after_reshape_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
        k_heads_intermediate_after_reshape_mem_cfg = k_heads_pre_rot_1BKD.memory_config()

        q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=self.reshape_output_q_mem_cfg)
        k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=self.reshape_output_k_mem_cfg)

        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode="decode", in_sharded=True, out_sharded=True)
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode="decode", in_sharded=True, out_sharded=True)

        q_heads_pre_rot_1BQD = ttnn.to_memory_config(
            q_heads_pre_rot_1BQD, memory_config=q_heads_intermediate_after_reshape_mem_cfg
        )
        k_heads_pre_rot_1BKD = ttnn.to_memory_config(
            k_heads_pre_rot_1BKD, memory_config=k_heads_intermediate_after_reshape_mem_cfg
        )

        q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.ROW_MAJOR_LAYOUT)
        k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)

        q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, q_shape)
        k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 8, 8, 128])

        q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=rm_mem_cfg_q)
        k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=rm_mem_cfg_k)
        return q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD

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
        x: ttnn.Tensor,
        current_pos,
        rot_mats=None,
        page_table=None,
        kv_cache=None,
    ) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        if self.use_prefetcher:
            xqkv_fused_sharded = ttnn.matmul(  # [1, 1, 32, 1280]
                x,
                self.wqkv,
                program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
                memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
                global_cb=self.prefetcher_setup.global_circular_buffer,
                dtype=ttnn.bfloat16,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id,
            )
        else:
            # No-prefetch (Blackhole) per-device QKV: column-sharded activation @ interleaved DRAM
            # wqkv [k_local, qkv_n_local], auto program config + interleaved output (a DRAM-sharded
            # program config hangs here). Each mesh column holds a K-fractured partial; they are
            # summed across columns on device (line_all_reduce(cluster_axis=1)) just before head
            # creation, once the tensor is in the width-sharded L1 create-head layout.
            x_in = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
            xqkv_fused_sharded = ttnn.linear(
                x_in,
                self.wqkv_per_device,
                program_config=None,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=None,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                # bf8 so the column all-reduce input matches persistent_buffers[1] (bf8) CB sizing.
                dtype=ttnn.bfloat8_b,
            )
            if x_in is not x:
                ttnn.deallocate(x_in)
        ttnn.deallocate(x)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        if not self.use_prefetcher:
            xqkv_fused_interleaved = ttnn.to_memory_config(xqkv_fused_sharded, memory_config=ttnn.L1_MEMORY_CONFIG)
            fqkv_shape = xqkv_fused_interleaved.shape
            if fqkv_shape[3] > self._qkv_n_local:
                xqkv_fused_interleaved = ttnn.slice(
                    xqkv_fused_interleaved,
                    [0, 0, 0, 0],
                    [fqkv_shape[0], fqkv_shape[1], fqkv_shape[2], self._qkv_n_local],
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                fqkv_shape = xqkv_fused_interleaved.shape
            xqkv_fused_interleaved = ttnn.reshape(
                xqkv_fused_interleaved,
                (1, 1, self.batch_size_per_device_group, fqkv_shape[3]),
                (1, 1, 32, fqkv_shape[3]),
            )
            if xqkv_fused_interleaved.layout != ttnn.TILE_LAYOUT:
                xqkv_fused_interleaved = ttnn.to_layout(
                    xqkv_fused_interleaved,
                    ttnn.TILE_LAYOUT,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
            # nlp_create_qkv_heads_decode only honors batch_offset/slice_size in its WIDTH_SHARDED
            # program factory; width-shard the fused tensor so batch_offset selects users
            # [col*8:col*8+8] per column.
            create_head_input_memcfg = self._create_head_input_memcfg_nlp
            if create_head_input_memcfg is not None:
                xqkv_fused_create_head_in = ttnn.to_memory_config(
                    xqkv_fused_interleaved, memory_config=create_head_input_memcfg
                )
                ttnn.deallocate(xqkv_fused_interleaved)
            else:
                xqkv_fused_create_head_in = xqkv_fused_interleaved
            # Sum the K-fractured QKV partials across mesh columns on device (sum + replicate),
            # the device equivalent of the former host column-sum. This keeps the whole decode path
            # on device and trace-capturable. The width-sharded [32, head_dim]x(qkv_n_local/head_dim)
            # L1 layout matches persistent_buffers[1] for the cluster-axis-1 all-reduce.
            # bf8 input matches the bf8 interim buffer CB sizing; output bf16 because
            # nlp_create_qkv_heads_decode only accepts bf16/fp32.
            xqkv_fused_create_head_in = self.tt_ccl.line_all_reduce(
                xqkv_fused_create_head_in,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=xqkv_fused_create_head_in.memory_config(),
                dtype=ttnn.bfloat16,
                use_optimal_ccl_for_llama=True,
            )
            (
                q_heads_pre_rot_1BQD,
                k_heads_pre_rot_1BKD,
                v_heads_1BKD,
            ) = ttnn.experimental.nlp_create_qkv_heads_decode(
                xqkv_fused_create_head_in,
                num_heads=self.n_local_heads,
                num_kv_heads=self.n_local_kv_heads,
                memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
                batch_offset=self.batch_offset_tt_tensor,
                slice_size=self.slice_size,
            )
            ttnn.deallocate(xqkv_fused_create_head_in)
        else:
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

        if self.qk_norm:
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD = self._apply_decode_qk_norm(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD
            )

        ttnn.deallocate(xqkv_fused_sharded)

        # Q, K Rotary Embeddings
        if self.use_prefetcher:
            q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
            )  # [1, 8, 8, 128], [1, 8, 8, 128]
        else:
            # No-prefetcher decode still requires HEIGHT_SHARDED cos/sin for RoPE.
            # get_rot_mats returns [1, 1, local_batch, head_dim], which is the
            # format expected by the non-fused decode rotary op. get_rm_rot_mats
            # expands to [1, expanded_batch, heads, head_dim] for the fused path;
            # keep a fallback for callers that still provide that layout.
            if rot_mats[0].shape[1] == 1:
                q_rot_cos = ttnn.to_layout(rot_mats[0], ttnn.TILE_LAYOUT)
                q_rot_sin = ttnn.to_layout(rot_mats[1], ttnn.TILE_LAYOUT)
                k_rot_cos = q_rot_cos
                k_rot_sin = q_rot_sin
            else:
                q_rot_end = [
                    1,
                    q_heads_pre_rot_1BQD.shape[1],
                    q_heads_pre_rot_1BQD.shape[2],
                    self.head_dim,
                ]
                k_rot_end = [
                    1,
                    k_heads_pre_rot_1BKD.shape[1],
                    k_heads_pre_rot_1BKD.shape[2],
                    self.head_dim,
                ]
                q_rot_cos = ttnn.slice(
                    rot_mats[0],
                    [0, 0, 0, 0],
                    q_rot_end,
                    memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
                )
                q_rot_sin = ttnn.slice(
                    rot_mats[1],
                    [0, 0, 0, 0],
                    q_rot_end,
                    memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
                )
                k_rot_cos = ttnn.slice(
                    rot_mats[0],
                    [0, 0, 0, 0],
                    k_rot_end,
                    memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
                )
                k_rot_sin = ttnn.slice(
                    rot_mats[1],
                    [0, 0, 0, 0],
                    k_rot_end,
                    memory_config=self.model_config["CREATE_HEAD_OUTPUT_MEMCFG"],
                )
                q_rot_cos = ttnn.to_layout(q_rot_cos, ttnn.TILE_LAYOUT)
                q_rot_sin = ttnn.to_layout(q_rot_sin, ttnn.TILE_LAYOUT)
                k_rot_cos = ttnn.to_layout(k_rot_cos, ttnn.TILE_LAYOUT)
                k_rot_sin = ttnn.to_layout(k_rot_sin, ttnn.TILE_LAYOUT)
            q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
                q_heads_pre_rot_1BQD,
                q_rot_cos,
                q_rot_sin,
                self.transformation_mats["decode"],
                is_decode_mode=True,
            )
            k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
                k_heads_pre_rot_1BKD,
                k_rot_cos,
                k_rot_sin,
                self.transformation_mats["decode"],
                is_decode_mode=True,
            )
        ttnn.deallocate(q_heads_pre_rot_1BQD)
        ttnn.deallocate(k_heads_pre_rot_1BKD)

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
        if not self.use_prefetcher:
            v_update_mem_cfg = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        {
                            ttnn.CoreRange(
                                ttnn.CoreCoord(5, 0),
                                ttnn.CoreCoord(6, 3),
                            )
                        }
                    ),
                    [32, self.head_dim],
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            v_heads_1BKD = ttnn.to_memory_config(v_heads_1BKD, v_update_mem_cfg)
        ttnn.experimental.paged_fused_update_cache(
            keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
        )
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

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
                compute_kernel_config=self.sdpa_decode_compute_kernel_config,
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
                compute_kernel_config=self.sdpa_decode_compute_kernel_config,
                memory_config=sdpa_out_mem_cfg,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )
        ttnn.deallocate(q_heads_1BQD)

        if not self.use_prefetcher:
            # Device SDPA output is batch-first [1, B_local, H_local, D] (8 users/col). Order matters:
            # nlp_concat_heads_decode pads batch to 32, so gather users across cols FIRST (8 -> 32),
            # then concat heads. Mimic tt_transformers.tt_all_gather: all_gather_async with
            # persistent_output_tensor=None AND a barrier semaphore (passing the persistent SDPA
            # buffer with barrier=None throws map::at here).
            _ccl = self.tt_ccl
            _ca = 1
            _gidx = _ccl.gather_idx[_ca]
            _ag_sems = [
                _ccl.gather_semaphore_handles[_ca][_gidx],
                _ccl.gather_semaphore_handles[_ca][(_gidx + 1) % _ccl.num_cbs],
            ]
            gathered_users = ttnn.experimental.all_gather_async(  # [1, 32, H_local, D]
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=_ca,
                mesh_device=self.mesh_device,
                topology=self.model_config["CCL_TOPOLOGY"],
                multi_device_global_semaphore=_ag_sems,
                persistent_output_tensor=None,
                barrier_semaphore=_ccl.get_and_cycle_barrier_semaphore_handle(_ca),
                num_links=1,
                memory_config=self.model_config["GATHER_USERS_MEMCFG"](self.cluster_shape[1]),
                subdevice_id=_ccl.worker_sub_device_id,
            )
            _ccl.gather_idx[_ca] = (_gidx + 1) % _ccl.num_cbs
            try:
                concat_sub_core_grids = gathered_users.memory_config().shard_spec.grid
            except Exception:
                concat_sub_core_grids = None
            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(  # [1, 1, 32, 1024]
                gathered_users,
                num_heads=self.n_local_heads,
                sub_core_grids=concat_sub_core_grids,
            )
            ttnn.deallocate(gathered_users)
        else:
            attn_output_cat = self.tt_ccl.all_gather_concat(  # [1, 1, 32, 1024]
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
                num_heads=self.n_local_heads,
            )
        ttnn.deallocate(attn_output_1G4D_sharded)

        # WO matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        use_replicated_full_wo = (
            self.is_qwen and not self.use_prefetcher and attn_output_cat.shape[-1] == self.n_heads * self.head_dim
        )
        if not self.use_prefetcher:
            # BH no-prefetch cannot use the ring WO program here: its static CB
            # allocation exceeds L1 once concat produces the correct decode shape.
            attn_output_cat_for_matmul = ttnn.to_memory_config(attn_output_cat, ttnn.DRAM_MEMORY_CONFIG)
            dense_out_ttnn = ttnn.linear(
                attn_output_cat_for_matmul,
                self.wo_interleaved_col_sharded if use_replicated_full_wo else self.wo_interleaved,
                program_config=None,
                memory_config=None,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                global_cb=None,
                dtype=ttnn.bfloat8_b,
                sub_device_id=None,
            )
            attn_output_cat_for_matmul.deallocate(True)
        else:
            dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
                attn_output_cat,
                self.wo,
                program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
                memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
                global_cb=self.prefetcher_setup.global_circular_buffer,
                dtype=ttnn.bfloat8_b,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id,
            )
        # [1, 1, 32, 2304]
        if use_replicated_full_wo:
            dense_out_reduced = dense_out_ttnn
        else:
            if not self.use_prefetcher and dense_out_ttnn.memory_config().buffer_type == ttnn.BufferType.DRAM:
                # The device all_reduce (ttnn.experimental.all_reduce_async) rejects DRAM input on
                # Blackhole; bring the WO output into the all_reduce's L1 layout first (mirrors the
                # MLP FF2 path). The no-prefetch WO matmul emits DRAM.
                dense_out_ttnn = ttnn.to_memory_config(dense_out_ttnn, self.model_config["DECODE_RESIDUAL_MEMCFG"])
            dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
                dense_out_ttnn,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(dense_out_ttnn)

        return dense_out_reduced

    def forward_prefill(
        self,
        x_11SH,
        rot_mats,
        user_id: int = 0,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        chunk_start_idx_tensor=None,
        kv_cache=None,
        batch_size=1,
    ):
        if batch_size > 1:
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        # Wormhole keeps main's fixed prefill link count (3); Blackhole uses the mesh link budget.
        prefill_num_links = self.model_config["GALAXY_NUM_LINKS"] if self.is_blackhole else 3
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > 2048:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // 2048, 2048, -1])

        if self.use_prefetcher:
            xqkv = ttnn.linear(
                x_11SH,
                self.wqkv_interleaved,
                dtype=self.ccl_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
            )
        else:
            # No-prefetch (Blackhole) per-device QKV: column-fractured activation (K=k_local=1280)
            # @ interleaved DRAM wqkv [k_local, qkv_n_local]. The padded ring weight expects K=1536;
            # the unpadded per-device weight matches the activation and keeps N=qkv_n_local so the
            # downstream line_all_reduce(cluster_axis=1) + nlp_create_qkv_heads stay correct.
            xqkv = ttnn.linear(
                x_11SH,
                self.wqkv_per_device,
                dtype=self.ccl_dtype,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                program_config=None,
            )
        # Minimal matmul is giving bad outputs for seqlen > 128
        # xqkv = ttnn.experimental.minimal_matmul(
        #     input_tensor=x_11SH,
        #     weight_tensor=self.wqkv_interleaved,
        #     config=self.model_config["XQKV_PREFILL_MINIMAL_PROGCFG"](seq_len),
        #     compute_kernel_config=self.compute_kernel_config_hifi2,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        # )

        ttnn.deallocate(x_11SH)

        xqkv_fused = self.tt_ccl.line_all_reduce(
            xqkv,
            cluster_axis=1,
            num_links=prefill_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="QKV",
            batch_size=batch_size,
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

        if self.qk_norm:
            q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")

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

        if self.qk_norm:
            k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        # k_heads_1KSD = k_heads_1KSD_pre_rot
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

        if not page_table:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)

        if page_table:
            # Use chunk_page_table only for prefix-cached prefill (chunk_start_idx > 0).
            # For non-prefix prefill, ignore chunk_page_table (trace may pass a dummy) and use page_table.
            use_chunk_for_fill = chunk_start_idx is not None and chunk_start_idx > 0
            fill_page_table = chunk_page_table if (use_chunk_for_fill and chunk_page_table is not None) else page_table

            # Each shard gets one row, which is locally at index 0
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill, fill_page_table, batch_idx=0)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill, fill_page_table, batch_idx=0)

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
        ring_distributed_sdpa = seq_len > 1024 and batch_size == 1 and (chunk_start_idx is None or chunk_start_idx == 0)
        use_chunked_sdpa = chunk_start_idx is not None and chunk_start_idx > 0

        if ring_distributed_sdpa:
            attn_output_1QSD = ttnn.transformer.ring_distributed_scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                ring_size=4,  # Number of devices in the ring topology (4 devices per row in 8x4 mesh)
                scale=self.scale,
                compute_kernel_config=self.compute_kernel_config_hifi4,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx=0),
                page_table=None,
                chunk_start_idx=None,
            )
        else:
            # When using prefix caching (chunk_start_idx provided), use chunked SDPA with KV cache tensors.
            # Flexible path: chunk_start_idx_tensor so one trace works for any chunk_start at replay.
            if use_chunked_sdpa:
                assert page_table is not None, "page_table must be provided for prefix caching"
                assert (
                    chunk_start_idx_tensor is not None
                ), "prefix caching requires chunk_start_idx_tensor for flexible SDPA"
                page_size = self.paged_attention_config.block_size if self.paged_attention_config else 32
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx_tensor=chunk_start_idx_tensor,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    program_config=self.model_config["SDPA_PROGCFG_FLEXIBLE_CHUNK"](seq_len, page_size),
                )

                # Replicate active column's data to all columns for correct RMSNORM behavior.
                # Chunked SDPA writes only to the column for this user_id; we zero others and all-reduce so every column has the same output.
                # Pre-computed column_mask: [1, 1, 1, 32] per device, 1.0 on owning column, 0.0 on others.
                # Stored on tt_ccl by the generator before forward; slice to scalar for broadcast.
                column_mask = self.tt_ccl._prefill_column_mask
                mask = ttnn.slice(column_mask, [0, 0, 0, 0], [1, 1, 1, 1])
                # attn_output_84SD: zero out inactive columns (multiply by 0); active column unchanged (multiply by 1).
                attn_output_84SD = ttnn.multiply(attn_output_84SD, mask)
                # line_all_reduce along columns: sum = active column's data (others 0); replicate to all columns so shape/values match for downstream.
                attn_output_84SD = self.tt_ccl.line_all_reduce(
                    attn_output_84SD,
                    cluster_axis=1,
                    num_links=prefill_num_links,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    buffer_key="ATTN_REPLICATE",
                )

                # Reshape from [1, 1, seq_len, head_dim] to [1, n_local_heads, seq_len, head_dim]
                attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])
            else:
                attn_output_1QSD = ttnn.transformer.scaled_dot_product_attention(
                    q_heads_1QSD_8b,
                    k_heads_1KSD_8b,
                    v_heads_1VSD_8b,
                    is_causal=True,
                    scale=self.scale,
                    compute_kernel_config=self.compute_kernel_config_hifi4,
                    program_config=self.model_config["SDPA_PROGCFG"](
                        seq_len // batch_size if seq_len // batch_size == 128 else seq_len
                    ),
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
        # In the prefix-cached path, attn_output_1QSD is a reshape/view of the
        # persistent ATTN_REPLICATE all-gather buffer. Deallocating it here also
        # deallocates the cached buffer, so later layers/warmup iterations see a
        # Tensor object that exists but is no longer allocated.
        if not use_chunked_sdpa:
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

        ## For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        if seq_len < 4096 or batch_size > 1:
            output_11SH = ttnn.linear(
                attn_output_11SH,
                self.wo_interleaved,
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
            )
        else:
            output_11SH = ttnn.experimental.minimal_matmul(
                input_tensor=attn_output_11SH,
                weight_tensor=self.wo_interleaved,
                config=self.model_config["WO_PREFILL_MINIMAL_PROGCFG"](seq_len),
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        output_11SH_reduced = self.tt_ccl.line_all_reduce(
            output_11SH,
            cluster_axis=0,
            num_links=prefill_num_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO_AG" if seq_len <= 4096 else "WO",
        )
        output_11SH.deallocate()

        return output_11SH_reduced

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
        chunk_start_idx_tensor=None,
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
                chunk_start_idx_tensor=chunk_start_idx_tensor,
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
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor
