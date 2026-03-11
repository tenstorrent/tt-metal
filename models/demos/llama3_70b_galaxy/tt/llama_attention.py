# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

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

        # OLMo has 5 local Q heads (not 8), requiring non-fused RoPE in decode mode
        self.is_olmo = getattr(configuration, "is_olmo", False)
        self.cluster_shape = configuration.cluster_shape

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
        self.qk_norm = configuration.qk_norm

        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats

        self.model_config = configuration.get_model_config()
        self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip

        # Sliding window attention support (OLMo: 3 sliding + 1 full pattern)
        # For non-OLMo models, sliding_window_size will be None (full attention)
        self.layer_num = layer_num
        if hasattr(configuration, "get_sliding_window_size"):
            self.sliding_window_size = configuration.get_sliding_window_size(layer_num)
        else:
            self.sliding_window_size = None

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

            # OLMo decode fix: Pad Q from 5 to 8 heads per device
            # This is required because fused RoPE needs num_heads * head_dim = 1024
            # OLMo: 5 local Q heads × 128 = 640 (doesn't meet constraint)
            # Padded: 8 local Q heads × 128 = 1024 (meets constraint)
            if self.is_olmo and self.n_local_heads < 8:
                # wq_selected shape: [n_local_heads * head_dim, dim] = [640, 5120]
                # Pad to [8 * head_dim, dim] = [1024, 5120]
                pad_heads = 8 - self.n_local_heads  # 8 - 5 = 3 heads to pad
                pad_size = pad_heads * self.head_dim  # 3 * 128 = 384
                old_shape = wq_selected.shape
                wq_selected = torch.nn.functional.pad(wq_selected, (0, 0, 0, pad_size), value=0.0)
                if i == 0:
                    print(
                        f"OLMo Q padding: {old_shape} -> {wq_selected.shape} (added {pad_size} for {pad_heads} extra heads)"
                    )

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        if self.is_olmo:
            print(f"OLMo qkv_cat shape: {qkv_cat.shape} (expected [1, 1, 5120, 10240] for padded Q)")

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

        # OLMo: Save unpadded WO for prefill before padding for decode
        pt_wo_unpadded = pt_wo if self.is_olmo else None

        # OLMo decode fix: Pad WO input dimension from 5120 to 8192 to match padded Q heads
        # Original: 5120 (40 heads * 128), Per device: 640
        # Padded: 8192 (1024 per device * 8 devices) to match 8 padded Q heads
        if self.is_olmo and self.n_local_heads < 8:
            # pt_wo shape: [1, 1, n_heads * head_dim, dim] = [1, 1, 5120, 5120]
            # Pad to: [1, 1, 8192, 5120] for padded Q heads (8 per device)
            target_wo_k = 1024 * self.num_devices_per_group  # 1024 * 8 = 8192
            pad_size = target_wo_k - self.n_heads * self.head_dim  # 8192 - 5120 = 3072
            pt_wo = torch.nn.functional.pad(pt_wo, (0, 0, 0, pad_size), value=0.0)
            print(f"OLMo WO padding: {self.n_heads * self.head_dim} -> {pt_wo.shape[-2]} (added {pad_size})")

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            configuration.dim // configuration.num_devices, configuration.dim
        )

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_WO_RING_MEMCFG"]
            if (self.use_fused_all_gather_matmul or self.TG)
            else wo_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d_prefetcher")
            if (self.use_fused_all_gather_matmul or self.TG)
            else cache_name("wo"),
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
        # OLMo: Create unpadded WO for prefill (decode uses padded WO for fused RoPE compatibility)
        if pt_wo_unpadded is not None:
            self.wo_interleaved_unpadded = ttnn.as_tensor(
                pt_wo_unpadded,
                dtype=ttnn.bfloat8_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(2, 3) if (self.use_fused_all_gather_matmul or self.TG) else (3, 2),
                    mesh_shape=configuration.cluster_shape,
                ),
                cache_file_name=cache_name("wo_width_sharded_2d_dram_unpadded"),
            )
            print(f"OLMo: Created unpadded WO for prefill with K={pt_wo_unpadded.shape[-2]}")
        else:
            self.wo_interleaved_unpadded = None
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        self.scale = self.head_dim**-0.5
        # Apply YaRN attention factor (mscale) if present (OLMo uses YaRN RoPE)
        if hasattr(configuration, "yarn_attention_factor"):
            self.scale *= configuration.yarn_attention_factor
        if tt_ccl.mode == "decode":
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
        if tt_ccl.mode == "decode" and prefetcher_setup is not None:
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

    def _debug_check_attn(self, name, tensor):
        """Check tensor for Inf/NaN in attention module."""
        import os

        if os.environ.get("DEBUG_DECODE", "0") != "1":
            return
        try:
            from loguru import logger

            torch_tensor = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
            )
            has_inf = torch.isinf(torch_tensor).any().item()
            has_nan = torch.isnan(torch_tensor).any().item()
            max_val = torch_tensor.float().abs().max().item()
            status = "OK" if not (has_inf or has_nan) else "BAD"
            logger.info(
                f"    ATTN [{status}] {name}: shape={list(torch_tensor.shape)}, max={max_val:.4e}, Inf={has_inf}, NaN={has_nan}"
            )
        except Exception as e:
            from loguru import logger

            logger.error(f"    ATTN [ERROR] {name}: {e}")

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
        # Ensure input is tilized for matmul (fused_rms_minimal may output ROW_MAJOR)
        print(f"DEBUG forward_decode: x layout before tilize = {x.get_layout()}, shape = {x.shape}")
        if x.get_layout() != ttnn.TILE_LAYOUT:
            print(f"DEBUG: Converting x to TILE_LAYOUT")
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)
        print(f"DEBUG forward_decode: x layout after tilize = {x.get_layout()}")

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        print(f"DEBUG: wqkv layout = {self.wqkv.get_layout()}, shape = {self.wqkv.shape}")
        xqkv_fused_sharded = ttnn.matmul(  # [1, 1, 32, 1280]
            x,  # [1, 1, 32, 1280]
            self.wqkv,
            program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
            memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
            compute_kernel_config=self.compute_kernel_config_hifi2,
            global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
            dtype=ttnn.bfloat16,
            sub_device_id=self.prefetcher_setup.worker_sub_device_id if self.prefetcher_setup is not None else None,
        )
        ttnn.deallocate(x)
        self._debug_check_attn("xqkv_fused_sharded", xqkv_fused_sharded)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        # NOTE: OLMo decode is BLOCKED due to llama_rs_create_heads kernel bug
        # The kernel ignores num_kv_heads=1 and outputs K/V with 8 heads instead of 1,
        # causing NaN values. See BRINGUP_LOG.md for details.
        # For now, use the same path as Llama/Qwen (which works for 8:1 GQA ratio)
        if False:  # Disabled - OLMo decode workaround (incomplete)
            pass
        else:
            # Standard path for Llama/Qwen
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
        self._debug_check_attn("q_heads_pre_rot", q_heads_pre_rot_1BQD)
        self._debug_check_attn("k_heads_pre_rot", k_heads_pre_rot_1BKD)
        self._debug_check_attn("v_heads", v_heads_1BKD)

        if self.qk_norm:
            rm_mem_cfg_q = q_heads_pre_rot_1BQD.memory_config()
            rm_mem_cfg_k = k_heads_pre_rot_1BKD.memory_config()

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, memory_config=self.reshape_intermediate_q_mem_cfg
            )
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, memory_config=self.reshape_intermediate_k_mem_cfg
            )

            # Reshape and prepare tensors for QK norm
            q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 1, 64, 128])  # [1, 8, 8, 128] => [1, 1, 64, 128]
            k_heads_pre_rot_1BKD = ttnn.view(
                k_heads_pre_rot_1BKD, [1, 1, 64, 128]
            )  # [1, 8, 1 (8), 128]] => [1, 1, 64, 128]

            q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.TILE_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)

            q_heads_intermediate_after_reshape_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
            k_heads_intermediate_after_reshape_mem_cfg = k_heads_pre_rot_1BKD.memory_config()

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, memory_config=self.reshape_output_q_mem_cfg
            )
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, memory_config=self.reshape_output_k_mem_cfg
            )

            # Apply QK norm
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

            q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 8, 8, 128])
            k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 8, 8, 128])  # ==> [1, 8, 1 (8), 128]

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=rm_mem_cfg_q)
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=rm_mem_cfg_k)

        # print("done create qkv heads")
        ttnn.deallocate(xqkv_fused_sharded)

        # Q, K Rotary Embeddings
        # Note: Fused RoPE requires num_heads * head_dim = 1024 for row-major tensors
        # Llama/Qwen have 8 local Q heads (8*128=1024) ✓
        # OLMo has 5 local Q heads padded to 8 (8*128=1024) ✓ for Q, but K has 1 head (1*128=128) ✗
        # For OLMo, use non-fused RoPE to handle different Q/K head counts
        print(f"DEBUG RoPE: Q shape={q_heads_pre_rot_1BQD.shape}, K shape={k_heads_pre_rot_1BKD.shape}")
        print(f"DEBUG RoPE: Q layout={q_heads_pre_rot_1BQD.get_layout()}, K layout={k_heads_pre_rot_1BKD.get_layout()}")
        if self.is_olmo:
            # OLMo: Fused RoPE requires K[-2] * K[-1] == 1024
            # K has 1 KV head: [1, batch, 1, 128] -> 1*128 = 128 ≠ 1024
            # Expand K heads from 1 to 8, apply fused RoPE, then slice back to 1 head
            print("DEBUG: OLMo decode - expanding K from 1 to 8 heads for fused RoPE")
            k_shape = k_heads_pre_rot_1BKD.shape
            k_mem_config = k_heads_pre_rot_1BKD.memory_config()

            # Get K's original shard grid (non-overlapping with Q)
            k_shard_grid = k_mem_config.shard_spec.grid
            print(f"DEBUG: K original shard grid: {k_shard_grid}")

            # Move K to DRAM (interleaved) to allow repeat without sharding constraints
            k_interleaved = ttnn.to_memory_config(k_heads_pre_rot_1BKD, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k_heads_pre_rot_1BKD)

            # Tile K along the num_heads dimension: repeat 8 times [1, 8, 1, 128] -> [1, 8, 8, 128]
            k_expanded = ttnn.repeat(k_interleaved, ttnn.Shape([1, 1, 8, 1]))
            ttnn.deallocate(k_interleaved)
            print(f"DEBUG: K expanded shape={k_expanded.shape}")

            # Create a HEIGHT_SHARDED memory config for K using its ORIGINAL non-overlapping grid
            # Shard shape changes from [1, 128] to [8, 128] to accommodate 8 heads
            k_expanded_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    k_shard_grid,  # Use K's original grid (non-overlapping with Q)
                    [8, 128],  # 8 heads * 128 head_dim per batch item
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            k_expanded_sharded = ttnn.to_memory_config(k_expanded, k_expanded_mem_config)
            ttnn.deallocate(k_expanded)

            print(f"DEBUG: Q shard grid: {q_heads_pre_rot_1BQD.memory_config().shard_spec.grid}")
            print(f"DEBUG: K expanded shard grid: {k_expanded_sharded.memory_config().shard_spec.grid}")

            # Apply fused RoPE (now K[-2]*K[-1] = 8*128 = 1024 ✓)
            q_heads_1BQD, k_heads_expanded = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_expanded_sharded, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
            )
            ttnn.deallocate(q_heads_pre_rot_1BQD)
            ttnn.deallocate(k_expanded_sharded)

            # Slice K back to 1 head (all 8 copies are identical after RoPE)
            # Move to DRAM first to avoid sharding issues with slice
            k_heads_expanded_dram = ttnn.to_memory_config(k_heads_expanded, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k_heads_expanded)
            k_heads_1BKD = ttnn.slice(k_heads_expanded_dram, [0, 0, 0, 0], [1, k_shape[1], 1, k_shape[3]])
            ttnn.deallocate(k_heads_expanded_dram)
            # Move back to original memory config
            k_heads_1BKD = ttnn.to_memory_config(k_heads_1BKD, k_mem_config)
            print(f"DEBUG: K sliced back shape={k_heads_1BKD.shape}")
        else:
            # Llama/Qwen: Use fused RoPE (both Q and K have 8 heads)
            q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
            )  # [1, 8, 8, 128], [1, 8, 8, 128]
            ttnn.deallocate(q_heads_pre_rot_1BQD)
            ttnn.deallocate(k_heads_pre_rot_1BKD)
        self._debug_check_attn("q_heads_post_rope", q_heads_1BQD)
        self._debug_check_attn("k_heads_post_rope", k_heads_1BKD)
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
        if self.is_olmo:
            # OLMo: Use separate cache updates (non-fused version doesn't have 8-heads requirement)
            # paged_fused_update_cache has hardcoded Llama70b shapes (requires 8 KV heads)
            # Non-fused paged_update_cache requires TILE layout AND sharded
            # Move to DRAM, tilize, then reshard with tile-aligned shape
            k_dram = ttnn.to_memory_config(k_heads_1BKD, ttnn.DRAM_MEMORY_CONFIG)
            v_dram = ttnn.to_memory_config(v_heads_1BKD, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k_heads_1BKD)
            ttnn.deallocate(v_heads_1BKD)
            k_tiled = ttnn.to_layout(k_dram, ttnn.TILE_LAYOUT)
            v_tiled = ttnn.to_layout(v_dram, ttnn.TILE_LAYOUT)
            ttnn.deallocate(k_dram)
            ttnn.deallocate(v_dram)
            # Reshard with tile-aligned shape for paged_update_cache
            # K/V shape after TILE padding: [1, 8, 32, 128] (padded num_kv_heads=32, head_dim=128)
            # For HEIGHT_SHARDED with 8 cores: shard_height = 8*32/8 = 32, shard_width = 128
            # Note: Column 7 is dispatch core (COL axis), use rows instead to get 8 cores
            kv_shard_mem_config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(
                    ttnn.CoreRangeSet(
                        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 7))]
                    ),  # column 0, rows 0-7
                    [32, 128],  # shard shape: one batch item's worth (padded to tile)
                    ttnn.ShardOrientation.ROW_MAJOR,
                ),
            )
            k_sharded = ttnn.to_memory_config(k_tiled, kv_shard_mem_config)
            v_sharded = ttnn.to_memory_config(v_tiled, kv_shard_mem_config)
            ttnn.deallocate(k_tiled)
            ttnn.deallocate(v_tiled)
            ttnn.experimental.paged_update_cache(keys, k_sharded, update_idxs_tensor=current_pos, page_table=page_table)
            ttnn.experimental.paged_update_cache(
                values, v_sharded, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.deallocate(k_sharded)
            ttnn.deallocate(v_sharded)
        else:
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
                sliding_window_size=self.sliding_window_size,  # OLMo: 4096 for sliding layers, None for full
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
                sliding_window_size=self.sliding_window_size,  # OLMo: 4096 for sliding layers, None for full
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"],
                memory_config=sdpa_out_mem_cfg,
            )

        ttnn.deallocate(q_heads_1BQD)
        self._debug_check_attn("sdpa_output", attn_output_1G4D_sharded)

        print(f"DEBUG SDPA output shape: {attn_output_1G4D_sharded.shape}")
        print(f"DEBUG SDPA output mem_config: {attn_output_1G4D_sharded.memory_config()}")

        # OLMo decode: Use 8 padded heads to match fused RoPE (num_heads * head_dim = 1024)
        # For other models with 8 local heads, n_local_heads == 8 already
        decode_num_heads = 8 if (self.is_olmo and self.n_local_heads < 8) else self.n_local_heads

        print(f"DEBUG all_gather_concat: num_heads={decode_num_heads}")

        # OLMo decode: all_gather_concat segfaults with small batches (batch=32, batch_per_column=8)
        # Use host-side all_gather + slice + reshape workaround
        if self.is_olmo:
            # SDPA output per device: [1, 8, 32, 128] (batch_per_column=8, 32_padded_heads, head_dim=128)
            # Note: SDPA pads heads to tile size (32), but only decode_num_heads (8) are valid
            print(f"DEBUG OLMo decode: SDPA output shape = {attn_output_1G4D_sharded.shape}")
            print(f"DEBUG OLMo decode: SDPA output mem_config = {attn_output_1G4D_sharded.memory_config()}")

            # Step 1: all_gather across column axis (4 devices) to get all 32 users
            # [1, 8, 32, 128] × 4 → [1, 32, 32, 128]
            attn_gathered = self.tt_ccl.line_all_gather_host(
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,  # Output to DRAM for slice/reshape
            )
            ttnn.deallocate(attn_output_1G4D_sharded)
            print(f"DEBUG OLMo decode: after all_gather shape = {attn_gathered.shape}")

            # Step 2: slice to keep only valid heads (8 out of 32 padded)
            # [1, 32, 32, 128] -> [1, 32, 8, 128]
            batch = attn_gathered.shape[1]  # 32
            head_dim = attn_gathered.shape[3]  # 128
            attn_sliced = ttnn.slice(attn_gathered, [0, 0, 0, 0], [1, batch, decode_num_heads, head_dim])
            ttnn.deallocate(attn_gathered)
            print(f"DEBUG OLMo decode: after slice shape = {attn_sliced.shape}")

            # Step 3: reshape [1, 32, 8, 128] -> [1, 1, 32, 1024]
            # Combine local_heads (8) with head_dim (128) to get 1024
            attn_reshaped = ttnn.reshape(attn_sliced, [1, 1, batch, decode_num_heads * head_dim])
            ttnn.deallocate(attn_sliced)
            print(f"DEBUG OLMo decode: after reshape shape = {attn_reshaped.shape}")
            print(f"DEBUG OLMo decode: after reshape layout = {attn_reshaped.get_layout()}")

            # Step 4: tilize if needed (from_torch in all_gather_host uses TILE, but reshape might change it)
            if attn_reshaped.get_layout() != ttnn.TILE_LAYOUT:
                print("DEBUG OLMo decode: tilizing...")
                # First move to DRAM interleaved for tilize
                attn_dram = ttnn.to_memory_config(attn_reshaped, ttnn.DRAM_MEMORY_CONFIG)
                ttnn.deallocate(attn_reshaped)
                attn_tiled = ttnn.to_layout(attn_dram, ttnn.TILE_LAYOUT)
                ttnn.deallocate(attn_dram)
            else:
                print("DEBUG OLMo decode: already tiled")
                attn_tiled = attn_reshaped

            # Step 5: Keep in DRAM for WO matmul (avoid sharding grid mismatch)
            # This is slower but avoids complex grid matching issues
            print(f"DEBUG OLMo decode: using DRAM for WO matmul input")
            attn_output_cat = ttnn.to_memory_config(attn_tiled, ttnn.DRAM_MEMORY_CONFIG)
            if attn_tiled is not attn_reshaped:
                ttnn.deallocate(attn_tiled)
        else:
            # Standard path for Llama/Qwen - use all_gather_concat
            attn_output_cat = self.tt_ccl.all_gather_concat(  # [1, 1, 32, 1024]
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
                num_heads=decode_num_heads,
            )
            ttnn.deallocate(attn_output_1G4D_sharded)
        # print("done concat heads")

        # Ensure attn_output_cat is tilized for WO matmul
        if attn_output_cat.get_layout() != ttnn.TILE_LAYOUT:
            # Must convert to interleaved memory before tilizing sharded tensors
            attn_output_cat = ttnn.to_memory_config(attn_output_cat, ttnn.DRAM_MEMORY_CONFIG)
            attn_output_cat = ttnn.to_layout(attn_output_cat, ttnn.TILE_LAYOUT)
            # Convert back to sharded for WO matmul
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat, self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"]
            )

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        if self.is_olmo:
            # OLMo: use simpler matmul with DRAM interleaved tensors
            # Use wo_interleaved to avoid circular buffer issues
            dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
                attn_output_cat,
                self.wo_interleaved,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(attn_output_cat)
            self._debug_check_attn("wo_matmul_out", dense_out_ttnn)
            # OLMo: Use device-side all_reduce with OLMo-specific sharded config
            # Reshard to SHARDED_WO_OUT_RING_MEMCFG_OLMO (10 cores × 128 = 1280)
            dense_out_sharded = ttnn.to_memory_config(
                dense_out_ttnn, self.model_config["SHARDED_WO_OUT_RING_MEMCFG_OLMO"]
            )
            ttnn.deallocate(dense_out_ttnn)
            dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
                dense_out_sharded,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(dense_out_sharded)
        else:
            dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
                attn_output_cat,
                self.wo,
                program_config=self.model_config["WO_DECODE_RING_PROGCFG"],
                memory_config=self.model_config["SHARDED_WO_OUT_RING_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
                global_cb=self.prefetcher_setup.global_circular_buffer if self.model_config["USE_PREFETCHER"] else None,
                dtype=ttnn.bfloat8_b,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id if self.prefetcher_setup is not None else None,
            )
            ttnn.deallocate(attn_output_cat)
            self._debug_check_attn("wo_matmul_out", dense_out_ttnn)
            # [1, 1, 32, 2304]
            dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
                dense_out_ttnn,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                use_optimal_ccl_for_llama=True,
            )
            ttnn.deallocate(dense_out_ttnn)
        self._debug_check_attn("attn_output_final", dense_out_reduced)

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
            num_links=3,
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
                sliding_window_size=self.sliding_window_size,  # OLMo: 4096 or None for full attention
            )
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
                sliding_window_size=self.sliding_window_size,  # OLMo: 4096 or None for full attention
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

        ## For shorter sequence lengths use the original matmul since it performs better than the minimal matmul
        # OLMo: Use unpadded WO for prefill (padded WO is for decode with padded Q heads)
        wo_weight = self.wo_interleaved_unpadded if self.wo_interleaved_unpadded is not None else self.wo_interleaved
        if seq_len < 4096 or batch_size > 1:
            output_11SH = ttnn.linear(
                attn_output_11SH,
                wo_weight,
                compute_kernel_config=self.compute_kernel_config_hifi2_fp16,
                dtype=ttnn.bfloat8_b,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
            )
        else:
            output_11SH = ttnn.experimental.minimal_matmul(
                input_tensor=attn_output_11SH,
                weight_tensor=wo_weight,
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
            num_links=3,
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
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor
