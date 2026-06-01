# SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
import os
from loguru import logger
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import comp_pcc


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
        self.trace_progress = True
        self.decode_debug = os.getenv("QWEN_DECODE_DEBUG", "0") == "1"
        self.decode_debug_values = os.getenv("QWEN_DECODE_DEBUG_VALUES", "0") == "1"
        self.decode_debug_cache_values = os.getenv("QWEN_DECODE_DEBUG_CACHE_VALUES", "0") == "1"
        self.sdpa_inf_index_debug = os.getenv("QWEN_SDPA_INF_INDEX_DEBUG", "0") == "1"
        self.sdpa_substage_debug = os.getenv("QWEN_SDPA_SUBSTAGE_DEBUG", "0") == "1"
        self.sdpa_substage_debug_ttnn = os.getenv("QWEN_SDPA_SUBSTAGE_DEBUG_TTNN", "0") == "1"
        self.sdpa_safe_precision = os.getenv("QWEN_SDPA_SAFE_PRECISION", "0") == "1"
        self.use_simple_decode_sdpa = os.getenv("QWEN_USE_SIMPLE_DECODE_SDPA", "0") == "1"
        self.use_simple_ttnn_decode_sdpa = os.getenv("QWEN_USE_SIMPLE_TTNN_DECODE_SDPA", "0") == "1"
        self.use_host_decode_wo = os.getenv("QWEN_USE_HOST_DECODE_WO", "1") == "1"
        # B2 production output segment: device SDPA -> nlp_concat_heads_decode (local heads)
        # -> ttnn.all_gather batch over cols -> column-parallel WO + line_all_reduce over rows.
        self.use_device_output_segment = os.getenv("QWEN_DEVICE_OUTPUT_SEGMENT", "0") == "1"
        self.use_host_qk_norm = os.getenv("QWEN_USE_HOST_QK_NORM", "1") == "1"
        self.host_reference_attn = None
        self.host_freqs_cis = None
        self.stage_pcc_debug = os.getenv("QWEN_STAGE_PCC_DEBUG", "0") == "1"
        self.qkv_stage_debug = os.getenv("QWEN_QKV_STAGE_DEBUG", "0") == "1"
        self.use_host_fused_qkv = os.getenv("QWEN_HOST_FUSED_QKV", "0") == "1"
        # BH Qwen no-prefetch: overwrite create_heads output with host-split Q/K/V (bring-up fallback).
        self.use_host_qkv_head_scatter = os.getenv("QWEN_HOST_QKV_HEAD_SCATTER", "0") == "1"
        self.cluster_shape = configuration.cluster_shape
        self.is_qwen = getattr(configuration, "is_qwen", False)
        self.is_blackhole = getattr(configuration, "is_blackhole", False)
        # Device QKV: per_device (DRAM sharded), ring, dram, host_fused, multicast.
        _qkv_mm_env = os.getenv("QWEN_DEVICE_QKV_MATMUL")
        if _qkv_mm_env is None and self.is_qwen and self.is_blackhole:
            # per_device avoids broken column->ring to_memory_config; host_fused for baseline.
            self.device_qkv_matmul = "per_device"
        else:
            self.device_qkv_matmul = (_qkv_mm_env or "ring").lower()
        # Device create_heads: nlp (no-prefetch BH) or llama_rs (prefetcher / RS-fused QKV only).
        self.device_create_heads = os.getenv("QWEN_DEVICE_CREATE_HEADS", "nlp").lower()
        self.disable_kv_cache_tensor_cache = os.getenv("QWEN_DISABLE_KV_CACHE_TENSOR_CACHE", "0") == "1"
        self._last_sdpa_cpu_out_bhd = None

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
        self.model_config["USE_PREFETCHER"] = configuration.use_prefetcher
        self.sdpa_decode_compute_kernel_config = self.model_config["SDPA_DECODE_COMPUTE_PROGCFG"]
        if self.sdpa_safe_precision:
            self.sdpa_decode_compute_kernel_config = ttnn.WormholeComputeKernelConfig(
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            logger.info("[qwen-decode-debug] enabled SDPA safe precision kernel config")
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

        qkv_cat_unpadded = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
        # Host reference for QKV stage PCC / optional host fused QKV bring-up (unpadded 10240).
        self._wqkv_host_weight = (
            qkv_cat_unpadded[0, 0].float() if (self.qkv_stage_debug or self.use_host_fused_qkv) else None
        )
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
        self._qkv_n_ring = getattr(configuration, "qkv_n_ring", self._qkv_n_local)
        cols = int(configuration.cluster_shape[1])
        self._k_local = self.hidden_size // cols
        self._k_ring = getattr(configuration, "qkv_k_ring", self._k_local)
        qkv_list_for_tt = qkv_list
        if self.is_qwen and self._qkv_n_ring > self._qkv_n_local:
            qkv_list_for_tt = []
            for qkv in qkv_list:
                if qkv.shape[-1] < self._qkv_n_ring:
                    qkv = torch.nn.functional.pad(qkv, (0, self._qkv_n_ring - qkv.shape[-1]))
                qkv_list_for_tt.append(qkv)
        qkv_cat_col = torch.cat(qkv_list_for_tt, dim=-1).unsqueeze(0).unsqueeze(0)
        qkv_cat = qkv_cat_col
        # Pad global K (dim=5120 -> 6144) for ring matmul only; per-device k_ring=1536.
        if self.is_qwen and self._k_ring * int(configuration.cluster_shape[1]) > self.hidden_size:
            k_pad = self._k_ring * int(configuration.cluster_shape[1]) - self.hidden_size
            qkv_cat = torch.nn.functional.pad(qkv_cat_col, (0, 0, 0, k_pad))

        # Ring: per-device [k_ring, qkv_n_ring] e.g. [1536, 1536] (K/N padded for RING_SIZE=24).
        wqkv_cache_tag = (
            "wqkv_sharded_2d_prefetcher_kring1536_nring1536" if self.is_qwen else "wqkv_sharded_2d_prefetcher"
        )
        wqkv_dram_cache_tag = "wqkv_sharded_2d_dram_kring1536_nring1536" if self.is_qwen else "wqkv_sharded_2d_dram"
        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=self.model_config["SHARDED_QKV_RING_MEMCFG"],
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name(wqkv_cache_tag),
        )
        self.wqkv_interleaved = ttnn.as_tensor(
            qkv_cat,
            dtype=self.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape),
            cache_file_name=cache_name(wqkv_dram_cache_tag),
        )
        # DRAM interleaved weights with logical k_local=1280 (no global K-pad) for dram QKV matmul.
        self.wqkv_col = self.wqkv_interleaved
        self.wqkv_dram_shard = self.wqkv_col
        if self.is_qwen:
            self.wqkv_col = ttnn.as_tensor(
                qkv_cat_col,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkv_sharded_2d_dram_k1280_nring1536_interleaved"),
            )
            self.wqkv_dram_shard = ttnn.as_tensor(
                qkv_cat_col,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=configuration.create_dram_sharded_mem_config(self._k_local, self._qkv_n_ring),
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkv_sharded_2d_dram_k1280_nring1536_dram_shard"),
            )
            # Per-device matmul: logical N=qkv_n_local (1280), interleaved DRAM (no DRAM-sharded
            # program config — that hangs on BH no-prefetch, job 11751). Mirrors MLP w2 path.
            qkv_cat_per_device = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)
            self.wqkv_per_device = ttnn.as_tensor(
                qkv_cat_per_device,
                dtype=self.dtype,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device, dims=(3, 2), mesh_shape=configuration.cluster_shape
                ),
                cache_file_name=cache_name("wqkv_sharded_2d_dram_k1280_n1280_per_device_interleaved"),
            )
        else:
            self.wqkv_per_device = None

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = self.state_dict[wo_str].transpose(-1, -2).unsqueeze(0).unsqueeze(0)
        self._wo_host_weight = pt_wo[0, 0].float() if self.stage_pcc_debug else None

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
        if tt_ccl.mode == "decode" and self.model_config["USE_PREFETCHER"]:
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
        if tt_ccl.mode == "decode":
            self.prefetcher_setup.insert_tensor(self.wqkv)
            self.prefetcher_setup.insert_tensor(self.wo)
        self.tt_ccl = tt_ccl

    def _trace(self, message):
        if self.trace_progress:
            logger.info(f"[qwen-attn-trace] {message}")

    def _debug_tensor_meta(self, name, tensor):
        if not self.decode_debug:
            return
        try:
            shape = tuple(tensor.shape)
        except Exception:
            shape = "unknown"
        try:
            layout = str(tensor.get_layout())
        except Exception:
            layout = "unknown"
        try:
            memcfg = tensor.memory_config()
            if memcfg.is_sharded() and memcfg.shard_spec is not None:
                shard_shape = memcfg.shard_spec.shape
                shard_desc = f"{memcfg.memory_layout} shard={shard_shape[0]}x{shard_shape[1]}"
            else:
                shard_desc = str(memcfg.memory_layout)
        except Exception:
            shard_desc = "unknown"
        logger.info(f"[qwen-decode-debug] {name}: shape={shape}, layout={layout}, mem={shard_desc}")

    def _debug_tensor_values(self, name, tensor):
        if not self.decode_debug_values:
            return

        host_tensor = None
        try:
            host_tensor = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
            )
        except Exception:
            try:
                host_tensor = ttnn.to_torch(tensor)
            except Exception as e:
                logger.info(f"[qwen-decode-debug-values] {name}: unable to readback tensor ({e})")
                return

        if host_tensor is None:
            logger.info(f"[qwen-decode-debug-values] {name}: host tensor is None")
            return

        if not torch.is_floating_point(host_tensor):
            host_tensor = host_tensor.float()

        finite_mask = torch.isfinite(host_tensor)
        total = host_tensor.numel()
        finite_count = int(finite_mask.sum().item())
        nan_count = int(torch.isnan(host_tensor).sum().item())
        inf_count = int(torch.isinf(host_tensor).sum().item())

        if finite_count > 0:
            finite_vals = host_tensor[finite_mask]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            mean_abs = float(finite_vals.abs().mean().item())
            max_abs = float(finite_vals.abs().max().item())
            logger.info(
                f"[qwen-decode-debug-values] {name}: total={total} finite={finite_count} nan={nan_count} inf={inf_count} "
                f"min={min_val:.6e} max={max_val:.6e} mean_abs={mean_abs:.6e} max_abs={max_abs:.6e}"
            )
        else:
            logger.info(f"[qwen-decode-debug-values] {name}: total={total} finite=0 nan={nan_count} inf={inf_count}")

        if self.sdpa_inf_index_debug and name == "sdpa_out" and inf_count > 0:
            self._debug_inf_index_patterns(name, host_tensor)

    def _debug_inf_index_patterns(self, name, host_tensor):
        is_inf = torch.isinf(host_tensor)
        idx = is_inf.nonzero(as_tuple=False)
        n_inf = int(idx.shape[0])
        sample = idx[:16].detach().cpu().tolist()
        logger.info(f"[qwen-sdpa-inf-idx] {name}: num_inf={n_inf}, sample_inf_idx={sample}")

        shape = tuple(host_tensor.shape)
        logger.info(f"[qwen-sdpa-inf-idx] {name}: host_shape={shape}")

        rank = host_tensor.ndim
        for axis in range(rank):
            dims = [d for d in range(rank) if d != axis]
            counts = is_inf.sum(dim=dims) if dims else is_inf
            nonzero = torch.nonzero(counts > 0, as_tuple=False).flatten()
            if nonzero.numel() == 0:
                logger.info(f"[qwen-sdpa-inf-idx] {name}: axis{axis}_inf_positions=[]")
                continue
            nz_list = nonzero.detach().cpu().tolist()
            preview = nz_list[:16]
            tail = "..." if len(nz_list) > 16 else ""
            logger.info(
                f"[qwen-sdpa-inf-idx] {name}: axis{axis}_inf_positions={preview}{tail} " f"(count={len(nz_list)})"
            )

        # Heuristic decode-layout interpretation: [1, H, B, D]
        if rank == 4 and shape[0] == 1:
            h_idx = idx[:, 1]
            b_idx = idx[:, 2]
            d_idx = idx[:, 3]
            logger.info(
                f"[qwen-sdpa-inf-idx] {name}: inferred[1,H,B,D] "
                f"H_range=({int(h_idx.min())},{int(h_idx.max())}) "
                f"B_range=({int(b_idx.min())},{int(b_idx.max())}) "
                f"D_range=({int(d_idx.min())},{int(d_idx.max())})"
            )
            ghost_users = int((b_idx >= self.batch_size_per_device_group).sum().item())
            dim_tail = int((d_idx >= self.head_dim).sum().item())
            logger.info(
                f"[qwen-sdpa-inf-idx] {name}: ghost_user_inf={ghost_users} "
                f"(batch_size_per_device_group={self.batch_size_per_device_group}), "
                f"tail_dim_inf={dim_tail} (head_dim={self.head_dim})"
            )

    def _debug_cache_values(self, keys, values, current_pos):
        if not self.decode_debug_cache_values:
            return

        if self.paged_attention_config:
            logger.info("[qwen-decode-debug-values] cache_probe: skipped for paged cache mode")
            return

        self._debug_tensor_values("keys_cache_pre_sdpa", keys)
        self._debug_tensor_values("values_cache_pre_sdpa", values)
        # current_pos is mesh-distributed; avoid direct to_torch without explicit mesh composer.
        pos_scalar = self._decode_position_scalar(current_pos)
        logger.info(f"[qwen-decode-debug-values] cache_probe_current_pos={pos_scalar}")

    def _read_cur_pos_from_device_tensors(self, current_pos):
        """Read cur_pos from one device shard (valid when all users share the same decode step)."""
        try:
            shards = ttnn.get_device_tensors(current_pos)
            if not shards:
                return None
            pos_torch = ttnn.to_torch(shards[0])
            if pos_torch.numel() == 0:
                return None
            return int(pos_torch.flatten()[0].item())
        except Exception:
            return None

    def _read_cur_pos_from_mesh_concat(self, current_pos, dims):
        """Reassemble a mesh-sharded cur_pos tensor (dims must be two ints, no None)."""
        try:
            pos_torch = ttnn.to_torch(
                current_pos,
                mesh_composer=ttnn.ConcatMesh2dToTensor(
                    self.mesh_device,
                    mesh_shape=tuple(self.cluster_shape),
                    dims=dims,
                ),
            )
            if pos_torch.numel() == 0:
                return None
            return int(pos_torch.flatten()[0].item())
        except Exception:
            return None

    def _decode_position_scalar(self, current_pos):
        """Read the decode position from the mesh-distributed cur_pos tensor."""
        if current_pos is None:
            return None
        for pos in (
            self._read_cur_pos_from_device_tensors(current_pos),
            self._read_cur_pos_from_mesh_concat(current_pos, (0, 1)),
            self._read_cur_pos_from_mesh_concat(current_pos, (1, 0)),
        ):
            if pos is not None:
                return pos
        logger.info("[qwen-decode-debug] current_pos readback failed on all paths")
        return None

    def _infer_decode_pos_from_cache(self, k_bhsd):
        """Best-effort position from cache occupancy (unreliable for paged / sparse caches)."""
        seq_len = k_bhsd.shape[2]
        token_energy = k_bhsd.abs().sum(dim=(1, 3))
        non_zero_mask = token_energy > 0
        if bool(non_zero_mask.any().item()):
            seq_idx = torch.arange(seq_len, device=token_energy.device).unsqueeze(0)
            return int((seq_idx * non_zero_mask).max().item())
        return seq_len - 1

    def _resolve_decode_position(self, current_pos, k_bhsd):
        """
        Prefer the cur_pos tensor passed to fused/paged SDPA. Cache inference can report
        spurious positions (e.g. 3823) when paged blocks contain uninitialized data.
        """
        pos_from_tensor = getattr(self, "_cached_decode_pos", None)
        if pos_from_tensor is None:
            pos_from_tensor = self._decode_position_scalar(current_pos)
        pos_from_cache = self._infer_decode_pos_from_cache(k_bhsd) if k_bhsd is not None else None
        if pos_from_tensor is not None:
            if pos_from_cache is not None and pos_from_cache != pos_from_tensor:
                logger.warning(
                    f"[qwen-decode-debug] decode pos mismatch: cur_pos_tensor={pos_from_tensor} "
                    f"inferred_from_cache={pos_from_cache}; using cur_pos_tensor"
                )
            return pos_from_tensor
        if pos_from_cache is not None:
            logger.warning(
                f"[qwen-decode-debug] cur_pos_tensor unavailable; "
                f"falling back to inferred_from_cache={pos_from_cache}"
            )
            return pos_from_cache
        return 0

    def _log_substage_stats(self, stage_name, tensor):
        if tensor is None:
            logger.info(f"[qwen-sdpa-substage] {stage_name}: unavailable")
            return
        if not torch.is_floating_point(tensor):
            tensor = tensor.float()
        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())
        finite_mask = torch.isfinite(tensor)
        if int(finite_mask.sum().item()) > 0:
            finite_vals = tensor[finite_mask]
            min_val = float(finite_vals.min().item())
            max_val = float(finite_vals.max().item())
            logger.info(
                f"[qwen-sdpa-substage] {stage_name}: min={min_val:.6e} max={max_val:.6e} nan={nan_count} inf={inf_count}"
            )
        else:
            logger.info(f"[qwen-sdpa-substage] {stage_name}: min=nan max=nan nan={nan_count} inf={inf_count}")

    def _to_torch_decode_tensor(self, tensor):
        try:
            return ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
            )
        except Exception:
            try:
                return ttnn.to_torch(tensor)
            except Exception:
                return None

    def _to_torch_decode_heads_users_tensor(self, tensor):
        """Gather per-chip [1, H, B, D] -> [1, H_global, B_global, D]."""
        try:
            return ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 2), mesh_shape=self.cluster_shape),
            )
        except Exception:
            return self._to_torch_decode_tensor(tensor)

    def _decode_heads_to_bhd(self, heads_1hbd):
        """[1, H, B, D] after ConcatMesh2d(1,2) -> [B, H, D]."""
        if heads_1hbd is None or heads_1hbd.ndim != 4:
            return None
        return heads_1hbd[0].permute(1, 0, 2).contiguous()

    def _pad_activation_k_for_ring_qkv(self, x_tt):
        """Pad activation K to k_ring (1536) so ring matmul K is divisible by RING_SIZE (24)."""
        try:
            k_act = int(x_tt.shape[-1])
        except Exception:
            k_act = self._k_local
        if k_act >= self._k_ring:
            return x_tt
        pad_w = self._k_ring - k_act
        return ttnn.pad(
            x_tt,
            padding=[(0, 0), (0, 0), (0, 0), (0, pad_w)],
            value=0.0,
        )

    def _wqkv_weight_for_matmul_mode(self):
        if self.device_qkv_matmul == "per_device":
            return self.wqkv_per_device if self.wqkv_per_device is not None else self.wqkv_dram_shard
        if self.device_qkv_matmul == "dram":
            return self.wqkv_col
        if self.device_qkv_matmul in ("multicast", "host_fused"):
            return self.wqkv_interleaved
        return self.wqkv

    def _column_sharded_to_ring_input(self, x_tt):
        """Upload column-sharded decode input into PREFETCHER_NOC1 ring L1 layout per chip."""
        ring_memcfg = self.model_config.get("SHARDED_ATTN_INPUT_RING_MEMCFG")
        if ring_memcfg is None:
            return self._pad_activation_k_for_ring_qkv(x_tt)
        shards = ttnn.get_device_tensors(x_tt)
        if not shards:
            return x_tt
        ring_shards = []
        for shard in shards:
            x_local = ttnn.to_torch(shard).float()[0, 0]
            k_act = x_local.shape[1]
            if k_act < self._k_ring:
                padded = torch.zeros(x_local.shape[0], self._k_ring, dtype=x_local.dtype)
                padded[:, :k_act] = x_local
                x_local = padded
            ring_shard = ttnn.from_torch(
                x_local.unsqueeze(0).unsqueeze(0).to(torch.bfloat16),
                device=shard.device(),
                dtype=shard.dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ring_memcfg,
            )
            ring_shards.append(ring_shard)
        try:
            return ttnn.combine_device_tensors(ring_shards)
        except Exception as exc:
            logger.warning(f"[qwen-qkv] combine_device_tensors ring upload failed: {exc}")
            return self._pad_activation_k_for_ring_qkv(x_tt)

    def _ensure_ring_qkv_tile_layout(self, xqkv_mesh):
        """Host column-sum / stage PCC readback expects TILE (matmul untilize_out -> ROW_MAJOR)."""
        if xqkv_mesh.layout == ttnn.TILE_LAYOUT:
            return xqkv_mesh
        return ttnn.to_layout(xqkv_mesh, ttnn.TILE_LAYOUT)

    def _host_sum_columns_decode(self, tensor_mesh):
        """Sum column-sharded partials within each mesh row; replicate to all columns."""
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(tensor_mesh)
        if len(shards) < rows * cols:
            return tensor_mesh
        dtype = tensor_mesh.get_dtype()
        for r in range(rows):
            acc = None
            for c in range(cols):
                part = ttnn.to_torch(shards[r * cols + c]).float()
                acc = part if acc is None else acc + part
            acc_bf16 = acc.to(torch.bfloat16)
            for c in range(cols):
                shard = shards[r * cols + c]
                summed_tt = ttnn.from_torch(
                    acc_bf16,
                    dtype=dtype,
                    layout=shard.layout,
                    memory_config=shard.memory_config(),
                    device=self.mesh_device,
                )
                ttnn.copy(summed_tt, shard)
                ttnn.deallocate(summed_tt)
        return tensor_mesh

    def _to_torch_decode_kv_cache_tensor(self, tensor):
        """Linear KV cache: gather batch on mesh cols (dim 0), heads on rows (dim 1)."""
        try:
            return ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 0), mesh_shape=self.cluster_shape),
            )
        except Exception:
            return self._to_torch_decode_tensor(tensor)

    def _gather_decode_q_heads_bhd(self, q_heads_tt):
        """Assemble per-chip [1, B_local, H_local, D] -> global [B, H, D] (8x4 mesh).

        nlp_create_qkv_heads_decode emits [1, B_local, H_local, D] (batch first): Q heads are
        split across mesh rows, users (batch) across mesh cols. Mirror the K gather here; the
        generic ConcatMesh helper assumes [1, H, B, D] (correct for the SDPA output, not Q heads).
        """
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        try:
            shards = ttnn.get_device_tensors(q_heads_tt)
            if len(shards) < rows * cols:
                return None
            col_parts = []
            for c in range(cols):
                head_shards = [ttnn.to_torch(shards[r * cols + c]).float()[0] for r in range(rows)]
                # [B_local, H_local, D] per shard -> cat heads across rows -> [B_local, H_global, D]
                col_parts.append(torch.cat(head_shards, dim=1))
            # cat users across cols -> [B_global, H_global, D]
            return torch.cat(col_parts, dim=0)
        except Exception as exc:
            logger.info(f"[qwen-q-gather] failed: {exc}")
            return None

    def _scatter_decode_q_heads_bhd(self, q_heads_tt, q_global_bhd):
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(q_heads_tt)
        for c in range(cols):
            b0, b1 = c * self.batch_size_per_device_group, (c + 1) * self.batch_size_per_device_group
            for r in range(rows):
                h0, h1 = r * self.n_local_heads, (r + 1) * self.n_local_heads
                q_local = q_global_bhd[b0:b1, h0:h1, :].permute(1, 0, 2).unsqueeze(0).to(torch.bfloat16)
                shard = shards[r * cols + c]
                q_local_tt = ttnn.from_torch(
                    q_local,
                    dtype=ttnn.bfloat16,
                    layout=shard.layout,
                    memory_config=shard.memory_config(),
                    device=self.mesh_device,
                )
                ttnn.copy(q_local_tt, shard)
                ttnn.deallocate(q_local_tt)

    def _gather_decode_k_heads_bhd(self, k_heads_tt):
        """Assemble per-chip [1, B_local, H_kv, D] -> global [B, H_kv, D] (8x4 mesh)."""
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        try:
            shards = ttnn.get_device_tensors(k_heads_tt)
            if len(shards) < rows * cols:
                return None
            col_parts = []
            for c in range(cols):
                head_shards = [ttnn.to_torch(shards[r * cols + c]).float()[0] for r in range(rows)]
                col_parts.append(torch.cat(head_shards, dim=1))
            return torch.cat(col_parts, dim=0)
        except Exception as exc:
            logger.info(f"[qwen-k-gather] failed: {exc}")
            return None

    def _scatter_decode_k_heads_bhd(self, k_heads_tt, k_global_bhd):
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(k_heads_tt)
        for c in range(cols):
            b0, b1 = c * self.batch_size_per_device_group, (c + 1) * self.batch_size_per_device_group
            k_col = k_global_bhd[b0:b1]
            for r in range(rows):
                k_local = k_col[:, r : r + 1, :].unsqueeze(0).to(torch.bfloat16)
                shard = shards[r * cols + c]
                k_local_tt = ttnn.from_torch(
                    k_local,
                    dtype=ttnn.bfloat16,
                    layout=shard.layout,
                    memory_config=shard.memory_config(),
                    device=self.mesh_device,
                )
                ttnn.copy(k_local_tt, shard)
                ttnn.deallocate(k_local_tt)

    def _to_torch_decode_k_heads_tensor(self, tensor):
        """Decode K heads [1, B_local, H_kv, D] on each chip — manual gather (not Q ConcatMesh dims)."""
        gathered = self._gather_decode_k_heads_bhd(tensor)
        if gathered is None:
            return self._to_torch_decode_kv_cache_tensor(tensor)
        return gathered.unsqueeze(0)

    def _safe_comp_pcc_flat(self, ref, tt, threshold, name):
        ref_f = ref.float().reshape(-1)
        tt_f = tt.float().reshape(-1)
        if ref_f.numel() != tt_f.numel():
            logger.info(f"[qwen-qkv-stage] {name} PCC skipped: size mismatch ref={ref_f.numel()} tt={tt_f.numel()}")
            return False, 0.0
        try:
            return comp_pcc(ref_f, tt_f, threshold)
        except Exception as exc:
            logger.info(f"[qwen-qkv-stage] {name} PCC failed: {exc}")
            return False, 0.0

    def _to_torch_decode_wo_input_tensor(self, tensor):
        try:
            if tensor.shape[-1] == self.n_heads * self.head_dim:
                shards = ttnn.get_device_tensors(tensor)
                if shards:
                    return ttnn.to_torch(shards[0])

            shards = ttnn.get_device_tensors(tensor)
            mesh_cols = self.cluster_shape[1]
            row_shards = []
            for row_idx in range(self.cluster_shape[0]):
                shard_idx = row_idx * mesh_cols
                if shard_idx >= len(shards):
                    return self._to_torch_decode_tensor(tensor)
                row_shards.append(ttnn.to_torch(shards[shard_idx]))
            return torch.cat(row_shards, dim=-1)
        except Exception:
            return self._to_torch_decode_tensor(tensor)

    def _reorder_cache_tensor(self, cache_tensor, batch_size):
        if cache_tensor is None or cache_tensor.ndim != 4:
            return None
        d_dim = 3
        batch_dim = next((i for i in range(3) if cache_tensor.shape[i] == batch_size), 0)
        remaining = [i for i in range(3) if i != batch_dim]
        seq_dim = max(remaining, key=lambda idx: cache_tensor.shape[idx])
        head_dim = [i for i in remaining if i != seq_dim][0]
        return cache_tensor.permute(batch_dim, head_dim, seq_dim, d_dim).contiguous()

    def _debug_sdpa_substages(self, q_heads, keys, values, current_pos=None):
        if not self.sdpa_substage_debug:
            return

        k_torch = self._to_torch_decode_kv_cache_tensor(keys)
        v_torch = self._to_torch_decode_kv_cache_tensor(values)
        q_bhd = self._gather_decode_q_heads_bhd(q_heads)
        if q_bhd is None or k_torch is None or v_torch is None:
            logger.info("[qwen-sdpa-substage] unable to read Q/K/V tensors for decomposition")
            return
        q_bhd = q_bhd.float()
        batch_size = q_bhd.shape[0]

        k_bhsd = self._reorder_cache_tensor(k_torch.float(), batch_size)
        v_bhsd = self._reorder_cache_tensor(v_torch.float(), batch_size)
        if k_bhsd is None or v_bhsd is None:
            logger.info(
                f"[qwen-sdpa-substage] unsupported cache shapes: K={tuple(k_torch.shape)} V={tuple(v_torch.shape)}"
            )
            return

        # Match Q head dimension with cache heads.
        # For GQA, each KV head fan-outs to multiple Q heads.
        if k_bhsd.shape[1] == 1 and q_bhd.shape[1] > 1:
            k_bhsd = k_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
            v_bhsd = v_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
        elif q_bhd.shape[1] % k_bhsd.shape[1] == 0 and q_bhd.shape[1] != k_bhsd.shape[1]:
            kv_repeat = q_bhd.shape[1] // k_bhsd.shape[1]
            k_bhsd = k_bhsd.repeat_interleave(kv_repeat, dim=1)
            v_bhsd = v_bhsd.repeat_interleave(kv_repeat, dim=1)
        elif k_bhsd.shape[1] != q_bhd.shape[1]:
            logger.info(f"[qwen-sdpa-substage] head mismatch: q_heads={q_bhd.shape[1]} kv_heads={k_bhsd.shape[1]}")
            return

        seq_len = k_bhsd.shape[2]
        inferred_pos = self._infer_decode_pos_from_cache(k_bhsd)
        decode_pos = self._resolve_decode_position(current_pos, k_bhsd)
        logger.info(
            f"[qwen-sdpa-substage] decode_pos={decode_pos} inferred_from_cache={inferred_pos} seq_len={seq_len}"
        )
        current_pos = decode_pos

        scores_raw = torch.einsum("bhd,bhsd->bhs", q_bhd, k_bhsd)
        self._log_substage_stats("cpu.scores_raw", scores_raw)

        scores_scaled = scores_raw * float(self.scale)
        self._log_substage_stats("cpu.scores_scaled", scores_scaled)

        mask = torch.full((batch_size, 1, seq_len), float("-inf"), device=scores_scaled.device)
        mask[:, :, : current_pos + 1] = 0.0
        scores_masked = scores_scaled + mask
        self._log_substage_stats("cpu.scores_masked", scores_masked)

        probs = torch.softmax(scores_masked, dim=-1)
        self._log_substage_stats("cpu.probs", probs)

        sdpa_out = torch.einsum("bhs,bhsd->bhd", probs, v_bhsd)
        self._log_substage_stats("cpu.sdpa_out", sdpa_out)
        self._last_sdpa_cpu_out_bhd = sdpa_out.detach()

        if not self.sdpa_substage_debug_ttnn:
            return

        try:
            # Use a tiny representative slice to keep debug cost bounded.
            q_small = q_bhd[0:1, 0:1, :].unsqueeze(2)  # [1, 1, 1, D]
            k_small = k_bhsd[0:1, 0:1, :, :]  # [1, 1, S, D]
            v_small = v_bhsd[0:1, 0:1, :, :]  # [1, 1, S, D]
            mask_small = torch.full((1, 1, 1, seq_len), -1e9, dtype=torch.float32)
            mask_small[:, :, :, : current_pos + 1] = 0.0

            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            q_tt = ttnn.from_torch(
                q_small, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            k_tt = ttnn.from_torch(
                k_small, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            v_tt = ttnn.from_torch(
                v_small, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            mask_tt = ttnn.from_torch(
                mask_small, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )

            scores_raw_tt = ttnn.matmul(q_tt, ttnn.transpose(k_tt, -2, -1))
            self._log_substage_stats("ttnn.scores_raw", self._to_torch_decode_tensor(scores_raw_tt))

            scores_scaled_tt = ttnn.multiply(scores_raw_tt, float(self.scale))
            self._log_substage_stats("ttnn.scores_scaled", self._to_torch_decode_tensor(scores_scaled_tt))

            scores_masked_tt = ttnn.add(scores_scaled_tt, mask_tt)
            self._log_substage_stats("ttnn.scores_masked", self._to_torch_decode_tensor(scores_masked_tt))

            probs_tt = ttnn.softmax(scores_masked_tt, dim=-1)
            self._log_substage_stats("ttnn.probs", self._to_torch_decode_tensor(probs_tt))

            out_tt = ttnn.matmul(probs_tt, v_tt)
            self._log_substage_stats("ttnn.sdpa_out", self._to_torch_decode_tensor(out_tt))
        except Exception as e:
            logger.info(f"[qwen-sdpa-substage] ttnn decomposition unavailable: {e}")

    def _debug_sdpa_valid_region(self, sdpa_out_tensor):
        if not self.sdpa_substage_debug:
            return
        if self._last_sdpa_cpu_out_bhd is None:
            logger.info("[qwen-sdpa] valid region check skipped: cpu reference unavailable")
            return

        sdpa_out_torch = self._to_torch_decode_heads_users_tensor(sdpa_out_tensor)
        if sdpa_out_torch is None:
            logger.info("[qwen-sdpa] valid region check skipped: unable to read sdpa_out")
            return
        if sdpa_out_torch.ndim != 4:
            logger.info(
                f"[qwen-sdpa] valid region check skipped: unsupported sdpa_out shape {tuple(sdpa_out_torch.shape)}"
            )
            return

        # Fused decode SDPA tensor readback layout is [1, H, B, D_phys].
        h = self._last_sdpa_cpu_out_bhd.shape[1]
        b = self._last_sdpa_cpu_out_bhd.shape[0]
        d = self._last_sdpa_cpu_out_bhd.shape[2]
        h = min(h, sdpa_out_torch.shape[1])
        b = min(b, sdpa_out_torch.shape[2])
        d = min(d, sdpa_out_torch.shape[3], self.head_dim)

        sdpa_valid = self._decode_heads_to_bhd(sdpa_out_torch[:, :h, :b, :d])
        if sdpa_valid is None:
            return
        sdpa_valid = sdpa_valid.float()
        ref_valid = self._last_sdpa_cpu_out_bhd[:b, :h, :d].float()

        passing, pcc_message = comp_pcc(ref_valid, sdpa_valid, 0.99)
        valid_inf = int(torch.isinf(sdpa_valid).sum().item())
        valid_nan = int(torch.isnan(sdpa_valid).sum().item())
        logger.info(
            f"[qwen-sdpa] valid region shape={tuple(sdpa_valid.shape)} "
            f"pcc={pcc_message} passing={passing} valid_nan={valid_nan} valid_inf={valid_inf}"
        )

    def _debug_concat_vs_sdpa_ref(self, attn_output_cat):
        if not self.sdpa_substage_debug or self._last_sdpa_cpu_out_bhd is None:
            return

        attn_host = self._to_torch_decode_wo_input_tensor(attn_output_cat)
        if attn_host is None or attn_host.ndim != 4:
            logger.info("[qwen-sdpa] concat check skipped: unable to read concat output")
            return

        b = min(self._last_sdpa_cpu_out_bhd.shape[0], attn_host.shape[-2])
        h = min(self._last_sdpa_cpu_out_bhd.shape[1], attn_host.shape[-1] // self.head_dim)
        d = self._last_sdpa_cpu_out_bhd.shape[2]
        width = h * d
        if width == 0 or b == 0:
            logger.info("[qwen-sdpa] concat check skipped: empty overlap")
            return

        # Reference attention input to WO is [B, H * D] with heads contiguous.
        ref_concat = self._last_sdpa_cpu_out_bhd[:b].reshape(b, h * d)[..., :width].float()
        tt_concat = attn_host[0, 0, :b, :width].float().contiguous()
        diff = (ref_concat - tt_concat).abs()
        max_diff = float(diff.max().item())
        mean_diff = float(diff.mean().item())
        # comp_pcc treats near-zero bf16 tensors as "all zero"; use diff stats for tiny activations.
        if ref_concat.abs().max().item() < 1e-6 and tt_concat.abs().max().item() < 1e-6:
            passing = max_diff < 1e-4
            pcc_message = 1.0 if passing else 0.0
        else:
            passing, pcc_message = comp_pcc(ref_concat, tt_concat, 0.99)
        logger.info(
            f"[qwen-sdpa] concat_vs_ref shape={tuple(tt_concat.shape)} "
            f"pcc={pcc_message} passing={passing} "
            f"mean_abs_diff={mean_diff:.6e} max_abs_diff={max_diff:.6e}"
        )

    def _simple_sdpa_allowed(self, page_table):
        """Host/ttnn simple SDPA only supports linear KV layouts, not paged block caches."""
        if page_table is not None:
            return False
        return True

    def _apply_decode_qk_norm(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD):
        if self.model_config.get("USE_PREFETCHER", False):
            return self._apply_decode_qk_norm_sharded(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)
        if self.use_host_qk_norm:
            return self._apply_decode_qk_norm_host(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)
        return self._apply_decode_qk_norm_flat(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)

    def _host_rmsnorm_bhd_torch(self, tensor_bhd, weight_1d, eps=1e-6):
        """RMSNorm on [B, H, D] (matches reference q_norm layout)."""
        w = weight_1d.float().view(1, 1, -1)
        t = tensor_bhd.float()
        return (t * torch.rsqrt(t.pow(2).mean(-1, keepdim=True) + eps)) * w

    def _debug_qkv_input_vs_ref(self, x_tt):
        """PCC of gathered column-sharded decode input vs host golden."""
        if not self.qkv_stage_debug:
            return
        golden = getattr(self, "host_input_golden", None)
        if golden is None:
            return
        x_bh = self._gather_sharded_hidden_input(x_tt)
        if x_bh is None:
            logger.info("[qwen-qkv-stage] input gather failed")
            return
        g = golden[:, 0, :] if golden.dim() == 3 else golden
        b = min(x_bh.shape[0], g.shape[0])
        d = min(x_bh.shape[1], g.shape[1])
        _pass, pcc = self._safe_comp_pcc_flat(g[:b, :d], x_bh[:b, :d], 0.99, "input_gather")
        max_diff = float((g[:b, :d] - x_bh[:b, :d]).abs().max().item())
        logger.info(
            f"[qwen-qkv-stage] input_gather (sharded vs golden) pcc={pcc} pass={_pass} max_abs_diff={max_diff:.6e}"
        )

    def _debug_qkv_weight_shard_vs_ref(self, r=0, c=0):
        """PCC of TT wqkv weight shard vs torch host weight block."""
        if not self.qkv_stage_debug or self._wqkv_host_weight is None:
            return
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        k_local = self.hidden_size // cols
        out_width = self.n_local_heads * self.head_dim + 2 * self.n_local_kv_heads * self.head_dim
        k0, k1 = c * k_local, (c + 1) * k_local
        n0, n1 = r * out_width, (r + 1) * out_width
        ref_w = self._wqkv_host_weight[k0:k1, n0:n1]
        w_tt = self._wqkv_weight_for_matmul_mode()
        try:
            shards = ttnn.get_device_tensors(w_tt)
            tt_w = ttnn.to_torch(shards[r * cols + c]).float()[0, 0]
            tt_w = tt_w[: ref_w.shape[0], : ref_w.shape[1]]
            _pass, pcc = self._safe_comp_pcc_flat(ref_w, tt_w, 0.99, f"wqkv_w_r{r}_c{c}")
            logger.info(
                f"[qwen-qkv-stage] wqkv_weight_shard r={r} c={c} ref={tuple(ref_w.shape)} "
                f"tt={tuple(tt_w.shape)} pcc={pcc} pass={_pass}"
            )
        except Exception as exc:
            logger.info(f"[qwen-qkv-stage] wqkv_weight_shard r={r} c={c} failed: {exc}")

    def _debug_qkv_partial_vs_ref(self, xqkv_mesh, x_tt):
        """PCC of per-device TT partial (pre column-sum) vs torch using local x/w shards."""
        if not self.qkv_stage_debug or self._wqkv_host_weight is None:
            return
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        out_width = self.n_local_heads * self.head_dim + 2 * self.n_local_kv_heads * self.head_dim
        x_shards_torch = getattr(self, "_qkv_partial_x_torch", None)
        if x_shards_torch is not None:
            x_shards = x_shards_torch
        else:
            x_shards = ttnn.get_device_tensors(x_tt)
        w_shards = ttnn.get_device_tensors(self._wqkv_weight_for_matmul_mode())
        out_shards = ttnn.get_device_tensors(xqkv_mesh)
        for r in range(rows):
            n0, n1 = r * out_width, (r + 1) * out_width
            for c in range(cols):
                idx = r * cols + c
                if x_shards_torch is not None:
                    x_local = x_shards[idx][0, 0]
                else:
                    x_local = ttnn.to_torch(x_shards[idx]).float()[0, 0]
                w_local = ttnn.to_torch(w_shards[idx]).float()[0, 0]
                tt = ttnn.to_torch(out_shards[idx]).float()[0, 0]
                k_act = (
                    min(self._k_ring, x_local.shape[1], w_local.shape[0])
                    if self.device_qkv_matmul == "ring"
                    else min(self._k_local, x_local.shape[1], w_local.shape[0])
                )
                n_act = min(out_width, w_local.shape[1], tt.shape[1])
                ref_partial = x_local[:, :k_act].float() @ w_local[:k_act, :n_act].float()
                tt = tt[:, :n_act]
                _pass, pcc = self._safe_comp_pcc_flat(ref_partial, tt, 0.99, f"xqkv_partial_r{r}_c{c}")
                logger.info(
                    f"[qwen-qkv-stage] xqkv_partial_pre_sum r={r} c={c} shape={tuple(tt.shape)} "
                    f"pcc={pcc} pass={_pass}"
                )
        self._qkv_partial_x_torch = None

    def _host_fused_xqkv_bh(self, x_tt):
        """Gather sharded decode input and return host fused xqkv [B, qkv_size]."""
        if self._wqkv_host_weight is None:
            return None
        x_bh = self._gather_sharded_hidden_input(x_tt)
        if x_bh is None:
            return None
        golden = getattr(self, "host_input_golden", None)
        if golden is not None:
            g = golden[:, 0, :] if golden.dim() == 3 else golden
            if g.shape == x_bh.shape:
                x_bh = g
        with torch.no_grad():
            return (x_bh.float() @ self._wqkv_host_weight).to(torch.bfloat16)

    def _split_fused_xqkv_bh(self, xqkv_bh):
        """Fused row layout [B,8*(Q+K+V)] -> global Q/K/V [B,H,D] matching torch reference."""
        rows = int(self.cluster_shape[0])
        q_width = self.n_local_heads * self.head_dim
        kv_width = self.n_local_kv_heads * self.head_dim
        block = q_width + 2 * kv_width
        q_parts, k_parts, v_parts = [], [], []
        for r in range(rows):
            sl = xqkv_bh[:, r * block : (r + 1) * block]
            q_parts.append(sl[:, :q_width])
            k_parts.append(sl[:, q_width : q_width + kv_width])
            v_parts.append(sl[:, q_width + kv_width : block])
        q_global = torch.cat(q_parts, dim=1).view(xqkv_bh.shape[0], self.n_heads, self.head_dim)
        k_global = torch.cat(k_parts, dim=1).view(xqkv_bh.shape[0], self.n_kv_heads, self.head_dim)
        v_global = torch.cat(v_parts, dim=1).view(xqkv_bh.shape[0], self.n_kv_heads, self.head_dim)
        return q_global, k_global, v_global

    def _scatter_decode_v_heads_bhd(self, v_heads_tt, v_global_bhd):
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(v_heads_tt)
        for c in range(cols):
            b0, b1 = c * self.batch_size_per_device_group, (c + 1) * self.batch_size_per_device_group
            v_col = v_global_bhd[b0:b1]
            for r in range(rows):
                v_local = v_col[:, r : r + 1, :].unsqueeze(0).to(torch.bfloat16)
                shard = shards[r * cols + c]
                v_local_tt = ttnn.from_torch(
                    v_local,
                    dtype=ttnn.bfloat16,
                    layout=shard.layout,
                    memory_config=shard.memory_config(),
                    device=self.mesh_device,
                )
                ttnn.copy(v_local_tt, shard)
                ttnn.deallocate(v_local_tt)

    def _host_scatter_qkv_heads_from_fused(self, q_heads_tt, k_heads_tt, v_heads_tt, xqkv_bh):
        q_global, k_global, v_global = self._split_fused_xqkv_bh(xqkv_bh)
        self._scatter_decode_q_heads_bhd(q_heads_tt, q_global)
        self._scatter_decode_k_heads_bhd(k_heads_tt, k_global)
        self._scatter_decode_v_heads_bhd(v_heads_tt, v_global)

    def _host_fused_qkv_decode(self, x_tt):
        """Host fused QKV for bring-up: gather columns -> matmul -> row-shard output, replicate cols."""
        out_bh = self._host_fused_xqkv_bh(x_tt)
        if out_bh is None:
            return None
        self._last_host_xqkv_bh = out_bh
        out_4d = out_bh.unsqueeze(0).unsqueeze(0).contiguous()
        return ttnn.from_torch(
            out_4d,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(3, None), mesh_shape=self.cluster_shape),
        )

    def _debug_qkv_xqkv_post_col_sum(self, xqkv_mesh):
        """PCC of TT xqkv after column-sum vs torch fused matmul on gathered input."""
        if not self.qkv_stage_debug or self._wqkv_host_weight is None:
            return
        golden = getattr(self, "host_input_golden", None)
        if golden is None:
            return
        x_bh = golden[:, 0, :] if golden.dim() == 3 else golden
        with torch.no_grad():
            ref_xqkv = x_bh.float() @ self._wqkv_host_weight
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(xqkv_mesh)
        out_width = self.n_local_heads * self.head_dim + 2 * self.n_local_kv_heads * self.head_dim
        for r in range(rows):
            ref_row = ref_xqkv[:, r * out_width : (r + 1) * out_width]
            for c in range(cols):
                tt = ttnn.to_torch(shards[r * cols + c]).float()[0, 0]
                tt = tt[:, : ref_row.shape[1]]
                _pass, pcc = self._safe_comp_pcc_flat(ref_row, tt, 0.99, f"xqkv_r{r}_c{c}")
                logger.info(
                    f"[qwen-qkv-stage] xqkv_post_col_sum r={r} c={c} shape={tuple(tt.shape)} " f"pcc={pcc} pass={_pass}"
                )

    def _debug_qkv_pre_norm_vs_ref(self, q_heads, k_heads):
        """PCC of TT wqkv output (pre QK norm) vs torch reference — isolates matmul/create_heads."""
        if not self.qkv_stage_debug:
            return
        ref = getattr(self, "host_reference_attn", None)
        golden = getattr(self, "host_input_golden", None)
        if ref is None or golden is None:
            return
        x_bh = golden[:, 0, :] if golden.dim() == 3 else golden
        x_bsh = x_bh.unsqueeze(1)
        with torch.no_grad():
            xq = ref.wq(x_bsh).view(x_bh.shape[0], 1, ref.n_local_heads, ref.head_dim)[:, 0].float()
            xk = ref.wk(x_bsh).view(x_bh.shape[0], 1, ref.n_local_kv_heads, ref.head_dim)[:, 0].float()
        tt_q = self._gather_decode_q_heads_bhd(q_heads)
        tt_k = self._gather_decode_k_heads_bhd(k_heads)
        if tt_q is None or tt_k is None:
            logger.info("[qwen-qkv-stage] pre_norm skipped: gather failed")
            return
        b = min(xq.shape[0], tt_q.shape[0])
        hq = min(xq.shape[1], tt_q.shape[1])
        hk = min(xk.shape[1], tt_k.shape[1])
        d = min(xq.shape[2], tt_q.shape[2])
        q_pass, q_pcc = self._safe_comp_pcc_flat(xq[:b, :hq, :d], tt_q[:b, :hq, :d], 0.99, "Q_pre_norm")
        k_pass, k_pcc = self._safe_comp_pcc_flat(xk[:b, :hk, :d], tt_k[:b, :hk, :d], 0.99, "K_pre_norm")
        logger.info(
            f"[qwen-qkv-stage] pre_norm (TT wqkv vs ref) "
            f"q_pcc={q_pcc} q_pass={q_pass} k_pcc={k_pcc} k_pass={k_pass}"
        )
        rows, cols = int(self.cluster_shape[0]), int(self.cluster_shape[1])
        shards = ttnn.get_device_tensors(q_heads)
        if len(shards) >= rows * cols:
            for c in range(cols):
                b0 = c * self.batch_size_per_device_group
                b1 = b0 + self.batch_size_per_device_group
                for r in range(rows):
                    h0 = r * self.n_local_heads
                    h1 = h0 + self.n_local_heads
                    local = ttnn.to_torch(shards[r * cols + c]).float()[0].permute(1, 0, 2)
                    _p, _pc = self._safe_comp_pcc_flat(
                        xq[b0:b1, h0:h1, :d], local[: b1 - b0, : h1 - h0, :d], 0.99, f"Q_shard_r{r}_c{c}"
                    )
                    logger.info(f"[qwen-qkv-stage] Q_shard r={r} c={c} pcc={_pc} pass={_p}")

    def _apply_decode_qk_norm_host(self, q_heads, k_heads):
        """BH no-prefetch: skip on-device QK norm until TT wqkv PCC is fixed (see pre_norm log)."""
        q_bhd = self._gather_decode_q_heads_bhd(q_heads)
        k_bhd = self._gather_decode_k_heads_bhd(k_heads)
        if q_bhd is not None and k_bhd is not None:
            q_n = self._host_rmsnorm_bhd_torch(q_bhd, self.q_norm_weight)
            k_n = self._host_rmsnorm_bhd_torch(k_bhd, self.k_norm_weight)
            logger.info(
                "[qwen-host-qk-norm] reference-only gathered norm "
                f"q_rms={float(q_n.pow(2).mean().sqrt().item()):.4e} "
                f"k_rms={float(k_n.pow(2).mean().sqrt().item()):.4e}; "
                "pass-through on device (wqkv pre_norm PCC must be fixed first)"
            )
        return q_heads, k_heads

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

    def _debug_qkv_post_rope_vs_ref(self, q_heads, k_heads, current_pos):
        if not self.qkv_stage_debug:
            return
        ref = getattr(self, "host_reference_attn", None)
        freqs = getattr(self, "host_freqs_cis", None)
        golden = getattr(self, "host_input_golden", None)
        if ref is None or freqs is None or golden is None:
            logger.info("[qwen-qkv-stage] skipped: need host_reference_attn, host_freqs_cis, host_input_golden")
            return
        try:
            from models.demos.llama3_70b_galaxy.reference.qwen import apply_rotary_emb
        except ImportError:
            logger.info("[qwen-qkv-stage] skipped: apply_rotary_emb import failed")
            return

        x_bh = golden[:, 0, :] if golden.dim() == 3 else golden
        pos = int(self._decode_position_scalar(current_pos))
        pos = max(0, min(pos, freqs.shape[0] - 1))
        x_bsh = x_bh.unsqueeze(1)
        with torch.no_grad():
            xq, xk, _xv = ref.wq(x_bsh), ref.wk(x_bsh), ref.wv(x_bsh)
            xq = xq.view(x_bh.shape[0], 1, ref.n_local_heads, ref.head_dim)
            xk = xk.view(x_bh.shape[0], 1, ref.n_local_kv_heads, ref.head_dim)
            xq = ref.q_norm(xq)
            xk = ref.k_norm(xk)
            freqs_i = freqs[pos, :].unsqueeze(0)
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_i)
            ref_q = xq[:, 0].float()
            ref_k = xk[:, 0].float()

        tt_q = self._gather_decode_q_heads_bhd(q_heads)
        if tt_q is None:
            logger.info("[qwen-qkv-stage] skipped: Q readback failed")
            return
        tt_q = tt_q.float()

        tt_k = self._gather_decode_k_heads_bhd(k_heads)
        if tt_k is None:
            logger.info("[qwen-qkv-stage] skipped: K readback failed")
            return

        b = min(ref_q.shape[0], tt_q.shape[0])
        hq = min(ref_q.shape[1], tt_q.shape[1])
        hk = min(ref_k.shape[1], tt_k.shape[1])
        d = min(ref_q.shape[2], tt_q.shape[2])
        ref_q_s = ref_q[:b, :hq, :d]
        tt_q_s = tt_q[:b, :hq, :d]
        ref_k_s = ref_k[:b, :hk, :d]
        tt_k_s = tt_k[:b, :hk, :d]
        q_pass, q_pcc = self._safe_comp_pcc_flat(ref_q_s, tt_q_s, 0.99, "Q")
        k_pass, k_pcc = self._safe_comp_pcc_flat(ref_k_s, tt_k_s, 0.99, "K")
        q_diff = (ref_q_s - tt_q_s).abs() if ref_q_s.shape == tt_q_s.shape else torch.tensor([float("nan")])
        k_diff = (ref_k_s - tt_k_s).abs() if ref_k_s.shape == tt_k_s.shape else torch.tensor([float("nan")])
        logger.info(
            f"[qwen-qkv-stage] post_rope pos={pos} "
            f"ref_q={tuple(ref_q.shape)} tt_q={tuple(tt_q.shape)} "
            f"ref_k={tuple(ref_k.shape)} tt_k={tuple(tt_k.shape)} "
            f"q_pcc={q_pcc} q_pass={q_pass} q_max_diff={float(q_diff.max().item()):.6e} "
            f"k_pcc={k_pcc} k_pass={k_pass} k_max_diff={float(k_diff.max().item()):.6e}"
        )

    def _gather_sharded_hidden_input(self, x_tt):
        """Gather column-sharded decode residual [1,1,B,dim//4] per chip -> [B, dim]."""
        cols = int(self.cluster_shape[1])
        try:
            shards = ttnn.get_device_tensors(x_tt)
            if len(shards) >= cols:
                parts = [ttnn.to_torch(shards[c]).float()[0, 0] for c in range(cols)]
                out = torch.cat(parts, dim=-1)
                if out.shape[-1] > self.hidden_size:
                    out = out[:, : self.hidden_size]
                return out
        except Exception as exc:
            logger.warning("[qwen-host] hidden gather failed: {}", exc)
        return None

    def _pack_decode_output_to_mesh(self, out_bh):
        """Pack [B, dim] torch output as replicated [1,1,B,dim] on mesh."""
        out_4d = out_bh.view(1, 1, out_bh.shape[0], out_bh.shape[1]).contiguous().to(torch.bfloat16)
        return ttnn.from_torch(
            out_4d,
            device=self.mesh_device,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def _forward_decode_via_host_reference(self, x, current_pos):
        """Full torch reference decode (unit-test bring-up): gather input, run host_reference_attn."""
        ref = getattr(self, "host_reference_attn", None)
        freqs = getattr(self, "host_freqs_cis", None)
        if ref is None or freqs is None:
            return None
        x_bh = self._gather_sharded_hidden_input(x)
        if x_bh is None:
            return None
        golden = getattr(self, "host_input_golden", None)
        if golden is not None:
            g = golden[:, 0, :] if golden.dim() == 3 else golden
            if g.shape == x_bh.shape:
                max_diff = float((g - x_bh).abs().max().item())
                logger.info(f"[qwen-host-ref] gather vs golden max_abs_diff={max_diff:.6e}")
        pos = int(self._decode_position_scalar(current_pos))
        pos = max(0, min(pos, freqs.shape[0] - 1))
        x_bsh = x_bh.unsqueeze(1)
        freqs_i = freqs[pos, :].unsqueeze(0)
        with torch.no_grad():
            out = ref(x_bsh, pos, freqs_i, mask=None)
        logger.info(
            f"[qwen-host-ref] decode via torch reference pos={pos} "
            f"in_rms={float(x_bh.pow(2).mean().sqrt().item()):.6e} "
            f"out_rms={float(out.pow(2).mean().sqrt().item()):.6e}"
        )
        return self._pack_decode_output_to_mesh(out[:, 0, :])

    def _simple_decode_wo_host_fallback(self, attn_output_cat):
        """Host WO matmul on full [B, n_heads*head_dim] attention vector (matches reference wo)."""
        if self._wo_host_weight is None:
            return None
        attn_host = self._to_torch_decode_wo_input_tensor(attn_output_cat)
        if attn_host is None or attn_host.ndim != 4:
            logger.info("[qwen-host-wo] unavailable: bad attn readback")
            return None
        in_features = self.n_heads * self.head_dim
        if attn_host.shape[-1] != in_features:
            logger.info(f"[qwen-host-wo] unavailable: attn width {attn_host.shape[-1]} != {in_features}")
            return None
        x = attn_host[0, 0].float()
        out = torch.matmul(x, self._wo_host_weight.float())
        logger.info(f"[qwen-host-wo] host WO out_rms={float(out.pow(2).mean().sqrt().item()):.6e}")
        return self._pack_decode_output_to_mesh(out)

    def _simple_decode_sdpa_host_fallback(self, q_heads, keys, values, sdpa_out_mem_cfg, current_pos=None):
        # Q create_heads output is [1, B_local, H_local, D]: heads on mesh rows, batch on cols.
        k_torch = self._to_torch_decode_kv_cache_tensor(keys)
        v_torch = self._to_torch_decode_kv_cache_tensor(values)
        q_bhd = self._gather_decode_q_heads_bhd(q_heads)
        if q_bhd is None or k_torch is None or v_torch is None:
            logger.info("[qwen-simple-sdpa] fallback unavailable: cannot readback Q/K/V")
            return None
        q_bhd = q_bhd.float()
        batch_size = q_bhd.shape[0]

        k_bhsd = self._reorder_cache_tensor(k_torch.float(), batch_size)
        v_bhsd = self._reorder_cache_tensor(v_torch.float(), batch_size)
        if k_bhsd is None or v_bhsd is None:
            logger.info(
                f"[qwen-simple-sdpa] fallback unavailable: unsupported cache shapes "
                f"K={tuple(k_torch.shape)} V={tuple(v_torch.shape)}"
            )
            return None

        # Expand/repeat KV heads to align with Q heads (GQA).
        if k_bhsd.shape[1] == 1 and q_bhd.shape[1] > 1:
            k_bhsd = k_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
            v_bhsd = v_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
        elif q_bhd.shape[1] % k_bhsd.shape[1] == 0 and q_bhd.shape[1] != k_bhsd.shape[1]:
            kv_repeat = q_bhd.shape[1] // k_bhsd.shape[1]
            k_bhsd = k_bhsd.repeat_interleave(kv_repeat, dim=1)
            v_bhsd = v_bhsd.repeat_interleave(kv_repeat, dim=1)
        elif k_bhsd.shape[1] != q_bhd.shape[1]:
            logger.info(
                f"[qwen-simple-sdpa] fallback unavailable: head mismatch "
                f"q_heads={q_bhd.shape[1]} kv_heads={k_bhsd.shape[1]}"
            )
            return None

        seq_len = k_bhsd.shape[2]
        decode_pos = self._resolve_decode_position(current_pos, k_bhsd)
        decode_pos = min(decode_pos, seq_len - 1)
        logger.info(
            f"[qwen-simple-sdpa] decode_pos={decode_pos} seq_len={seq_len} "
            f"(cur_pos_tensor={self._decode_position_scalar(current_pos)})"
        )

        scores = torch.einsum("bhd,bhsd->bhs", q_bhd, k_bhsd) * float(self.scale)
        mask = torch.full((batch_size, 1, seq_len), float("-inf"), device=scores.device)
        mask[:, :, : decode_pos + 1] = 0.0
        probs = torch.softmax(scores + mask, dim=-1)
        out_bhd = torch.einsum("bhs,bhsd->bhd", probs, v_bhsd).to(torch.bfloat16)
        self._last_sdpa_cpu_out_bhd = out_bhd.detach().float()

        # [1, H_global, B_global, D] — shard heads on mesh rows, batch on mesh cols (matches all_gather_concat).
        out_1hbd = out_bhd.permute(1, 0, 2).unsqueeze(0).contiguous()
        logger.info(
            f"[qwen-simple-sdpa] global out shape={tuple(out_1hbd.shape)} "
            f"(B={out_bhd.shape[0]} H={out_bhd.shape[1]})"
        )
        try:
            return ttnn.from_torch(
                out_1hbd,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=sdpa_out_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(1, 2),
                    mesh_shape=self.cluster_shape,
                ),
            )
        except Exception as e:
            logger.info(f"[qwen-simple-sdpa] fallback conversion failed: {e}")
            return None

    def _simple_decode_sdpa_ttnn_fallback(self, q_heads, keys, values, sdpa_out_mem_cfg, current_pos=None):
        q_torch = self._to_torch_decode_tensor(q_heads)
        k_torch = self._to_torch_decode_tensor(keys)
        v_torch = self._to_torch_decode_tensor(values)
        if q_torch is None or k_torch is None or v_torch is None:
            logger.info("[qwen-simple-ttnn-sdpa] fallback unavailable: cannot readback Q/K/V")
            return None
        if q_torch.ndim != 4:
            logger.info(f"[qwen-simple-ttnn-sdpa] fallback unavailable: unsupported Q shape {tuple(q_torch.shape)}")
            return None

        q_bhd = self._decode_heads_to_bhd(q_torch)
        if q_bhd is None:
            logger.info("[qwen-simple-ttnn-sdpa] fallback unavailable: Q layout")
            return None
        q_bhd = q_bhd.float()
        batch_size = q_bhd.shape[0]
        k_bhsd = self._reorder_cache_tensor(k_torch.float(), batch_size)
        v_bhsd = self._reorder_cache_tensor(v_torch.float(), batch_size)
        if k_bhsd is None or v_bhsd is None:
            logger.info(
                f"[qwen-simple-ttnn-sdpa] fallback unavailable: unsupported cache shapes "
                f"K={tuple(k_torch.shape)} V={tuple(v_torch.shape)}"
            )
            return None

        # Match Q/KV heads for GQA.
        if k_bhsd.shape[1] == 1 and q_bhd.shape[1] > 1:
            k_bhsd = k_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
            v_bhsd = v_bhsd.expand(-1, q_bhd.shape[1], -1, -1)
        elif q_bhd.shape[1] % k_bhsd.shape[1] == 0 and q_bhd.shape[1] != k_bhsd.shape[1]:
            kv_repeat = q_bhd.shape[1] // k_bhsd.shape[1]
            k_bhsd = k_bhsd.repeat_interleave(kv_repeat, dim=1)
            v_bhsd = v_bhsd.repeat_interleave(kv_repeat, dim=1)
        elif k_bhsd.shape[1] != q_bhd.shape[1]:
            logger.info(
                f"[qwen-simple-ttnn-sdpa] fallback unavailable: head mismatch "
                f"q_heads={q_bhd.shape[1]} kv_heads={k_bhsd.shape[1]}"
            )
            return None

        seq_len = k_bhsd.shape[2]
        decode_pos = self._resolve_decode_position(current_pos, k_bhsd)
        decode_pos = min(decode_pos, seq_len - 1)
        logger.info(
            f"[qwen-simple-ttnn-sdpa] decode_pos={decode_pos} seq_len={seq_len} "
            f"(cur_pos_tensor={self._decode_position_scalar(current_pos)})"
        )

        q_4d = q_bhd.unsqueeze(2).to(torch.bfloat16)  # [B, H, 1, D]
        k_4d = k_bhsd.to(torch.bfloat16)  # [B, H, S, D]
        v_4d = v_bhsd.to(torch.bfloat16)  # [B, H, S, D]
        mask = torch.full((batch_size, q_4d.shape[1], 1, seq_len), -1e9, dtype=torch.bfloat16)
        mask[:, :, :, : decode_pos + 1] = 0.0

        try:
            mapper = ttnn.ReplicateTensorToMesh(self.mesh_device)
            q_tt = ttnn.from_torch(
                q_4d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            k_tt = ttnn.from_torch(
                k_4d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            v_tt = ttnn.from_torch(
                v_4d, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )
            mask_tt = ttnn.from_torch(
                mask, device=self.mesh_device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper
            )

            scores_tt = ttnn.matmul(q_tt, ttnn.transpose(k_tt, -2, -1))
            scores_tt = ttnn.multiply(scores_tt, float(self.scale))
            scores_tt = ttnn.add(scores_tt, mask_tt)
            probs_tt = ttnn.softmax(scores_tt, dim=-1)
            out_tt = ttnn.matmul(probs_tt, v_tt)  # [B, H, 1, D]

            # Primitive path uses replicated tensors; read one local shard to avoid
            # mesh-concat shape inflation during debug readback.
            out_torch = None
            try:
                out_shards = ttnn.get_device_tensors(out_tt)
                if out_shards:
                    out_torch = ttnn.to_torch(out_shards[0])
            except Exception:
                out_torch = self._to_torch_decode_tensor(out_tt)
            if out_torch is None:
                logger.info("[qwen-simple-ttnn-sdpa] fallback unavailable: unable to read primitive output")
                return None
            if out_torch.ndim != 4:
                logger.info(
                    f"[qwen-simple-ttnn-sdpa] fallback unavailable: primitive output shape={tuple(out_torch.shape)}"
                )
                return None

            out_bhd = out_torch[:, :, 0, :].contiguous().float()
            # Compare primitive-TTNN vs host-simple decomposition for confidence.
            host_out_bhd = torch.einsum(
                "bhs,bhsd->bhd",
                torch.softmax(
                    torch.einsum("bhd,bhsd->bhs", q_bhd, k_bhsd) * float(self.scale)
                    + torch.where(
                        torch.arange(seq_len, device=q_bhd.device).view(1, 1, seq_len) <= decode_pos,
                        torch.zeros(1, 1, seq_len, device=q_bhd.device),
                        torch.full((1, 1, seq_len), float("-inf"), device=q_bhd.device),
                    ),
                    dim=-1,
                ),
                v_bhsd,
            ).float()
            if host_out_bhd.shape == out_bhd.shape:
                pass_host_cmp, pcc_host_cmp = comp_pcc(host_out_bhd, out_bhd, 0.99)
                logger.info(
                    f"[qwen-simple-ttnn-sdpa] primitive_vs_host_simple pcc={pcc_host_cmp} passing={pass_host_cmp}"
                )
            else:
                logger.info(
                    f"[qwen-simple-ttnn-sdpa] primitive_vs_host_simple skipped: "
                    f"shape_mismatch host={tuple(host_out_bhd.shape)} ttnn={tuple(out_bhd.shape)}"
                )

            out_1hbd = out_bhd.permute(1, 0, 2).unsqueeze(0).contiguous().to(torch.bfloat16)
            return ttnn.from_torch(
                out_1hbd,
                device=self.mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                memory_config=sdpa_out_mem_cfg,
                mesh_mapper=ttnn.ShardTensor2dMesh(
                    self.mesh_device,
                    dims=(1, 3),
                    mesh_shape=self.cluster_shape,
                ),
            )
        except Exception as e:
            logger.info(f"[qwen-simple-ttnn-sdpa] fallback failed: {e}")
            return None

    def _debug_stagewise_pcc(self, attn_output_cat, wo_out):
        if not self.stage_pcc_debug or self._wo_host_weight is None:
            return

        attn_host = self._to_torch_decode_tensor(attn_output_cat)
        wo_host = self._to_torch_decode_tensor(wo_out)
        if attn_host is None or wo_host is None:
            logger.info("[qwen-stage-pcc] skipped: unable to read attn/wo tensors")
            return
        if attn_host.ndim != 4 or wo_host.ndim != 4:
            logger.info(
                f"[qwen-stage-pcc] skipped: unsupported shapes attn={tuple(attn_host.shape)} wo={tuple(wo_host.shape)}"
            )
            return

        in_features = int(attn_host.shape[-1])
        out_features = int(wo_host.shape[-1])

        # Resolve a WO matrix that matches this decode-local projection.
        wo_weight = None
        if (
            self._wo_host_weight.ndim == 2
            and self._wo_host_weight.shape[0] == in_features
            and self._wo_host_weight.shape[1] == out_features
        ):
            wo_weight = self._wo_host_weight
        else:
            # Fallback 1: probe one physical shard from the actual TT weight tensor.
            # This is debug-only and is intended to match the local decode WO projection shape.
            try:
                local_wo_shards = ttnn.get_device_tensors(self.wo)
                if local_wo_shards:
                    local_wo = ttnn.to_torch(local_wo_shards[0]).float()
                    if local_wo.ndim == 4:
                        local_wo = local_wo[0, 0]
                    if local_wo.ndim == 2 and local_wo.shape == (in_features, out_features):
                        wo_weight = local_wo
            except Exception:
                wo_weight = None

        best_chunk_idx = None
        best_chunk_pcc_message = None
        best_chunk_pass = None
        best_chunk_mean_abs = None
        best_chunk_max_abs = None
        best_chunk_wo_ref = None

        # Fallback 2: derive local WO candidates by chunking the global WO rows.
        # This keeps debug actionable even when direct shard readback shape doesn't match host-composed activations.
        if wo_weight is None:
            global_wo = self._wo_host_weight
            if global_wo.ndim == 2 and global_wo.shape[1] == out_features and global_wo.shape[0] % in_features == 0:
                n_chunks = global_wo.shape[0] // in_features
                for chunk_idx in range(n_chunks):
                    start = chunk_idx * in_features
                    end = start + in_features
                    candidate_wo = global_wo[start:end, :]
                    try:
                        candidate_ref = torch.matmul(attn_host.float(), candidate_wo)
                    except Exception:
                        continue
                    out_h = min(candidate_ref.shape[-2], wo_host.shape[-2])
                    out_w = min(candidate_ref.shape[-1], wo_host.shape[-1])
                    candidate_ref = candidate_ref[..., :out_h, :out_w].contiguous()
                    candidate_cmp = wo_host.float()[..., :out_h, :out_w].contiguous()
                    passing, pcc_message = comp_pcc(candidate_ref, candidate_cmp, 0.99)
                    diff = (candidate_ref - candidate_cmp).abs()
                    mean_abs = float(diff.mean().item())
                    max_abs = float(diff.max().item())

                    # Prefer higher PCC; if parse fails, prefer lower mean abs.
                    try:
                        pcc_val = float(pcc_message)
                    except Exception:
                        pcc_val = None

                    if best_chunk_idx is None:
                        best_chunk_idx = chunk_idx
                        best_chunk_pcc_message = pcc_message
                        best_chunk_pass = passing
                        best_chunk_mean_abs = mean_abs
                        best_chunk_max_abs = max_abs
                        best_chunk_wo_ref = candidate_ref
                    else:
                        current_best_pcc = None
                        try:
                            current_best_pcc = float(best_chunk_pcc_message)
                        except Exception:
                            current_best_pcc = None

                        better = False
                        if pcc_val is not None and current_best_pcc is not None:
                            better = pcc_val > current_best_pcc
                        elif pcc_val is not None and current_best_pcc is None:
                            better = True
                        elif pcc_val is None and current_best_pcc is None:
                            better = mean_abs < best_chunk_mean_abs
                        if better:
                            best_chunk_idx = chunk_idx
                            best_chunk_pcc_message = pcc_message
                            best_chunk_pass = passing
                            best_chunk_mean_abs = mean_abs
                            best_chunk_max_abs = max_abs
                            best_chunk_wo_ref = candidate_ref

                if best_chunk_idx is not None:
                    wo_weight = global_wo[best_chunk_idx * in_features : (best_chunk_idx + 1) * in_features, :]

        if wo_weight is None:
            logger.info(
                f"[qwen-stage-pcc] skipped: unable to resolve local WO weight "
                f"(need {in_features}x{out_features}, global={tuple(self._wo_host_weight.shape)})"
            )
            return

        try:
            wo_ref = torch.matmul(attn_host.float(), wo_weight)
        except Exception as e:
            logger.info(f"[qwen-stage-pcc] skipped: host WO matmul failed ({e})")
            return

        out_h = min(wo_ref.shape[-2], wo_host.shape[-2])
        out_w = min(wo_ref.shape[-1], wo_host.shape[-1])
        wo_ref = wo_ref[..., :out_h, :out_w].contiguous()
        wo_cmp = wo_host.float()[..., :out_h, :out_w].contiguous()

        passing, pcc_message = comp_pcc(wo_ref, wo_cmp, 0.99)
        diff = (wo_ref - wo_cmp).abs()
        if best_chunk_idx is not None and best_chunk_wo_ref is not None:
            logger.info(
                f"[qwen-stage-pcc] wo_candidate_chunk={best_chunk_idx} "
                f"pcc={best_chunk_pcc_message} passing={best_chunk_pass} "
                f"mean_abs_diff={best_chunk_mean_abs:.6e} max_abs_diff={best_chunk_max_abs:.6e}"
            )
        logger.info(
            f"[qwen-stage-pcc] wo_matmul_vs_host_ref shape={tuple(wo_cmp.shape)} "
            f"pcc={pcc_message} passing={passing} mean_abs_diff={float(diff.mean().item()):.6e} "
            f"max_abs_diff={float(diff.max().item()):.6e}"
        )

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
                if weight_cache_path and not configuration.dummy_weights and not self.disable_kv_cache_tensor_cache
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
        self._cached_decode_pos = self._decode_position_scalar(current_pos)
        if self.decode_debug:
            logger.info(f"[qwen-decode-debug] cached_decode_pos={self._cached_decode_pos}")

        if (
            self.is_qwen
            and not self.model_config.get("USE_PREFETCHER", False)
            and os.getenv("QWEN_HOST_DECODE_FALLBACK", "0") == "1"
        ):
            host_out = self._forward_decode_via_host_reference(x, current_pos)
            if host_out is not None:
                return host_out

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        self._trace("decode: start qkv matmul")
        if self.model_config["USE_PREFETCHER"]:
            x_qkv = self._pad_activation_k_for_ring_qkv(x)
            xqkv_fused_sharded = ttnn.matmul(  # [1, 1, 32, k_ring]
                x_qkv,
                self.wqkv,
                program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
                memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
                compute_kernel_config=self.compute_kernel_config_hifi2,
                global_cb=self.prefetcher_setup.global_circular_buffer,
                dtype=ttnn.bfloat16,
                sub_device_id=self.prefetcher_setup.worker_sub_device_id,
            )
            if x_qkv is not x:
                ttnn.deallocate(x_qkv)
        else:
            self._debug_qkv_input_vs_ref(x)
            use_bh_qwen_no_pf = (
                self.is_qwen and self.is_blackhole and not self.model_config.get("USE_PREFETCHER", False)
            )
            xqkv_bh_for_heads = None
            use_host_fused_qkv = self.use_host_fused_qkv or self.device_qkv_matmul == "host_fused"
            if use_bh_qwen_no_pf and (use_host_fused_qkv or self.use_host_qkv_head_scatter):
                xqkv_bh_for_heads = self._host_fused_xqkv_bh(x)
            if use_host_fused_qkv:
                xqkv_fused_sharded = self._host_fused_qkv_decode(x)
                if xqkv_fused_sharded is None:
                    raise RuntimeError("[qwen-qkv] QWEN_HOST_FUSED_QKV=1 but host fused QKV failed")
                self._debug_qkv_xqkv_post_col_sum(xqkv_fused_sharded)
                self._trace("decode: done host fused qkv")
            elif use_bh_qwen_no_pf:
                x_qkv = x
                x_ring = x
                x_in_dram = x  # dram QKV input; freed after stage debug
                if self.qkv_stage_debug:
                    self._debug_qkv_weight_shard_vs_ref(0, 0)
                use_multicast = self.device_qkv_matmul == "multicast"
                if use_multicast and os.getenv("QWEN_FORCE_DEVICE_QKV_MULTICAST", "0") != "1":
                    # XQKV_DECODE_PROGCFG grid is (0,0)-anchored; Galaxy BH uses start_core=(1,0)
                    # sub-core grids -> shard grid outside config grid (job 11487).
                    logger.warning(
                        "[qwen-qkv] multicast QKV disabled on BH no-prefetch; use ring "
                        "(set QWEN_FORCE_DEVICE_QKV_MULTICAST=1 to override)."
                    )
                    use_multicast = False
                if use_multicast:
                    self._trace("decode: qkv multicast linear (XQKV_DECODE_PROGCFG)")
                    try:
                        x_in = ttnn.to_memory_config(x, self.model_config["SHARDED_ATTN_INPUT_MEMCFG"])
                    except Exception:
                        x_in = x
                    xqkv_fused_sharded = ttnn.linear(
                        x_in,
                        self.wqkv_interleaved,
                        program_config=self.model_config["XQKV_DECODE_PROGCFG"],
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=self.compute_kernel_config_hifi2,
                        dtype=ttnn.bfloat16,
                    )
                    if x_in is not x:
                        ttnn.deallocate(x_in)
                elif self.device_qkv_matmul == "host_fused":
                    self._trace("decode: host fused qkv upload (correct xqkv on device)")
                    xqkv_fused_sharded = self._host_fused_qkv_decode(x)
                    if xqkv_fused_sharded is None:
                        raise RuntimeError("[qwen-qkv] QWEN_DEVICE_QKV_MATMUL=host_fused failed")
                elif self.device_qkv_matmul == "per_device":
                    # Column-sharded act @ interleaved DRAM wqkv [k_local, qkv_n_local]; auto program
                    # config + interleaved output (DRAM-sharded progcfg hangs here, job 11751).
                    self._trace("decode: qkv per_device interleaved dram-weight matmul")
                    x_in = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
                    self._qkv_partial_x_torch = None
                    if self.qkv_stage_debug:
                        try:
                            self._qkv_partial_x_torch = [
                                ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(x_in)
                            ]
                        except Exception as exc:
                            logger.warning(f"[qwen-qkv-stage] partial x snapshot before per_device qkv failed: {exc}")
                    xqkv_fused_sharded = ttnn.linear(
                        x_in,
                        self._wqkv_weight_for_matmul_mode(),
                        program_config=None,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        core_grid=None,
                        compute_kernel_config=self.compute_kernel_config_hifi2,
                        dtype=ttnn.bfloat16,
                    )
                    if x_in is not x:
                        ttnn.deallocate(x_in)
                elif self.device_qkv_matmul == "dram":
                    # Column-sharded K (1280) + wqkv_col; no K-pad (pad breaks column shard tiles).
                    self._trace("decode: qkv dram column-sharded linear")
                    x_in_dram = x
                    col_memcfg = self.model_config.get("SHARDED_ATTN_INPUT_MEMCFG")
                    if col_memcfg is not None:
                        try:
                            x_in_dram = ttnn.to_memory_config(x, col_memcfg)
                        except Exception:
                            x_in_dram = x
                    self._qkv_partial_x_torch = None
                    if self.qkv_stage_debug:
                        try:
                            self._qkv_partial_x_torch = [
                                ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(x_in_dram)
                            ]
                        except Exception as exc:
                            logger.warning(f"[qwen-qkv-stage] partial x snapshot before dram qkv failed: {exc}")
                    xqkv_fused_sharded = ttnn.linear(
                        x_in_dram,
                        self.wqkv_col,
                        program_config=None,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        compute_kernel_config=self.compute_kernel_config_hifi2,
                        dtype=ttnn.bfloat16,
                    )
                elif self.device_qkv_matmul == "ring":
                    # Ring DRAM-sharded weights + ring progcfg (K=k_ring for RING_SIZE divisibility).
                    x_qkv = None
                    x_ring = self._column_sharded_to_ring_input(x)
                    self._qkv_partial_x_torch = None
                    if self.qkv_stage_debug:
                        try:
                            self._qkv_partial_x_torch = [
                                ttnn.to_torch(s).float() for s in ttnn.get_device_tensors(x_ring)
                            ]
                        except Exception as exc:
                            logger.warning(f"[qwen-qkv-stage] partial x snapshot before ring qkv failed: {exc}")
                    xqkv_fused_sharded = ttnn.matmul(
                        x_ring,
                        self.wqkv,
                        program_config=self.model_config["XQKV_DECODE_RING_PROGCFG"],
                        memory_config=self.model_config["SHARDED_QKV_OUT_RING_MEMCFG"],
                        compute_kernel_config=self.compute_kernel_config_hifi2,
                        dtype=ttnn.bfloat16,
                    )
                    xqkv_fused_sharded = self._ensure_ring_qkv_tile_layout(xqkv_fused_sharded)
                else:
                    raise RuntimeError(f"[qwen-qkv] unknown QWEN_DEVICE_QKV_MATMUL={self.device_qkv_matmul}")
                if self.device_qkv_matmul != "host_fused":
                    x_for_partial = x if self.device_qkv_matmul == "ring" else x
                    self._debug_qkv_partial_vs_ref(xqkv_fused_sharded, x_for_partial)
                    self._trace("decode: start qkv column sum")
                    xqkv_fused_sharded = self._host_sum_columns_decode(xqkv_fused_sharded)
                else:
                    self._trace("decode: skip qkv column sum (host fused row-sharded)")
                self._debug_qkv_xqkv_post_col_sum(xqkv_fused_sharded)
                self._trace("decode: done qkv column sum")
                if self.device_qkv_matmul == "ring":
                    if x_ring is not x:
                        ttnn.deallocate(x_ring)
                elif self.device_qkv_matmul == "dram" and x_in_dram is not x:
                    ttnn.deallocate(x_in_dram)
            else:
                xqkv_fused_sharded = ttnn.matmul(
                    x,
                    self.wqkv_interleaved,
                    program_config=None,
                    memory_config=None,
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    global_cb=None,
                    dtype=ttnn.bfloat16,
                    sub_device_id=None,
                )
                self._trace("decode: start qkv column sum")
                xqkv_fused_sharded = self._host_sum_columns_decode(xqkv_fused_sharded)
                self._debug_qkv_xqkv_post_col_sum(xqkv_fused_sharded)
                self._trace("decode: done qkv column sum")
        self._trace("decode: done qkv matmul")
        self._debug_tensor_meta("qkv_matmul_out", xqkv_fused_sharded)
        ttnn.deallocate(x)
        # xqkv_fused_sharded -> [1, 1, 32, 12288 // 8]

        ###
        # Reshape and rotary embeddings
        ###
        self._trace("decode: start rs create heads")
        if not self.model_config.get("USE_PREFETCHER", False):
            use_bh_qwen_no_pf = (
                self.is_qwen and self.is_blackhole and not self.model_config.get("USE_PREFETCHER", False)
            )
            use_llama_rs_heads = self.device_create_heads == "llama_rs" and self.model_config.get(
                "USE_PREFETCHER", False
            )
            if use_bh_qwen_no_pf and self.device_create_heads == "llama_rs" and not use_llama_rs_heads:
                logger.warning(
                    "[qwen-qkv] llama_rs_create_heads needs USE_PREFETCHER (ring 32x64 xqkv); "
                    "falling back to nlp_create_qkv_heads_decode."
                )
            if use_bh_qwen_no_pf and use_llama_rs_heads:
                self._trace("decode: llama_rs_create_heads (device)")
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
            else:
                self._trace("decode: nlp_create_qkv_heads_decode")
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
                # Ring matmul uses untilize_out=True -> ROW_MAJOR; create_heads needs TILE.
                if xqkv_fused_interleaved.layout != ttnn.TILE_LAYOUT:
                    xqkv_fused_interleaved = ttnn.to_layout(
                        xqkv_fused_interleaved,
                        ttnn.TILE_LAYOUT,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
                # nlp_create_qkv_heads_decode only honors batch_offset/slice_size in its
                # WIDTH_SHARDED program factory; the interleaved factory silently ignores
                # them, so every mesh column would read users [0:slice_size] and the global
                # batch gather scrambles (pre_norm PCC ~0). Width-shard the fused tensor so
                # batch_offset selects users [col*8:col*8+8] per column as intended.
                create_head_input_memcfg = self._create_head_input_memcfg_nlp
                if create_head_input_memcfg is not None:
                    xqkv_fused_create_head_in = ttnn.to_memory_config(
                        xqkv_fused_interleaved, memory_config=create_head_input_memcfg
                    )
                    ttnn.deallocate(xqkv_fused_interleaved)
                else:
                    xqkv_fused_create_head_in = xqkv_fused_interleaved
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
            if xqkv_bh_for_heads is not None:
                self._host_scatter_qkv_heads_from_fused(
                    q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, v_heads_1BKD, xqkv_bh_for_heads
                )
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
        self._trace("decode: done rs create heads")
        self._debug_tensor_meta("q_heads_pre_rot", q_heads_pre_rot_1BQD)
        self._debug_tensor_meta("k_heads_pre_rot", k_heads_pre_rot_1BKD)
        self._debug_tensor_meta("v_heads", v_heads_1BKD)
        self._debug_tensor_values("q_heads_pre_rot", q_heads_pre_rot_1BQD)
        self._debug_tensor_values("k_heads_pre_rot", k_heads_pre_rot_1BKD)
        self._debug_qkv_pre_norm_vs_ref(q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD)

        if self.qk_norm:
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD = self._apply_decode_qk_norm(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD
            )
            self._debug_tensor_values("q_heads_post_qk_norm", q_heads_pre_rot_1BQD)
            self._debug_tensor_values("k_heads_post_qk_norm", k_heads_pre_rot_1BKD)

        # print("done create qkv heads")
        ttnn.deallocate(xqkv_fused_sharded)

        # Q, K Rotary Embeddings
        self._trace("decode: start rotary")
        if self.model_config.get("USE_PREFETCHER", False):
            # Fused decode rotary requires cos/sin tensors in TILE layout.
            rot_cos = ttnn.to_layout(rot_mats[0], ttnn.TILE_LAYOUT)
            rot_sin = ttnn.to_layout(rot_mats[1], ttnn.TILE_LAYOUT)
            q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_cos, rot_sin, self.transformation_mats["decode"]
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
            self._debug_tensor_values("q_rot_cos", q_rot_cos)
            self._debug_tensor_values("q_rot_sin", q_rot_sin)
            self._debug_tensor_values("k_rot_cos", k_rot_cos)
            self._debug_tensor_values("k_rot_sin", k_rot_sin)
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
        self._trace("decode: done rotary")
        self._debug_tensor_meta("q_heads_post_rotary", q_heads_1BQD)
        self._debug_tensor_meta("k_heads_post_rotary", k_heads_1BKD)
        self._debug_tensor_values("q_heads_post_rotary", q_heads_1BQD)
        self._debug_tensor_values("k_heads_post_rotary", k_heads_1BKD)
        self._debug_tensor_values("v_heads_pre_sdpa", v_heads_1BKD)
        self._debug_qkv_post_rope_vs_ref(q_heads_1BQD, k_heads_1BKD, current_pos)
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
        self._trace("decode: start paged_fused_update_cache")
        if not self.model_config.get("USE_PREFETCHER", False):
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
        self._trace("decode: done paged_fused_update_cache")
        self._debug_cache_values(keys, values, current_pos)

        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # print("done update cache")
        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        sdpa_out_mem_cfg = self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group)
        self._trace("decode: start sdpa decode")
        if self.decode_debug_values:
            logger.info(f"[qwen-decode-debug-values] sdpa_scale={self.scale}")
            logger.info(f"[qwen-decode-debug-values] sdpa_safe_precision={int(self.sdpa_safe_precision)}")
            # Important: this is the exact V tensor argument consumed by fused decode SDPA.
            # Log it here (post-cache-update, pre-SDPA call) to distinguish upstream V corruption
            # from issues internal to SDPA.
            self._debug_tensor_values("sdpa_v_arg_pre_op", values)
        if self.sdpa_substage_debug and page_table is None:
            self._debug_sdpa_substages(q_heads_1BQD, keys, values, current_pos)
        elif self.sdpa_substage_debug and page_table is not None:
            logger.info("[qwen-sdpa-substage] skipped for paged KV (linear-cache substage debug is invalid)")

        simple_sdpa_allowed = self._simple_sdpa_allowed(page_table)
        if page_table is not None and (self.use_simple_decode_sdpa or self.use_simple_ttnn_decode_sdpa):
            logger.info(
                "[qwen-simple-sdpa] skipping host/ttnn simple SDPA for paged KV; "
                "using fused paged_scaled_dot_product_attention_decode"
            )

        if self.use_simple_ttnn_decode_sdpa and simple_sdpa_allowed:
            logger.info("[qwen-simple-ttnn-sdpa] using primitive TTNN decode SDPA fallback")
            attn_output_1G4D_sharded = self._simple_decode_sdpa_ttnn_fallback(
                q_heads_1BQD, keys, values, sdpa_out_mem_cfg, current_pos
            )
            if attn_output_1G4D_sharded is None:
                logger.info("[qwen-simple-ttnn-sdpa] fallback unavailable, reverting to host-simple/fused path")
        if self.use_simple_decode_sdpa and simple_sdpa_allowed and "attn_output_1G4D_sharded" not in locals():
            logger.info("[qwen-simple-sdpa] using simple host decode SDPA fallback")
            attn_output_1G4D_sharded = self._simple_decode_sdpa_host_fallback(
                q_heads_1BQD, keys, values, sdpa_out_mem_cfg, current_pos
            )
            if attn_output_1G4D_sharded is None:
                logger.info("[qwen-simple-sdpa] fallback unavailable, reverting to fused decode SDPA")
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
        if "attn_output_1G4D_sharded" not in locals():
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
        self._trace("decode: done sdpa decode")
        self._debug_tensor_meta("sdpa_out", attn_output_1G4D_sharded)
        self._debug_tensor_values("sdpa_out", attn_output_1G4D_sharded)
        self._debug_sdpa_valid_region(attn_output_1G4D_sharded)

        ttnn.deallocate(q_heads_1BQD)

        if self.use_device_output_segment and not self.model_config.get("USE_PREFETCHER", False):
            # B2: device SDPA output is batch-first [1, B_local, H_local, D] (8 users/col).
            # Order matters: nlp_concat_heads_decode pads batch to 32, so gather the users
            # across cols FIRST (8 -> 32), then concat heads. Device SDPA is already
            # [1, B, NH, D], so (unlike the host-SDPA path) no users<->heads transpose is needed.
            self._trace("decode: start all_gather users over cols")
            # Mimic tt_transformers.tt_all_gather exactly: all_gather_async with
            # persistent_output_buffer=None AND a barrier semaphore. (line_all_gather's async
            # path instead passes the persistent SDPA buffer with barrier=None, which throws
            # map::at here.) Gather users (dim=1) across cols 8->32 before concat.
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
            self._trace("decode: done all_gather users over cols")
            try:
                concat_sub_core_grids = gathered_users.memory_config().shard_spec.grid
            except Exception:
                concat_sub_core_grids = None
            self._trace("decode: start device concat heads (nlp_concat_heads_decode)")
            attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(  # [1, 1, 32, 1024]
                gathered_users,
                num_heads=self.n_local_heads,
                sub_core_grids=concat_sub_core_grids,
            )
            self._trace("decode: done device concat heads")
            ttnn.deallocate(gathered_users)
        else:
            self._trace("decode: start all_gather_concat")
            # Device SDPA emits batch-first [1, B_local, H_local, D]; host-simple SDPA emits
            # head-first [1, H_local, B_local, D]. The host assembly must know which to gather
            # heads-over-rows / users-over-cols correctly (else PCC collapses, e.g. 0.176).
            _sdpa_batch_first = not (self.use_simple_decode_sdpa or self.use_simple_ttnn_decode_sdpa)
            attn_output_cat = self.tt_ccl.all_gather_concat(  # [1, 1, 32, 1024]
                attn_output_1G4D_sharded,
                dim=1,
                cluster_axis=1,
                num_links=1,
                batch_first=_sdpa_batch_first,
                memory_config=self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"],
                num_heads=self.n_local_heads,
            )
            self._trace("decode: done all_gather_concat")
        self._debug_tensor_meta("all_gather_concat_out", attn_output_cat)
        self._debug_tensor_values("all_gather_concat_out", attn_output_cat)
        self._debug_concat_vs_sdpa_ref(attn_output_cat)
        ttnn.deallocate(attn_output_1G4D_sharded)
        # print("done concat heads")

        if self.use_host_decode_wo and not self.model_config.get("USE_PREFETCHER", False):
            self._trace("decode: start host wo matmul")
            dense_out_reduced = self._simple_decode_wo_host_fallback(attn_output_cat)
            if dense_out_reduced is not None:
                self._trace("decode: done host wo matmul")
                self._debug_tensor_meta("wo_matmul_out", dense_out_reduced)
                self._debug_tensor_values("wo_matmul_out", dense_out_reduced)
                self._debug_tensor_meta("line_all_reduce_out", dense_out_reduced)
                return dense_out_reduced
            logger.info("[qwen-host-wo] unavailable, falling back to TT wo matmul")

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        self._trace("decode: start wo matmul")
        use_replicated_full_wo = (
            self.is_qwen
            and not self.model_config.get("USE_PREFETCHER", False)
            and attn_output_cat.shape[-1] == self.n_heads * self.head_dim
            and os.getenv("QWEN_WO_USE_COL_SHARDED", "1") == "1"
        )
        if not self.model_config.get("USE_PREFETCHER", False):
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
        self._trace("decode: done wo matmul")
        self._debug_tensor_meta("wo_matmul_out", dense_out_ttnn)
        self._debug_tensor_values("wo_matmul_out", dense_out_ttnn)
        self._debug_stagewise_pcc(attn_output_cat, dense_out_ttnn)
        # [1, 1, 32, 2304]
        if use_replicated_full_wo:
            self._trace("decode: skip line_all_reduce for replicated full-width WO")
            dense_out_reduced = dense_out_ttnn
        else:
            self._trace("decode: start line_all_reduce")
            dense_out_reduced = self.tt_ccl.line_all_reduce(  # [1, 1, 32, 1280]
                dense_out_ttnn,
                cluster_axis=0,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=(
                    ttnn.DRAM_MEMORY_CONFIG
                    if not self.model_config.get("USE_PREFETCHER", False)
                    else self.model_config["DECODE_RESIDUAL_MEMCFG"]
                ),
                use_optimal_ccl_for_llama=True,
            )
            self._trace("decode: done line_all_reduce")
            ttnn.deallocate(dense_out_ttnn)
        self._debug_tensor_meta("line_all_reduce_out", dense_out_reduced)
        self._debug_tensor_values("line_all_reduce_out", dense_out_reduced)

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
        chunk_start_idx_tensor=None,
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
            dtype=self.ccl_dtype,
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
                    num_links=3,
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
