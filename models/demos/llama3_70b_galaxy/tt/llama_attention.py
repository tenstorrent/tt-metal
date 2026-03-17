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
        self.n_local_heads_padded = 8 if (self.is_olmo and self.n_local_heads < 8) else self.n_local_heads
        self.cluster_shape = configuration.cluster_shape
        self.capture_intermediates = False  # Set by parent decoder for PCC debug
        self.captured = {}  # Stores reconstructed torch tensors

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

        qkv_list_unpadded = []
        qkv_list_padded = []
        need_padding = self.is_olmo and self.n_local_heads < 8
        for i in range(self.num_devices_per_group):
            wq_selected = torch.chunk(self.state_dict[wq_str], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(self.state_dict[wk_str], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(self.state_dict[wv_str], self.num_devices_per_group, dim=0)[i]

            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)
            qkv_list_unpadded.append(torch.cat([wq, wk, wv], dim=-1))

            if need_padding:
                pad_heads = 8 - self.n_local_heads
                pad_size = pad_heads * self.head_dim
                wq_padded = torch.transpose(
                    torch.nn.functional.pad(wq_selected, (0, 0, 0, pad_size), value=0.0), -2, -1
                )
                qkv_list_padded.append(torch.cat([wq_padded, wk, wv], dim=-1))

        qkv_cat_unpadded = torch.cat(qkv_list_unpadded, dim=-1).unsqueeze(0).unsqueeze(0)
        qkv_cat_padded = (
            torch.cat(qkv_list_padded, dim=-1).unsqueeze(0).unsqueeze(0) if need_padding else qkv_cat_unpadded
        )
        if self.is_olmo:
            print(f"OLMo qkv_cat unpadded: {qkv_cat_unpadded.shape}, padded: {qkv_cat_padded.shape}")

        # Ring stuff
        # Llama3: 9216, 12288
        # Qwen3: 6144, 12288

        # Llama3: [1, 1, 8192, 10240] -> [2304, 1536]
        # Qwen3: [1, 1, 5120, 10240] -> [1280, 1536]
        self.wqkv = ttnn.as_tensor(
            qkv_cat_padded,
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
            qkv_cat_unpadded,
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

        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        if f"{q_norm_str}.weight" in self.state_dict:
            self.qk_norm = True

            if self.is_olmo:
                # OLMo QK-norm: norm applied AFTER head split (post llama_rs_create_heads).
                # q_norm [5120] = 40 heads × 128, k_norm [1024] = 8 KV heads × 128.
                # Per device (8 TP on axis-0): 5 real Q heads + 3 padded, 1 K head.
                #
                # K norm is exact (1 head, weight [128]).
                # Q norm uses per-head normalization with correct per-head weights:
                #   reshape Q[1,8,32,128] → [1,1,256,128], rms_norm with ones[128],
                #   reshape back, multiply by per-head weight [1,8,1,128].
                SHARD_HEIGHT = 32
                q_norm_full = self.state_dict[q_norm_str + ".weight"]  # [5120]
                k_norm_full = self.state_dict[k_norm_str + ".weight"]  # [1024]
                n_real_q_heads = self.n_local_heads  # 5
                n_padded_q_heads = self.n_local_heads_padded  # 8
                n_kv_heads_local = self.n_local_kv_heads  # 1
                n_tp = self.num_devices_per_group  # 8

                # K norm weight: [1024] → per device [128] → format [1, 1, 4, 32]
                k_dim = k_norm_full.shape[0]
                k_norm_torch = (
                    k_norm_full.unsqueeze(0).view(1, 1, k_dim).reshape([1, 1, k_dim // SHARD_HEIGHT, SHARD_HEIGHT])
                )
                self.olmo_k_norm_weight = ttnn.as_tensor(
                    k_norm_torch,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(2, None), mesh_shape=configuration.cluster_shape
                    ),
                )

                # Q rms_norm "ones" weight: [128] replicated on all devices
                # Used for normalization step (weight=1 means just normalize, no scale)
                ones_torch = torch.ones(1, 1, self.head_dim // SHARD_HEIGHT, SHARD_HEIGHT)
                self.olmo_q_norm_ones = ttnn.as_tensor(
                    ones_torch,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                )

                # Q per-head scaling weight: [1, n_padded*n_tp, 1, 128] → shard to [1, 8, 1, 128] per device
                # Each device's 5 real heads get their trained weight; 3 padded heads get ones
                total_padded_heads = n_padded_q_heads * n_tp  # 64
                q_head_weight_torch = torch.ones(1, total_padded_heads, 1, self.head_dim)
                for dev in range(n_tp):
                    for h in range(n_real_q_heads):
                        global_head = dev * n_real_q_heads + h
                        local_idx = dev * n_padded_q_heads + h
                        w_start = global_head * self.head_dim
                        q_head_weight_torch[0, local_idx, 0, :] = q_norm_full[w_start : w_start + self.head_dim]
                self.olmo_q_head_weight = ttnn.as_tensor(
                    q_head_weight_torch,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(1, None), mesh_shape=configuration.cluster_shape
                    ),
                )

                # Q per-head scaling weight for prefill: [n_real*n_tp, head_dim] → shard to [n_real, head_dim] per device
                # Shape [1, n_real*n_tp, 1, head_dim] = [1, 40, 1, 128], sharded across row devices
                q_head_weight_prefill = torch.ones(1, n_real_q_heads * n_tp, 1, self.head_dim)
                for dev in range(n_tp):
                    for h in range(n_real_q_heads):
                        global_head = dev * n_real_q_heads + h
                        local_idx = dev * n_real_q_heads + h
                        w_start = global_head * self.head_dim
                        q_head_weight_prefill[0, local_idx, 0, :] = q_norm_full[w_start : w_start + self.head_dim]
                self.olmo_q_head_weight_prefill = ttnn.as_tensor(
                    q_head_weight_prefill,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(1, None), mesh_shape=configuration.cluster_shape
                    ),
                )
                self.olmo_q_per_device_dim = q_norm_full.shape[0] // n_tp  # 640
                self.olmo_k_per_device_dim = k_dim // n_tp  # 128

                # Store raw CPU weights for host-side global norm in prefill
                self.olmo_q_norm_cpu = q_norm_full.clone()  # [5120]
                self.olmo_k_norm_cpu = k_norm_full.clone()  # [1024]

                # Full-dim Q norm weight: normalize over all local Q dims (640 per row device)
                # Shape [1, 1, 5120//32, 32] = [1, 1, 160, 32], sharded to [1, 1, 20, 32] per row device
                # dims=(2, None): d0=2 shards over axis 0 (8 row devices) → 160/8=20 per device ✓
                q_dim = q_norm_full.shape[0]  # 5120
                q_norm_full_2d = q_norm_full.view(1, 1, q_dim // SHARD_HEIGHT, SHARD_HEIGHT)
                self.olmo_q_norm_weight_full_prefill = ttnn.as_tensor(
                    q_norm_full_2d,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(2, None), mesh_shape=configuration.cluster_shape
                    ),
                )

                # Full-dim K norm weight: normalize over all local K dims (128 per row device)
                # Shape [1, 1, 1024//32, 32] = [1, 1, 32, 32], sharded to [1, 1, 4, 32] per row device
                # dims=(2, None): d0=2 shards over axis 0 (8 row devices) → 32/8=4 per device ✓
                k_norm_full_2d = k_norm_full.view(1, 1, k_dim // SHARD_HEIGHT, SHARD_HEIGHT)
                self.olmo_k_norm_weight_full_prefill = ttnn.as_tensor(
                    k_norm_full_2d,
                    device=self.mesh_device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device, dims=(2, None), mesh_shape=configuration.cluster_shape
                    ),
                )
            else:
                # Qwen3-style per-head QK-norm (weight [head_dim]=128, applied after head split)
                self.reshape_intermediate_q_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(64, 128),
                    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(1, 0))]),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.reshape_intermediate_k_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(64, 128),
                    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(3, 0), ttnn.CoreCoord(3, 0))]),
                    strategy=ttnn.ShardStrategy.HEIGHT,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.reshape_output_q_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(64, 32),
                    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 0), ttnn.CoreCoord(2, 1))]),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                self.reshape_output_k_mem_cfg = ttnn.create_sharded_memory_config(
                    shape=(64, 32),
                    core_grid=ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(1, 2), ttnn.CoreCoord(2, 3))]),
                    strategy=ttnn.ShardStrategy.WIDTH,
                    orientation=ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                block_w = 128 // 4 // 32
                subblock_w = 1
                while subblock_w > 0:
                    if block_w % subblock_w == 0:
                        break
                    subblock_w -= 1
                self.norm_program_cfg = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=[2, 2],
                    subblock_w=subblock_w,
                    block_h=2,
                    block_w=block_w,
                    inplace=False,
                )
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

            ttnn_shape = list(tensor.shape)
            torch_tensor = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
            )
            has_inf = torch.isinf(torch_tensor).any().item()
            has_nan = torch.isnan(torch_tensor).any().item()
            max_val = torch_tensor.float().abs().max().item()
            status = "OK" if not (has_inf or has_nan) else "BAD"
            logger.info(
                f"    ATTN [{status}] {name}: ttnn_shape={ttnn_shape}, concat_shape={list(torch_tensor.shape)}, max={max_val:.4e}, Inf={has_inf}, NaN={has_nan}"
            )
        except Exception as e:
            from loguru import logger

            logger.error(f"    ATTN [ERROR] {name}: {e}")

    def _capture_attn(self, name, tensor):
        """Capture a tensor as CPU torch for PCC debug. Only active when capture_intermediates=True.

        Uses dims=(0, 1) to match unit test convention for create_heads output.
        Also logs the ttnn per-device shape for debugging.
        """
        if not self.capture_intermediates:
            return
        try:
            from loguru import logger

            logger.info(f"  [capture] {name}: ttnn_shape={list(tensor.shape)}")
            t = ttnn.to_torch(
                tensor,
                mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(0, 1), mesh_shape=self.cluster_shape),
            ).float()
            self.captured[name] = t
        except Exception as e:
            from loguru import logger

            logger.warning(f"  [attn capture] {name}: failed — {e}")

    def _capture_prefill_attn(self, name, tensor):
        """Capture a prefill-path tensor for PCC debug using per-device extraction."""
        if not self.capture_intermediates:
            return
        try:
            from loguru import logger

            logger.info(f"  [prefill_attn capture] {name}: ttnn_shape={list(tensor.shape)}")
            device_tensors = ttnn.get_device_tensors(tensor)
            per_dev = []
            for dt in device_tensors:
                per_dev.append(ttnn.to_torch(dt).float())
            n_rows, n_cols = self.cluster_shape
            # Arrange: device_tensors order is row-major: (0,0), (0,1), ..., (0,3), (1,0), ...
            grid = {}
            for idx, t in enumerate(per_dev):
                r, c = idx // n_cols, idx % n_cols
                grid[(r, c)] = t
            self.captured[name] = grid
            # Also log first device shape
            logger.info(f"    per-dev shape: {list(per_dev[0].shape)}, n_devices={len(per_dev)}")
        except Exception as e:
            from loguru import logger

            logger.warning(f"  [prefill_attn capture] {name}: failed — {e}")

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
        if x.get_layout() != ttnn.TILE_LAYOUT:
            x = ttnn.to_layout(x, ttnn.TILE_LAYOUT)

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
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
        if self.is_olmo and self.qk_norm:
            # OLMo QK-norm: keep fused reduce+heads, apply norm post-split.
            # llama_rs_create_heads → Q[1,8,32,128], K[1,1,32,128], V[1,1,32,128]
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
            ttnn.deallocate(xqkv_fused_sharded)

            # Save original non-overlapping memory configs (from llama_rs_create_heads)
            q_mem_cfg = q_heads_pre_rot_1BQD.memory_config()
            k_mem_cfg = k_heads_pre_rot_1BKD.memory_config()

            self._capture_attn("q_pre_norm", q_heads_pre_rot_1BQD)
            self._capture_attn("k_pre_norm", k_heads_pre_rot_1BKD)
            self._capture_attn("v_heads", v_heads_1BKD)

            # ---- K global norm: L1 INTERLEAVED (no DRAM roundtrip) ----
            # K exits create_heads as [1,8,1,128] HEIGHT_SHARDED in L1 with [1,128] shard (2KB total).
            # Move to L1 INTERLEAVED before to_layout (sub-tile [1,128] shards can't tile-convert directly).
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                k_heads_pre_rot_1BKD, self.model_config["OLMO_K_NORM_L1_MEMCFG"]
            )
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)
            k_stats = ttnn.rms_norm_pre_all_gather(
                k_heads_pre_rot_1BKD,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=self.model_config["OLMO_K_NORM_L1_MEMCFG"],
            )
            k_stats_gathered = self._olmo_qk_norm_all_gather(k_stats, cluster_axis=0)
            ttnn.deallocate(k_stats)
            k_heads_pre_rot_1BKD = ttnn.rms_norm_post_all_gather(
                k_heads_pre_rot_1BKD,
                k_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_k_norm_weight,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=self.model_config["OLMO_K_NORM_L1_MEMCFG"],
            )
            ttnn.deallocate(k_stats_gathered)
            k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)
            k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, k_mem_cfg)

            # ---- Q global norm: distributed rms_norm over 8 row devices (5120 total elements) ----
            # create_heads output: [1, batch(8), padded_heads(8), head_dim(128)]
            # batch is in dim 1 (8 users per device group), heads in dim 2.
            # Normalize over 5 real heads × 128 = 640 per device (globally 5120 across 8 row devices).
            # Move to L1 INTERLEAVED (eliminates DRAM roundtrip for slice/reshape/tilize/norm ops).
            q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                q_heads_pre_rot_1BQD, self.model_config["OLMO_Q_NORM_L1_MEMCFG"]
            )
            q_batch = q_heads_pre_rot_1BQD.shape[1]  # batch dimension

            # Slice 5 real Q heads in dim 2: [1, batch, 8, 128] → [1, batch, 5, 128]
            q_real = ttnn.slice(q_heads_pre_rot_1BQD, [0, 0, 0, 0], [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_heads_pre_rot_1BQD)

            # Reshape [1, batch, 5, 128] → [1, 1, batch, 640]: flatten heads into last dim
            q_flat = ttnn.reshape(q_real, [1, 1, q_batch, self.n_local_heads * self.head_dim])
            ttnn.deallocate(q_real)
            q_flat = ttnn.to_layout(q_flat, ttnn.TILE_LAYOUT, memory_config=self.model_config["OLMO_Q_NORM_L1_MEMCFG"])

            # Distributed global Q norm: all_gather on cluster_axis=0 (8 row devices)
            q_stats = ttnn.rms_norm_pre_all_gather(
                q_flat,
                dtype=ttnn.bfloat16,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=self.model_config["OLMO_Q_NORM_L1_MEMCFG"],
            )
            q_stats_gathered = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
            ttnn.deallocate(q_stats)
            q_flat = ttnn.rms_norm_post_all_gather(
                q_flat,
                q_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_q_norm_weight_full_prefill,
                compute_kernel_config=self.compute_kernel_config_hifi2,
                memory_config=self.model_config["OLMO_Q_NORM_L1_MEMCFG"],
            )
            ttnn.deallocate(q_stats_gathered)
            q_flat = ttnn.to_layout(q_flat, ttnn.ROW_MAJOR_LAYOUT)

            # Undo reshape [1, 1, batch, 640] → [1, batch, 5, 128]
            q_real_normed = ttnn.reshape(q_flat, [1, q_batch, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_flat)

            # Pad 3 zero heads in dim 2: [1, batch, 5, 128] → [1, batch, 8, 128]
            n_pad = self.n_local_heads_padded - self.n_local_heads
            if n_pad > 0:
                q_heads_pre_rot_1BQD = ttnn.pad(q_real_normed, [(0, 0), (0, 0), (0, n_pad), (0, 0)], value=0.0)
                ttnn.deallocate(q_real_normed)
            else:
                q_heads_pre_rot_1BQD = q_real_normed

            self._capture_attn("q_post_norm", q_heads_pre_rot_1BQD)
            self._capture_attn("k_post_norm", k_heads_pre_rot_1BKD)

            q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, q_mem_cfg)
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
            ttnn.deallocate(xqkv_fused_sharded)

            if self.qk_norm:
                # Qwen3-style per-head QK-norm (after head split)
                rm_mem_cfg_q = q_heads_pre_rot_1BQD.memory_config()
                rm_mem_cfg_k = k_heads_pre_rot_1BKD.memory_config()
                q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                    q_heads_pre_rot_1BQD, memory_config=self.reshape_intermediate_q_mem_cfg
                )
                k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                    k_heads_pre_rot_1BKD, memory_config=self.reshape_intermediate_k_mem_cfg
                )
                q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 1, 64, 128])
                k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 1, 64, 128])
                q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.TILE_LAYOUT)
                k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.TILE_LAYOUT)
                q_imm_cfg = q_heads_pre_rot_1BQD.memory_config()
                k_imm_cfg = k_heads_pre_rot_1BKD.memory_config()
                q_heads_pre_rot_1BQD = ttnn.to_memory_config(
                    q_heads_pre_rot_1BQD, memory_config=self.reshape_output_q_mem_cfg
                )
                k_heads_pre_rot_1BKD = ttnn.to_memory_config(
                    k_heads_pre_rot_1BKD, memory_config=self.reshape_output_k_mem_cfg
                )
                q_heads_pre_rot_1BQD = self.q_norm(
                    q_heads_pre_rot_1BQD, mode="decode", in_sharded=True, out_sharded=True
                )
                k_heads_pre_rot_1BKD = self.k_norm(
                    k_heads_pre_rot_1BKD, mode="decode", in_sharded=True, out_sharded=True
                )
                q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=q_imm_cfg)
                k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=k_imm_cfg)
                q_heads_pre_rot_1BQD = ttnn.to_layout(q_heads_pre_rot_1BQD, ttnn.ROW_MAJOR_LAYOUT)
                k_heads_pre_rot_1BKD = ttnn.to_layout(k_heads_pre_rot_1BKD, ttnn.ROW_MAJOR_LAYOUT)
                q_heads_pre_rot_1BQD = ttnn.view(q_heads_pre_rot_1BQD, [1, 8, 8, 128])
                k_heads_pre_rot_1BKD = ttnn.view(k_heads_pre_rot_1BKD, [1, 8, 8, 128])
                q_heads_pre_rot_1BQD = ttnn.to_memory_config(q_heads_pre_rot_1BQD, memory_config=rm_mem_cfg_q)
                k_heads_pre_rot_1BKD = ttnn.to_memory_config(k_heads_pre_rot_1BKD, memory_config=rm_mem_cfg_k)

        self._debug_check_attn("q_heads_pre_rot", q_heads_pre_rot_1BQD)
        self._debug_check_attn("k_heads_pre_rot", k_heads_pre_rot_1BKD)
        self._debug_check_attn("v_heads", v_heads_1BKD)

        # Q, K Rotary Embeddings
        # Note: Fused RoPE requires num_heads * head_dim = 1024 for row-major tensors
        # Llama/Qwen have 8 local Q heads (8*128=1024) ✓
        # OLMo has 5 local Q heads padded to 8 (8*128=1024) ✓ for Q, but K has 1 head (1*128=128) ✗
        # For OLMo, use non-fused RoPE to handle different Q/K head counts
        if self.is_olmo:
            # OLMo: Fused RoPE requires K[-2] * K[-1] == 1024
            # K has 1 KV head: [1, batch, 1, 128] -> 1*128 = 128 ≠ 1024
            # Expand K heads from 1 to 8, apply fused RoPE, then slice back to 1 head
            k_shape = k_heads_pre_rot_1BKD.shape
            k_mem_config = k_heads_pre_rot_1BKD.memory_config()

            # Get K's original shard grid (non-overlapping with Q)
            k_shard_grid = k_mem_config.shard_spec.grid
            # Move K to DRAM (interleaved) to allow repeat without sharding constraints
            k_interleaved = ttnn.to_memory_config(k_heads_pre_rot_1BKD, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(k_heads_pre_rot_1BKD)

            # Tile K along the num_heads dimension: repeat 8 times [1, 8, 1, 128] -> [1, 8, 8, 128]
            k_expanded = ttnn.repeat(k_interleaved, ttnn.Shape([1, 1, 8, 1]))
            ttnn.deallocate(k_interleaved)

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
            self._capture_attn("k_expanded_post_rope", k_heads_expanded_dram)
            k_heads_1BKD = ttnn.slice(k_heads_expanded_dram, [0, 0, 0, 0], [1, k_shape[1], 1, k_shape[3]])
            ttnn.deallocate(k_heads_expanded_dram)
            self._capture_attn("k_post_rope_sliced", k_heads_1BKD)
            # Move back to original memory config
            k_heads_1BKD = ttnn.to_memory_config(k_heads_1BKD, k_mem_config)
        else:
            # Llama/Qwen: Use fused RoPE (both Q and K have 8 heads)
            q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
            )  # [1, 8, 8, 128], [1, 8, 8, 128]
            ttnn.deallocate(q_heads_pre_rot_1BQD)
            ttnn.deallocate(k_heads_pre_rot_1BKD)
        self._debug_check_attn("q_heads_post_rope", q_heads_1BQD)
        self._debug_check_attn("k_heads_post_rope", k_heads_1BKD)
        self._capture_attn("q_post_rope", q_heads_1BQD)
        self._capture_attn("v_from_create_heads", v_heads_1BKD)
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
        self._capture_attn("sdpa_out", attn_output_1G4D_sharded)
        self._capture_attn("values_cache", values)  # KV cache values for debug

        # OLMo decode: Use 8 padded heads to match fused RoPE (num_heads * head_dim = 1024)
        # For other models with 8 local heads, n_local_heads == 8 already
        decode_num_heads = 8 if (self.is_olmo and self.n_local_heads < 8) else self.n_local_heads

        if self.is_olmo:
            # OLMo SDPA→WO: slice padded heads, then all_gather batch across col devices.
            # SDPA output per device: [1, B=8, NH_padded=32, D=128] (batch=8 per col device,
            #   heads tile-padded from 8→32, head_dim=128).
            # llama_rs_create_heads did reduce_scatter splitting batch 32→8 across 4 col devices.
            # We reverse this: slice to 5 real heads, reshape, all_gather → [1, 1, 32, 640].
            sdpa_dram = ttnn.to_memory_config(attn_output_1G4D_sharded, ttnn.DRAM_MEMORY_CONFIG)
            ttnn.deallocate(attn_output_1G4D_sharded)
            sdpa_dram = ttnn.to_layout(sdpa_dram, ttnn.ROW_MAJOR_LAYOUT)

            batch_per_dev = sdpa_dram.shape[1]  # 8 (batch_size_per_device_group)

            # Slice to 5 real Q heads in dim2: [1, 8, 32, 128] → [1, 8, 5, 128]
            sdpa_real = ttnn.slice(sdpa_dram, [0, 0, 0, 0], [1, batch_per_dev, self.n_local_heads, self.head_dim])
            ttnn.deallocate(sdpa_dram)

            # Reshape [1, 8, 5, 128] → [1, 1, 8, 640] (flatten heads*head_dim)
            sdpa_flat = ttnn.reshape(sdpa_real, [1, 1, batch_per_dev, self.n_local_heads * self.head_dim])
            ttnn.deallocate(sdpa_real)

            # All_gather across col devices (cluster_axis=1) in dim2: [1, 1, 8, 640] → [1, 1, 32, 640]
            attn_output_cat = self.tt_ccl.line_all_gather(
                sdpa_flat,
                dim=2,
                cluster_axis=1,
                num_links=self.model_config["GALAXY_NUM_LINKS"],
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            ttnn.deallocate(sdpa_flat)
            self._debug_check_attn("attn_output_cat", attn_output_cat)
            self._capture_attn("wo_input", attn_output_cat)
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
            if self.is_olmo:
                # OLMo: keep in DRAM, just tilize (wo_interleaved is DRAM too)
                attn_output_cat = ttnn.to_layout(attn_output_cat, ttnn.TILE_LAYOUT)
            else:
                # Must convert to interleaved memory before tilizing sharded tensors
                attn_output_cat = ttnn.to_memory_config(attn_output_cat, ttnn.DRAM_MEMORY_CONFIG)
                attn_output_cat = ttnn.to_layout(attn_output_cat, ttnn.TILE_LAYOUT)
                # Convert back to sharded for WO matmul
                attn_output_cat = ttnn.to_memory_config(
                    attn_output_cat, self.model_config["SHARDED_ATTN_WO_INPUT_RING_MEMCFG"]
                )

        # Original matmul on each device [1, 1, 32, 1024] @ [1, 1, 1024, 2048]
        if self.is_olmo:
            # OLMo: use unpadded wo (640 rows = 5 real heads × 128 per col device)
            # attn_output_cat is [1, 1, 32, 640] (sliced to 5 real heads only)
            wo_decode = (
                self.wo_interleaved_unpadded if self.wo_interleaved_unpadded is not None else self.wo_interleaved
            )
            dense_out_ttnn = ttnn.matmul(  # [1, 1, 32, 1280]
                attn_output_cat,
                wo_decode,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b,
            )
            ttnn.deallocate(attn_output_cat)
            self._debug_check_attn("wo_matmul_out", dense_out_ttnn)
            # wo weight dims=(2,3): K(5120 heads) over rows(axis 0), N(5120 model) over cols(axis 1).
            # Each device computes partial_Y = X[32,640_i] @ W[640_i,1280_j].
            # All-reduce over rows (axis 0) sums partial K contributions → correct output per col.
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
        self._capture_attn("attn_out_final", dense_out_reduced)

        # print("done all reduce")

        return dense_out_reduced

    def _olmo_qk_norm_global(self, heads_tensor, weight_tensor, local_dim, n_heads):
        """OLMo global QK-norm via host roundtrip for exact match with HF.

        HF normalizes Q over 5120 dims and K over 1024 dims before head split.
        We read local heads to host, concatenate across row devices to get the full
        dimension, apply RMSNorm globally, then push each device's portion back.
        Prefill-only (not traced).
        """
        import torch

        seq = heads_tensor.shape[2]
        q_flat = ttnn.reshape(heads_tensor, [1, 1, seq, local_dim])
        q_flat = ttnn.to_layout(q_flat, ttnn.TILE_LAYOUT)

        q_host = ttnn.to_torch(
            q_flat,
            mesh_composer=ttnn.ConcatMesh2dToTensor(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
        ).float()
        ttnn.deallocate(q_flat)
        ttnn.deallocate(heads_tensor)

        n_rows, n_cols = self.cluster_shape
        col_w = q_host.shape[3] // n_cols

        q_global = torch.cat([q_host[:, r : r + 1, :, :col_w] for r in range(n_rows)], dim=-1)
        full_dim = q_global.shape[-1]

        is_q = full_dim == self.olmo_q_norm_cpu.shape[0]
        cpu_weight = self.olmo_q_norm_cpu if is_q else self.olmo_k_norm_cpu

        variance = q_global.pow(2).mean(-1, keepdim=True)
        q_normed_global = q_global * torch.rsqrt(variance + 1e-6) * cpu_weight

        result = torch.zeros(1, n_rows, seq, n_cols * col_w, dtype=torch.bfloat16)
        for r in range(n_rows):
            chunk = q_normed_global[:, :, :, r * col_w : (r + 1) * col_w].bfloat16()
            for c in range(n_cols):
                result[:, r, :, c * col_w : (c + 1) * col_w] = chunk

        result_dev = ttnn.from_torch(
            result,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            mesh_mapper=ttnn.ShardTensor2dMesh(self.mesh_device, dims=(1, 3), mesh_shape=self.cluster_shape),
        )
        return ttnn.reshape(result_dev, [1, n_heads, seq, self.head_dim])

    def _olmo_qk_norm_all_gather_single_axis(self, stats_tensor, cluster_axis):
        """All-gather stats along a single axis using async API with semaphores."""
        ccl = self.tt_ccl
        idx = ccl.gather_idx[cluster_axis]
        sem_handles = ccl.gather_semaphore_handles[cluster_axis]
        if isinstance(sem_handles[idx], list):
            semaphores = sem_handles[idx]
        else:
            semaphores = [sem_handles[idx], sem_handles[(idx + 1) % ccl.num_cbs]]
        barrier_semaphore = ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis)
        result = ttnn.experimental.all_gather_async(
            stats_tensor,
            dim=3,
            cluster_axis=cluster_axis,
            mesh_device=ccl.mesh_device,
            topology=ttnn.Topology.Linear,
            multi_device_global_semaphore=semaphores,
            persistent_output_tensor=None,
            barrier_semaphore=barrier_semaphore,
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            subdevice_id=ccl.worker_sub_device_id,
        )
        ccl.gather_idx[cluster_axis] = (idx + 1) % ccl.num_cbs
        return result

    def _olmo_qk_norm_all_gather(self, stats_tensor, cluster_axis=0):
        """All-gather QK-norm stats across row devices (axis 0, 8 devices).

        After create_heads' all-reduce, all 4 cols have identical Q/K data.
        Only need to gather across 8 rows to collect stats for global RMS
        (5120 Q dims = 8 rows × 640, or 1024 K dims = 8 rows × 128).
        """
        return self._olmo_qk_norm_all_gather_single_axis(stats_tensor, cluster_axis=0)

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
        skip_input_dealloc=False,
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

        if not skip_input_dealloc:
            ttnn.deallocate(x_11SH)

        xqkv_fused = self.tt_ccl.line_all_reduce(
            xqkv,
            cluster_axis=1,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="QKV",
            batch_size=batch_size,
        )
        ttnn.deallocate(xqkv)

        if seq_len > 2048:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        # OLMo QK-norm is applied AFTER nlp_create_qkv_heads (per-head, like decode path)

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

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

        ###
        # Rotary embeddings
        ###

        self._capture_prefill_attn("v_heads", v_heads_1VSD)

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot_bf8 = q_heads_1QSD_pre_rot
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(q_heads_1QSD_pre_rot_bf8)

        self._capture_prefill_attn("q_before_norm", q_heads_1QSD_pre_rot)

        if self.qk_norm and not self.is_olmo:
            q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")
        elif self.is_olmo and self.qk_norm:
            # OLMo QK-norm: distributed RMS across 8 row devices for exact global normalization.
            # HF normalizes Q over 5120 dims (all 40 heads) before head split.
            # Each row device has 640 dims (5 heads × 128). Use rms_norm_pre/post_all_gather
            # with all_gather on cluster_axis=0 to compute global variance over 5120 dims.
            #
            # Q shape is [1, n_heads, seq, head_dim]. Must transpose to [1, seq, n_heads, head_dim]
            # before flattening so heads are contiguous per position (not per head across positions).
            q_seq = q_heads_1QSD_pre_rot.shape[2]
            q_transposed = ttnn.transpose(q_heads_1QSD_pre_rot, 1, 2)  # [1, seq, 5, 128]
            ttnn.deallocate(q_heads_1QSD_pre_rot)
            q_flat = ttnn.reshape(q_transposed, [1, 1, q_seq, self.n_local_heads * self.head_dim])
            ttnn.deallocate(q_transposed)
            q_flat = ttnn.to_layout(q_flat, ttnn.TILE_LAYOUT)
            q_stats = ttnn.rms_norm_pre_all_gather(
                q_flat, compute_kernel_config=self.compute_kernel_config_hifi2, dtype=ttnn.bfloat16
            )
            q_stats_gathered = self._olmo_qk_norm_all_gather(q_stats, cluster_axis=0)
            ttnn.deallocate(q_stats)
            q_normed_flat = ttnn.rms_norm_post_all_gather(
                q_flat,
                q_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_q_norm_weight_full_prefill,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )
            ttnn.deallocate(q_flat)
            q_unflat = ttnn.reshape(q_normed_flat, [1, q_seq, self.n_local_heads, self.head_dim])
            ttnn.deallocate(q_normed_flat)
            q_heads_1QSD_pre_rot = ttnn.transpose(q_unflat, 1, 2)  # [1, 5, seq, 128]
            ttnn.deallocate(q_unflat)

        self._capture_prefill_attn("q_after_norm", q_heads_1QSD_pre_rot)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        self._capture_prefill_attn("q_after_rope", q_heads_1QSD)
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot_bf8 = k_heads_1KSD_pre_rot
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)
            ttnn.deallocate(k_heads_1KSD_pre_rot_bf8)

        self._capture_prefill_attn("k_before_norm", k_heads_1KSD_pre_rot)

        if self.qk_norm and not self.is_olmo:
            k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")
        elif self.is_olmo and self.qk_norm:
            # OLMo K norm: distributed RMS across 8 row devices for exact global normalization.
            # HF normalizes K over 1024 dims (all 8 KV heads). Each row device has 128.
            # all_gather on cluster_axis=0 to compute global variance over 1024 dims.
            k_heads_1KSD_pre_rot = ttnn.to_layout(k_heads_1KSD_pre_rot, ttnn.TILE_LAYOUT)
            k_stats = ttnn.rms_norm_pre_all_gather(
                k_heads_1KSD_pre_rot, compute_kernel_config=self.compute_kernel_config_hifi2, dtype=ttnn.bfloat16
            )
            k_stats_gathered = self._olmo_qk_norm_all_gather(k_stats, cluster_axis=0)
            ttnn.deallocate(k_stats)
            k_heads_1KSD_pre_rot = ttnn.rms_norm_post_all_gather(
                k_heads_1KSD_pre_rot,
                k_stats_gathered,
                epsilon=1e-6,
                weight=self.olmo_k_norm_weight_full_prefill,
                compute_kernel_config=self.compute_kernel_config_hifi2,
            )

        self._capture_prefill_attn("k_after_norm", k_heads_1KSD_pre_rot)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        self._capture_prefill_attn("k_after_rope", k_heads_1KSD)
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

        output_11SH_reduced = self.tt_ccl.line_all_reduce(
            output_11SH,
            cluster_axis=0,
            num_links=self.model_config["GALAXY_NUM_LINKS"],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            buffer_key="WO_AG" if seq_len <= 4096 else "WO",
        )
        output_11SH.deallocate()

        self._capture_prefill_attn("wo_out", output_11SH_reduced)

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
        skip_input_dealloc=False,
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
                skip_input_dealloc=skip_input_dealloc,
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
