# SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.ccl import tt_all_gather, tt_all_reduce
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange


class Attention(LightweightModule):
    def __init__(
        self,
        mesh_device,
        tt_ccl,
        args,
        state_dict,
        weight_cache_path,
        layer_num,
        dtype,
        transformation_mats,
        configuration,
        paged_attention_config=None,
        use_paged_kv_cache=False,
    ):
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
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
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset
        self.num_device_groups = self.num_devices // self.n_kv_heads
        self.num_devices_per_group = self.n_kv_heads if self.TG else self.num_devices
        self.batch_size_per_device_group = (
            max(self.max_batch_size // self.num_device_groups, 1) if self.TG else self.max_batch_size
        )

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        self.arch_name = configuration.arch_name
        # TODO: Fix this once all-gather supports < tile_size
        if self.TG:
            weight = torch.zeros(1, 32, 8, 32)
            for i in range(32):
                col = i % 4  # This determines which group of 8 to select
                weight[:, i, :, col * 8 : (col + 1) * 8] = torch.eye(8)

            self.slice_mat = ttnn.from_torch(
                weight,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=1),
            )
            user_selection_matrix = torch.eye(8, 8)
            user_selection_matrix = torch.nn.functional.pad(user_selection_matrix, (0, 24), "constant", 0)  # (8, 32)
            user_selection_matrix = [user_selection_matrix] * 4
            user_selection_matrix = torch.block_diag(*user_selection_matrix)  # (32, 128)
            self.user_selection_matrix = ttnn.from_torch(
                user_selection_matrix,
                dtype=ttnn.bfloat4_b,
                layout=ttnn.TILE_LAYOUT,
                device=self.mesh_device,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
            )

        self.dtype = dtype

        self.max_seq_len = configuration.max_seq_len
        self.grid_size = configuration.max_grid_size

        self.compute_kernel_config_hifi2 = configuration.compute_kernel_config_hifi2
        self.compute_kernel_config_hifi2_fp16 = configuration.compute_kernel_config_hifi2_fp16

        self.compute_kernel_config_hifi4 = configuration.compute_kernel_config_hifi4

        self.transformation_mats = transformation_mats
        self.is_sliding = (
            configuration.layer_types[layer_num] == "sliding_attention" if configuration.layer_types else False
        )
        self.sliding_window = configuration.sliding_window if self.is_sliding else None

        self.model_config = configuration.get_model_config()
        self.ccl_topology = configuration.ccl_topology()
        self.is_multichip = configuration.is_multichip
        self.activation_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION
        )
        self.wqkv_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV
        )
        self.wo_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO
        )
        self.kv_cache_dtype = self.model_config["DECODERS_OPTIMIZATIONS"].get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE
        )
        self.li_qkv_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = self.model_config["DECODERS_OPTIMIZATIONS"].get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}")

        wq_str = f"{layer_name}.wq"
        wk_str = f"{layer_name}.wk"
        wv_str = f"{layer_name}.wv"
        wo_str = f"{layer_name}.wo"
        q_norm_str = f"{layer_name}.q_norm"
        k_norm_str = f"{layer_name}.k_norm"

        # Initialize bias tensors as None
        self.wqkv_bias_decode = None
        self.wqkv_bias_prefill = None

        # Create combined QKV bias if present in state dict
        if f"{wq_str}.bias" in state_dict:
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], configuration.num_devices)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], configuration.num_devices)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(configuration.num_devices)
                ],
                dim=-1,
            )
            # Prefill can use broadcasting on the bias add so wants a 1d tensor
            self.wqkv_bias_prefill = ttnn.as_tensor(
                qkv_bias,
                device=self.mesh_device,
                mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                layout=ttnn.TILE_LAYOUT,
                cache_file_name=cache_name("wqkv_bias_prefill_sharded"),
            )
            # as_tensor returns (32, dim) which is incorrect, this reshape updates the padded size to the correct size
            self.wqkv_bias_prefill = ttnn.reshape(
                self.wqkv_bias_prefill,
                (1, 1, 1, self.wqkv_bias_prefill.shape[-1]),
                (1, 1, self.wqkv_bias_prefill.shape[-2], self.wqkv_bias_prefill.shape[-1]),
            )

            # Broadcasting does not seem to be supported inside execute_trace so expand to the whole batch size
            # Create a list of bias tensors for each multiple of tile_size up to max_batch_size
            self.wqkv_bias_decode = []
            for batch_size in range(
                configuration.tile_size,
                configuration.tile_padded_batch_rows + configuration.tile_size,
                configuration.tile_size,
            ):
                qkv_bias_decode = qkv_bias.unsqueeze(0).expand(batch_size, -1)
                bias_tensor = ttnn.as_tensor(
                    qkv_bias_decode,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensorToMesh(self.mesh_device, dim=-1),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=cache_name(f"wqkv_bias_decode_sharded_{batch_size}"),
                )
                self.wqkv_bias_decode.append(bias_tensor)

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
            wq_selected = torch.chunk(state_dict[f"{wq_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wk_selected = torch.chunk(state_dict[f"{wk_str}.weight"], self.num_devices_per_group, dim=0)[i]
            wv_selected = torch.chunk(state_dict[f"{wv_str}.weight"], self.num_devices_per_group, dim=0)[i]

            # Transpose the selected chunks
            wq = torch.transpose(wq_selected, -2, -1)
            wk = torch.transpose(wk_selected, -2, -1)
            wv = torch.transpose(wv_selected, -2, -1)

            qkv = torch.cat([wq, wk, wv], dim=-1)
            qkv_list.append(qkv)

        qkv_cat = torch.cat(qkv_list, dim=-1).unsqueeze(0).unsqueeze(0)

        self.wqkv = ttnn.as_tensor(
            qkv_cat,
            dtype=self.wqkv_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if self.TG else wqkv_mem_config,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, dims=(3, 2) if self.TG else (2, 3), mesh_shape=configuration.cluster_shape
            ),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        def norm_reshard(x, norm, mode):
            """Hack until RMSNorm supports height-sharded output config"""
            if mode == "decode":
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode)
            if mode == "decode":
                x = ttnn.to_memory_config(x, mem_cfg, dtype=x.dtype)
            return x

        if f"{q_norm_str}.weight" in state_dict:
            fn_q_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix q_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=q_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"]
                tt_ccl=self.tt_ccl,
            )
            self.q_norm = lambda x, mode: norm_reshard(x, fn_q_norm, mode)
        else:
            self.q_norm = lambda x, mode: x

        if f"{k_norm_str}.weight" in state_dict:
            fn_k_norm = RMSNorm(
                device=self.mesh_device,
                dim=self.head_dim,
                eps=configuration.norm_eps,
                state_dict=state_dict,
                state_dict_prefix=None,  # we already prefix k_norm_str
                weight_cache_path=None if configuration.dummy_weights else weight_cache_path,
                weight_dtype=ttnn.bfloat16,
                weight_key=k_norm_str,
                add_unit_offset=self.rms_norm_add_unit_offset,
                is_distributed=False,
                sharded_program_config=None,  # FIXME: add height-sharded support. self.model_config["SHARDED_NORM_ATTN_PRGM_CFG"],
                sharded_output_config=None,  # FIXME: add height-sharded support. self.model_config["CREATE_QKV_DECODE_SHARD"],
                tt_ccl=self.tt_ccl,
            )
            self.k_norm = lambda x, mode: norm_reshard(x, fn_k_norm, mode)
        else:
            self.k_norm = lambda x, mode: x

        # For ring topology we can use all gather matmul for wo
        self.use_fused_all_gather_matmul = self.model_config["USE_FUSED_ALL_GATHER_MATMUL"]
        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        wo_mem_config = configuration.create_dram_sharded_mem_config(
            (configuration.n_heads * configuration.head_dim) // configuration.num_devices, configuration.dim
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
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

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
                dtype=self.kv_cache_dtype,
                layout=self.model_config["ATTN_W_LAYOUT_TILE"],
                device=self.mesh_device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
                cache_file_name=(
                    f"{weight_cache_path}/kvcache_{k_or_v.shape}"
                    if weight_cache_path and not configuration.dummy_weights
                    else None
                ),
            )
            for k_or_v in [cache_k, cache_v]
        ]

    def to_qk_fused_memory_config(self, q_tensor: ttnn.Tensor, k_tensor: ttnn.Tensor):
        """
        Convert Q and K tensors to height-sharded memory layouts suitable for
        fused QK ops such as rotary_embedding_llama_fused_qk and the subsequent
        QK matmul/attention score computation.

        This function:
        - Infers the number of Q heads and KV heads from the input tensors
        - Shards Q and K along the batch dimension using HEIGHT sharding
        - Places Q and K on disjoint core regions to avoid overlap (assuming a row size of 8 cores):
            - For q_batch = 32: 32 cores for Q fill 4 complete rows (32 // 8 = 4), so K starts at row 4, column 0
            - For q_batch = 1: 1 core for Q starts at (0,0), so K starts at column 1, row 0
        - Uses row-major shard orientation with explicit shard shapes

        The resulting memory layouts are compatible with fused attention
        kernels that expect Q and K to be distributed across separate
        core ranges while preserving per-head contiguity.

        Args:
            q_tensor (ttnn.Tensor):
                Query tensor with shape [..., batch, num_q_heads, head_dim].

            k_tensor (ttnn.Tensor):
                Key tensor with shape [..., batch, num_kv_heads, head_dim].

        Returns:
            Tuple[ttnn.Tensor, ttnn.Tensor]:
                (q_tensor, k_tensor) converted to sharded memory configurations.
        """
        n_q_heads = q_tensor.shape[2]
        n_kv_heads = k_tensor.shape[2]
        q_batch = q_tensor.shape[1]
        k_batch = k_tensor.shape[1]
        assert q_batch == k_batch

        row_size = 8  # We assume a row size of 8 cores
        k_start_core = ttnn.CoreCoord(q_batch % row_size, q_batch // row_size)

        q_core_grid = ttnn.CoreRangeSet({num_to_corerange(q_batch)})
        k_core_grid = ttnn.CoreRangeSet({num_to_corerange(k_batch, start_core=k_start_core)})

        q_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_q_heads), self.head_dim),
            core_grid=q_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        k_mem_config = ttnn.create_sharded_memory_config(
            shape=(nearest_32(n_kv_heads), self.head_dim),
            core_grid=k_core_grid,
            strategy=ttnn.ShardStrategy.HEIGHT,
            orientation=ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        q_tensor = ttnn.to_memory_config(q_tensor, q_mem_config)
        k_tensor = ttnn.to_memory_config(k_tensor, k_mem_config)
        return q_tensor, k_tensor

    def forward_decode(self, x: ttnn.Tensor, current_pos, rot_mats=None, page_table=None, kv_cache=None) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """

        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###

        xqkv_fused_sharded = ttnn.linear(
            x,
            self.wqkv,
            # bias=self.wqkv_bias,
            memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
            program_config=self.model_config["XQKV_DECODE_PROGCFG"],
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
        )
        # FIXME: File bug against dram-sharded matmuls with bias
        if self.wqkv_bias_decode:
            # select the bias tensor based on the number of tiles in the rows
            # WARNING: must not change the batch size between compiling and executing a trace
            num_tiles = int(math.ceil(xqkv_fused_sharded.shape[-2] / self.tile_size))
            xqkv_fused_sharded = xqkv_fused_sharded + self.wqkv_bias_decode[num_tiles - 1]

        ttnn.deallocate(x)
        xqkv_fused = tt_all_reduce(
            xqkv_fused_sharded,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=self.model_config["QKV_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[1]),
            sharded=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )

        if self.TG:
            # TODO: Slice the fused_query_key_value tensor get batch=8
            xqkv_fused = ttnn.matmul(
                self.slice_mat,
                xqkv_fused,
                dtype=ttnn.bfloat16,
                memory_config=self.model_config["CREATE_HEAD_INPUT_MEMCFG"],
            )
        else:
            # bfloat16 is required by nlp_create_qkv_heads_decode
            xqkv_fused = ttnn.sharded_to_interleaved(xqkv_fused_sharded, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

        ttnn.deallocate(xqkv_fused_sharded)

        # Reshape such that true unpadded batch is tracked in shape
        fqkv_shape = xqkv_fused.shape
        xqkv_fused = ttnn.reshape(
            xqkv_fused, (1, 1, self.batch_size_per_device_group, fqkv_shape[3]), (1, 1, 32, fqkv_shape[3])
        )

        ###
        # Reshape and rotary embeddings
        ###
        (
            q_heads_pre_rot_1BQD,
            k_heads_pre_rot_1BKD,
            v_heads_1BKD,
        ) = ttnn.experimental.nlp_create_qkv_heads_decode(
            xqkv_fused,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_kv_heads,
            memory_config=self.model_config["CREATE_QKV_DECODE_SHARD"],
        )

        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode="decode")
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode="decode")

        ttnn.deallocate(xqkv_fused)

        # Q, K Rotary Embeddings
        if self.args.use_qk_fused:
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD = self.to_qk_fused_memory_config(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD
            )

            q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
                q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
            )
        else:
            # Q Rotary Embeddings
            q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
                q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
            )

            # K Rotary Embeddings
            k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
                k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
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

        if self.args.use_qk_fused:
            ttnn.experimental.paged_fused_update_cache(
                keys, k_heads_1BKD, values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        else:
            ttnn.experimental.paged_update_cache(
                keys, k_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
            ttnn.experimental.paged_update_cache(
                values, v_heads_1BKD, update_idxs_tensor=current_pos, page_table=page_table
            )
        ttnn.deallocate(k_heads_1BKD)
        ttnn.deallocate(v_heads_1BKD)

        # NOTE: Varying the batch size will result in slightly different outputs.
        # For example, a prompt w/ 1 user vs, the same prompt repeated N times for N users, will produce different outputs
        # This is because the SDPA op in decode mode has different number of reductions depending on batch size
        # Which leads to slightly different outputs from attention (due to accumulated errors)
        if page_table:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
        else:
            attn_output_1G4D = ttnn.transformer.scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=self.model_config["SDPA_DECODE_PROGCFG"],
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)

        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.model_config["SCORES_BATCHED_MM_OUTPUT_MEMCFG"](self.batch_size_per_device_group),
        )
        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        if self.use_fused_all_gather_matmul:
            attn_output_cat = ttnn.to_memory_config(
                attn_output_cat, self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"]
            )

            # Fused AGMM only valid for ring topology
            if self.ccl_topology == ttnn.Topology.Ring:
                _, dense_out_sharded = ttnn.experimental.all_gather_matmul_async(
                    attn_output_cat,
                    self.wo,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    all_gather_core_grid_offset=(0, 4),
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    num_links=1,
                    memory_config_ag=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                    memory_config_mm=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
                    compute_kernel_config=self.compute_kernel_config_hifi2,
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )
            else:
                all_gather_output = ttnn.experimental.all_gather_async(
                    attn_output_cat,
                    persistent_output_buffer=None,
                    dim=3,
                    multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                    num_links=1,
                    topology=self.ccl_topology,
                    memory_config=self.model_config["ATTN_ALL_GATHER_MATMUL_OUTPUT_MEMCFG"],
                    barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                    chunks_per_sync=10,
                    num_workers_per_link=2,
                    num_buffers_per_channel=2,
                )

                dense_out_sharded = ttnn.linear(
                    all_gather_output,
                    self.wo,
                    memory_config=self.model_config["DECODE_RESIDUAL_MEMCFG"],
                    program_config=self.model_config["ATTN_ALL_GATHER_MATMUL_PROGCFG"],
                    compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
                )

                ttnn.deallocate(all_gather_output)
            ttnn.deallocate(attn_output_cat)
            dense_out_sharded = ttnn.to_memory_config(dense_out_sharded, self.model_config["DECODE_RESIDUAL_MEMCFG"])
            return dense_out_sharded

        else:
            attn_output = tt_all_gather(
                attn_output_cat,
                self.mesh_device,
                self.tt_ccl,
                dim=2,
                cluster_axis=1,
                num_links=2,
                memory_config=self.model_config["GATHER_USERS_MEMCFG"](list(self.mesh_device.shape)[1]),
                sharded=True,
                # dtype=self.ccl_dtype,  # Running bf16 until we have SDPA output bfp8 df; otherwise we have two sharded to interleaved/interleaved to sharded conversions
            )
            if self.TG:
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

            # TODO: Fix this once self.TG supports dram-sharded matmuls
            dense_out_sharded = ttnn.matmul(
                attn_output,
                self.wo,
                core_grid=ttnn.CoreGrid(y=4, x=8) if self.TG else None,
                program_config=self.model_config["ATTN_OUTPUT_PROGCFG"] if not self.TG else None,
                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                dtype=ttnn.bfloat8_b if self.TG else None,
                compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
            )

            ttnn.deallocate(attn_output_cat)

            # All reduce
            dense_out_reduced = tt_all_reduce(
                dense_out_sharded,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                dim=0 if (self.TG and self.hidden_size < 8192) else 3,
                topology=self.ccl_topology,
                memory_config=(
                    (
                        self.model_config["SELF_OUT_REDUCE_SCATTER_MEMCFG"]
                        if self.hidden_size == 8192
                        else self.model_config["SELF_OUT_GATHERED_MEMCFG"](list(self.mesh_device.shape)[0])
                    )
                    if self.TG
                    else self.model_config["DECODE_RESIDUAL_MEMCFG"]
                ),
                sharded=True,
                dtype=self.ccl_dtype,
                use_composite=True if self.hidden_size == 8192 else False,
            )

            if not self.TG:
                dense_out_reduced = ttnn.to_memory_config(
                    dense_out_reduced, self.model_config["DECODE_RESIDUAL_MEMCFG"]
                )

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
    ):
        seq_len = x_11SH.shape[-2]
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            if seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
                raise ValueError(f"seq_len {seq_len} must be divisible by {self.MAX_QKV_MM_SEQ_LEN}")
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        xqkv_fused = ttnn.linear(
            x_11SH,
            self.wqkv,
            dtype=self.ccl_dtype if self.TG else self.activation_dtype or ttnn.bfloat16,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
            program_config=self.model_config["XQKV_PREFILL_PROGCFG"](seq_len),
        )

        # FIXME: surely ttnn.linear bias should work?
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        xqkv_fused = tt_all_reduce(
            xqkv_fused,
            self.mesh_device,
            self.tt_ccl,
            cluster_axis=1,
            num_reduce_scatter_links=self.num_reduce_scatter_links,
            num_all_gather_links=self.num_all_gather_links,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=self.ccl_dtype,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        ttnn.deallocate(x_11SH)

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

        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode="prefill")
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode="prefill")

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )
        ttnn.deallocate(q_heads_1QSD_pre_rot)

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:  # Rotary embeddings require bfloat16 inputs
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

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
        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)
        ttnn.deallocate(k_heads_1KSD)

        # sharding k_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)

        ttnn.deallocate(v_heads_1VSD)

        # sharding v_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self.model_config["KV_PREFILL_MEM_CFG"](seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        if self.TG:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table:
            # In the case that the tokens have been padded along the seq len dimension, we need to fill the cache with the unpadded k/v values.
            # Assume that the page table does not have padding, so we can use it to get the unpadded page len.
            block_size = keys_BKSD.shape[2]
            # If chunked prefill, use chunk_page_table if given, otherwise use page_table.
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced, fill_page_table, batch_idx=user_id)
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
        if seq_len >= self.min_kv_prefill_shard_seqlen and not self.TG and not page_table:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        if chunk_start_idx is not None:
            if self.sliding_window is not None:
                raise NotImplementedError("Sliding window not supported for chunked prefill SDPA")
            attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                input_tensor_q=q_heads_1QSD_8b,
                input_tensor_k=keys_BKSD,
                input_tensor_v=values_BKSD,
                page_table_tensor=page_table,
                chunk_start_idx=chunk_start_idx,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len, chunk_start_idx),
            )
        else:
            attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                sliding_window_size=self.sliding_window,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.model_config["SDPA_PROGCFG"](seq_len),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)
        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        # Non fused All Gather Matmul
        if self.use_fused_all_gather_matmul:  # is true for Ring topology
            attn_output_11SH = ttnn.experimental.all_gather_async(
                attn_output_11SH,
                persistent_output_buffer=None,
                dim=3,
                multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(),
                num_links=1,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(),
                chunks_per_sync=10,
                num_workers_per_link=2,
                num_buffers_per_channel=2,
            )

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.model_config["WO_PREFILL_PROGCFG"](seq_len),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce-scatter
        if not self.use_fused_all_gather_matmul:
            output_11SH = tt_all_reduce(
                output_11SH,
                self.mesh_device,
                self.tt_ccl,
                cluster_axis=0,
                dim=0 if self.TG else 3,
                num_reduce_scatter_links=self.num_reduce_scatter_links,
                num_all_gather_links=self.num_all_gather_links,
                topology=self.ccl_topology,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=self.ccl_dtype,
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
