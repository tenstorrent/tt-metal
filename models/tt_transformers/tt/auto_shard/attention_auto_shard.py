# SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

import math

import torch

import ttnn
from models.common.lightweightmodule import LightweightModule
from models.common.rmsnorm import RMSNorm
from models.common.utility_functions import nearest_32
from models.tt_transformers.tt.auto_shard.ccl_auto_shard import (
    all_reduce,
    core_grid_program_config,
    log_default_grid,
    matmul_reduce_scatter,
)
from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.model_config import OpGroup, TensorGroup, num_to_corerange
from models.tt_transformers.tt.auto_shard.cost_model.sharding import (
    AttentionShapes,
    MemoryParams,
    cache_tag,
    select_sharding,
    workload_from_config,
)


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
        prefetcher=None,
        toy=False,
        auto_shard=None,  # FIXME: auto-sharding
    ):
        print("Auto Sharding Attention")
        super().__init__()
        self.args = args
        self.mesh_device = mesh_device
        self.tt_ccl = tt_ccl
        self.num_devices = configuration.num_devices
        self.hidden_size = configuration.dim
        self.n_heads = configuration.n_heads
        self.head_dim = configuration.head_dim
        self.max_seq_len = configuration.max_seq_len
        self.max_batch_size = configuration.max_batch_size
        self.n_kv_heads = configuration.n_kv_heads
        self.paged_attention_config = paged_attention_config
        self.ccl_dtype = configuration.ccl_dtype
        self.MAX_QKV_MM_SEQ_LEN = configuration.MAX_QKV_MM_SEQ_LEN
        self.tile_size = configuration.tile_size
        self.rms_norm_add_unit_offset = configuration.rms_norm_add_unit_offset
        self.configuration = configuration

        # FIXME: auto-shard
        # One sharding descriptor drives the whole module: it fixes the wqkv/wo mesh placements
        # and the two all-reduce axes, so decode and prefill run a single code path on any mesh.
        shapes = AttentionShapes(
            self.n_heads, self.n_kv_heads, self.head_dim, self.hidden_size, configuration.qkv_size
        )
        # Rank placements for the real run's workload (prefill = max prompt that fits, decode =
        # max_generated_tokens), falling back to params.py when the demo didn't plumb it. Drop any
        # placement whose per-device DRAM footprint (weights + KV cache, all layers) won't fit;
        # raises a clear OOM error here instead of failing mid-allocation on device.
        prefill_len, decode_steps = workload_from_config(configuration)
        self.sharding = select_sharding(
            self.mesh_device,
            shapes,
            tuple(configuration.cluster_shape),
            memory_params=MemoryParams.from_config(configuration),
            prefill_len=prefill_len,
            decode_steps=decode_steps,
        )
        
        
        self.num_devices_per_group = self.sharding.num_intermediate_shards  # heads split -> n_local_heads
        self.num_device_groups = self.sharding.num_model_shards            # orthogonal (batch/model) axis
        self.batch_size_per_device_group = self.max_batch_size      # batch is not split on small meshes

        self.n_local_heads = self.n_heads // self.num_devices_per_group
        self.n_local_kv_heads = self.n_kv_heads // self.num_devices_per_group

        # The KV-prefill height-shard sizes to the actual local KV heads (from num_intermediate_shards),
        # not the model_config default's cluster_shape[1] assumption -- so any placement fits, including
        # full replication (num_intermediate_shards=1, so n_local_kv_heads = n_kv_heads).
        self.min_kv_prefill_shard_seqlen = (ttnn.TILE_SIZE * 8 * 8) / self.n_local_kv_heads

        self.use_qk_fused = configuration.use_qk_fused
        self.use_hf_rope = configuration.use_hf_rope
        self.arch_name = configuration.arch_name
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

        # When prefetcher is enabled, use consistent dtypes across all layers to avoid
        # race conditions caused by different block sizes
        use_prefetcher = prefetcher is not None

        decoders_optimizations = self.args.decoders_optimizations
        self.activation_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.ACTIVATION, prefetcher=use_prefetcher
        )
        self.wqkv_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WQKV, prefetcher=use_prefetcher
        )
        self.wo_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.WO, prefetcher=use_prefetcher
        )
        self.kv_cache_dtype = decoders_optimizations.get_tensor_dtype(
            decoder_id=layer_num, tensor=TensorGroup.KV_CACHE, prefetcher=use_prefetcher
        )
        self.li_qkv_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_DECODE, configuration=configuration
        )
        self.sdpa_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_DECODE, configuration=configuration
        )
        self.li_o_decode_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_DECODE, configuration=configuration
        )
        self.sdpa_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.SDPA_PREFILL, configuration=configuration
        )
        self.li_qkv_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_QKV_PREFILL, configuration=configuration
        )
        self.li_o_prefill_compute_kernel_cfg = decoders_optimizations.get_math_fidelity(
            decoder_id=layer_num, op=OpGroup.LI_O_PREFILL, configuration=configuration
        )

        layer_name = configuration.get_state_dict_prefix(self.__class__.__name__, layer_num)
        # Tagged with the mesh shape AND the placement, so layouts can't clobber each other -- see
        # sharding.cache_tag. Without a cache every layer redoes the full torch -> ttnn conversion
        # (tilize + quantize + shard) on every run, which is most of this path's start-up cost.
        tag = cache_tag(self.sharding, configuration.cluster_shape)
        if configuration.dummy_weights or (weight_cache_path is None):
            cache_name = lambda _: None
        else:
            cache_name = lambda name: weight_cache_path / (f"{layer_name}.{name}_{tag}")

        # Select rotary embedding implementation for decode
        if self.use_hf_rope and self.use_qk_fused:
            raise NotImplementedError("Fused QK is not implemented for HF-style rope")
        if self.use_hf_rope:
            self.rotary_embedding_decode = self._hf_rope_decode
        elif self.use_qk_fused:
            self.rotary_embedding_decode = self._mllama_rope_fused_qk_decode
        else:
            self.rotary_embedding_decode = self._mllama_rope_decode

        # Select rotary embedding implementation for prefill
        if self.use_hf_rope:
            self.rotary_embedding_prefill = self._hf_rope_prefill
        else:
            self.rotary_embedding_prefill = self._mllama_rope_prefill

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
            # AUTO-SHARD FIX: chunk by num_devices_per_group (the head/intermediate shard count),
            # matching the wqkv weight below -- NOT by the total chip count. The two agree only on a
            # 1D mesh; on a 2x2 the weight splits qkv 2 ways while num_devices=4 split the bias 4
            # ways, and the prefill bias add blew up (2304 vs 1152).
            qkv_bias = torch.concat(
                [
                    torch.concat(
                        [
                            torch.chunk(state_dict[f"{wq_str}.bias"], self.num_devices_per_group)[i],
                            torch.chunk(state_dict[f"{wk_str}.bias"], self.num_devices_per_group)[i],
                            torch.chunk(state_dict[f"{wv_str}.bias"], self.num_devices_per_group)[i],
                        ],
                        dim=-1,
                    )
                    for i in range(self.num_devices_per_group)
                ],
                dim=-1,
            )

            # The bias rides the same placement as wqkv: split the qkv width over intermediate_axis,
            # and hold one copy along model_axis. Dims are negative so this works at either rank
            # (prefill bias is a row, decode's is batch rows); -1/-2 mirror col_dims' 3/2.
            bias_dims = [None, None]
            if self.sharding.placement.intermediate_axis is not None:
                bias_dims[self.sharding.placement.intermediate_axis] = -1
            if self.sharding.placement.model_axis is not None:
                bias_dims[self.sharding.placement.model_axis] = -2
            bias_dims = tuple(bias_dims)

            def bias_to_mesh(b, cache_key):
                # The bias is added to the qkv matmul output *before* the all-reduce over
                # model_axis, so a replicated copy would be summed num_model_shards times. Give the
                # real bias to rank 0 of model_axis and zeros to the rest: the reduce then sums to
                # exactly one bias, with no scaling and no reordering of the add.
                rows = b.shape[-2]
                stacked = torch.zeros(rows * self.num_device_groups, b.shape[-1], dtype=b.dtype)
                stacked[:rows] = b
                return ttnn.as_tensor(
                    stacked,
                    device=self.mesh_device,
                    mesh_mapper=ttnn.ShardTensor2dMesh(
                        self.mesh_device,
                        dims=bias_dims,
                        mesh_shape=configuration.cluster_shape,
                    ),
                    dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    layout=ttnn.TILE_LAYOUT,
                    cache_file_name=cache_name(cache_key),
                )

            # Prefill can use broadcasting on the bias add so wants a single row
            self.wqkv_bias_prefill = bias_to_mesh(qkv_bias.unsqueeze(0), "wqkv_bias_prefill_sharded")
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
                self.wqkv_bias_decode.append(
                    bias_to_mesh(qkv_bias_decode, f"wqkv_bias_decode_sharded_{batch_size}")
                )

        # when splitting the devices, we need to make sure that the number of heads is divisible by the number of devices
        assert self.n_heads % self.num_devices_per_group == 0
        assert self.n_kv_heads % self.num_devices_per_group == 0
        assert configuration.qkv_size % self.num_devices_per_group == 0
        assert configuration.dim % self.num_devices_per_group == 0

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
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=self.sharding.col_dims,
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wqkv_sharded_2d"),
        )

        def norm_reshard(x, norm, mode, norm_config):
            """Hack until RMSNorm supports height-sharded output config"""
            if mode == Mode.DECODE:
                mem_cfg = x.memory_config()
                x = ttnn.to_memory_config(x, ttnn.L1_MEMORY_CONFIG, dtype=x.dtype)
            x = norm(x, mode, norm_config=norm_config)
            if mode == Mode.DECODE:
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
                tt_ccl=self.tt_ccl,
            )
            self.q_norm = lambda x, mode, norm_config: norm_reshard(x, fn_q_norm, mode, norm_config)
        else:
            self.q_norm = lambda x, mode, norm_config: x

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
                tt_ccl=self.tt_ccl,
            )
            self.k_norm = lambda x, mode, norm_config: norm_reshard(x, fn_k_norm, mode, norm_config)
        else:
            self.k_norm = lambda x, mode, norm_config: x

        pt_wo = state_dict[f"{wo_str}.weight"].transpose(-1, -2).unsqueeze(0).unsqueeze(0)

        self.wo = ttnn.as_tensor(
            pt_wo,
            dtype=self.wo_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device,
                dims=self.sharding.row_dims,
                mesh_shape=configuration.cluster_shape,
            ),
            cache_file_name=cache_name("wo_width_sharded_2d"),
        )
        if not use_paged_kv_cache:
            # vLLM provides its own kv cache
            self.init_kv_cache(configuration, weight_cache_path)

        if configuration.query_pre_attn_scalar is not None:
            self.scale = configuration.query_pre_attn_scalar**-0.5
        else:
            self.scale = self.head_dim**-0.5

    def _kv_prefill_mem_config(self, seq_len):
        """Height-sharded config for the KV cache fill, sized to this placement's local KV heads."""
        return ttnn.create_sharded_memory_config(
            ((self.n_local_kv_heads * seq_len // (8 * 8)), self.head_dim),
            ttnn.CoreGrid(y=8, x=8),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
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
                dtype=self.kv_cache_dtype,
                layout=self.args.get_attn_weights_layout(),
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
        - Places Q and K on disjoint core regions to avoid overlap within sub_core_grids
        - Uses row-major shard orientation with explicit shard shapes

        The resulting memory layouts are compatible with fused attention
        kernels that expect Q and K to be distributed across separate
        core ranges while preserving per-head contiguity.

        Args:
            q_tensor (ttnn.Tensor):
                Query tensor with shape [..., batch, num_q_heads, head_dim].

            k_tensor (ttnn.Tensor):
                Key tensor with shape [..., batch, num_kv_heads, head_dim].

            sub_core_grids (ttnn.CoreRangeSet):
                The available core grids to place Q and K tensors on.

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

    def _mllama_rope_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        # Q Rotary Embeddings
        q_heads_1BQD = ttnn.experimental.rotary_embedding_llama(
            q_heads_pre_rot_1BQD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )

        # K Rotary Embeddings
        k_heads_1BKD = ttnn.experimental.rotary_embedding_llama(
            k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"], is_decode_mode=True
        )
        return q_heads_1BQD, k_heads_1BKD

    def _mllama_rope_fused_qk_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD = self.to_qk_fused_memory_config(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD
        )

        q_heads_1BQD, k_heads_1BKD = ttnn.experimental.rotary_embedding_llama_fused_qk(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats[0], rot_mats[1], self.transformation_mats["decode"]
        )
        return q_heads_1BQD, k_heads_1BKD

    def _hf_rope_decode(self, q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos):
        if q_heads_pre_rot_1BQD.dtype != ttnn.bfloat16:
            q_heads_pre_rot_1BQD = ttnn.typecast(q_heads_pre_rot_1BQD, dtype=ttnn.bfloat16)
        if k_heads_pre_rot_1BKD.dtype != ttnn.bfloat16:
            k_heads_pre_rot_1BKD = ttnn.typecast(k_heads_pre_rot_1BKD, dtype=ttnn.bfloat16)

        q_heads_1BQD = ttnn.experimental.rotary_embedding_hf(
            q_heads_pre_rot_1BQD,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=True,
        )
        k_heads_1BKD = ttnn.experimental.rotary_embedding_hf(
            k_heads_pre_rot_1BKD,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=True,
        )
        return q_heads_1BQD, k_heads_1BKD

    def _mllama_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        # FIXME: (for 2d) rotary_embedding_llama requires bf16 inputs. On 2D meshes the QKV all-reduce
        # downcasts to ccl_dtype (bf8), so cast back here (matches _hf_rope_prefill).
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)
        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_llama(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )

        k_heads_1KSD = ttnn.experimental.rotary_embedding_llama(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            self.transformation_mats["prefill"],
            is_decode_mode=False,
        )

        return q_heads_1QSD, k_heads_1KSD

    def _hf_rope_prefill(self, q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats):
        if q_heads_1QSD_pre_rot.dtype != ttnn.bfloat16:
            q_heads_1QSD_pre_rot = ttnn.typecast(q_heads_1QSD_pre_rot, dtype=ttnn.bfloat16)

        q_heads_1QSD = ttnn.experimental.rotary_embedding_hf(
            q_heads_1QSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=False,
        )

        if k_heads_1KSD_pre_rot.dtype != ttnn.bfloat16:
            k_heads_1KSD_pre_rot = ttnn.typecast(k_heads_1KSD_pre_rot, dtype=ttnn.bfloat16)

        k_heads_1KSD = ttnn.experimental.rotary_embedding_hf(
            k_heads_1KSD_pre_rot,
            rot_mats[0],
            rot_mats[1],
            is_decode_mode=False,
        )

        return q_heads_1QSD, k_heads_1KSD

    def forward_decode(self, x: ttnn.Tensor, current_pos, rot_mats=None, page_table=None, kv_cache=None) -> ttnn.Tensor:
        """
        x: (seq_len, 1, batch, dim)
        current_pos: (batch_size), current token position in the sequence for each user
        """
        ###
        # QKV matmuls
        # Use HiFi2 for DRAM-sharded matmuls as they are otherwise flop-bound. Loses 1 bit of activation precision.
        ###
        # Bias (select by row-tile count) is added to the matmul output before the reduce, whether
        # unfused or fused. Compute it up front from x's seq dim (== the matmul output's rows).
        wqkv_bias = None
        if self.wqkv_bias_decode:
            # WARNING: must not change the batch size between compiling and executing a trace
            num_tiles = int(math.ceil(x.shape[-2] / self.tile_size))
            wqkv_bias = self.wqkv_bias_decode[num_tiles - 1]

        # --- unfused path (kept for reference) ---
        # xqkv_fused = ttnn.linear(
        #     x,
        #     self.wqkv,
        #     # wqkv is DRAM-interleaved + 2D-sharded, so a plain matmul with a DRAM output (like prefill).
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
        #     dtype=self.activation_dtype or ttnn.bfloat16,
        # )
        # if wqkv_bias is not None:  # FIXME: File bug against dram-sharded matmuls with bias
        #     xqkv_fused = xqkv_fused + wqkv_bias
        # ttnn.deallocate(x)
        # # Reduce the QKV partials over the hidden axis; a no-op when the hidden isn't split.
        # # replicate=True: every chip needs the full qkv to create its heads.
        # xqkv_fused = all_reduce(xqkv_fused, self.mesh_device, axis=self.sharding.reduce_col_over,
        #                         replicate=True, dtype=self.ccl_dtype, topology=self.ccl_topology)

        # --- fused path: matmul + reduce_scatter overlapped (bias folded in, added pre-reduce;
        # replicate=True -> trailing all_gather gives every chip the full qkv for head creation). ---
        # ATTN_QKV_CORES=<n> forces the QKV matmul core grid; unset = ttnn default.
        log_default_grid(x, self.wqkv, "ATTN qkv")
        qkv_pc = core_grid_program_config(
            self.args, "ATTN_QKV_CORES", x.shape[-2], x.shape[-1], self.wqkv.shape[-1], "ATTN qkv"
        )
        xqkv_fused = matmul_reduce_scatter(
            x,
            self.wqkv,
            self.mesh_device,
            self.tt_ccl,
            axis=self.sharding.reduce_col_over,
            replicate=True,
            bias=wqkv_bias,
            dtype=self.ccl_dtype,
            compute_kernel_config=self.li_qkv_decode_compute_kernel_cfg,
            program_config=qkv_pc,
            topology=self.ccl_topology,
            label="ATTN qkv [decode]",
        )
        ttnn.deallocate(x)

        # bfloat16 (in L1) is required by nlp_create_qkv_heads_decode.
        xqkv_fused = ttnn.to_memory_config(xqkv_fused, ttnn.L1_MEMORY_CONFIG, ttnn.bfloat16)

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
            memory_config=self.args.get_attn_create_head_output_mem_config(Mode.DECODE, None),
        )
        norm_config = self.args.get_norm_config("attn", Mode.DECODE, None)
        q_heads_pre_rot_1BQD = self.q_norm(q_heads_pre_rot_1BQD, mode=Mode.DECODE, norm_config=norm_config)
        k_heads_pre_rot_1BKD = self.k_norm(k_heads_pre_rot_1BKD, mode=Mode.DECODE, norm_config=norm_config)
        ttnn.deallocate(xqkv_fused)

        # Q, K Rotary Embeddings
        q_heads_1BQD, k_heads_1BKD = self.rotary_embedding_decode(
            q_heads_pre_rot_1BQD, k_heads_pre_rot_1BKD, rot_mats, current_pos
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

        if self.use_qk_fused:
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
        sdpa_decode_prog_cfg = self.args.get_attn_sdpa_decode_program_config(None)
        if page_table is not None:
            attn_output_1G4D = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                q_heads_1BQD,
                keys,
                values,
                page_table_tensor=page_table,
                cur_pos_tensor=current_pos,
                scale=self.scale,
                sliding_window_size=self.sliding_window,
                program_config=sdpa_decode_prog_cfg,
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
                program_config=sdpa_decode_prog_cfg,
                compute_kernel_config=self.sdpa_decode_compute_kernel_cfg,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,  # FIXME: why not L1 height sharded e.g. SCORES_BATCHED_MM_OUTPUT_MEMCFG?
            )

        ttnn.deallocate(q_heads_1BQD)
        attn_output_11BH = ttnn.to_memory_config(
            attn_output_1G4D,
            memory_config=self.args.get_attn_sdpa_output_mem_config(
                Mode.DECODE, self.batch_size_per_device_group, None
            ),
        )

        attn_output_cat = ttnn.experimental.nlp_concat_heads_decode(
            attn_output_11BH,
            num_heads=self.n_local_heads,
        )
        ttnn.deallocate(attn_output_11BH)
        ttnn.deallocate(attn_output_1G4D)

        # Heads are sharded on the head axis and replicated across the group axis, so there is no
        # users all-gather: each device runs its own Wo matmul and the head-split partials are
        # summed by the Wo all-reduce below.
        attn_output = ttnn.to_memory_config(attn_output_cat, ttnn.DRAM_MEMORY_CONFIG)

        # wo is DRAM-interleaved + 2D-sharded -> plain matmul, DRAM output.

        # --- unfused path (kept for reference) ---
        # dense_out = ttnn.linear(
        #     attn_output,
        #     self.wo,
        #     program_config=None,
        #     memory_config=ttnn.DRAM_MEMORY_CONFIG,
        #     compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
        # )
        # ttnn.deallocate(attn_output_cat)
        # # Reduce the Wo partials over the head axis; a no-op when the heads aren't split.
        # # replicate=False: on a line the output stays split along the head axis (a reduce_scatter).
        # dense_out = all_reduce(dense_out, self.mesh_device, axis=self.sharding.reduce_row_over,
        #                        replicate=False, dtype=self.ccl_dtype, topology=self.ccl_topology)

        # --- fused path: matmul + reduce_scatter overlapped (replicate=False -> no trailing
        # all_gather; the head-split result is what the decoder's residual expects). ---
        log_default_grid(attn_output, self.wo, "ATTN dense_out")
        dense_out = matmul_reduce_scatter(
            attn_output,
            self.wo,
            self.mesh_device,
            self.tt_ccl,
            axis=self.sharding.reduce_row_over,
            replicate=False,
            dtype=self.ccl_dtype,
            compute_kernel_config=self.li_o_decode_compute_kernel_cfg,
            topology=self.ccl_topology,
            label="ATTN dense_out [decode]",
        )
        ttnn.deallocate(attn_output_cat)

        return dense_out

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
        # For batched prefill, x_11SH has shape [B, 1, S, H] where B is batch_size
        # concat before QKV matmul, then reshape back to batch after
        batch_size = x_11SH.shape[0]
        if batch_size > 1:
            # Concatenate batch dimension into sequence for matmul compatibility
            x_11SH = ttnn.reshape(x_11SH, [1, 1, x_11SH.shape[-2] * x_11SH.shape[-3] * x_11SH.shape[-4], -1])

        seq_len = x_11SH.shape[-2]
        original_seq_len = seq_len  # Track original for later unpadding
        assert seq_len % 128 == 0 and seq_len > 0, "Seqlen must be divisible by 128"
        ###
        # QKV matmuls
        ###

        # reshaping long sequence to matmul fit on device
        # Pad seq_len to nearest multiple of MAX_QKV_MM_SEQ_LEN if needed
        if seq_len > self.MAX_QKV_MM_SEQ_LEN and seq_len % self.MAX_QKV_MM_SEQ_LEN != 0:
            padded_seq_len = (
                (seq_len + self.MAX_QKV_MM_SEQ_LEN - 1) // self.MAX_QKV_MM_SEQ_LEN
            ) * self.MAX_QKV_MM_SEQ_LEN
            pad_len = padded_seq_len - seq_len
            x_11SH = ttnn.pad(x_11SH, padding=[(0, 0), (0, 0), (0, pad_len), (0, 0)], value=0.0)
            seq_len = padded_seq_len

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            x_11SH = ttnn.reshape(x_11SH, [1, seq_len // self.MAX_QKV_MM_SEQ_LEN, self.MAX_QKV_MM_SEQ_LEN, -1])

        if self.args.use_minimal_qkv_prefill_matmul(seq_len):
            # TODO: current path
            xqkv_fused = ttnn.experimental.minimal_matmul(
                x_11SH,
                self.wqkv,
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                config=self.args.get_attn_qkv_program_config(Mode.PREFILL, seq_len, None),
            )
        else:
            xqkv_fused = ttnn.linear(
                x_11SH,
                self.wqkv,
                dtype=self.activation_dtype or ttnn.bfloat16,
                memory_config=self.args.get_attn_qkv_mm_mem_config(Mode.PREFILL, None),
                compute_kernel_config=self.li_qkv_prefill_compute_kernel_cfg,
                # AUTO-SHARD FIX: pass the real head-shard count so per_core_N matches the actual
                # per-device QKV output width (not the hardcoded cluster_shape[1] assumption).
                program_config=self.args.get_attn_qkv_program_config(
                    Mode.PREFILL, seq_len, None, num_head_shards=self.sharding.num_intermediate_shards
                ),
            )

        # FIXME: surely ttnn.linear bias should work?
        if self.wqkv_bias_prefill is not None:
            xqkv_fused = xqkv_fused + self.wqkv_bias_prefill

        # Reduce the QKV partials over the hidden axis; a no-op when the hidden isn't split.
        # replicate=True: every chip needs the full qkv to create its heads.
        xqkv_fused = all_reduce(
            xqkv_fused,
            self.mesh_device,
            axis=self.sharding.reduce_col_over,
            replicate=True,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )

        if seq_len > self.MAX_QKV_MM_SEQ_LEN:
            xqkv_fused = ttnn.reshape(xqkv_fused, [1, 1, seq_len, -1])

        # Slice back to original seq_len if we padded earlier
        if original_seq_len != seq_len:
            xqkv_fused = xqkv_fused[:, :, :original_seq_len, :]
            seq_len = original_seq_len

        if batch_size > 1:
            xqkv_fused = ttnn.reshape(xqkv_fused, [batch_size, 1, seq_len // batch_size, -1])

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

        norm_config = self.args.get_norm_config("attn", Mode.PREFILL, None)
        q_heads_1QSD_pre_rot = self.q_norm(q_heads_1QSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)
        k_heads_1KSD_pre_rot = self.k_norm(k_heads_1KSD_pre_rot, mode=Mode.PREFILL, norm_config=norm_config)

        ttnn.deallocate(xqkv_fused)

        ###
        # Rotary embeddings
        ###

        # Apply rotary embeddings using the selected implementation
        q_heads_1QSD, k_heads_1KSD = self.rotary_embedding_prefill(q_heads_1QSD_pre_rot, k_heads_1KSD_pre_rot, rot_mats)
        ttnn.deallocate(q_heads_1QSD_pre_rot)
        ttnn.deallocate(k_heads_1KSD_pre_rot)

        # Fill KV-Cache
        if kv_cache:
            keys_BKSD, values_BKSD = kv_cache[0], kv_cache[1]
        else:
            keys_BKSD, values_BKSD = self.layer_past[0], self.layer_past[1]

        k_heads_1KSD_8b = ttnn.typecast(k_heads_1KSD, dtype=keys_BKSD.dtype)
        ttnn.deallocate(k_heads_1KSD)

        # sharding k_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and page_table is None:
            k_fill = ttnn.interleaved_to_sharded(k_heads_1KSD_8b, self._kv_prefill_mem_config(seq_len))
        else:
            k_fill = k_heads_1KSD_8b

        v_heads_1VSD_8b = ttnn.typecast(v_heads_1VSD, dtype=values_BKSD.dtype)

        ttnn.deallocate(v_heads_1VSD)

        # sharding v_fill to deal with update_cache memory limitation
        if seq_len >= self.min_kv_prefill_shard_seqlen and page_table is None:
            v_fill = ttnn.interleaved_to_sharded(v_heads_1VSD_8b, self._kv_prefill_mem_config(seq_len))
        else:
            v_fill = v_heads_1VSD_8b

        # Subset to this group's chips only under DATA parallelism, where each group holds a
        # different user's cache and writing every chip would clobber the other users.
        #
        # num_device_groups is num_model_shards, a TENSOR-parallel axis, so the old condition
        # (both > 1) also fired for pure TP -- i0/m1 on a 2x2. There the striding is wrong: the
        # cache is ReplicateTensorToMesh sized for this chip's head group, and k_fill is already
        # split on the head axis and identical along the model axis (the column matmul's all-reduce
        # replicates on a 2D mesh), so every chip already holds exactly what it must store. Striding
        # by num_model_shards writes one chip per head group and leaves the other half of the mesh
        # with an unfilled cache -- Qwen2.5-7B on 2x2 at i0/m1 generates correctly for ~60 tokens and
        # then degenerates, which is that stale cache taking over as decode leans on it.
        #
        # batch is not split in this path (batch_size_per_device_group == max_batch_size), so this
        # is currently always False; kept as the real condition rather than deleted.
        batch_is_split_across_groups = self.batch_size_per_device_group < self.max_batch_size
        if batch_is_split_across_groups and self.num_device_groups > 1 and self.num_devices_per_group > 1:
            k_fill = self.prefill_prepare_tensor_for_kv_cache(k_fill, user_id)
            v_fill = self.prefill_prepare_tensor_for_kv_cache(v_fill, user_id)
        if page_table is not None:
            # In the case that the tokens have been padded along the seq len dimension, we need to fill the cache with the unpadded k/v values.
            # Assume that the page table does not have padding, so we can use it to get the unpadded page len.
            block_size = keys_BKSD.shape[2]
            # If chunked prefill, use chunk_page_table if given, otherwise use page_table.
            fill_page_table = chunk_page_table if chunk_page_table is not None else page_table

        if batch_size > 1:
            # For batched prefill, loop over VALID users only and fill each user's cache separately
            # k_fill/v_fill have shape [padded_batch, n_kv_heads, seq_len_per_user, head_dim]
            # The paged_fill_cache kernel reads batch_idx_ptr[0] for all positions,
            # so we must call it once per user with their specific K/V slice
            #
            # IMPORTANT: user_id is a list of valid slot indices for batched prefill.
            # Empty slots have page_table entries of -1, so we must skip them to avoid
            # writing to invalid memory blocks.
            seq_len_per_user = k_fill.shape[2]
            page_len = fill_page_table.shape[1] * block_size

            # user_id is a list of valid slot indices (e.g., [0, 1, 2, ..., N-1] for N users)
            # Each slot index tells us which row in k_fill and page_table to use
            valid_slots = user_id if isinstance(user_id, (list, tuple)) else list(range(batch_size))

            for slot_idx in valid_slots:
                # Extract this slot's K/V slice: [1, n_kv_heads, seq_len_per_user, head_dim]
                k_user = k_fill[slot_idx : slot_idx + 1, :, :, :]
                v_user = v_fill[slot_idx : slot_idx + 1, :, :, :]

                # Slice to page length if needed (same as single-user path)
                k_user_sliced = k_user[:, :, :page_len, :] if page_len < seq_len_per_user else k_user
                v_user_sliced = v_user[:, :, :page_len, :] if page_len < seq_len_per_user else v_user

                # Fill cache for this specific slot with scalar batch_idx
                ttnn.experimental.paged_fill_cache(keys_BKSD, k_user_sliced, fill_page_table, batch_idx=slot_idx)
                ttnn.experimental.paged_fill_cache(values_BKSD, v_user_sliced, fill_page_table, batch_idx=slot_idx)
        elif page_table is not None:
            # Single user path with page_table
            page_len = fill_page_table.shape[1] * block_size
            k_fill_sliced = k_fill[:, :, :page_len, :] if page_len < k_fill.shape[2] else k_fill
            v_fill_sliced = v_fill[:, :, :page_len, :] if page_len < v_fill.shape[2] else v_fill
            ttnn.experimental.paged_fill_cache(keys_BKSD, k_fill_sliced, fill_page_table, batch_idx=user_id)
            ttnn.experimental.paged_fill_cache(values_BKSD, v_fill_sliced, fill_page_table, batch_idx=user_id)
        else:
            # Single user path without page_table
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
        if seq_len >= self.min_kv_prefill_shard_seqlen and page_table is None:
            ttnn.deallocate(k_fill)
            ttnn.deallocate(v_fill)

        # SDPA
        q_heads_1QSD_8b = ttnn.typecast(q_heads_1QSD, dtype=self.activation_dtype or ttnn.bfloat8_b)
        ttnn.deallocate(q_heads_1QSD)

        if chunk_start_idx is not None:
            if self.sliding_window is not None:
                raise NotImplementedError("Sliding window not supported for chunked prefill SDPA")
            if isinstance(chunk_start_idx, ttnn.Tensor):
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx=None,
                    chunk_start_idx_tensor=chunk_start_idx,
                    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                    program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, 0, None),
                )
            else:
                attn_output_84SD = ttnn.transformer.chunked_scaled_dot_product_attention(
                    input_tensor_q=q_heads_1QSD_8b,
                    input_tensor_k=keys_BKSD,
                    input_tensor_v=values_BKSD,
                    page_table_tensor=page_table,
                    chunk_start_idx=chunk_start_idx,
                    compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                    program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, seq_len, chunk_start_idx, None),
                )
        else:
            # For batched prefill, the actual per-user seq_len is seq_len // batch_size
            # since the tensors have shape [batch_size, n_heads, seq_len_per_user, head_dim]
            sdpa_seq_len = seq_len // batch_size if batch_size > 1 else seq_len
            attn_output_84SD = ttnn.transformer.scaled_dot_product_attention(
                q_heads_1QSD_8b,
                k_heads_1KSD_8b,
                v_heads_1VSD_8b,
                is_causal=True,
                sliding_window_size=self.sliding_window,
                scale=self.scale,
                compute_kernel_config=self.sdpa_prefill_compute_kernel_cfg,
                program_config=self.args.get_attn_sdpa_program_config(Mode.PREFILL, sdpa_seq_len, None, None),
            )

        # deallocate keys and values
        ttnn.deallocate(q_heads_1QSD_8b)
        ttnn.deallocate(k_heads_1KSD_8b)
        ttnn.deallocate(v_heads_1VSD_8b)

        # For single-user prefill, reshape to expected format for nlp_concat_heads
        # For batched prefill (batch_size > 1), skip this reshape - nlp_concat_heads handles [B, H, S, D]
        # IMPORTANT: Reshaping [B, H, S, D] to [1, H, B*S, D] BEFORE concat_heads would scramble data
        # because batch and sequence dimensions are separated by heads. Must reshape AFTER concat_heads.
        if batch_size == 1:
            attn_output_1QSD = ttnn.reshape(attn_output_84SD, [1, self.n_local_heads, -1, self.head_dim])
        else:
            attn_output_1QSD = attn_output_84SD

        ###
        # Output matmul
        ###
        attn_output_11SH = ttnn.experimental.nlp_concat_heads(
            attn_output_1QSD,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        ttnn.deallocate(attn_output_1QSD)

        # For batched prefill, reshape to concatenate batch dimension into sequence
        # This MUST happen AFTER nlp_concat_heads to preserve correct data layout
        # nlp_concat_heads outputs [B, 1, S_per_user, H*D], reshape to [1, 1, B*S, H*D]
        if batch_size > 1:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, 1, seq_len, -1])

        # reshaping long sequence to matmul fit on device
        if seq_len > 1024:
            attn_output_11SH = ttnn.reshape(attn_output_11SH, [1, seq_len // 1024, 1024, -1])

        output_11SH = ttnn.linear(
            attn_output_11SH,
            self.wo,
            compute_kernel_config=self.li_o_prefill_compute_kernel_cfg,
            dtype=self.activation_dtype or ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            program_config=self.args.get_attn_wo_program_config(Mode.PREFILL, seq_len, None),
        )

        if seq_len > 1024:
            output_11SH = ttnn.reshape(output_11SH, [1, 1, seq_len, -1])
        ttnn.deallocate(attn_output_11SH)

        # Reduce the Wo partials over the head axis; a no-op when the heads aren't split.
        # replicate=False: on a line the output stays split along the head axis (a reduce_scatter).
        output_11SH = all_reduce(
            output_11SH,
            self.mesh_device,
            axis=self.sharding.reduce_row_over,
            replicate=False,
            dtype=self.ccl_dtype,
            topology=self.ccl_topology,
        )

        return output_11SH

    def forward(
        self,
        x,
        current_pos,
        rot_mats=None,
        user_id=0,
        mode=Mode.DECODE,
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        kv_cache=None,
    ):
        if mode == Mode.PREFILL:
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
        single_column_tensors = tensors[user_id // self.batch_size_per_device_group :: self.num_device_groups]
        # Create multi-device tensor
        multi_device_tensor = ttnn.combine_device_tensors(tensors=single_column_tensors)

        return multi_device_tensor
