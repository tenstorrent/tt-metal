# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

import math

import torch
from transformers.configuration_utils import PretrainedConfig

import ttnn
from models.common.modules.tt_ccl import get_tt_ccl


class ttMLA:
    def __init__(
        self,
        config: PretrainedConfig,
        state_dict: dict[str, torch.Tensor],
        mesh_device: ttnn.MeshDevice,
        layer_idx: int = 0,
        seq_len: int = 1024,
        sp_axis: int = 0,
        tp_axis: int = 1,
    ):
        self.config = config
        self.mesh_device = mesh_device
        self.layer_idx = layer_idx

        # Extract dimensions from config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.kv_lora_rank = config.kv_lora_rank
        self.q_lora_rank = config.q_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        rope_factor = config.rope_scaling["factor"]
        mscale = config.rope_scaling["mscale"]

        self.scale = self.qk_head_dim**-0.5
        if rope_factor > 1.0:
            mscale = 0.1 * mscale * math.log(rope_factor) + 1.0
            self.scale = self.scale * mscale * mscale

        self.default_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,
            packer_l1_acc=True,
        )

        self.hifi4_fp32_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        self.ring_sdpa_compute_grid = (
            mesh_device.compute_with_storage_grid_size().x,
            mesh_device.compute_with_storage_grid_size().y - 1,
        )
        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.ring_sdpa_compute_grid,
            q_chunk_size=32,
            k_chunk_size=32,
            exp_approx_mode=False,
        )

        # Create CCL object for semaphore management
        self.tt_ccl = get_tt_ccl(mesh_device)
        self.tp_factor = mesh_device.shape[tp_axis]

        # ring attention setup
        persistent_v_shard_dims = [None, None]
        persistent_v_shard_dims[tp_axis] = 1  # TP heads
        persistent_k_shard_dims = [None, None]

        ag_output_shape_k = (1, 1, seq_len, self.kv_lora_rank + self.qk_rope_head_dim)
        ag_output_shape_v = (1, self.num_heads, seq_len, self.v_head_dim)

        self.persistent_k_output_buffer = ttnn.from_torch(
            torch.zeros(ag_output_shape_k),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=persistent_k_shard_dims
            ),
        )

        self.persistent_v_output_buffer = ttnn.from_torch(
            torch.zeros(ag_output_shape_v),
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=persistent_v_shard_dims
            ),
        )

        # dummy joint tensors
        dummy_joint_q = torch.zeros(1, self.num_heads, 0, self.kv_lora_rank + self.qk_rope_head_dim)
        dummy_joint_k = torch.zeros(1, 1, 0, self.kv_lora_rank + self.qk_rope_head_dim)
        dummy_joint_v = torch.zeros(1, self.num_heads, 0, self.v_head_dim)

        joint_qv_shard_dims = [None, None]
        joint_qv_shard_dims[tp_axis] = 1
        self.tt_dummy_joint_q = ttnn.from_torch(
            dummy_joint_q,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=joint_qv_shard_dims
            ),
        )

        self.tt_dummy_joint_k = ttnn.from_torch(
            dummy_joint_k,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        self.tt_dummy_joint_v = ttnn.from_torch(
            dummy_joint_v,
            device=self.mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat8_b,  # hardcoded for now
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(
                self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=joint_qv_shard_dims
            ),
        )

        # Load weights to TT device
        self._load_weights(state_dict)

    def _load_weights(self, state_dict: dict[str, torch.Tensor]):
        """
        Load weights from state dict and convert to TT tensors.

        Expected keys in state_dict:
        - q_a_proj.weight
        - q_a_layernorm.weight
        - q_b_proj.weight
        - kv_a_proj_with_mqa.weight
        - kv_a_layernorm.weight
        - kv_b_proj.weight
        - o_proj.weight
        """

        # Mesh Device = (sp x tp)
        q_a_ln_weight = state_dict["q_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        self.q_a_layernorm_weight = self._to_tt_tensor(
            q_a_ln_weight,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        kv_a_ln_weight = state_dict["kv_a_layernorm.weight"].reshape(1, 1, -1, ttnn.TILE_SIZE)
        self.kv_a_layernorm_weight = self._to_tt_tensor(
            kv_a_ln_weight,
            ttnn.bfloat16,
            ttnn.ROW_MAJOR_LAYOUT,
            ttnn.ReplicateTensorToMesh(self.mesh_device),
        )

        q_a_proj = state_dict["q_a_proj.weight"]
        q_a_proj = q_a_proj.transpose(-2, -1)

        shard_dims = [None, None]
        tp_axis = 1
        sp_axis = 0
        shard_dims[tp_axis] = 0
        mesh_mapper = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
        )
        self.q_a_proj_weight = self._to_tt_tensor(q_a_proj, ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_mapper)

        shard_dims[tp_axis] = 1
        shard_dims[sp_axis] = None
        mesh_mapper_q_b_proj = ttnn.ShardTensor2dMesh(
            self.mesh_device, mesh_shape=tuple(self.mesh_device.shape), dims=shard_dims
        )
        self.q_b_proj_weight = self._to_tt_tensor(
            state_dict["q_b_proj.weight"].transpose(-2, -1),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_mapper_q_b_proj,
        )

        self.kv_a_proj_with_mqa_weight = self._to_tt_tensor(
            state_dict["kv_a_proj_with_mqa.weight"].transpose(-2, -1),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_mapper,
        )
        kv_b_proj_weights = state_dict["kv_b_proj.weight"].reshape(
            1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )

        torch_weights_k = kv_b_proj_weights[..., : self.qk_nope_head_dim, :].transpose(-2, -1)
        torch_weights_v = kv_b_proj_weights[..., self.qk_nope_head_dim :, :]

        shard_dims[tp_axis] = 1
        shard_dims[sp_axis] = None
        self.wkv_b1_weight = self._to_tt_tensor(
            torch_weights_k.transpose(-2, -1),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_mapper_q_b_proj,
        )
        self.wkv_b2_weight = self._to_tt_tensor(
            torch_weights_v.transpose(-2, -1),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            mesh_mapper_q_b_proj,
        )

        self.o_proj_weight = self._to_tt_tensor(
            state_dict["o_proj.weight"].transpose(-2, -1), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, mesh_mapper
        )

        print(f"✓ Loaded {len(state_dict)} weights in MLA layer {self.layer_idx} to TT device")

    def _to_tt_tensor(
        self, tensor: torch.Tensor, dtype: ttnn.DataType, layout: ttnn.Layout, mesh_mapper: ttnn.TensorToMesh
    ) -> ttnn.Tensor:
        return ttnn.from_torch(
            tensor,
            device=self.mesh_device,
            dtype=dtype,
            layout=layout,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_mapper,
        )

    def get_weight_shapes(self) -> dict[str, tuple]:
        return {
            "q_a_proj.weight": tuple(self.q_a_proj_weight.shape),
            "q_a_layernorm.weight": tuple(self.q_a_layernorm_weight.shape),
            "q_b_proj.weight": tuple(self.q_b_proj_weight.shape),
            "kv_a_proj_with_mqa.weight": tuple(self.kv_a_proj_with_mqa_weight.shape),
            "kv_a_layernorm.weight": tuple(self.kv_a_layernorm_weight.shape),
            "wkv_b1_weight": tuple(self.wkv_b1_weight.shape),
            "wkv_b2_weight": tuple(self.wkv_b2_weight.shape),
            "o_proj.weight": tuple(self.o_proj_weight.shape),
        }

    # Expects ativation in form of:
    # [1, batch_size == 1, seq_len // sp_factor, hidden_size // tp_factor]
    def forward(self, hidden_states: ttnn.Tensor, rope_tensors: dict) -> ttnn.Tensor:
        mesh_size = self.mesh_device.shape
        sp_factor = mesh_size[0]
        tp_factor = mesh_size[1]

        num_heads_local = self.num_heads // tp_factor
        seq_len_local = hidden_states.shape[2]

        # q_projection
        tt_q = ttnn.linear(
            hidden_states,
            self.q_a_proj_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # All reduce
        tt_q = ttnn.experimental.reduce_scatter_minimal_async(
            tt_q,
            persistent_output_buffers=None,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
        )
        tt_q = ttnn.experimental.all_gather_async(
            tt_q,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
        )

        # rmsnorm
        tt_q = ttnn.rms_norm(
            tt_q,
            weight=self.q_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        tt_q = ttnn.linear(
            tt_q,
            self.q_b_proj_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # convert to
        # [batch (1), num_heads_local, seq_len_local, qk_head_dim]
        tt_q, _, _ = ttnn.experimental.nlp_create_qkv_heads(
            tt_q,
            num_heads=num_heads_local,
            num_kv_heads=0,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # split rope and nope
        tt_q_nope = ttnn.slice(tt_q, [0, 0, 0, 0], [1, num_heads_local, seq_len_local, self.qk_nope_head_dim])
        tt_q_rope = ttnn.slice(
            tt_q, [0, 0, 0, self.qk_nope_head_dim], [1, num_heads_local, seq_len_local, self.qk_head_dim]
        )

        tt_q_nope = ttnn.linear(
            tt_q_nope,
            self.wkv_b1_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        tt_q_rope = ttnn.experimental.rotary_embedding_llama(
            tt_q_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # concat rope and nope
        tt_q = ttnn.concat([tt_q_nope, tt_q_rope], dim=-1)
        tt_q = ttnn.typecast(tt_q, dtype=ttnn.bfloat8_b)

        # kv
        tt_kv = ttnn.linear(
            hidden_states,
            self.kv_a_proj_with_mqa_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        # All reduce
        tt_kv = ttnn.experimental.all_gather_async(
            tt_kv,
            dim=1,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_ag_semaphore_handles(cluster_axis=1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
        )
        tt_kv = ttnn.experimental.fast_reduce_nc(
            tt_kv, dims=[1], output=None, compute_kernel_config=self.hifi4_fp32_compute_kernel_config
        )

        # split rope and nope
        tt_kv_nope = ttnn.slice(tt_kv, [0, 0, 0, 0], [1, 1, seq_len_local, self.kv_lora_rank])
        tt_kv_rope = ttnn.slice(
            tt_kv, [0, 0, 0, self.kv_lora_rank], [1, 1, seq_len_local, self.kv_lora_rank + self.qk_rope_head_dim]
        )
        ttnn.deallocate(tt_kv)

        tt_kv_nope = ttnn.rms_norm(
            tt_kv_nope,
            weight=self.kv_a_layernorm_weight,
            epsilon=self.config.rms_norm_eps,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        tt_kv_rope = ttnn.experimental.rotary_embedding_llama(
            tt_kv_rope,
            rope_tensors["cos_matrix"],
            rope_tensors["sin_matrix"],
            rope_tensors["trans_matrix"],
            is_decode_mode=False,
        )

        # concat rope and nope
        tt_kvpe = ttnn.concat([tt_kv_nope, tt_kv_rope], dim=-1)
        ttnn.deallocate(tt_kv_rope)
        tt_kvpe = ttnn.typecast(tt_kvpe, dtype=ttnn.bfloat8_b)

        # expand v with wkv_b2
        # workaround for #37416
        tt_v_latent_post_repeat = ttnn.repeat(tt_kv_nope, [1, num_heads_local, 1, 1])
        ttnn.deallocate(tt_kv_nope)

        tt_v_embedding = ttnn.linear(
            tt_v_latent_post_repeat,
            self.wkv_b2_weight,
            dtype=ttnn.bfloat8_b,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )

        attn_out, _, _ = ttnn.transformer.ring_joint_scaled_dot_product_attention(
            tt_q,
            tt_kvpe,
            tt_v_embedding,
            self.tt_dummy_joint_q,
            self.tt_dummy_joint_k,
            self.tt_dummy_joint_v,
            persistent_output_buffer_k=self.persistent_k_output_buffer,
            persistent_output_buffer_v=self.persistent_v_output_buffer,
            joint_strategy="rear",
            logical_n=seq_len_local * sp_factor,
            program_config=self.ring_sdpa_program_config,
            compute_kernel_config=self.default_compute_kernel_config,
            dim=2,
            multi_device_global_semaphore=self.tt_ccl.ring_attention_ccl_semaphore_handles,
            num_links=1,
            cluster_axis=0,
            mesh_device=self.mesh_device,
            topology=ttnn.Topology.Linear,
            subdevice_id=self.tt_ccl.worker_sub_device_id,
            ccl_core_grid_offset=self.tt_ccl.ring_attention_ccl_core_grid_offset,
            is_causal=True,
            scale=self.scale,
        )

        v_out = ttnn.experimental.nlp_concat_heads(attn_out, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        v_out = ttnn.linear(
            v_out,
            self.o_proj_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self.default_compute_kernel_config,
        )
        out = ttnn.experimental.reduce_scatter_minimal_async(
            v_out,
            dim=3,
            multi_device_global_semaphore=self.tt_ccl.get_and_cycle_rs_semaphore_handles(cluster_axis=1),
            barrier_semaphore=self.tt_ccl.get_and_cycle_barrier_semaphore_handle(cluster_axis=1),
            num_links=1,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Linear,
            cluster_axis=1,
        )
        return out
