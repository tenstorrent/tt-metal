# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ....layers.normalization import DistributedRMSNorm
from ....layers.linear import ColParallelLinear
from ....utils.substate import substate
from ....utils.tensor import bf16_tensor
from models.common.utility_functions import is_blackhole


class WanAttention:
    # Map from (is_blackhole, sp_factor, tp_factor) -> (q_chunk_size, k_chunk_size)
    sdpa_chunk_size_map = {
        (False, 2, 4): (256, 256),
        (False, 8, 4): (256, 256),
        (True, 2, 2): (128, 512),
        (True, 8, 4): (128, 512),
    }
    default_sdpa_chunk_size = (256, 256)

    def __init__(
        self,
        dim,
        num_heads,
        qk_norm=True,
        eps=1e-5,
        mesh_device=None,
        ccl_manager=None,
        parallel_config=None,
        is_fsdp=False,
    ):
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.n_local_heads = self.num_heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
            "mesh_axis": parallel_config.tensor_parallel.mesh_axis,
            "ccl_manager": ccl_manager,
        }

        self.norm_q = DistributedRMSNorm(**rms_kwargs)
        self.norm_k = DistributedRMSNorm(**rms_kwargs)

        # Unfused qkv because this might be cross attention
        self.to_q = ColParallelLinear(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.to_k = ColParallelLinear(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )
        self.to_v = ColParallelLinear(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.to_out = ColParallelLinear(
            dim,
            dim,
            bias=True,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.dummy_joint_input = bf16_tensor(torch.zeros((1, self.n_local_heads, 0, self.head_dim)), device=mesh_device)

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=full_grid,
            q_chunk_size=256,
            k_chunk_size=256,
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        ring_sdpa_chunk_size = self.sdpa_chunk_size_map.get(
            (
                is_blackhole(),
                self.parallel_config.sequence_parallel.factor,
                self.parallel_config.tensor_parallel.factor,
            ),
            self.default_sdpa_chunk_size,
        )

        self.ring_sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=ring_sdpa_chunk_size[0],
            k_chunk_size=ring_sdpa_chunk_size[1],
            exp_approx_mode=False,  # NOTE: False is more correct
        )

        self.sdpa_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=False,
            fp32_dest_acc_en=False,  # NOTE: Set to True if there's a correctness issue
        )

        self.rope_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        device_grid = self.mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(x=device_grid.x, y=device_grid.y)

        self.mm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

    def to_cached_state_dict(self, path_prefix):
        cache_dict = {}

        # Cache normalization layers
        norm_q_cache = self.norm_q.to_cached_state_dict(path_prefix + "norm_q.")
        norm_k_cache = self.norm_k.to_cached_state_dict(path_prefix + "norm_k.")

        # Add norm prefixes to all keys
        for key, value in norm_q_cache.items():
            cache_dict[f"norm_q.{key}"] = value
        for key, value in norm_k_cache.items():
            cache_dict[f"norm_k.{key}"] = value

        # Cache linear layers
        to_q_cache = self.to_q.to_cached_state_dict(path_prefix + "to_q.")
        to_k_cache = self.to_k.to_cached_state_dict(path_prefix + "to_k.")
        to_v_cache = self.to_v.to_cached_state_dict(path_prefix + "to_v.")
        to_out_cache = self.to_out.to_cached_state_dict(path_prefix + "to_out.")

        # Add linear layer prefixes to all keys
        for key, value in to_q_cache.items():
            cache_dict[f"to_q.{key}"] = value
        for key, value in to_k_cache.items():
            cache_dict[f"to_k.{key}"] = value
        for key, value in to_v_cache.items():
            cache_dict[f"to_v.{key}"] = value
        for key, value in to_out_cache.items():
            cache_dict[f"to_out.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.norm_q.from_cached_state_dict(substate(cache_dict, "norm_q"))
        self.norm_k.from_cached_state_dict(substate(cache_dict, "norm_k"))

        self.to_q.from_cached_state_dict(substate(cache_dict, "to_q"))
        self.to_k.from_cached_state_dict(substate(cache_dict, "to_k"))
        self.to_v.from_cached_state_dict(substate(cache_dict, "to_v"))
        self.to_out.from_cached_state_dict(substate(cache_dict, "to_out"))

    def load_state_dict(self, state_dict):
        self.norm_q.load_state_dict(substate(state_dict, "norm_q"))
        self.norm_k.load_state_dict(substate(state_dict, "norm_k"))

        self.to_q.load_state_dict(substate(state_dict, "to_q"))
        self.to_k.load_state_dict(substate(state_dict, "to_k"))
        self.to_v.load_state_dict(substate(state_dict, "to_v"))

        self.to_out.load_state_dict(substate(state_dict, "to_out.0"))

    def __call__(self, spatial_1BND, N, prompt_1BLP=None, rope_cos=None, rope_sin=None, trans_mat=None):
        """
        spatial_1BND: fractured N on SP, fracturd D on TP
        prompt_1BLP: replicated on SP, replicated D on TP (optional)
        rope_cos: fractured on SP, TP
        rope_sin: fractured on SP, TP
        trans_mat: replicated

        If prompt_1BLP is not provided, run self-attention.
        Otherwise, run cross-attention on prompt.

        Outputs:
        spatial_1BND: fractured N on SP, fractured D on TP
        """

        if rope_cos is not None:
            # If ROPE is given, this is self-attention
            assert rope_sin is not None
            assert trans_mat is not None
            assert prompt_1BLP is None

        if self.parallel_config.tensor_parallel.factor > 1:
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        kv_input = prompt_1BLP if prompt_1BLP is not None else spatial_1BND

        # Project spatial
        q_1BNF = self.to_q(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)
        k_1BNF = self.to_k(kv_input, compute_kernel_config=self.mm_compute_kernel_config)
        v_1BNF = self.to_v(kv_input, compute_kernel_config=self.mm_compute_kernel_config)

        # Norm spatial before splitting heads
        q_BHNE = self.norm_q(
            q_1BNF, num_heads_per_device=self.n_local_heads, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat
        )
        k_BHNE = self.norm_k(
            k_1BNF, num_heads_per_device=self.n_local_heads, rope_cos=rope_cos, rope_sin=rope_sin, trans_mat=trans_mat
        )

        def create_heads(inp):
            out, _, _ = ttnn.experimental.nlp_create_qkv_heads(
                inp,
                num_heads=self.n_local_heads,
                num_kv_heads=0,
                transpose_k_heads=False,
            )
            return out

        v_BHNE = create_heads(v_1BNF)

        # Rope

        if prompt_1BLP is None:
            # Self attention
            if self.parallel_config.sequence_parallel.factor > 1:
                # HACK: pass null joint inputs to take advantage of ring attention, even though this is self-attention.
                spatial_BHNE, prompt_BHLE, _lse = ttnn.transformer.ring_joint_scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    self.dummy_joint_input,
                    self.dummy_joint_input,
                    self.dummy_joint_input,
                    persistent_output_buffer_k=self.ccl_manager.get_ag_ping_pong_buffer(
                        k_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    persistent_output_buffer_v=self.ccl_manager.get_ag_ping_pong_buffer(
                        v_BHNE.shape, 2, self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    joint_strategy="rear",
                    logical_n=N,
                    program_config=self.ring_sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                    dim=2,
                    multi_device_global_semaphore=self.ccl_manager.get_ag_ping_pong_semaphore(
                        self.parallel_config.sequence_parallel.mesh_axis
                    ),
                    num_links=self.ccl_manager.num_links,
                    cluster_axis=self.parallel_config.sequence_parallel.mesh_axis,
                    mesh_device=self.mesh_device,
                    topology=ttnn.Topology.Linear,  # RJA always uses Linear topology
                    subdevice_id=self.ccl_manager.ccl_sub_device_id,
                    ccl_core_grid_offset=(0, self.sdpa_worker_grid[1]),
                )
            else:
                spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                    q_BHNE,
                    k_BHNE,
                    v_BHNE,
                    is_causal=False,
                    program_config=self.sdpa_program_config,
                    compute_kernel_config=self.sdpa_compute_kernel_config,
                )
        else:
            # Cross attention
            spatial_BHNE = ttnn.transformer.scaled_dot_product_attention(
                q_BHNE,
                k_BHNE,
                v_BHNE,
                is_causal=False,
                program_config=self.sdpa_program_config,
                compute_kernel_config=self.sdpa_compute_kernel_config,
            )

        spatial_1BND = ttnn.transformer.concatenate_heads(spatial_BHNE)
        spatial_1BND = ttnn.unsqueeze(spatial_1BND, 0)

        if self.parallel_config.tensor_parallel.factor > 1:
            # Gather spatial on TP axis before projection
            spatial_1BND = self.ccl_manager.all_gather_persistent_buffer(
                spatial_1BND, dim=3, mesh_axis=self.parallel_config.tensor_parallel.mesh_axis
            )

        spatial_1BND = self.to_out(spatial_1BND, compute_kernel_config=self.mm_compute_kernel_config)

        return spatial_1BND
