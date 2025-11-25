# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import torch
import ttnn
from ....layers.normalization import RMSNorm
from ....layers.linear import ColParallelLinear
from ....utils.substate import substate


class HunyuanAttention:
    def __init__(
        self,
        hidden_dim,
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        eps=1e-5,
        mesh_device=None,
        init=False,
        ccl_manager=None,
        parallel_config=None,
        is_fsdp=False,
    ):
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.eps = eps

        self.mesh_device = mesh_device
        self.ccl_manager = ccl_manager
        self.parallel_config = parallel_config

        self.n_local_heads = self.num_attention_heads // self.parallel_config.tensor_parallel.factor
        self.n_local_key_value_heads = self.num_key_value_heads // self.parallel_config.tensor_parallel.factor

        fsdp_mesh_axis = self.parallel_config.sequence_parallel.mesh_axis if is_fsdp else None

        rms_kwargs = {
            "embedding_dim": head_dim,
            "norm_eps": eps,
            "norm_elementwise_affine": True,
            "bias": False,
            "mesh_device": mesh_device,
        }

        self.norm_q = RMSNorm(**rms_kwargs)
        self.norm_k = RMSNorm(**rms_kwargs)

        self.to_qkv = ColParallelLinear(
            hidden_dim,
            (num_attention_heads + 2 * num_key_value_heads) * head_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        self.to_out = ColParallelLinear(
            (num_attention_heads + 2 * num_key_value_heads) * head_dim,
            hidden_dim,
            bias=False,
            mesh_device=mesh_device,
            mesh_axis=parallel_config.tensor_parallel.mesh_axis,
            fsdp_mesh_axis=fsdp_mesh_axis,
            ccl_manager=ccl_manager,
        )

        full_grid = self.mesh_device.compute_with_storage_grid_size()
        self.sdpa_worker_grid = (full_grid.x, full_grid.y - 1)
        self.sdpa_program_config = ttnn.SDPAProgramConfig(
            compute_with_storage_grid_size=self.sdpa_worker_grid,
            q_chunk_size=256,
            k_chunk_size=512,
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

        self.rmsnorm_compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
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
        to_qkv_cache = self.to_qkv.to_cached_state_dict(path_prefix + "to_qkv.")
        to_out_cache = self.to_out.to_cached_state_dict(path_prefix + "to_out.")

        # Add linear layer prefixes to all keys
        for key, value in to_qkv_cache.items():
            cache_dict[f"to_qkv.{key}"] = value
        for key, value in to_out_cache.items():
            cache_dict[f"to_out.{key}"] = value

        return cache_dict

    def from_cached_state_dict(self, cache_dict):
        self.norm_q.from_cached_state_dict(substate(cache_dict, "norm_q"))
        self.norm_k.from_cached_state_dict(substate(cache_dict, "norm_k"))
        self.norm_added_q.from_cached_state_dict(substate(cache_dict, "norm_added_q"))
        self.norm_added_k.from_cached_state_dict(substate(cache_dict, "norm_added_k"))

        self.to_qkv.from_cached_state_dict(substate(cache_dict, "to_qkv"))
        self.to_out.from_cached_state_dict(substate(cache_dict, "to_out"))

    def load_state_dict(self, state_dict):
        def reshape_and_merge_qkv(q_state, k_state, v_state):
            # Rearrange QKV projections such column-fracturing shards the heads
            def _merge_tensors(q, k, v):
                n_dev = self.parallel_config.tensor_parallel.factor
                q, k, v = q.T, k.T, v.T
                q = q.reshape(q.shape[0], n_dev, self.n_local_heads, self.head_dim)
                k = k.reshape(k.shape[0], n_dev, self.n_local_key_value_heads, self.head_dim)
                v = v.reshape(v.shape[0], n_dev, self.n_local_key_value_heads, self.head_dim)
                qkv = torch.cat([q, k, v], dim=2)
                qkv = qkv.reshape(
                    qkv.shape[0], n_dev * (self.n_local_heads + 2 * self.n_local_key_value_heads) * self.head_dim
                )
                qkv = qkv.T
                return qkv

            weight = _merge_tensors(q_state["weight"], k_state["weight"], v_state["weight"])

            out_state = {"weight": weight}
            if "bias" in q_state:
                bias = _merge_tensors(
                    q_state["bias"].unsqueeze(-1), k_state["bias"].unsqueeze(-1), v_state["bias"].unsqueeze(-1)
                )
                bias = bias.squeeze(-1)
                out_state["bias"] = bias
            return out_state

        self.norm_q.load_state_dict(substate(state_dict, "norm_q"))
        self.norm_k.load_state_dict(substate(state_dict, "norm_k"))

        qkv_state = reshape_and_merge_qkv(
            substate(state_dict, "to_q"), substate(state_dict, "to_k"), substate(state_dict, "to_v")
        )
        self.to_qkv.load_state_dict(qkv_state)

        self.to_out.load_state_dict(substate(state_dict, "to_out.0"))

    def __call__(self, x_1BSD, rope_cos, rope_sin, trans_mat):
        """
        x_1BSD: fractured N on SP, replicated D on TP
        rope_cos: fractured on SP, TP
        rope_sin: fractured on SP, TP
        trans_mat: replicated

        Outputs:
        x_1BNF: fractured N on SP, replicated D on TP
        """

        # (1, batch, seq_len, hidden_dim) -> (1, batch, seq_len, (num_attention_heads + num_key_value_heads) * head_dim)
        qkv_1BSH = self.to_qkv(x_1BSD, compute_kernel_config=self.mm_compute_kernel_config)

        # Note that we asssume an interleaved structure: [1, batch, seq_len, (q_heads, k_head, v_head) x num_groups]
        # Taking hunyuan values, we have that with TP=8, each device gets 32/8 = 4 q_heads, 8/8 = 1 k_head/q_head
        # so something like q_1BSH = [1, batch, seq_len, 4 * head_dim], k_1BSH = [1, batch, seq_len, 1 * head_dim], v_1BSH = [1, batch, seq_len, 1 * head_dim]
        q_1QSH, k_1KSH, v_1VSH = ttnn.experimental.nlp_create_qkv_heads(
            qkv_1BSH,
            num_heads=self.n_local_heads,
            num_kv_heads=self.n_local_key_value_heads,
            transpose_k_heads=False,
        )

        # QK norm, note that since the heads
        q_1QSH = self.norm_q(q_1QSH, compute_kernel_config=self.rmsnorm_compute_kernel_config)
        k_1KSH = self.norm_k(k_1KSH, compute_kernel_config=self.rmsnorm_compute_kernel_config)

        # Rope
        # q_1QSH = ttnn.experimental.rotary_embedding_llama(
        #     q_1QSH, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
        # )
        # k_1KSH = ttnn.experimental.rotary_embedding_llama(
        #     k_1KSH, rope_cos, rope_sin, trans_mat, compute_kernel_config=self.rope_compute_kernel_config
        # )
