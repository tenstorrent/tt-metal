# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""
MiniMax-M2.5 MoE block — Galaxy mesh (8,4) with EP=8 + TP=4.

All routing on device (trace-safe): sigmoid + bias + topk + scatter.
Follows gpt_oss TopKRouter pattern — no CPU fallback.

EP slicing: pre-computed gather matrix [E, E_local] per device.
"""

import torch

import ttnn
from models.demos.gpt_oss.config import MeshConfig
from models.demos.gpt_oss.tt.ccl import CCLManager
from models.demos.gpt_oss.tt.experts.operations import apply_expert_parallel_allreduce, apply_tensor_parallel_allreduce

from .model_config import MiniMaxM2TTConfig


class TtMiniMaxMoE:
    """MiniMax-M2.5 MoE block — device-only routing (trace-safe)."""

    def __init__(
        self,
        device,
        state_dict: dict,
        config: MiniMaxM2TTConfig,
        layer_idx: int,
        mesh_config: MeshConfig = None,
        ccl_manager: CCLManager = None,
        weight_cache_path=None,
    ):
        self.config = config
        self.device = device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self._is_mesh = isinstance(device, ttnn.MeshDevice)

        E = config.num_local_experts
        FF = config.intermediate_size
        H = config.hidden_size

        ep = mesh_config.ep if mesh_config else 1
        tp = mesh_config.tp if mesh_config else 1
        E_local = E // ep

        prefix = f"model.layers.{layer_idx}.block_sparse_moe."
        cache_prefix = weight_cache_path / f"layer{layer_idx}.moe" if weight_cache_path else None

        self._E = E
        self._E_local = E_local
        self._FF = FF
        self._H = H
        self._ep = ep
        self._tp = tp

        self._gate_compute_config = None

        rep_mapper = ttnn.ReplicateTensorToMesh(device) if self._is_mesh else None

        # ---- Router gate weight [H, E] on device ----
        gate_w = state_dict[prefix + "gate.weight"].to(torch.bfloat16)  # [E, H]
        gate_cache = cache_prefix / "gate_weight" if cache_prefix else None
        self.gate_weight = ttnn.as_tensor(
            gate_w.T,  # [H, E] for ttnn.linear
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_mapper,
            cache_file_name=gate_cache,
        )

        # ---- Routing bias [1, E] in float32 (matched to float32 sigmoid path) ----
        routing_bias = state_dict[prefix + "e_score_correction_bias"].float()
        bias_cache = cache_prefix / "routing_bias" if cache_prefix else None
        self.routing_bias = ttnn.as_tensor(
            routing_bias.unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=rep_mapper,
            cache_file_name=bias_cache,
        )

        # ---- EP gather matrix for EP slicing ----
        if ep > 1 and self._is_mesh:
            gather_matrices = torch.zeros(ep, E, E_local, dtype=torch.bfloat16)
            for r in range(ep):
                for j in range(E_local):
                    gather_matrices[r, r * E_local + j, j] = 1.0
            ep_gather_mapper = ttnn.ShardTensor2dMesh(device, dims=(0, None), mesh_shape=device.shape)
            gather_cache = cache_prefix / "ep_gather_mat" if cache_prefix else None
            self.ep_gather_mat = ttnn.as_tensor(
                gather_matrices,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ep_gather_mapper,
                cache_file_name=gather_cache,
            )
        else:
            self.ep_gather_mat = None

        # ---- Expert weights: EP + TP 2D sharding ----
        if self._is_mesh:
            ep_tp_col_mapper = ttnn.ShardTensor2dMesh(device, dims=(1, -1), mesh_shape=device.shape)
            ep_tp_row_mapper = ttnn.ShardTensor2dMesh(device, dims=(1, -2), mesh_shape=device.shape)
        else:
            ep_tp_col_mapper = None
            ep_tp_row_mapper = None

        w1_stack = torch.stack([state_dict[prefix + f"experts.{j}.w1.weight"] for j in range(E)])
        w2_stack = torch.stack([state_dict[prefix + f"experts.{j}.w2.weight"] for j in range(E)])
        w3_stack = torch.stack([state_dict[prefix + f"experts.{j}.w3.weight"] for j in range(E)])

        w1_4d = w1_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)
        w3_4d = w3_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)
        w2_4d = w2_stack.transpose(1, 2).unsqueeze(0).to(torch.bfloat16)

        w1_cache = cache_prefix / "w1" if cache_prefix else None
        self.w1 = ttnn.as_tensor(
            w1_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_col_mapper,
            cache_file_name=w1_cache,
        )
        w3_cache = cache_prefix / "w3" if cache_prefix else None
        self.w3 = ttnn.as_tensor(
            w3_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_col_mapper,
            cache_file_name=w3_cache,
        )
        w2_cache = cache_prefix / "w2" if cache_prefix else None
        self.w2 = ttnn.as_tensor(
            w2_4d,
            dtype=config.weight_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ep_tp_row_mapper,
            cache_file_name=w2_cache,
        )

    # ------------------------------------------------------------------
    # Device Router (trace-safe, follows gpt_oss TopKRouter pattern)
    # ------------------------------------------------------------------

    def _route(self, x_flat: ttnn.Tensor, T_pad: int):
        """Device-only routing: linear → sigmoid → bias → topk → scatter → normalize.

        Sigmoid and bias addition are computed in float32 for topk selection
        accuracy (matching HF reference), then cast back to bfloat16 for
        topk/scatter which require bfloat16.
        """
        E = self._E
        E_local = self._E_local

        x_2d = ttnn.reshape(x_flat, (T_pad, self._H))
        if self._gate_compute_config is None:
            self._gate_compute_config = ttnn.init_device_compute_kernel_config(
                x_2d.device().arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
        logits = ttnn.linear(
            x_2d,
            self.gate_weight,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            compute_kernel_config=self._gate_compute_config,
        )

        logits_f32 = ttnn.typecast(logits, ttnn.float32)
        logits.deallocate(True)

        rw_f32 = ttnn.sigmoid(logits_f32)
        logits_f32.deallocate(True)

        scores_f32 = ttnn.add(rw_f32, self.routing_bias)

        scores = ttnn.typecast(scores_f32, ttnn.bfloat16)
        scores_f32.deallocate(True)

        top_k = self.config.num_experts_per_tok
        topk_vals, top_k_idx = ttnn.topk(scores, k=top_k, dim=-1)
        scores.deallocate(True)

        rw = ttnn.typecast(rw_f32, ttnn.bfloat16)
        rw_f32.deallocate(True)

        # Build binary mask [T_pad, E] via scatter.
        # zeros_like/ones_like are NOT trace-safe (host→device write), so
        # use device-side arithmetic: mul(x,0) for zeros, add(mul(x,0),1) for ones.
        zeros_e = ttnn.mul(rw, 0.0)
        ones_src = ttnn.add(ttnn.mul(topk_vals, 0.0), 1.0)
        topk_vals.deallocate(True)
        mask = ttnn.scatter(zeros_e, dim=1, index=top_k_idx, src=ones_src)
        top_k_idx.deallocate(True)
        ones_src.deallocate(True)

        selected_rw = ttnn.mul(rw, mask)
        rw.deallocate(True)
        mask.deallocate(True)

        rw_sum = ttnn.sum(selected_rw, dim=-1, keepdim=True)
        rw_sum = ttnn.reciprocal(rw_sum)
        sparse_w = ttnn.mul(selected_rw, rw_sum)
        selected_rw.deallocate(True)
        rw_sum.deallocate(True)

        # EP gather: [T_pad, E] @ [E, E_local] → [T_pad, E_local]
        if self.ep_gather_mat is not None:
            sparse_4d = ttnn.reshape(sparse_w, (1, 1, T_pad, E))
            gather_4d = ttnn.reshape(self.ep_gather_mat, (1, 1, E, E_local))
            local_w = ttnn.matmul(sparse_4d, gather_4d, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            sparse_4d.deallocate(True)
            local_w = ttnn.reshape(local_w, (1, T_pad, E_local, 1))
            local_w = ttnn.permute(local_w, (0, 2, 1, 3))
            sparse_w.deallocate(True)
        else:
            local_w = ttnn.reshape(sparse_w, (1, T_pad, E_local, 1))
            local_w = ttnn.permute(local_w, (0, 2, 1, 3))
            sparse_w.deallocate(True)

        return local_w  # [1, E_local, T_pad, 1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        cfg = self.config
        H = self._H
        tp = self._tp
        ep = self._ep
        E_local = self._E_local

        orig_shape = list(x.shape)
        if len(orig_shape) == 3:
            T = orig_shape[0] * orig_shape[1]
        elif len(orig_shape) == 4:
            T = orig_shape[0] * orig_shape[1] * orig_shape[2]
        else:
            raise ValueError(f"Unexpected x.shape {orig_shape}")

        T_pad = max(32, ((T + 31) // 32) * 32)

        x_flat = ttnn.reshape(x, (1, 1, T, H))
        if T_pad != T:
            x_flat = ttnn.pad(x_flat, padding=[(0, 0), (0, 0), (0, T_pad - T), (0, 0)], value=0.0)

        routing_tt = self._route(x_flat, T_pad)

        # ------ Expand x for batched expert matmul ------
        x_exp = ttnn.repeat(x_flat, ttnn.Shape([1, E_local, 1, 1]))

        # ------ Gate/Up projections ------
        gate = ttnn.matmul(x_exp, self.w1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        up = ttnn.matmul(x_exp, self.w3, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        x_exp.deallocate(True)

        # ------ SwiGLU ------
        gate_silu = ttnn.silu(gate)
        gate.deallocate(True)
        hidden = ttnn.mul(gate_silu, up)
        gate_silu.deallocate(True)
        up.deallocate(True)

        # ------ Down projection ------
        down = ttnn.matmul(hidden, self.w2, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        hidden.deallocate(True)

        # ------ Apply routing weights ------
        down = ttnn.mul(down, routing_tt)
        routing_tt.deallocate(True)

        # ------ Sum over local experts ------
        out = ttnn.unsqueeze_to_4D(ttnn.experimental.fast_reduce_nc(down, dims=[1]))
        down.deallocate(True)

        # ------ EP all-reduce ------
        if self._is_mesh and self.mesh_config and self.ccl_manager and ep > 1:
            out = apply_expert_parallel_allreduce(out, self.mesh_config, self.ccl_manager)

        out = ttnn.unsqueeze_to_4D(out)

        # ------ TP all-reduce ------
        if self._is_mesh and self.mesh_config and self.ccl_manager and tp > 1:
            out = apply_tensor_parallel_allreduce(out, self.mesh_config, self.device, T_pad, self.ccl_manager)

        # ------ Reshape back ------
        if T_pad != T:
            out = ttnn.slice(out, (0, 0, 0, 0), (1, 1, T, H))

        out = ttnn.reshape(out, orig_shape)
        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
