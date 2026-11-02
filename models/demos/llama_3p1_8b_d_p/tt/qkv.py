# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel Llama-3.1 Q/K/V projection for Galaxy prefill."""

from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import validate_mesh
from models.demos.llama_3p1_8b_d_p.tt.rope import hf_to_meta

_LOCAL_QKV_WIDTH = (layout.local_q_heads + 2 * layout.local_kv_heads) * Model.HEAD_DIM
_WEIGHT_SHAPES = {
    "q_proj.weight": (Model.NUM_ATTENTION_HEADS * Model.HEAD_DIM, Model.EMB_SIZE),
    "k_proj.weight": (Model.NUM_KEY_VALUE_HEADS * Model.HEAD_DIM, Model.EMB_SIZE),
    "v_proj.weight": (Model.NUM_KEY_VALUE_HEADS * Model.HEAD_DIM, Model.EMB_SIZE),
}


class QKVProjection:
    """Project an SP-sharded activation to TP-local Q, K, and V heads.

    The constructor accepts raw HF weights and converts each Q/K head to Meta adjacent-pair
    coordinates exactly once before packing the per-TP-column Q_i|K_i|V_i groups.
    """

    def __init__(self, mesh_device, mesh_config, state_dict):
        validate_mesh(mesh_device, mesh_config, "QKV")
        host_weights = self._validate_weights(state_dict)
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(3, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        packed = self._pack_raw_hf_weights(host_weights)
        self.qkv_weight = ttnn.from_torch(
            packed,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mesh_config.column_parallel(mesh_device),
        )

    @staticmethod
    def _validate_weights(state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"QKV state_dict must be a mapping, got {type(state_dict).__name__}")
        missing = [name for name in _WEIGHT_SHAPES if name not in state_dict]
        if missing:
            raise ValueError(f"QKV state_dict is missing required weights: {', '.join(missing)}")
        weights = {}
        for name, expected_shape in _WEIGHT_SHAPES.items():
            weight = state_dict[name]
            if not isinstance(weight, torch.Tensor):
                raise ValueError(f"QKV {name} must be a host torch.Tensor, got {type(weight).__name__}")
            if weight.device.type != "cpu":
                raise ValueError(f"QKV {name} must be a host CPU tensor, got device {weight.device}")
            if tuple(weight.shape) != expected_shape:
                raise ValueError(f"QKV {name} must have shape {expected_shape}, got {tuple(weight.shape)}")
            weights[name] = weight.to(torch.bfloat16)
        return weights

    @staticmethod
    def _convert_qk_heads(raw_weight, num_heads):
        # HF Linear is [heads*head_dim, input]. Put head_dim last for hf_to_meta, then restore
        # [heads, head_dim, input]. Keeping heads separate prevents conversion across head boundaries.
        per_head = raw_weight.reshape(num_heads, Model.HEAD_DIM, Model.EMB_SIZE).transpose(-2, -1)
        return hf_to_meta(per_head).transpose(-2, -1).contiguous()

    @classmethod
    def _pack_raw_hf_weights(cls, weights):
        q_heads = cls._convert_qk_heads(weights["q_proj.weight"], Model.NUM_ATTENTION_HEADS)
        k_heads = cls._convert_qk_heads(weights["k_proj.weight"], Model.NUM_KEY_VALUE_HEADS)
        v_heads = weights["v_proj.weight"].reshape(Model.NUM_KEY_VALUE_HEADS, Model.HEAD_DIM, Model.EMB_SIZE)
        groups = []
        for tp_coord in range(layout.tp):
            q0 = tp_coord * layout.local_q_heads
            q_local = q_heads[q0 : q0 + layout.local_q_heads].reshape(
                layout.local_q_heads * Model.HEAD_DIM, Model.EMB_SIZE
            )
            k_local = k_heads[tp_coord].reshape(Model.HEAD_DIM, Model.EMB_SIZE)
            v_local = v_heads[tp_coord].reshape(Model.HEAD_DIM, Model.EMB_SIZE)
            groups.append(torch.cat((q_local, k_local, v_local), dim=0).transpose(-2, -1))
        packed = torch.cat(groups, dim=-1).contiguous()
        return packed.reshape(1, 1, Model.EMB_SIZE, layout.tp * _LOCAL_QKV_WIDTH)

    def _validate_input(self, x):
        if not isinstance(x, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(x):
            raise ValueError("QKV input must be a device ttnn.Tensor")
        if x.device() != self.mesh_device:
            raise ValueError("QKV input must reside on the constructor mesh")
        expected_shape = (1, 1, layout.local_sequence, Model.EMB_SIZE)
        if tuple(x.shape) != expected_shape:
            raise ValueError(f"QKV input must have local shape {expected_shape}, got {tuple(x.shape)}")
        if len(ttnn.get_device_tensors(x)) != layout.num_devices:
            raise ValueError(f"QKV input must cover {layout.num_devices} mesh devices")
        if x.dtype != ttnn.bfloat16:
            raise ValueError(f"QKV input must be bfloat16, got {x.dtype}")
        if x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"QKV input must use TILE_LAYOUT, got {x.layout}")
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"QKV input must use interleaved DRAM, got {x.memory_config()}")

    def __call__(self, x):
        self._validate_input(x)
        fused = ttnn.matmul(
            x,
            self.qkv_weight,
            program_config=self.program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        q, k, v = ttnn.experimental.nlp_create_qkv_heads(
            fused,
            num_heads=layout.local_q_heads,
            num_kv_heads=layout.local_kv_heads,
            transpose_k_heads=False,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            kv_tied=False,
        )
        fused.deallocate(True)
        return q, k, v
