# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense tensor-parallel Llama-3.1 SwiGLU MLP for Galaxy prefill."""

from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig

_MESH_SHAPE = (4, 8)
_SP = 4
_TP = 8
_LOCAL_SEQUENCE = 256
_HIDDEN_SIZE = Llama31_8BConfig.EMB_SIZE
_INTERMEDIATE_SIZE = Llama31_8BConfig.INTERMEDIATE_SIZE
_WEIGHT_SHAPES = {
    "gate_proj.weight": (_INTERMEDIATE_SIZE, _HIDDEN_SIZE),
    "up_proj.weight": (_INTERMEDIATE_SIZE, _HIDDEN_SIZE),
    "down_proj.weight": (_HIDDEN_SIZE, _INTERMEDIATE_SIZE),
}


class MLP:
    """Run dense SwiGLU with TP-sharded projections and a TP all-reduce."""

    def __init__(self, mesh_device, mesh_config, state_dict):
        self._validate_mesh(mesh_device, mesh_config)
        host_weights = self._validate_weights(state_dict)
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.fabric_links = self._validate_tp_fabric_links()

        self.compute_kernel_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.gate_up_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(7, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=8,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )
        self.down_program_config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
            compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=4,
            per_core_M=1,
            per_core_N=16,
            transpose_mcast=False,
            fused_activation=None,
            fuse_batch=False,
        )

        self.gate_weight = self._upload_weight(
            host_weights["gate_proj.weight"], mesh_config.column_parallel(mesh_device)
        )
        self.up_weight = self._upload_weight(host_weights["up_proj.weight"], mesh_config.column_parallel(mesh_device))
        self.down_weight = self._upload_weight(host_weights["down_proj.weight"], mesh_config.row_parallel(mesh_device))

    @staticmethod
    def _validate_mesh(mesh_device, mesh_config):
        required = ("mesh_shape", "tp", "tp_axis", "sp_axis", "sp")
        missing = [name for name in required if not hasattr(mesh_config, name)]
        if missing:
            raise ValueError(f"MLP mesh_config is missing: {', '.join(missing)}")
        if tuple(mesh_config.mesh_shape) != _MESH_SHAPE:
            raise ValueError(f"MLP requires mesh_shape={_MESH_SHAPE}, got {tuple(mesh_config.mesh_shape)}")
        if (mesh_config.sp, mesh_config.tp, mesh_config.sp_axis, mesh_config.tp_axis) != (_SP, _TP, 0, 1):
            raise ValueError(
                "MLP requires SP=4 on mesh axis 0 and TP=8 on mesh axis 1; "
                f"got SP={mesh_config.sp}, TP={mesh_config.tp}, "
                f"sp_axis={mesh_config.sp_axis}, tp_axis={mesh_config.tp_axis}"
            )
        actual_shape = tuple(mesh_device.shape)
        if actual_shape != tuple(mesh_config.mesh_shape):
            raise ValueError(
                f"MLP mesh/config disagreement: device shape={actual_shape}, config shape={mesh_config.mesh_shape}"
            )
        if mesh_device.get_num_devices() != _SP * _TP:
            raise ValueError(f"MLP requires {_SP * _TP} mesh devices, got {mesh_device.get_num_devices()}")

    @staticmethod
    def _validate_weights(state_dict):
        if not isinstance(state_dict, Mapping):
            raise ValueError(f"MLP state_dict must be a mapping, got {type(state_dict).__name__}")
        missing = [name for name in _WEIGHT_SHAPES if name not in state_dict]
        if missing:
            raise ValueError(f"MLP state_dict is missing required weights: {', '.join(missing)}")
        weights = {}
        for name, expected_shape in _WEIGHT_SHAPES.items():
            weight = state_dict[name]
            if not isinstance(weight, torch.Tensor):
                raise ValueError(f"MLP {name} must be a host torch.Tensor, got {type(weight).__name__}")
            if weight.device.type != "cpu":
                raise ValueError(f"MLP {name} must be a host CPU tensor, got device {weight.device}")
            if tuple(weight.shape) != expected_shape:
                raise ValueError(f"MLP {name} must have shape {expected_shape}, got {tuple(weight.shape)}")
            weights[name] = weight
        return weights

    def _validate_tp_fabric_links(self):
        links = []
        for sp_coord in range(_SP):
            for tp_coord in range(_TP):
                src_coord = ttnn.MeshCoordinate([sp_coord, tp_coord])
                src_node = self.mesh_device.get_fabric_node_id(src_coord)
                for neighbor_tp in ((tp_coord - 1) % _TP, (tp_coord + 1) % _TP):
                    dst_coord = ttnn.MeshCoordinate([sp_coord, neighbor_tp])
                    dst_node = self.mesh_device.get_fabric_node_id(dst_coord)
                    forwarding_indices = tuple(ttnn.get_forwarding_link_indices(src_node, dst_node))
                    if 0 not in forwarding_indices or 1 not in forwarding_indices:
                        raise RuntimeError(
                            "MLP requires two live TP forwarding links (indices 0 and 1) on every ring edge; "
                            f"src={src_coord}/{src_node}, dst={dst_coord}/{dst_node}, "
                            f"available={forwarding_indices}"
                        )
                    links.append((sp_coord, tp_coord, neighbor_tp, forwarding_indices))
        return tuple(links)

    def _upload_weight(self, host_weight, mapper):
        # HF stores Linear weights as [out, in]. TTNN matmul consumes [in, out]. Transpose on the
        # host before applying column/row TP placement so each mapper shards the intended axis.
        return ttnn.from_torch(
            host_weight.to(torch.bfloat16).transpose(-2, -1).contiguous(),
            device=self.mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    def _validate_input(self, x):
        if not isinstance(x, ttnn.Tensor) or not ttnn.is_tensor_storage_on_device(x):
            raise ValueError("MLP input must be a device ttnn.Tensor")
        expected_shape = (1, 1, _LOCAL_SEQUENCE, _HIDDEN_SIZE)
        if tuple(x.shape) != expected_shape:
            raise ValueError(f"MLP input must have local shape {expected_shape}, got {tuple(x.shape)}")
        if len(ttnn.get_device_tensors(x)) != _SP * _TP:
            raise ValueError(f"MLP input must cover {_SP * _TP} mesh devices")
        if x.dtype != ttnn.bfloat16:
            raise ValueError(f"MLP input must be bfloat16, got {x.dtype}")
        if x.layout != ttnn.TILE_LAYOUT:
            raise ValueError(f"MLP input must use TILE_LAYOUT, got {x.layout}")
        if x.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"MLP input must use interleaved DRAM, got {x.memory_config()}")
        topology = ttnn.get_usable_topology(x, topology=ttnn.Topology.Ring, cluster_axis=self.mesh_config.tp_axis)
        if topology != ttnn.Topology.Ring:
            raise RuntimeError(f"MLP requires a live TP ring, but TTNN selected {topology}")

    def __call__(self, x):
        self._validate_input(x)
        gate = ttnn.matmul(
            x,
            self.gate_weight,
            program_config=self.gate_up_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        up = ttnn.matmul(
            x,
            self.up_weight,
            program_config=self.gate_up_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        product = ttnn.mul(
            gate,
            up,
            input_tensor_a_activations=[ttnn.UnaryOpType.SILU],
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
        )
        gate.deallocate(True)
        up.deallocate(True)

        partial = ttnn.matmul(
            product,
            self.down_weight,
            program_config=self.down_program_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        product.deallocate(True)
        output = ttnn.all_reduce(
            partial,
            cluster_axis=self.mesh_config.tp_axis,
            num_links=2,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            topology=ttnn.Topology.Ring,
        )
        partial.deallocate(True)
        return output
