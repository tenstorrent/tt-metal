# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Dense tensor-parallel Llama-3.1 SwiGLU MLP for Galaxy prefill."""

from collections.abc import Mapping

import torch

import ttnn
from models.demos.llama_3p1_8b_d_p.reference.llama_3p1_8b_config import Llama31_8BConfig as Model
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import PREFILL_LAYOUT as layout
from models.demos.llama_3p1_8b_d_p.tt.prefill_geometry import validate_mesh

_WEIGHT_SHAPES = {
    "gate_proj.weight": (Model.INTERMEDIATE_SIZE, Model.EMB_SIZE),
    "up_proj.weight": (Model.INTERMEDIATE_SIZE, Model.EMB_SIZE),
    "down_proj.weight": (Model.EMB_SIZE, Model.INTERMEDIATE_SIZE),
}


class MLP:
    """Run dense SwiGLU with TP-sharded projections and a TP all-reduce."""

    def __init__(self, mesh_device, mesh_config, state_dict):
        validate_mesh(mesh_device, mesh_config, "MLP")
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
        for sp_coord in range(layout.sp):
            for tp_coord in range(layout.tp):
                src_coord = ttnn.MeshCoordinate([sp_coord, tp_coord])
                src_node = self.mesh_device.get_fabric_node_id(src_coord)
                for neighbor_tp in ((tp_coord - 1) % layout.tp, (tp_coord + 1) % layout.tp):
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
        if x.device() != self.mesh_device:
            raise ValueError("MLP input must reside on the constructor mesh")
        expected_shape = (1, 1, layout.local_sequence, Model.EMB_SIZE)
        if tuple(x.shape) != expected_shape:
            raise ValueError(f"MLP input must have local shape {expected_shape}, got {tuple(x.shape)}")
        if len(ttnn.get_device_tensors(x)) != layout.num_devices:
            raise ValueError(f"MLP input must cover {layout.num_devices} mesh devices")
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
