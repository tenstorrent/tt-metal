# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Llama 3.1 8B dense SwiGLU MLP.

Llama uses a dense SwiGLU MLP (NOT MoE):
    gate = silu(gate_proj(x))  # gate_proj: [hidden, intermediate/tp]
    up = up_proj(x)            # up_proj: [hidden, intermediate/tp]
    hidden = gate * up          # element-wise with fused SiLU activation
    output = down_proj(hidden)  # down_proj: [intermediate/tp, hidden]

With TP=4: intermediate_local = 14336/4 = 3584.

gate_proj and up_proj are column-parallel. down_proj is row-parallel (followed by allreduce).

Uses HiFi4 + fp32_dest_acc_en=True for all matmuls (the bring-up default).
"""


import ttnn
from models.demos.llama_3_1_8b.tt.config import MeshConfig
from models.demos.llama_3_1_8b.utils.general_utils import get_cache_file_name


class MLP:
    """Llama 3.1 8B dense SwiGLU MLP with TP-parallel gate/up/down projections."""

    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        ccl_manager,
        tensor_cache_path=None,
        mesh_config=None,
        weight_dtype=ttnn.bfloat8_b,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config or MeshConfig(mesh_device.shape, tp=mesh_device.shape[1])
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size  # 4096

        # HiFi4 + fp32_dest_acc_en for bring-up accuracy
        self.compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        col_mesh_mapper = self.mesh_config.column_parallel(mesh_device)
        row_mesh_mapper = self.mesh_config.row_parallel(mesh_device)

        if state_dict:
            # gate_proj: [intermediate_size, hidden_size] -> transposed to [hidden_size, intermediate_size]
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-1, -2)
            # up_proj: same shape as gate_proj
            up_proj_weight = state_dict["up_proj.weight"].transpose(-1, -2)
            # down_proj: [hidden_size, intermediate_size] -> transposed to [intermediate_size, hidden_size]
            down_proj_weight = state_dict["down_proj.weight"].transpose(-1, -2)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        # gate_proj is column-parallel: [hidden_size, intermediate_size/tp]
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "gate_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # up_proj is column-parallel: [hidden_size, intermediate_size/tp]
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=col_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "up_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        # down_proj is row-parallel: [intermediate_size/tp, hidden_size]
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=weight_dtype,
            mesh_mapper=row_mesh_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, "down_proj"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        SwiGLU MLP forward.

        Args:
            hidden_states: Input tensor [1, 1, seq_len, hidden_size]

        Returns:
            Output tensor [1, 1, seq_len, hidden_size] (after allreduce)
        """
        # gate_proj: [1, 1, seq_len, intermediate_local]
        gate_out = ttnn.linear(
            hidden_states,
            self.gate_proj,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        # up_proj: [1, 1, seq_len, intermediate_local]
        up_out = ttnn.linear(
            hidden_states,
            self.up_proj,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )

        # Fused SiLU activation on gate_out and element-wise multiply with up_out:
        # result = silu(gate_out) * up_out
        hidden = ttnn.multiply(gate_out, up_out, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])
        gate_out.deallocate(True)
        up_out.deallocate(True)

        # down_proj: [1, 1, seq_len, hidden_size] (row-parallel)
        hidden = ttnn.typecast(hidden, ttnn.bfloat8_b)
        output = ttnn.matmul(
            hidden,
            self.down_proj,
            dtype=ttnn.bfloat16,
            compute_kernel_config=self.compute_kernel_config,
        )
        hidden.deallocate(True)

        # Tensor-parallel allreduce
        if self.mesh_config.tp > 1:
            output = self.mesh_config.allreduce(output, self.ccl_manager, pad_size=0, axis=self.mesh_config.tp_axis)

        return output
