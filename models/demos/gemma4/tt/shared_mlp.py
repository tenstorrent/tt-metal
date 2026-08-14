# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Gemma4 Shared/Dense MLP with GeGLU activation.

Each decoder layer has BOTH a shared MLP and routed MoE experts.
Architecture: down_proj(GELU(gate_proj(x)) * up_proj(x))
intermediate_size = 2112, no bias.

HF weight shapes:
  gate_proj.weight: [intermediate_size, hidden_size] = [2112, 2816]
  up_proj.weight:   [intermediate_size, hidden_size] = [2112, 2816]
  down_proj.weight: [hidden_size, intermediate_size] = [2816, 2112]
"""

import ttnn
from models.demos.gemma4.tt.ccl import ccl_allreduce
from models.demos.gemma4.tt.precision import dtype_to_str
from models.demos.gemma4.utils.general_utils import get_cache_file_name


class SharedMLP:
    def __init__(
        self,
        mesh_device,
        hf_config,
        state_dict,
        mesh_config,
        ccl_manager=None,
        dtype=ttnn.bfloat8_b,
        tensor_cache_path=None,
    ):
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        tp = mesh_config.tp if mesh_config else 1
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

        # Tag the cache filenames with the weight dtype so that flipping a
        # SharedMLP weight's dtype (e.g. bf16 → bfp8 for DRAM-pressure relief)
        # doesn't collide with a previously-cached file that holds the same
        # logical weight at a different dtype. The rest of the model's cache
        # entries are unaffected and stay reusable across runs.
        dtype_suffix = f"_{dtype_to_str(dtype)}"

        # Match math fidelity to the weight's mantissa width. Fidelity is the number of passes
        # the matrix unit makes over the operand mantissas, so it — not the storage format — is
        # what sets math time: a narrower dtype moves fewer bytes but issues the same MACs. At
        # bfp4 the weights carry 3 mantissa bits, so the extra passes of a higher fidelity spend
        # time capturing bits that are not there. Left at the op default for wider dtypes, whose
        # accuracy those passes do buy something for.
        self.compute_kernel_config = (
            ttnn.init_device_compute_kernel_config(
                mesh_device.arch(),
                math_fidelity=ttnn.MathFidelity.LoFi,
                math_approx_mode=False,
                fp32_dest_acc_en=False,
                packer_l1_acc=False,
            )
            if dtype == ttnn.bfloat4_b
            else None
        )

        if tp > 1:
            col_mapper = mesh_config.column_parallel(mesh_device)
            row_mapper = mesh_config.row_parallel(mesh_device)
        else:
            col_mapper = None
            row_mapper = None

        if state_dict:
            gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
            down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        else:
            gate_proj_weight = None
            up_proj_weight = None
            down_proj_weight = None

        # gate/up: column-parallel (shard output dim across TP devices)
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        # down: row-parallel (shard input dim, allreduce after)
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            device=mesh_device,
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

    def __call__(self, hidden_states):
        """
        GeGLU MLP forward with TP support.

        gate/up are column-parallel, down is row-parallel + allreduce.
        """
        # gate = GELU(x @ gate_proj)
        gate = ttnn.linear(hidden_states, self.gate_proj, compute_kernel_config=self.compute_kernel_config)
        gate = ttnn.gelu(gate, fast_and_approximate_mode=True)

        # up = x @ up_proj
        up = ttnn.linear(hidden_states, self.up_proj, compute_kernel_config=self.compute_kernel_config)

        # hidden = gate * up
        hidden = ttnn.mul(gate, up)
        gate.deallocate(True)
        up.deallocate(True)

        # output = hidden @ down_proj
        output = ttnn.linear(hidden, self.down_proj, compute_kernel_config=self.compute_kernel_config)
        hidden.deallocate(True)

        # Allreduce after row-parallel down_proj
        if self.mesh_config is not None and self.mesh_config.tp > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)

        return output
