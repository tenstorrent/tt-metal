# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dense MLP for Gemma4-31B prefill."""

import ttnn
from models.demos.gemma4_d_p.tt.attention.operations import prefill_short_lived_memcfg
from models.demos.gemma4_d_p.tt.ccl import ccl_allreduce
from models.demos.gemma4_d_p.tt.matmul_config import prefill_matmul_program_config
from models.demos.gemma4_d_p.tt.precision import dtype_to_str
from models.demos.gemma4_d_p.utils.general_utils import get_cache_file_name


class MLP:
    def __init__(
        self, mesh_config, hf_config, state_dict, ccl_manager=None, dtype=ttnn.bfloat8_b, tensor_cache_path=None
    ):
        mesh_device = mesh_config.device
        self.mesh_device = mesh_device
        self.mesh_config = mesh_config
        self.ccl_manager = ccl_manager
        self.hidden_size = hf_config.hidden_size
        self.intermediate_size = hf_config.intermediate_size

        grid = mesh_device.compute_with_storage_grid_size()
        self.core_grid = ttnn.CoreGrid(y=grid.y, x=grid.x)

        tp = mesh_config.tp_degree
        tp_suffix = f"_tp{tp}" if tp > 1 else ""

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
            col_mapper = mesh_config.column_parallel()
            row_mapper = mesh_config.row_parallel()
        else:
            col_mapper = None
            row_mapper = None

        gate_proj_weight = state_dict["gate_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        up_proj_weight = state_dict["up_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        down_proj_weight = state_dict["down_proj.weight"].transpose(-2, -1).unsqueeze(0).unsqueeze(0)
        common = dict(device=mesh_device, dtype=dtype, layout=ttnn.TILE_LAYOUT)
        self.gate_proj = ttnn.as_tensor(
            gate_proj_weight,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"gate_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **common,
        )
        self.up_proj = ttnn.as_tensor(
            up_proj_weight,
            mesh_mapper=col_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"up_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **common,
        )
        self.down_proj = ttnn.as_tensor(
            down_proj_weight,
            mesh_mapper=row_mapper,
            cache_file_name=get_cache_file_name(tensor_cache_path, f"down_proj.weight{tp_suffix}{dtype_suffix}"),
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **common,
        )

    def _matmul_configs(self, hidden_states, weight, fused_activation=None):
        """(program_config, compute_kernel_config) for one projection: explicit blocking with fp32
        accumulation on the widest column count that splits N evenly, or (None, the default compute
        config) for the core-grid path.

        With this blocking, accumulating in bf16 drifts long-context KV accuracy in the deep layers, so
        the explicit path accumulates in fp32. That halves the output subblock, which only pays off for
        short M: at a per-core M of 4 (chunk 8192 at CP8) the default config is as fast, so it is kept.
        packer_l1_acc accumulates the K-block partials in L1, a little faster with no accuracy cost.
        """
        grid = self.mesh_device.compute_with_storage_grid_size()
        n_tiles = weight.padded_shape[-1] // ttnn.TILE_SIZE
        grid_x = max(x for x in range(1, grid.x + 1) if n_tiles % x == 0)
        program_config = prefill_matmul_program_config(
            hidden_states, weight, grid_x, grid.y, fused_activation, fp32_dest_acc=True, max_per_core_m=2
        )
        if program_config is None:
            return None, self.compute_kernel_config
        compute_kernel_config = ttnn.init_device_compute_kernel_config(
            self.mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.LoFi,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        return program_config, compute_kernel_config

    def _project(self, hidden_states, weight, memory_config, gelu=False):
        """hidden_states @ weight, on the explicit config when there is one and the core grid otherwise.
        With gelu, the GELU is fused either way."""
        fused_activation = ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH) if gelu else None
        program_config, compute_kernel_config = self._matmul_configs(hidden_states, weight, fused_activation)
        return ttnn.linear(
            hidden_states,
            weight,
            program_config=program_config,
            compute_kernel_config=compute_kernel_config,
            core_grid=None if program_config is not None else self.core_grid,
            activation="gelu_tanh" if gelu and program_config is None else None,
            memory_config=memory_config,
        )

    def __call__(self, hidden_states):
        """Apply column-parallel gate/up projections and row-parallel down projection."""
        # All three intermediates are short-lived, deallocated in this call, and
        # touch no SDPA input and no collective, so they are L1 candidates.
        act_mc = prefill_short_lived_memcfg()

        gate = self._project(hidden_states, self.gate_proj, act_mc, gelu=True)
        up = self._project(hidden_states, self.up_proj, act_mc)
        hidden = ttnn.mul(gate, up, memory_config=act_mc)
        gate.deallocate(True)
        up.deallocate(True)
        # Pack output to DRAM ahead of ccl_allreduce.
        output = self._project(hidden, self.down_proj, ttnn.DRAM_MEMORY_CONFIG)
        hidden.deallocate(True)
        if self.mesh_config is not None and self.mesh_config.tp_degree > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)
        return output
