# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tensor-parallel dense MLP for Gemma4-31B prefill."""

import ttnn
from models.demos.gemma4_d_p.tt.attention.operations import prefill_short_lived_memcfg
from models.demos.gemma4_d_p.tt.ccl import ccl_allreduce
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

        # Load-bearing, not a tuning knob. ttnn only threads a matmul's activation into
        # the program config when a user grid is given; with no grid it picks the simple
        # config, which drops the activation and re-applies it as a standalone unary op
        # -- exactly the pass the fused GELU below exists to remove. The grid also picks
        # a better config in its own right: it is worth ~220us per layer here even
        # before the fusion, and it is what makes L1 activations a win rather than a 4x
        # loss. All three projections use it for that reason.
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

    def __call__(self, hidden_states):
        """Apply column-parallel gate/up projections and row-parallel down projection."""
        # All three intermediates are short-lived, deallocated in this call, and touch no
        # SDPA input and no collective, so they are L1 candidates. Measured on the grid
        # config below, L1 is faster for all three -- including `hidden`, down_proj's in0.
        # The L1-interleaved-in0 penalty that layer.py documents is a property of the
        # simple program config, not of matmul: it does not appear on this path.
        act_mc = prefill_short_lived_memcfg()

        # GELU rides on the gate matmul as a fused kernel activation. On its own it is a
        # full read and write of a [1024, 5376] tensor for one SFPU op per tile.
        # "gelu_tanh" resolves to the same UnaryOpType that GeluVariant.Tanh selects, so
        # this stays gelu_pytorch_tanh rather than the erf or LUT variant, and the result
        # is bit-identical to the separate gelu at the same core_grid.
        gate = ttnn.linear(
            hidden_states,
            self.gate_proj,
            compute_kernel_config=self.compute_kernel_config,
            activation="gelu_tanh",
            core_grid=self.core_grid,
            memory_config=act_mc,
        )
        up = ttnn.linear(
            hidden_states,
            self.up_proj,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
            memory_config=act_mc,
        )
        # mul takes its output config from the first input, so this only says out loud
        # what `gate` already decided -- but it is what keeps the two in step if either
        # side of the flag changes.
        hidden = ttnn.mul(gate, up, memory_config=act_mc)
        gate.deallocate(True)
        up.deallocate(True)
        # The output must be DRAM ahead of ccl_allreduce -- an L1 activation clashes with
        # the circular buffers the collective reserves -- and matmul would otherwise
        # inherit L1 from `hidden`, so DRAM is explicit here.
        output = ttnn.linear(
            hidden,
            self.down_proj,
            compute_kernel_config=self.compute_kernel_config,
            core_grid=self.core_grid,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        hidden.deallocate(True)
        if self.mesh_config is not None and self.mesh_config.tp_degree > 1:
            output = ccl_allreduce(output, self.mesh_config, self.ccl_manager)
        return output
