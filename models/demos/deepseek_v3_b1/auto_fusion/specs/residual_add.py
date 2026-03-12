# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
ResidualAdd micro-op specification.

Mirrors the contract in unified_kernels/residual_add.hpp:
  - Element-wise add: out[i] = in0[i] + in1[core_idx * out_w + i]
  - TRISC only (NCRISC/BRISC are noop)
  - ComputeCTArgs<out_w>
  - ComputeArgs{ in0_cb, in1_cb, out_cb, total_in1_tiles, core_idx }

The in1 CB contains the full [1, N] residual across all cores. Each core
indexes at offset core_idx * out_w.
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract, SDFRate

RESIDUAL_ADD = MicroOpSpec(
    name="ResidualAdd",
    header="unified_kernels/residual_add.hpp",
    namespace="deepseek_b1_ops",
    struct_name="ResidualAdd",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::ResidualAdd::ReaderCTArgs",
        rt_args_type="deepseek_b1_ops::ResidualAdd::ReaderArgs",
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="deepseek_b1_ops::ResidualAdd::WriterCTArgs",
        rt_args_type="deepseek_b1_ops::ResidualAdd::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type=("deepseek_b1_ops::ResidualAdd::ComputeCTArgs<" "{out_w}>"),
        rt_args_type="deepseek_b1_ops::ResidualAdd::ComputeArgs",
        named_ct_args=["in0", "in1", "out", "out_w", "total_in1_tiles", "core_idx"],
        rt_args_fields=[
            ("in0_cb", "ct:in0"),
            ("in1_cb", "ct:in1"),
            ("out_cb", "ct:out"),
            ("total_in1_tiles", "ct:total_in1_tiles"),
            ("core_idx", "ct:core_idx"),
        ],
        cb_reads=["in0", "in1"],
        cb_writes=["out"],
    ),
    cb_ports={
        "in0": CBPortSpec(
            CBDirection.INPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="out_w"),
        ),
        "in1": CBPortSpec(
            CBDirection.INPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="total_in1_tiles"),
        ),
        "out": CBPortSpec(
            CBDirection.OUTPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="out_w"),
        ),
    },
    # Op<CTArgs, IsActiveCore>
    op_template="Op<{CTArgs}, {is_active}>",
    risc_latency={"ncrisc": 0, "brisc": 0, "trisc": 200},
)
