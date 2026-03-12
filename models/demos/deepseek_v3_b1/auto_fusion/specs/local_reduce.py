# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
LocalReduce micro-op specification.

Mirrors the contract in unified_kernels/local_reduce.hpp:
  - Element-wise sum reduction of N tiles with optional SiLU
  - TRISC only (NCRISC/BRISC are noop)
  - ComputeCTArgs<NumTiles, ApplySilu>
  - ComputeArgs{ in_cb, out_cb }
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract, SDFRate

LOCAL_REDUCE = MicroOpSpec(
    name="LocalReduce",
    header="unified_kernels/local_reduce.hpp",
    namespace="deepseek_b1_ops",
    struct_name="LocalReduce",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::LocalReduce::ReaderCTArgs",
        rt_args_type="deepseek_b1_ops::LocalReduce::ReaderArgs",
        named_ct_args=["input_cb", "num_tiles"],
        setup_sharded=["input"],
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="deepseek_b1_ops::LocalReduce::WriterCTArgs",
        rt_args_type="deepseek_b1_ops::LocalReduce::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type=("deepseek_b1_ops::LocalReduce::ComputeCTArgs<" "{num_tiles}, {apply_silu}>"),
        rt_args_type="deepseek_b1_ops::LocalReduce::ComputeArgs",
        named_ct_args=["num_tiles", "apply_silu", "input_cb", "output_cb"],
        rt_args_fields=[
            ("in_cb", "ct:input_cb"),
            ("out_cb", "ct:output_cb"),
        ],
        cb_reads=["input"],
        cb_writes=["output"],
    ),
    cb_ports={
        "input": CBPortSpec(
            CBDirection.INPUT,
            is_sharded=True,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
        "output": CBPortSpec(
            CBDirection.OUTPUT,
            sdf_rate=SDFRate(tokens=1),
        ),
    },
    op_template="Op<{CTArgs}, {is_active}>",
    risc_latency={"ncrisc": 50, "brisc": 0, "trisc": 500},
)
