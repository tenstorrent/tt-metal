# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Matmul micro-op specification.

Mirrors the contract in unified_kernels/matmul.hpp:
  - NCRISC/BRISC: no-op (weights pre-loaded via setup_sharded_buffer)
  - TRISC: ComputeCTArgs<out_w, transpose, fused_activation>
  - ComputeArgs{ in0, in1, out, k_num_tiles }
  - Op<CTArgs, IsActiveCore, pop_in0, pop_in1>

Note: setup_sharded only includes in1 (weights). in0 may come from
mcast (not sharded) or from a sharded tensor, depending on the graph.
The graph controls which ports are external via cb_bindings.
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract

MATMUL = MicroOpSpec(
    name="Matmul",
    header="unified_kernels/matmul.hpp",
    namespace="deepseek_b1_ops",
    struct_name="Matmul",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::Matmul::ReaderCTArgs",
        rt_args_type="deepseek_b1_ops::Matmul::ReaderArgs",
        named_ct_args=["in0", "in1", "k_num_tiles", "out_w"],
        # Only in1 (weights) is unconditionally sharded.
        # in0 may come from mcast (internal) or be sharded (external).
        # The codegen skips setup_sharded for internal (non-external) ports.
        setup_sharded=["in1"],
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="deepseek_b1_ops::Matmul::WriterCTArgs",
        rt_args_type="deepseek_b1_ops::Matmul::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type=("deepseek_b1_ops::Matmul::ComputeCTArgs<" "{out_w}, {transpose}, {fused_activation}>"),
        rt_args_type="deepseek_b1_ops::Matmul::ComputeArgs",
        named_ct_args=["in0", "in1", "out", "k_num_tiles", "out_w", "transpose", "fused_activation"],
        rt_args_fields=[
            ("in0", "ct:in0"),
            ("in1", "ct:in1"),
            ("out", "ct:out"),
            ("k_num_tiles", "ct:k_num_tiles"),
        ],
        cb_reads=["in0", "in1"],
        cb_writes=["out"],
    ),
    cb_ports={
        "in0": CBPortSpec(CBDirection.INPUT),
        "in1": CBPortSpec(CBDirection.WEIGHT, is_sharded=True),
        "out": CBPortSpec(CBDirection.OUTPUT),
    },
    # Op<CTArgs, IsActiveCore, pop_in0, pop_in1>
    op_template="Op<{CTArgs}, {is_active}, {pop_in0}, {pop_in1}>",
    risc_latency={"ncrisc": 0, "brisc": 0, "trisc": 1000},
)
