# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
RMSNorm micro-op specification.

Mirrors the contract in:
  - unified_kernels/rmsnorm.hpp (Op class)
  - micro_ops/rmsnorm/kernels/rmsnorm_kernel.cpp (standalone kernel)
  - micro_ops/rmsnorm/op.py (host-side descriptor)
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract, SDFRate

RMSNORM = MicroOpSpec(
    name="RMSNorm",
    header="unified_kernels/rmsnorm.hpp",
    namespace="deepseek_b1_ops",
    struct_name="RMSNorm",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::RMSNorm::ReaderCTArgs",
        rt_args_type="deepseek_b1_ops::RMSNorm::ReaderArgs",
        named_ct_args=["input_cb", "gamma_cb", "num_tiles"],
        setup_sharded=["input", "gamma"],
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type="deepseek_b1_ops::RMSNorm::WriterCTArgs",
        rt_args_type="deepseek_b1_ops::RMSNorm::WriterArgs",
        is_noop=True,
    ),
    trisc=RISCContract(
        ct_args_type=(
            "deepseek_b1_ops::RMSNorm::ComputeCTArgs<"
            "{fp32_acc}, {num_tiles}, {rsqrt_fast_approx}, "
            "{input_cb}, {gamma_cb}, {output_cb}>"
        ),
        rt_args_type="deepseek_b1_ops::RMSNorm::ComputeArgs",
        named_ct_args=[
            "input_cb",
            "gamma_cb",
            "output_cb",
            "fp32_acc",
            "num_tiles",
            "rsqrt_fast_approx",
        ],
        # epsilon is uint32_t (packed float bits), scalar is float
        # Both passed via common runtime args matching standalone pattern
        rt_args_fields=[
            ("epsilon", "rt_uint32:epsilon"),
            ("scalar", "rt_float:scalar"),
        ],
        common_runtime_args=[
            ("epsilon", "uint32_t"),
            ("scalar", "float"),
        ],
        cb_reads=["input", "gamma"],
        cb_writes=["output"],
    ),
    cb_ports={
        "input": CBPortSpec(
            CBDirection.INPUT,
            is_sharded=True,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
        "gamma": CBPortSpec(
            CBDirection.WEIGHT,
            is_sharded=True,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
        "output": CBPortSpec(
            CBDirection.OUTPUT,
            sdf_rate=SDFRate(tokens=1),
        ),
    },
    # Op<CTArgs, IsActiveCore, pop_input>
    op_template="Op<{CTArgs}, {is_active}, true>",
)
