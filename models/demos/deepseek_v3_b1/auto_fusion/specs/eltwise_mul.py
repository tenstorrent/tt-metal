# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
EltwiseMul micro-op specification (simple binary multiply mode).

Mirrors the contract in unified_kernels/eltwise_mul.hpp with enable_scalar=0:
  - Simple binary multiply: out[i] = in0[i] * in1[i]
  - TRISC only (NCRISC/BRISC are noop for binary mode)
  - ComputeCTArgs with cb_in0_wait = cb_in0, cb_in1_wait = cb_in1, enable_scalar=0
"""

from models.demos.deepseek_v3_b1.auto_fusion.types import CBDirection, CBPortSpec, MicroOpSpec, RISCContract, SDFRate

ELTWISE_MUL = MicroOpSpec(
    name="EltwiseMul",
    header="unified_kernels/eltwise_mul.hpp",
    namespace="deepseek_b1_ops",
    struct_name="EltwiseMul",
    ncrisc=RISCContract(
        ct_args_type="deepseek_b1_ops::EltwiseMul::ReaderCTArgs",
        rt_args_type="",  # no RT args struct needed
        is_noop=True,
    ),
    brisc=RISCContract(
        ct_args_type=(
            "deepseek_b1_ops::EltwiseMul::WriterCTArgs<"
            "{out_cb}, {num_tiles}, "
            "0, 0, 0, "  # cb_scalar, cb_scalar_src, scalar_index_offset (unused)
            "0>"  # enable_scalar = 0
        ),
        rt_args_type="",
        is_noop=True,
        named_ct_args=["out_cb", "num_tiles"],
        cb_reads=["out"],
    ),
    trisc=RISCContract(
        ct_args_type=(
            "deepseek_b1_ops::EltwiseMul::ComputeCTArgs<"
            "{in0_cb}, {in1_cb}, {out_cb}, {num_tiles}, "
            "{in0_cb}, {in0_wait_tiles}, "  # cb_in0_wait, cb_in0_wait_tiles
            "{in1_cb}, {in1_wait_tiles}, "  # cb_in1_wait, cb_in1_wait_tiles
            "0, "  # cb_scalar (unused)
            "0, "  # fp32_dest_acc_en
            "0, "  # enable_scalar = 0 (simple binary)
            "{in1_tile_offset}>"  # cb_in1_tile_offset for coalesced CB
        ),
        rt_args_type="",
        named_ct_args=[
            "in0_cb",
            "in1_cb",
            "out_cb",
            "num_tiles",
            "in0_wait_tiles",
            "in1_wait_tiles",
            "in1_tile_offset",
        ],
        cb_reads=["in0", "in1"],
        cb_writes=["out"],
    ),
    cb_ports={
        "in0": CBPortSpec(
            CBDirection.INPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
        "in1": CBPortSpec(
            CBDirection.INPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
        "out": CBPortSpec(
            CBDirection.OUTPUT,
            sdf_rate=SDFRate(tokens=0, is_parametric=True, param_expr="num_tiles"),
        ),
    },
    # EltwiseMul::Op<CTArgs, IsActiveCore, PopInputs>
    op_template="Op<{CTArgs}, {is_active}, true>",
    risc_latency={"ncrisc": 0, "brisc": 0, "trisc": 300},
)
