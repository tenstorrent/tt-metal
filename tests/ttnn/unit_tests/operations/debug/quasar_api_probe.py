# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Compile-only probes of compute APIs on Quasar.

Each probe builds a ProgramSpec whose compute kernel calls an API (with the template arguments tt-blaze uses) inside
a branch that never runs, so a probe passing means the API compiles for the device; it says nothing about numerics.
Not a pytest file: run it as a script against Quasar, e.g. on craq-sim with TT_METAL_SIMULATOR, ARCH_NAME=quasar,
TT_METAL_SLOW_DISPATCH_MODE=1 and TT_METAL_LLK_ASSERTS=1:

    python tests/ttnn/unit_tests/operations/debug/quasar_api_probe.py [probe name ...]

It prints one line per probe (OK, or FAIL with the first compiler errors), then the results as JSON.
"""

import json
import re
import sys

import ttnn

E = ttnn.experimental

PROBES = {
    # name: (includes, body)
    "copy_tile": (["api/compute/tile_move_copy.h"], "copy_init(dfb::in0); copy_tile(dfb::in0, 0, 0);"),
    "pack_tile": (["api/compute/pack.h"], "pack_tile(0, dfb::out);"),
    "pack_block_contiguous": (
        ["api/compute/experimental/pack_block.h"],
        "pack_block_contiguous_init(dfb::out); pack_block_contiguous(0, dfb::out, 1);",
    ),
    "reconfig_full_operand": (
        ["api/compute/reconfig_data_format.h"],
        "reconfig_full_operand(dfb::in0, dfb::in1); reconfig_data_format_srca(dfb::in0); pack_reconfig_data_format(dfb::out);",
    ),
    "add_tiles/mul_tiles": (
        ["api/compute/eltwise_binary.h"],
        "add_tiles_init(dfb::in0, dfb::in1); add_tiles(dfb::in0, dfb::in1, 0, 0, 0); "
        "mul_tiles_init(dfb::in0, dfb::in1); mul_tiles(dfb::in0, dfb::in1, 0, 0, 0);",
    ),
    "add_reuse_dest": (
        ["api/compute/eltwise_binary.h"],
        "add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::in0); "
        "add_reuse_dest_tiles<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(dfb::in0, 0, 0);",
    ),
    "binop_with_scalar(mul_unary)": (
        ["api/compute/eltwise_unary/binop_with_scalar.h"],
        "binop_with_scalar_tile_init(); mul_unary_tile(0, 0x3f800000u);",
    ),
    "exp_tile": (["api/compute/eltwise_unary/exp.h"], "exp_tile_init(); exp_tile(0);"),
    "recip_tile": (["api/compute/eltwise_unary/recip.h"], "recip_tile_init(); recip_tile(0);"),
    "rsqrt_tile": (["api/compute/eltwise_unary/rsqrt.h"], "rsqrt_tile_init(); rsqrt_tile(0);"),
    "silu/sigmoid": (
        ["api/compute/compute_kernel_api.h"],
        "silu_tile_init(); silu_tile(0); sigmoid_tile_init(); sigmoid_tile(0);",
    ),
    "gelu_tile": (["api/compute/eltwise_unary/gelu.h"], "gelu_tile_init(); gelu_tile(0);"),
    "add_rsqrt_tile": (
        ["api/compute/experimental/add_rsqrt.h"],
        "add_rsqrt_tile_init(); add_rsqrt_tile<false, VectorMode::R, 4>(0, 0x3f800000u);",
    ),
    "mul_binary_tile(sfpu)": (
        ["api/compute/eltwise_binary_sfpu.h"],
        "mul_binary_tile_init(); mul_binary_tile(0, 1, 0);",
    ),
    "tilize": (
        ["api/compute/tilize.h"],
        "tilize_init(dfb::in0, 1, dfb::out); tilize_block(dfb::in0, 1, dfb::out); tilize_uninit(dfb::in0, dfb::out);",
    ),
    "pack_untilize_block": (
        ["api/compute/pack_untilize.h"],
        "pack_untilize_init<1, 1>(dfb::in0, dfb::out); pack_untilize_block<1, 1>(dfb::in0, 1, dfb::out, 0); "
        "pack_untilize_uninit(dfb::out);",
    ),
    "pack_untilize_dest": (
        ["api/compute/pack_untilize.h"],
        "pack_untilize_dest_init<1, 1>(dfb::out); pack_untilize_dest<1, 1>(dfb::out); pack_untilize_uninit(dfb::out);",
    ),
    "reduce": (
        ["api/compute/reduce.h"],
        "reduce_init<PoolType::MAX, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, dfb::out); "
        "reduce_tile<PoolType::MAX, ReduceDim::REDUCE_COL>(dfb::in0, dfb::in1, 0, 0, 0); reduce_uninit();",
    ),
    "bcast_scalar_mul": (
        ["api/compute/bcast.h"],
        "mul_bcast_scalar_init(dfb::in0, dfb::in1); mul_tiles_bcast<BroadcastType::SCALAR>(dfb::in0, dfb::in1, 0, 0, 0);",
    ),
    "bcast_cols_mul": (
        ["api/compute/bcast.h"],
        "mul_bcast_cols_init_short(dfb::in0, dfb::in1); mul_tiles_bcast_cols(dfb::in0, dfb::in1, 0, 0, 0);",
    ),
    "matmul_block": (
        ["api/compute/matmul.h"],
        "matmul_block_init(dfb::in0, dfb::in1, 0, 1, 1, 1); matmul_block(dfb::in0, dfb::in1, 0, 0, 0, 0, 1, 1, 1);",
    ),
    "custom_mm": (
        ["api/compute/experimental/custom_mm.h"],
        "custom_mm_block_init_short(dfb::in0, dfb::in1, dfb::out, 1);",
    ),
    "transpose_wh": (
        ["api/compute/transpose_wh.h"],
        "transpose_wh_init(dfb::in0, dfb::out); transpose_wh_tile(dfb::in0, 0, 0);",
    ),
    "typecast(bf16->fp32)": (
        ["api/compute/eltwise_unary/typecast.h"],
        "typecast_tile_init<(uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float32>(); "
        "typecast_tile<(uint32_t)DataFormat::Float16_b, (uint32_t)DataFormat::Float32>(0);",
    ),
    "rope_sfpu": (["api/compute/experimental/rope_sfpu.h"], "rope_sfpu_fused_init();"),
    "topk_xl": (["api/compute/experimental/topk_xl.h"], "topk_xl_init<32, true>();"),
    "t6_semaphore (compute_fence)": (
        ["api/compute/experimental/semaphore_compute_impl.h"],
        "PACK(t6_semaphore_get<p_stall::PACK>(semaphore::FPU_SFPU));",
    ),
    "mul_reduce_scalar": (
        ["api/compute/experimental/mul_reduce_scalar.h"],
        "mul_reduce_scalar_init(dfb::in0, dfb::in0);",
    ),
    "rmsnorm(experimental)": (
        ["api/compute/experimental/rmsnorm.h"],
        "rmsnorm_mul_bcast_scalar_reuse_tiles_init<1>(dfb::in0);",
    ),
    "deepseek_binary_dest_reuse": (
        ["api/compute/experimental/eltwise_mul_scalar.h"],
        "deepseek_binary_dest_reuse_tiles_init(dfb::in0);",
    ),
    "eltwise_add_scalar": (["api/compute/experimental/eltwise_add_scalar.h"], ""),
    "sdpa": (["api/compute/experimental/sdpa.h"], ""),
    "sdpa_custom_mm": (["api/compute/experimental/sdpa_custom_mm.h"], ""),
    "sdpa_weighted_reduce": (["api/compute/experimental/sdpa_weighted_reduce.h"], ""),
    "softmax_k": (["api/compute/experimental/softmax_k.h"], "softmax_k_init(dfb::in0);"),
    "sinkhorn": (["api/compute/experimental/sinkhorn.h"], ""),
    "hadamard": (["api/compute/experimental/hadamard.h"], ""),
    "sum_reduce_scalar": (["api/compute/experimental/sum_reduce_scalar.h"], ""),
    "pack_rows_to_addr": (["api/compute/experimental/pack_rows_to_addr.h"], ""),
    "custom_mm_reuse_dest_srcb": (["api/compute/experimental/custom_mm_reuse_dest_srcb.h"], ""),
    "sfpu_binary_bcast": (["api/compute/sfpu_binary_bcast.h"], ""),
    "generic_moe_gate": (["api/compute/experimental/generic_moe_gate.h"], ""),
}

DM_SOURCE = "void kernel_main() {}\n"


def compute_source(includes, body):
    inc = "\n".join(f'#include "{h}"' for h in includes)
    return f"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
{inc}
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
void kernel_main() {{
    compute_kernel_hw_startup(dfb::in0, dfb::in1, dfb::out);
    if (get_arg(args::never) == 0xDEADBEEFu) {{
        {body}
    }}
}}
"""


def run_probe(device, out, includes, body):
    core = ttnn.CoreCoord(0, 0)
    dfbs = [
        E.DataflowBufferSpec(unique_id=n, entry_size=2048, num_entries=2, data_format_metadata=ttnn.bfloat16)
        for n in ("in0", "in1", "out")
    ]
    spec = E.ProgramSpec(
        name="probe",
        kernels=[
            E.KernelSpec(
                unique_id="dm",
                source=E.KernelSpec.SourceCode(code=DM_SOURCE),
                hw_config=E.create_reader_datamovement_config(disable_dfb_implicit_sync_for_all=True),
                dfb_bindings=[E.producer_of("in0", "in0"), E.producer_of("in1", "in1"), E.consumer_of("out", "out")],
            ),
            E.KernelSpec(
                unique_id="compute",
                source=E.KernelSpec.SourceCode(code=compute_source(includes, body)),
                hw_config=E.ComputeHardwareConfig(),
                dfb_bindings=[E.consumer_of("in0", "in0"), E.consumer_of("in1", "in1"), E.producer_of("out", "out")],
                runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["never"]),
            ),
        ],
        dataflow_buffers=dfbs,
        work_units=[E.WorkUnitSpec(kernels=["dm", "compute"], target_nodes=core)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[E.ProgramRunArgs.KernelRunArgs(kernel="compute", runtime_arg_values={"never": {core: 0}})]
    )
    E.generic_op([out], spec, run_args)


def first_errors(text):
    errs = []
    for line in text.splitlines():
        if "error:" in line and line not in errs:
            errs.append(re.sub(r"^.*?(/tt_metal/|/blaze/)", r"\1", line.strip())[:260])
        if len(errs) >= 2:
            break
    return errs


def main():
    names = sys.argv[1:] or list(PROBES)
    device = ttnn.open_device(device_id=0)
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([32, 32]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
    )
    results = {}
    for name in names:
        includes, body = PROBES[name]
        try:
            run_probe(device, out, includes, body)
            results[name] = ["OK"]
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            log = re.search(r"Log: (\S+)", msg)
            text = msg
            if log:
                try:
                    text = open(log.group(1)).read()
                except OSError:
                    pass
            results[name] = ["FAIL"] + (first_errors(text) or [msg.splitlines()[0][:260]])
        print(name, "->", results[name], flush=True)
    ttnn.close_device(device)
    json.dump(results, sys.stdout, indent=1)


PROBES.update(
    {
        "binary_tiles_init+add/mul": (
            ["api/compute/eltwise_binary.h"],
            "binary_tiles_init<true, EltwiseBinaryType::ELWADD>(dfb::in0, dfb::in1); add_tiles(dfb::in0, dfb::in1, 0, 0, 0); "
            "binary_tiles_init<true, EltwiseBinaryType::ELWMUL>(dfb::in0, dfb::in1); mul_tiles(dfb::in0, dfb::in1, 0, 0, 0);",
        ),
        "bcast_cols/rows_mul": (
            ["api/compute/bcast.h"],
            "mul_bcast_cols_init(dfb::in0, dfb::in1); mul_tiles_bcast_cols(dfb::in0, dfb::in1, 0, 0, 0); "
            "mul_bcast_rows_init(dfb::in0, dfb::in1); mul_tiles_bcast_rows(dfb::in0, dfb::in1, 0, 0, 0);",
        ),
        "t6_semaphore(PACK_UNPACK)": (
            ["api/compute/experimental/semaphore_compute_impl.h"],
            "PACK(t6_semaphore_get<p_stall::PACK>(semaphore::PACK_UNPACK));",
        ),
        "reconfig_data_format": (
            ["api/compute/reconfig_data_format.h"],
            "reconfig_data_format_srca(dfb::in0); reconfig_data_format(dfb::in0, dfb::in1); pack_reconfig_data_format(dfb::out);",
        ),
    }
)


if __name__ == "__main__":
    main()
