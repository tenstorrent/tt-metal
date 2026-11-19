# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Unexecuted inline SFPU arithmetic probe; default emits source without hardware."""

import argparse
import ctypes
import hashlib
import json
import traceback
from pathlib import Path

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
NAMES = ("product", "product_error", "sum", "sum_error", "preliminary")


def fixtures():
    proof = json.loads((DOC / "v6_fma_expansion_proof.json").read_text())
    cases = [dict(name="constructed_midpoint", **{k: proof["midpoint_counterexample"][k] for k in ("a", "b", "c")})]
    for name, row in proof["fixtures"].items():
        for i, triple in enumerate(row["midpoint_examples"]):
            cases.append(dict(name=f"{name}_midpoint{i}", a=triple["a"], b=triple["b"], c=triple["acc"]))
    cases.extend(
        [
            dict(name="normal_cancellation", a=1.1250001192092896, b=0.9999999403953552, c=-1.125),
            dict(name="signed_zero", a=-0.0, b=1.0, c=-0.0),
            dict(name="unsupported_subnormal", a=2.0**-126, b=2.0**-23, c=0.0),
        ]
    )
    return cases


def graph():
    # Each arithmetic boundary is written to/reloaded from FP32 DST. This also
    # keeps vector live ranges short. The emitted assembly still needs inspection.
    expressions = []

    def op(kind, left, right):
        slot = 4 + len(expressions)
        expressions.append((slot, kind, left, right))
        return slot

    def split(x):
        t = op("*", 3, x)
        hi = op("-", t, op("-", t, x))
        return hi, op("-", x, hi)

    ah, al = split(0)
    bh, bl = split(1)
    p = op("*", 0, 1)
    e1 = op("-", p, op("*", ah, bh))
    e2 = op("-", e1, op("*", al, bh))
    e3 = op("-", e2, op("*", ah, bl))
    pe = op("-", op("*", al, bl), e3)
    s = op("+", p, 2)
    z = op("-", s, p)
    se = op("+", op("-", p, op("-", s, z)), op("-", 2, z))
    preliminary = op("+", s, op("+", se, pe))
    return expressions, (p, pe, s, se, preliminary)


def expected(case, expressions, outputs):
    f32 = lambda x: ctypes.c_float(x).value
    values = [case["a"], case["b"], case["c"], 4097.0]
    for slot, kind, left, right in expressions:
        assert slot == len(values)
        a, b = values[left], values[right]
        values.append(f32(a + b if kind == "+" else a - b if kind == "-" else a * b))
    return [values[i] for i in outputs]


def kernel_source(cases, expressions, outputs):
    switches = []
    for i, case in enumerate(cases):
        switches.append(
            f"case {i}: {{\n"
            + "\n".join(
                f"sfpi::dst_reg[{j}] = sfpi::vFloat({float(case[key]).hex()}f);"
                for j, key in enumerate(("a", "b", "c"))
            )
            + "\nbreak; }"
        )
    operations = "\n".join(
        f"{{ sfpi::vFloat a = sfpi::dst_reg[{left}]; sfpi::vFloat b = sfpi::dst_reg[{right}]; sfpi::dst_reg[{slot}] = a {kind} b; }}"
        for slot, kind, left, right in expressions
    )
    selections = "\n".join(
        f"case {i}: {{ sfpi::vFloat selected = sfpi::dst_reg[{slot}]; sfpi::dst_reg[0] = selected; break; }}"
        for i, slot in enumerate(outputs)
    )
    return (
        r"""
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#endif
void kernel_main() {
    CircularBuffer out(16);
    compute_kernel_hw_startup(0,16);
    for (uint32_t i=0; i<get_compile_time_arg_val(0); ++i) {
        tile_regs_acquire_math_clear();
#ifdef TRISC_MATH
        sfpu::_init_sfpu_config_reg();
        addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
        math::reset_counters(p_setrwc::SET_ABD_F);
        _llk_math_eltwise_sfpu_start_(1);
        switch(i/5) { CASES }
        sfpi::dst_reg[3] = sfpi::vFloat(4097.0f);
        OPERATIONS
        switch(i%5) { SELECTIONS }
        _llk_math_eltwise_sfpu_done_();
#endif
        tile_regs_commit();
        out.reserve_back(1);
        tile_regs_wait();
        pack_block(1,16,1);
        tile_regs_release_math_clear();
        out.push_back(1);
    }
}
""".replace(
            "CASES", "\n".join(switches)
        )
        .replace("OPERATIONS", operations)
        .replace("SELECTIONS", selections)
    )


WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
    Noc noc; DataflowBuffer partial(16);
    for(uint32_t i=0;i<get_arg_val<uint32_t>(1);++i) {
        partial.wait_front(1);
        noc.async_write(partial,output,4096,{}, {.page_id=i});
        noc.async_writes_flushed();partial.pop_front(1);
    }
    noc.async_write_barrier();
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-device", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists(), args.output
    cases = fixtures()
    expressions, outputs = graph()
    source = kernel_source(cases, expressions, outputs)
    result = dict(
        classification="primitive diagnostic, not decoder acceptance",
        status="source_emitted_unexecuted",
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compute_sha256=hashlib.sha256(source.encode()).hexdigest(),
        writer_sha256=hashlib.sha256(WRITER.encode()).hexdigest(),
        limits="Constants bypass input unpack; full corrected FMA/midpoint logic is not implemented; source is uncompiled until an explicit device run. Only first cell of each output tile is assessed. Subnormal case is outside bounded proof.",
        outputs=list(NAMES),
        scratch_last_slot=expressions[-1][0],
        cases=[dict(**case, expected=dict(zip(NAMES, expected(case, expressions, outputs)))) for case in cases],
    )
    args.output.with_suffix(".compute.cpp.snapshot").write_text(source)
    args.output.with_suffix(".writer.cpp.snapshot").write_text(WRITER)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        return
    import torch

    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        core = ttnn.CoreCoord(0, 0)
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
        unused = ttnn.from_torch(torch.zeros(1, 1, 32, 32), device=mesh, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
        count = len(cases) * len(NAMES)
        output = ttnn.empty((count, 1, 32, 32), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)
        cbs = [
            ttnn.CBDescriptor(
                total_size=pages * 4096,
                core_ranges=cores,
                format_descriptors=[
                    ttnn.CBFormatDescriptor(buffer_index=index, data_format=ttnn.float32, page_size=4096)
                ],
            )
            for index, pages in ((0, 2), (16, 2))
        ]
        runtime = ttnn.RuntimeArgs()
        runtime[0][0] = [output.buffer_address(), count]
        program = ttnn.ProgramDescriptor(
            kernels=[
                ttnn.KernelDescriptor(
                    kernel_source=WRITER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                    runtime_args=runtime,
                    config=ttnn.WriterConfigDescriptor(),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=source,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    compile_time_args=[count],
                    runtime_args=[],
                    config=ttnn.ComputeConfigDescriptor(
                        math_fidelity=ttnn.MathFidelity.HiFi4,
                        math_approx_mode=False,
                        fp32_dest_acc_en=True,
                        dst_full_sync_en=True,
                    ),
                ),
            ],
            semaphores=[],
            cbs=cbs,
        )
        ttnn.generic_op([unused, output], program)
        host = ttnn.to_torch(output)[:, 0, 0, 0].reshape(len(cases), len(NAMES))
        for i, case in enumerate(result["cases"]):
            want = torch.tensor(list(case["expected"].values()), dtype=torch.float32)
            case["actual"] = dict(zip(NAMES, host[i].tolist()))
            case["bit_equal"] = dict(zip(NAMES, (want.view(torch.int32) == host[i].view(torch.int32)).tolist()))
        result["status"] = "component_measured"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


if __name__ == "__main__":
    main()
