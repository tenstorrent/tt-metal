# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""BF16 streaming multiply plus native terminal cumsum; opt-in device diagnostic."""

import argparse
import ast
import copy
import json
import math
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_ordered_router_fusion_probe as router
from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_terminal_projection_scan_probe as terminal

E, REPO, REVISION = terminal.E, terminal.REPO, terminal.REVISION
sha, bits, schedule, fixtures = terminal.sha, terminal.bits, terminal.schedule, terminal.fixtures


def product_bits(expected, actual):
    result = bits(expected, actual)
    mismatch = expected.contiguous().view(torch.int32) != actual.contiguous().view(torch.int32)
    zero_only = mismatch & (expected == 0) & (actual == 0)
    result["signed_zero_only_differences"] = int(zero_only.sum())
    result["nonzero_differences"] = int((mismatch & ~zero_only).sum())
    ids = mismatch.flatten().nonzero().flatten()[:16]
    result["first_different_flat_indices"] = ids.tolist()
    result["expected_bits"] = expected.contiguous().view(torch.int32).flatten()[ids].tolist()
    result["actual_bits"] = actual.contiguous().view(torch.int32).flatten()[ids].tolist()
    return result


READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t coeff_offset=get_compile_time_arg_val(0);
 const auto weight=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto coeff=TensorAccessor(TensorAccessorArgs<coeff_offset>(),get_arg_val<uint32_t>(1));
 const uint32_t start=get_arg_val<uint32_t>(2), runs=get_arg_val<uint32_t>(3);
 const uint32_t ht=get_arg_val<uint32_t>(4), wt=get_arg_val<uint32_t>(5);
 Noc noc;DataflowBuffer a(0),b(1);
 for(uint32_t run=start;run<start+runs;++run) {
  for(uint32_t h=0;h<ht;++h) {
   a.reserve_back(1);b.reserve_back(1);
   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;
   const uint32_t column=(run/wt)*ht+h;
   noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});
   noc.async_read(coeff,b,2048,{.page_id=column},{.offset_bytes=0});
   noc.async_read_barrier();a.push_back(1);b.push_back(1);
  }
 }
}
"""
COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/bcast.h"
#include "api/compute/cumsum.h"
#include "api/compute/pack.h"
#include "api/dataflow/dataflow_buffer.h"
void kernel_main() {
 DataflowBuffer a(0),b(1),out(16);
 compute_kernel_hw_startup(0,1,16);
 constexpr uint32_t ht=get_compile_time_arg_val(0);
 constexpr bool products_only=get_compile_time_arg_val(1)!=0;
 const uint32_t runs=get_arg_val<uint32_t>(0);
 for(uint32_t run=0;run<runs;++run) {
  for(uint32_t h=0;h<ht;++h) {
   a.wait_front(1);b.wait_front(1);tile_regs_acquire();
   // FPU multiplication leaves the native cumsum SFPU carry registers intact.
   mul_bcast_cols_init(0,1);mul_tiles_bcast_cols(0,1,0,0,0);
   if constexpr(!products_only) {
    cumsum_tile_init<CumsumMode::Default>();
    cumsum_tile<CumsumMode::Default>(0,h==0);
   }
   tile_regs_commit();a.pop_front(1);b.pop_front(1);tile_regs_wait();
   if(products_only || h+1==ht) {
    out.reserve_back(1);pack_tile(0,16);out.push_back(1);
   }
   tile_regs_release();
  }
 }
}
"""
WRITER = terminal.WRITER
PRODUCT_WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t start=get_arg_val<uint32_t>(1), runs=get_arg_val<uint32_t>(2);
 const uint32_t ht=get_arg_val<uint32_t>(3),wt=get_arg_val<uint32_t>(4);
 Noc noc;DataflowBuffer out(16);
 for(uint32_t run=start;run<start+runs;++run) {
  for(uint32_t h=0;h<ht;++h) {
   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;
   out.wait_front(1);
   noc.async_write(out,output,4096,{}, {.page_id=tile});
   noc.async_write_barrier();out.pop_front(1);
  }
 }
}
"""


class StreamingProjectionScan:
    """BF16 column-broadcast product into FP32 DST followed by native cumsum."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, weight, coefficients, *, products_only=False):
        ttnn = self.ttnn
        shape = tuple(weight.shape)
        assert len(shape) >= 2 and shape[-2] in (512, 1024) and shape[-1] % 32 == 0
        assert tuple(coefficients.shape) == shape[:-1] + (1,)
        for tensor in (weight, coefficients):
            assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
            assert tensor.memory_config() in (ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG)
        ht, wt = shape[-2] // 32, shape[-1] // 32
        partition = schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y, _, _ in partition]
        )
        output = ttnn.empty(
            shape if products_only else shape[:-2] + (1, shape[-1]),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG if products_only else ttnn.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for x, y, start, count in partition:
            rd[x][y] = [weight.buffer_address(), coefficients.buffer_address(), start, count, ht, wt]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        wa = ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        ca = ttnn.TensorAccessorArgs(coefficients).get_compile_time_args()
        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(READER, ttnn.ReaderConfigDescriptor(), [1 + len(wa)] + wa + ca, rd),
                kernel(COMPUTE, config, [ht, int(products_only)], cr),
                kernel(
                    PRODUCT_WRITER if products_only else WRITER,
                    ttnn.WriterConfigDescriptor(),
                    ttnn.TensorAccessorArgs(output).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, ttnn.bfloat16),
                    (1, 2, 2048, ttnn.bfloat16),
                    (16, 1, 4096, ttnn.float32),
                    (17, 1, 4096, ttnn.float32),
                ]
            ],
        )
        ttnn.generic_op([weight, coefficients, output], program)
        return output


def extract(snapshot):
    sources = terminal.extract(snapshot)
    baseline = ast.parse(sources["baseline"]).body[0]
    candidate = copy.deepcopy(baseline)
    restorations = []
    for branch in candidate.body[:3]:
        i = next(
            i
            for i, n in enumerate(branch.body)
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "products"
        )
        old = copy.deepcopy(branch.body[i : i + 7])
        assert [ast.unparse(n).split(" = ")[0] for n in old] == [
            "products",
            "prefixes",
            "del products",
            "tail",
            "del prefixes",
            "partial",
            "del tail",
        ]
        restorations.append((i, old))
        branch.body[i : i + 7] = ast.parse("partial = self.streaming_projection_scan(weight, coefficients)").body
    restored = copy.deepcopy(candidate)
    for branch, (i, old) in zip(restored.body, restorations):
        branch.body[i : i + 1] = old
    assert ast.dump(restored) == ast.dump(baseline)
    inputs = copy.deepcopy(baseline)
    inputs.body = inputs.body[:3]
    for branch, (i, _) in zip(inputs.body, restorations):
        branch.body = branch.body[:i] + ast.parse("return weight, coefficients").body
    inputs.body += ast.parse("raise AssertionError('inputs helper requires a selected singleton row')").body
    sources["candidate"] = ast.unparse(ast.fix_missing_locations(candidate))
    sources["inputs"] = ast.unparse(ast.fix_missing_locations(inputs))
    return sources


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve previous evidence"
    sources = extract(args.snapshot)
    state, cases, captures = fixtures(args.checkpoint, args.random_cases)
    # BF16 finite products need at most 16 significand bits. Bound the exponent
    # range on these fixtures; this does not prove device FPU or signed-zero behavior.
    nonzero_weights = torch.cat([v.flatten().float().abs() for v in state.values()])
    nonzero_weights = nonzero_weights[nonzero_weights != 0]
    wmin, wmax = float(nonzero_weights.min()), float(nonzero_weights.max())
    product_bounds = []
    for case in cases:
        values = case["x"].float().abs().flatten()
        nonzero = values[values != 0]
        lower = float(nonzero.min()) * wmin if len(nonzero) else None
        upper = float(values.max()) * wmax
        assert lower is None or lower >= torch.finfo(torch.float32).tiny
        assert upper <= torch.finfo(torch.float32).max
        product_bounds.append(dict(name=case["name"], nonzero_lower_bound=lower, upper_bound=upper))
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    actual_x = cases[0]["x"].flatten()[order].float().reshape(4, 1, 512, 1)
    actual_w = state["mlp.gate.weight"].T[order].float().reshape(4, 1, 512, 256)
    actual_products = actual_x * actual_w
    leading_fixture = torch.stack((actual_products[:3, 0], actual_products[1:4, 0]))
    leading_weights = torch.stack((actual_w[:3, 0], actual_w[1:4, 0])).bfloat16()
    leading_coefficients = torch.stack((actual_x[:3, 0], actual_x[1:4, 0])).bfloat16()
    # CPU traversal proof includes unequal work groups, multiple leading axes, all widths/Ks.
    checks = []
    for shape in [
        (4, 1, 512, 256),
        (8, 1, 512, 2048),
        (4, 1, 512, 6208),
        (4, 1, 512, 6144),
        (2, 1, 1024, 6208),
        (2, 1, 1024, 6144),
        (2, 3, 512, 256),
    ]:
        ht, wt = shape[-2] // 32, shape[-1] // 32
        total = math.prod(shape[:-2]) * wt
        for gx, gy in [(8, 8), (11, 10)]:
            partition = schedule(total, gx, gy)
            runs = [r for _, _, start, count in partition for r in range(start, start + count)]
            assert runs == list(range(total))
            tiles = [(r // wt) * ht * wt + h * wt + r % wt for r in runs for h in range(ht)]
            assert sorted(tiles) == list(range(math.prod(shape) // 1024))
        checks.append(dict(shape=shape, runs=total, input_tiles=len(tiles), traversal_exact=True))
    result = dict(
        status="cpu_source_setup_pass",
        device_executed=False,
        captures=captures,
        classification="component diagnostic; no decoder acceptance or performance claim",
        revision=REVISION,
        script_sha256=sha(__file__),
        frozen_router_sha256=sha(router.__file__),
        frozen_terminal_sha256=sha(terminal.__file__),
        snapshot_sha256=sha(args.snapshot),
        snapshot=str(args.snapshot),
        random_seed=603636,
        fixture_cases=len(cases),
        cpu_fixture_normal_product_bounds=product_bounds,
        product_exactness_scope="BF16 inputs and normal-range fixture products; device FPU and signed zeros remain unproven",
        scheduling_checks=checks,
        ast_restoration_exact=True,
        changed_scan_sequences=3,
        source_runtime="BF16 FPU broadcast multiply into FP32 DST, then native cumsum; no materialized products or runtime host conversion",
        limits="CPU setup proves source scope/scheduling only; native SFPU arithmetic and performance require opt-in device checks",
        cases=[],
        timing=[],
    )
    result["dependencies"] = [
        dict(path=str(p), sha256=sha(p))
        for p in [
            REPO
            / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp",
            REPO / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp",
            REPO / "tt_metal/hw/inc/api/compute/cumsum.h",
            REPO / "tt_metal/hw/inc/api/compute/bcast.h",
            REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_math_binary_api.h",
            REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h",
        ]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, source in dict(
        reader=READER, compute=COMPUTE, writer=WRITER, product_writer=PRODUCT_WRITER, **sources
    ).items():
        Path(str(args.output) + f".{label}.snapshot").write_text(source)
    fixture_path = Path(str(args.output) + ".fixtures.pt")
    torch.save(
        dict(
            state=state,
            cases=cases,
            leading_fixture=leading_fixture,
            leading_weights=leading_weights,
            leading_coefficients=leading_coefficients,
        ),
        fixture_path,
    )
    result["fixtures"] = dict(path=str(fixture_path), sha256=sha(fixture_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SOURCE_SETUP_PASS", len(cases))
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") in (None, "", "0")
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=67108864
        )
        result.update(status="running", device_executed=True, device_closed=False)
        obj = types.SimpleNamespace(
            streaming_projection_scan=StreamingProjectionScan(ttnn, mesh),
            mesh_device=mesh,
            kind="linear_attention",
            decode_linear={},
            w={},
        )
        result["grid"] = [obj.streaming_projection_scan.grid.x, obj.streaming_projection_scan.grid.y]
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        exec(
            compile(sources["setup"], str(args.output) + ":setup", "exec"),
            dict(
                self=obj,
                torch=torch,
                ttnn=ttnn,
                mesh_device=mesh,
                upload=upload,
                DRAM=ttnn.DRAM_MEMORY_CONFIG,
                state_dict=state,
                pieces=[state[f"linear_attn.in_proj_{s}.weight"] for s in ("qkv", "z", "a", "b")],
            ),
        )
        calls = {}
        for label in ("baseline", "candidate", "products", "inputs"):
            namespace = dict(ttnn=ttnn, DRAM=ttnn.DRAM_MEMORY_CONFIG)
            exec(compile(sources[label], str(args.output) + ":" + label, "exec"), namespace)
            calls[label] = types.MethodType(namespace["linear"], obj)

        def call(label, dx, case, dtype):
            obj.linear = calls[label]
            with device_only():
                return obj.linear(dx, case["key"], dtype, **case["kwargs"])

        payload = {}
        leading_device = upload(leading_fixture, ttnn.float32)
        native_prefix = ttnn.cumsum(leading_device, -2, dtype=ttnn.float32, disable_compensated_sum=True)
        native_tail = native_prefix[:, :, 480:512, :]
        native_row = native_tail[:, :, 31:32, :]
        lw, lc = upload(leading_weights), upload(leading_coefficients)
        with device_only():
            generic_row = obj.streaming_projection_scan(lw, lc)
            generated_products = obj.streaming_projection_scan(lw, lc, products_only=True)
        result["multiple_leading_axes_products_fp32"] = product_bits(leading_fixture, ttnn.to_torch(generated_products))
        assert result["multiple_leading_axes_products_fp32"]["exact"]
        del lw, lc, generated_products
        result["multiple_leading_axes_fp32"] = bits(ttnn.to_torch(native_row), ttnn.to_torch(generic_row))
        save()
        assert result["multiple_leading_axes_fp32"]["exact"]
        del leading_device, native_prefix, native_tail, native_row, generic_row
        for case in cases:
            if args.case_filter and not any(s in case["name"] for s in args.case_filter):
                continue
            dx = upload(case["x"])
            row = dict(name=case["name"], input_shape=list(case["x"].shape), outputs={})
            payload[case["name"]] = {}
            if case["x"].shape[-2] == 1:
                products = call("products", dx, case, ttnn.float32)
                k = products.shape[-2]
                prefixes = ttnn.cumsum(
                    products, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
                )
                tail = prefixes[:, :, k - 32 : k, :]
                partial = tail[:, :, 31:32, :]
                weight, coefficients = call("inputs", dx, case, ttnn.float32)
                with device_only():
                    terminal_output = obj.streaming_projection_scan(weight, coefficients)
                    generated_products = obj.streaming_projection_scan(weight, coefficients, products_only=True)
                row["products_fp32"] = product_bits(ttnn.to_torch(products), ttnn.to_torch(generated_products))
                del generated_products, weight, coefficients
                hn, hf = ttnn.to_torch(partial), ttnn.to_torch(terminal_output)
                row["partial_fp32"] = bits(hn, hf)
                del products, prefixes, tail, partial, terminal_output
            for dtype, label in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
                native = call("baseline", dx, case, dtype)
                hn = ttnn.to_torch(native)
                del native
                candidate = call("candidate", dx, case, dtype)
                hf = ttnn.to_torch(candidate)
                del candidate
                row["outputs"][label] = bits(hn, hf)
                payload[case["name"]][label] = dict(native=hn, candidate=hf)
            result["cases"].append(row)
            save()
            assert (
                row.get("partial_fp32", {"exact": True})["exact"]
                and row.get("products_fp32", {"exact": True})["exact"]
                and all(v["exact"] for v in row["outputs"].values())
            ), case["name"]
            if args.timing and (case["name"].startswith("actual_227/") or case["name"].startswith("stacked_")):
                for variant in ("baseline", "candidate"):
                    for _ in range(3):
                        out = call(variant, dx, case, ttnn.bfloat16)
                    reference = ttnn.to_torch(out)
                    eager_samples = []
                    for _ in range(5):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call(variant, dx, case, ttnn.bfloat16)
                        ttnn.synchronize_device(mesh)
                        eager_samples.append((time.perf_counter() - start) * 1000)
                    assert bits(reference, ttnn.to_torch(out))["exact"]
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = call(variant, dx, case, ttnn.bfloat16)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        samples = []
                        for _ in range(10):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter()
                            for _ in range(10):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            samples.append((time.perf_counter() - start) * 1000 / 10)
                        repeated = bits(reference, ttnn.to_torch(out))
                        assert repeated["exact"]
                        result["timing"].append(
                            dict(
                                name=case["name"],
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                warmed_eager_median_ms=statistics.median(eager_samples),
                                warmed_eager_samples_ms=eager_samples,
                                after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
                    del out
                save()
        assert result["cases"], "No cases selected"
        torch.save(payload, Path(str(args.output) + ".device.pt"))
        result["status"] = "component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot", type=Path, default=E / "v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-cases", type=int, default=1)
    parser.add_argument(
        "--case-filter", nargs="+", help="Optional case-name substring selection; recorded cases show executed scope"
    )
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--timing", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
