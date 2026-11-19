# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Read original BF16 TILE activation bits directly into paired coefficient columns."""

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

from models.autoports.qwen_qwen3_6_35b_a3b.tests import v6_streaming_projection_scan_probe as stream
from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import verify_source

E, REPO, REVISION = stream.E, stream.REPO, stream.REVISION
sha, bits = stream.sha, stream.bits
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_integration.py.snapshot"
BASE_SHA = "700c10db90b23fac1d40427c9bbcd82a6924c7704699d54768a0160c2b641e05"
SETUP_SOURCE = E / "v6_inline_router_silu_chunk64_output_mkl_explicit_norm_v2.py.snapshot"
READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 constexpr uint32_t activation_offset=get_compile_time_arg_val(0);
 constexpr uint32_t output_offset=get_compile_time_arg_val(1);
 constexpr bool coefficients_only=get_compile_time_arg_val(2)!=0;
 const auto weight=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto activation=TensorAccessor(TensorAccessorArgs<activation_offset>(),get_arg_val<uint32_t>(1));
 const auto coefficient_output=TensorAccessor(TensorAccessorArgs<output_offset>(),get_arg_val<uint32_t>(6));
 const uint32_t start=get_arg_val<uint32_t>(2),runs=get_arg_val<uint32_t>(3);
 const uint32_t ht=get_arg_val<uint32_t>(4),wt=get_arg_val<uint32_t>(5);
 Noc noc;DataflowBuffer a(0),b(1),raw(2);
 // Initialize both contiguous coefficient CB pages before publishing either.
 // Only column0 changes afterward; padded columns stay zero on every reuse.
 b.reserve_back(2);
 auto* initial=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 for(uint32_t i=0;i<1024;++i) initial[i]=0;
 for(uint32_t run=start;run<start+runs;++run) {
  if(coefficients_only && run%wt!=0) continue;
  for(uint32_t h=0;h<ht;++h) {
   const uint32_t tile=(run/wt)*ht*wt+h*wt+run%wt;
   const uint32_t column=(run/wt)*ht+h;
   raw.reserve_back(1);b.reserve_back(1);
   if constexpr(!coefficients_only) {
    a.reserve_back(1);
    noc.async_read(weight,a,2048,{.page_id=tile},{.offset_bytes=0});
   }
   noc.async_read(activation,raw,2048,{.page_id=column},{.offset_bytes=0});
   noc.async_read_barrier();raw.push_back(1);raw.wait_front(1);
   auto* source=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(2));
   auto* target=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(1));
   // Original activation TILE row0, pair-flipped inside each 32-lane tile.
   // Coefficient TILE column0: face0 rows0..15, face2 rows16..31.
   for(uint32_t r=0;r<32;++r) {
    const uint32_t paired=r^1;
    const uint32_t src=(paired/16)*256+paired%16;
    const uint32_t dst=(r/16)*512+(r%16)*16;
    target[dst]=source[src];
   }
   raw.pop_front(1);b.push_back(1);
   if constexpr(coefficients_only) {
    b.wait_front(1);
    noc.async_write(b,coefficient_output,2048,{}, {.page_id=column});
    noc.async_write_barrier();b.pop_front(1);
   } else { a.push_back(1); }
  }
 }
}
"""


class DirectCoefficientStreamingScan:
    """Same streaming FPU/cumsum, with raw BF16 original-input coefficient construction."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, weight, activation, *, products_only=False, coefficients_only=False):
        t = self.ttnn
        shape = tuple(weight.shape)
        assert shape[-2] in (512, 1024) and shape[-1] % 32 == 0 and len(shape) == 4 and shape[1] == 1
        total_k = math.prod(shape[:-2]) * shape[-2]
        assert tuple(activation.shape) == (1, 1, total_k)
        assert not (products_only and coefficients_only)
        assert weight.dtype == activation.dtype == t.bfloat16 and weight.layout == activation.layout == t.TILE_LAYOUT
        assert weight.memory_config() in (t.L1_MEMORY_CONFIG, t.DRAM_MEMORY_CONFIG)
        # TensorAccessor carries original interleaved or sharded activation addressing.
        ht, wt = shape[-2] // 32, shape[-1] // 32
        partition = stream.schedule(math.prod(shape[:-2]) * wt, self.grid.x, self.grid.y)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partition])
        output_shape = (
            shape[:-1] + (1,) if coefficients_only else shape if products_only else shape[:-2] + (1, shape[-1])
        )
        output = t.empty(
            output_shape,
            device=self.mesh,
            dtype=t.bfloat16 if coefficients_only else t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG if products_only or coefficients_only else t.L1_MEMORY_CONFIG,
        )
        rd, wr, cr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        for x, y, start, count in partition:
            rd[x][y] = [
                weight.buffer_address(),
                activation.buffer_address(),
                start,
                count,
                ht,
                wt,
                output.buffer_address(),
            ]
            wr[x][y] = [output.buffer_address(), start, count, ht, wt]
            cr[x][y] = [count]
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )

        def kernel(source, conf, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=args,
                runtime_args=runtime,
            )

        accessors = [t.TensorAccessorArgs(value) for value in (weight, activation, output)]
        assert all(not a.get_common_runtime_args() for a in accessors)
        wa, xa, oa = [a.get_compile_time_args() for a in accessors]
        kernels = [
            kernel(
                READER,
                t.ReaderConfigDescriptor(),
                [3 + len(wa), 3 + len(wa) + len(xa), int(coefficients_only)] + wa + xa + oa,
                rd,
            )
        ]
        if not coefficients_only:
            kernels += [
                kernel(stream.COMPUTE, config, [ht, int(products_only)], cr),
                kernel(stream.PRODUCT_WRITER if products_only else stream.WRITER, t.WriterConfigDescriptor(), oa, wr),
            ]
        program = t.ProgramDescriptor(
            kernels=kernels,
            cbs=[
                t.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, dtype, page)],
                )
                for index, pages, page, dtype in [
                    (0, 2, 2048, t.bfloat16),
                    (1, 2, 2048, t.bfloat16),
                    (2, 1, 2048, t.bfloat16),
                    (16, 1, 4096, t.float32),
                    (17, 1, 4096, t.float32),
                ]
            ],
        )
        t.generic_op([weight, activation, output], program)
        return output


def sources():
    assert sha(BASE) == BASE_SHA
    assert verify_source(stream.__file__, "b555f5caae383fb2cf1976860e92b9af3f1f2e2e39ec560e4c43d4d05bb23354")
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    baseline = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "linear")
    candidate = copy.deepcopy(baseline)
    inputs = copy.deepcopy(baseline)
    inputs.body = inputs.body[1:4]
    for branch in inputs.body:
        i = next(
            i for i, n in enumerate(branch.body) if isinstance(n, ast.If) and ast.unparse(n.test) == "terminal_scan"
        )
        branch.body = branch.body[:i] + ast.parse("return weight, coefficients").body
    inputs.body += ast.parse("raise AssertionError('singleton inputs required')").body
    for index, branch in enumerate(candidate.body[1:4]):
        selected = []
        for n in branch.body:
            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) in ("projection_k", "blocks", "width", "weight"):
                selected.append(copy.deepcopy(n))
        assert [ast.unparse(n.targets[0]) for n in selected] == (["projection_k"] if index < 2 else []) + [
            "blocks",
            "width",
            "weight",
        ]
        selected += ast.parse("partial = self.direct_coefficient_scan(weight, x)").body
        selected += copy.deepcopy(branch.body[-3:])
        branch.body.insert(0, ast.If(ast.Name("terminal_scan", ast.Load()), selected, []))
    restored = copy.deepcopy(candidate)
    for branch in restored.body[1:4]:
        assert ast.unparse(branch.body[0].test) == "terminal_scan"
        del branch.body[0]
    assert ast.dump(restored) == ast.dump(baseline)
    return dict(
        baseline=ast.unparse(baseline),
        candidate=ast.unparse(ast.fix_missing_locations(candidate)),
        inputs=ast.unparse(ast.fix_missing_locations(inputs)),
        setup=stream.extract(SETUP_SOURCE)["setup"],
    )


def mapping_proof():
    reports = []
    # Exhaust every possible BF16 bit pattern without interpreting its numerical value.
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(-1, 32)
    row = torch.zeros((len(codes), 1024), dtype=torch.int16)
    row[:, :16] = codes[:, :16]
    row[:, 256:272] = codes[:, 16:]
    columns = torch.zeros_like(row)
    for r in range(32):
        paired = r ^ 1
        columns[:, (r // 16) * 512 + (r % 16) * 16] = row[:, (paired // 16) * 256 + paired % 16]
    recovered = torch.stack([columns[:, (r // 16) * 512 + (r % 16) * 16] for r in range(32)], dim=1)
    assert torch.equal(recovered, codes[:, torch.arange(32) ^ 1])
    for total_k in (2048, 4096):
        for k in (512, 1024):
            native = (
                torch.arange(total_k)
                .reshape(1, total_k // 32, 32)[:, :, torch.arange(32) ^ 1]
                .reshape(total_k // k, 1, k, 1)
            )
            direct = torch.tensor(
                [
                    ((block * (k // 32) + h) * 32) + (r ^ 1)
                    for block in range(total_k // k)
                    for h in range(k // 32)
                    for r in range(32)
                ]
            ).reshape_as(native)
            assert torch.equal(native, direct)
            reports.append(dict(total_k=total_k, projection_k=k, exact=True))
    return dict(all_65536_bf16_bit_patterns_exact=True, block_mapping=reports)


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve earlier evidence"
    code = sources()
    state, cases, captures = stream.fixtures(args.checkpoint, args.random_cases)
    for case in cases:
        if case["x"].shape[-2] == 1:
            case["kwargs"]["terminal_scan"] = True
    mapping = mapping_proof()
    extras = []
    for case in cases:
        case["placement"] = "dram"
        if case["name"].endswith("/out"):
            extra = copy.deepcopy(case)
            extra["name"] += "/width_sharded"
            extra["placement"] = "width_sharded"
            extras.append(extra)
        if case["name"] in ("actual_227/router", "actual_227/attention_k512", "actual_227/output_k512"):
            extra = copy.deepcopy(case)
            extra["name"] += "/l1"
            extra["placement"] = "l1"
            extras.append(extra)
    cases += extras
    result = dict(
        status="cpu_source_setup_pass",
        device_executed=False,
        classification="direct raw-bit BF16 coefficient reader component; integrated performance unmeasured",
        script_sha256=sha(__file__),
        base=str(BASE),
        base_sha256=BASE_SHA,
        streaming_helper=str(Path(stream.__file__)),
        streaming_helper_sha256=sha(stream.__file__),
        setup_snapshot=str(SETUP_SOURCE),
        setup_snapshot_sha256=sha(SETUP_SOURCE),
        captures=captures,
        fixture_cases=len(cases),
        seed=603636,
        full_linear_ast_restoration=True,
        full_attention_guard_unchanged=True,
        default_prefill_tail_unchanged=True,
        changed_arithmetic="None: raw BF16 coefficient bit relocation; frozen streaming FPU/cumsum and native reduction/cast retained",
        native_order="Direct FP32 unpack, copy block0 to dst0, then add_binary_tile(dst0,dst1,dst0) in ascending block order",
        precision_limit="Raw mapping source proof only; native coefficient/product/partial/output equality requires device checks",
        coefficient_mapping_proof=mapping,
        cases=[],
        timing=[],
    )
    native_dir = REPO / "ttnn/cpp/ttnn/operations/experimental/reduction/fast_reduce_nc/device"
    dependencies = [
        native_dir / name
        for name in [
            "fast_reduce_nc_program_factory.cpp",
            "fast_reduce_nc_device_operation.cpp",
            "kernels/reduce_nc.cpp",
            "kernels/reader_reduce_nc.cpp",
        ]
    ]
    dependencies += [
        REPO / "ttnn/cpp/ttnn/operations/copy/typecast/typecast.cpp",
        REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_typecast.h",
        REPO / "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp",
    ]
    dependencies += [
        REPO / "tt_metal/impl/buffers/tensor_accessor_args.cpp",
        REPO
        / "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/nlp_concat_heads_decode_device_operation.cpp",
    ]
    result["dependencies"] = [dict(path=str(p), sha256=sha(p)) for p in dependencies]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name, text in code.items():
        Path(str(args.output) + f".{name}.snapshot").write_text(text)
    for label, source in dict(
        reader=READER, compute=stream.COMPUTE, writer=stream.WRITER, product_writer=stream.PRODUCT_WRITER
    ).items():
        Path(str(args.output) + f".{label}.cpp.snapshot").write_text(source)
    fixture_path = Path(str(args.output) + ".fixtures.pt")
    torch.save(dict(state=state, cases=cases), fixture_path)
    result["fixtures"] = dict(path=str(fixture_path), sha256=sha(fixture_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SOURCE_SETUP_PASS", len(cases), "65536 coefficient bit patterns")
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
            streaming_projection_scan=stream.StreamingProjectionScan(ttnn, mesh),
            direct_coefficient_scan=DirectCoefficientStreamingScan(ttnn, mesh),
            mesh_device=mesh,
            kind="linear_attention",
            decode_linear={},
            w={},
        )
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
            compile(code["setup"], str(args.output) + ":setup", "exec"),
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
        for label in ("baseline", "candidate", "inputs"):
            ns = dict(ttnn=ttnn, DRAM=ttnn.DRAM_MEMORY_CONFIG)
            exec(compile(code[label], str(args.output) + ":" + label, "exec"), ns)
            calls[label] = types.MethodType(ns["linear"], obj)

        def call(label, dx, case, dtype):
            obj.linear = calls[label]
            with device_only():
                return obj.linear(dx, case["key"], dtype, **case["kwargs"])

        payload = {}
        for case in cases:
            if args.case_filter and not any(s in case["name"] for s in args.case_filter):
                continue
            dx = upload(case["x"])
            if case["placement"] == "l1":
                dx = ttnn.to_memory_config(dx, ttnn.L1_MEMORY_CONFIG)
            elif case["placement"] == "width_sharded":
                cores = ttnn.num_cores_to_corerangeset(32, mesh.compute_with_storage_grid_size(), row_wise=True)
                memory = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(cores, (32, 128), ttnn.ShardOrientation.ROW_MAJOR),
                )
                dx = ttnn.to_memory_config(dx, memory)
            row = dict(name=case["name"], shape=list(case["x"].shape), outputs={})
            payload[case["name"]] = {}
            if case["x"].shape[-2] == 1:
                weight, coefficient = call("inputs", dx, case, ttnn.float32)
                with device_only():
                    direct_coefficient = obj.direct_coefficient_scan(weight, dx, coefficients_only=True)
                row["coefficient_bits"] = bits(ttnn.to_torch(coefficient), ttnn.to_torch(direct_coefficient))
                del direct_coefficient
                native_product = ttnn.multiply(
                    weight,
                    coefficient,
                    dtype=ttnn.float32,
                    fast_and_approximate_mode=False,
                    memory_config=ttnn.L1_MEMORY_CONFIG,
                )
                with device_only():
                    direct_product = obj.direct_coefficient_scan(weight, dx, products_only=True)
                row["product_fp32"] = stream.product_bits(ttnn.to_torch(native_product), ttnn.to_torch(direct_product))
                del native_product, direct_product
                with device_only():
                    native_partial = obj.streaming_projection_scan(weight, coefficient)
                    direct_partial = obj.direct_coefficient_scan(weight, dx)
                row["partial_fp32"] = bits(ttnn.to_torch(native_partial), ttnn.to_torch(direct_partial))
                del native_partial, direct_partial, weight, coefficient

            for dtype, label in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
                old = call("baseline", dx, case, dtype)
                hn = ttnn.to_torch(old)
                placement = old.memory_config()
                del old
                new = call("candidate", dx, case, dtype)
                assert new.memory_config() == placement
                hf = ttnn.to_torch(new)
                del new
                row["outputs"][label] = bits(hn, hf)
                payload[case["name"]][label] = dict(native=hn, packed=hf)
            result["cases"].append(row)
            save()
            assert all(v["exact"] for v in row["outputs"].values()) and all(
                row.get(k, {"exact": True})["exact"] for k in ("coefficient_bits", "product_fp32", "partial_fp32")
            ), case["name"]
            if args.timing and case["name"].startswith("actual_227/"):
                for variant in ("baseline", "candidate"):
                    for _ in range(3):
                        output = call(variant, dx, case, ttnn.bfloat16)
                    reference = ttnn.to_torch(output)
                    eager = []
                    for _ in range(5):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        output = call(variant, dx, case, ttnn.bfloat16)
                        ttnn.synchronize_device(mesh)
                        eager.append((time.perf_counter() - start) * 1000)
                    output = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call(variant, dx, case, ttnn.bfloat16)
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
                        repeated = bits(reference, ttnn.to_torch(output))
                        assert repeated["exact"]
                        result["timing"].append(
                            dict(
                                name=case["name"],
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                warmed_eager_median_ms=statistics.median(eager),
                                warmed_eager_samples_ms=eager,
                                after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
                    del output
                save()
        assert result["cases"]
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--random-cases", type=int, default=1)
    p.add_argument("--case-filter", nargs="+")
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--timing", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
