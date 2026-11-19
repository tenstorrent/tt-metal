# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Terminal-only native router cumsum diagnostic. Hardware execution is opt-in."""

import argparse
import ast
import copy
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_ordered_router_probe import REVISION, setup_ordered

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
REPO = ROOT.parents[2]
READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto input=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t run=get_arg_val<uint32_t>(1);
 const uint32_t ht=get_arg_val<uint32_t>(2), wt=get_arg_val<uint32_t>(3);
 Noc noc; DataflowBuffer buffer(0);
 for(uint32_t h=0;h<ht;++h) {
  buffer.reserve_back(1);
  const uint32_t id=(run/wt)*ht*wt+h*wt+run%wt;
  noc.async_read(input,buffer,4096,{.page_id=id},{.offset_bytes=0});
  noc.async_read_barrier();buffer.push_back(1);
 }
}
"""
COMPUTE = r"""
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/cumsum.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/dataflow_buffer.h"
void kernel_main() {
 DataflowBuffer input(0), output(16);
 compute_kernel_hw_startup(0,16);
 constexpr uint32_t count=get_compile_time_arg_val(0);
 for(uint32_t j=0;j<count;++j) {
  input.wait_front(1);tile_regs_acquire();
  copy_init(0);copy_tile(0,0,0);
  cumsum_tile_init<CumsumMode::Default>();
  cumsum_tile<CumsumMode::Default>(0,j==0);
  tile_regs_commit();input.pop_front(1);tile_regs_wait();
  if(j+1==count) {
   output.reserve_back(1);pack_tile(0,16);output.push_back(1);
  }
  tile_regs_release();
 }
}
"""
WRITER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto output=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t id=get_arg_val<uint32_t>(1);
 Noc noc;DataflowBuffer final_tile(16), row(17);
 final_tile.wait_front(1);row.reserve_back(1);
 auto* src=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));
 auto* dst=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(17));
 for(uint32_t i=0;i<1024;++i) dst[i]=0;
 // Float32 tile faces: copy physical row31 to logical row0 without arithmetic.
 for(uint32_t i=0;i<16;++i) { dst[i]=src[752+i];dst[256+i]=src[1008+i]; }
 row.push_back(1);row.wait_front(1);
 noc.async_write(row,output,4096,{}, {.page_id=id});
 noc.async_write_barrier();row.pop_front(1);final_tile.pop_front(1);
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TerminalCumsum:
    """Router-specific terminal native scan; no runtime tensor host conversion."""

    def __init__(self, ttnn, mesh):
        self.ttnn, self.mesh = ttnn, mesh
        self.cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 3))])

    def __call__(self, products):
        ttnn = self.ttnn
        assert tuple(products.shape) == (4, 1, 512, 256)
        assert products.dtype == ttnn.float32 and products.layout == ttnn.TILE_LAYOUT
        assert products.memory_config() == ttnn.L1_MEMORY_CONFIG
        output = ttnn.empty(
            (4, 1, 1, 256),
            device=self.mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.L1_MEMORY_CONFIG,
        )
        rd, wr = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        for run in range(32):
            x, y = run % 8, run // 8
            rd[x][y] = [products.buffer_address(), run, 16, 8]
            wr[x][y] = [output.buffer_address(), run]
        config = ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            dst_full_sync_en=False,
        )
        modes = [ttnn.UnpackToDestMode.Default] * 64
        modes[0] = ttnn.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, conf, compile_args, runtime):
            return ttnn.KernelDescriptor(
                kernel_source=source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=self.cores,
                config=conf,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = ttnn.ProgramDescriptor(
            kernels=[
                kernel(
                    READER, ttnn.ReaderConfigDescriptor(), ttnn.TensorAccessorArgs(products).get_compile_time_args(), rd
                ),
                kernel(COMPUTE, config, [16], []),
                kernel(
                    WRITER, ttnn.WriterConfigDescriptor(), ttnn.TensorAccessorArgs(output).get_compile_time_args(), wr
                ),
            ],
            cbs=[
                ttnn.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=self.cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.float32, 4096)],
                )
                for index, pages in [(0, 2), (16, 1), (17, 1)]
            ],
        )
        ttnn.generic_op([products, output], program)
        return output


def methods(snapshot):
    """Clone only the already-selected branch; substitute exactly the six scan/crop statements."""
    tree = ast.parse(snapshot.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "linear")
    assert isinstance(original.body[0], ast.If)
    assert ast.unparse(original.body[0].test) == "ordered_router and x.shape[-2] == 1 and (key == 'mlp.gate.weight')"
    baseline = copy.deepcopy(original)
    baseline.body = baseline.body[0].body
    i = next(
        i for i, n in enumerate(baseline.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "prefixes"
    )
    old = baseline.body[i : i + 6]
    assert [ast.unparse(n).split(" = ")[0] for n in old] == [
        "prefixes",
        "del products",
        "tail",
        "del prefixes",
        "partial",
        "del tail",
    ]
    assert "disable_compensated_sum=True" in ast.unparse(old[0])
    candidate = copy.deepcopy(baseline)
    candidate.body[i : i + 6] = ast.parse("partial = self.terminal_cumsum(products)\ndel products").body
    products = copy.deepcopy(baseline)
    products.body = products.body[:i] + [ast.Return(ast.Name("products", ast.Load()))]
    restore = copy.deepcopy(candidate)
    restore.body[i : i + 2] = copy.deepcopy(old)
    assert ast.dump(restore) == ast.dump(baseline)
    return {
        k: ast.unparse(ast.fix_missing_locations(v))
        for k, v in [("baseline", baseline), ("candidate", candidate), ("products", products)]
    }


def bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    integer = torch.int32 if a.dtype == torch.float32 else torch.int16
    count = int((a.contiguous().view(integer) != b.contiguous().view(integer)).sum())
    return dict(bit_different=count, exact=count == 0, max_abs=float((a.float() - b.float()).abs().max()))


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve earlier evidence"
    sources = methods(args.snapshot)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.layers.34.mlp.gate.weight"
    weight_path = args.checkpoint / index[key]
    with safe_open(weight_path, framework="pt", device="cpu") as f:
        weight = f.get_tensor(key)
    assert weight.dtype == torch.bfloat16 and tuple(weight.shape) == (256, 2048)
    inputs, captures = {}, []
    for pos in [227, 738, 5799]:
        path = E / f"v6_inline_precision_stream_probe.json.pos{pos}.pt"
        cap = torch.load(path, map_location="cpu", weights_only=False)
        assert cap["layer"] == 34 and cap["position"] == pos
        inputs[f"actual_{pos}"] = cap["tt_values"]["post_norm"].reshape(1, 1, 2048).bfloat16()
        captures.append(dict(path=str(path), sha256=sha(path)))
    generator = torch.Generator().manual_seed(603635)
    for i in range(args.random_cases):
        inputs[f"random_{i}"] = (torch.randn((1, 1, 2048), generator=generator) * (2.0 ** ((i % 5) - 2))).bfloat16()
    inputs["zeros"] = torch.zeros((1, 1, 2048), dtype=torch.bfloat16)
    # Raw tile remapping and tile traversal proof, independent of numerical scan assumptions.
    tile = torch.arange(1024).reshape(2, 2, 16, 16).permute(0, 2, 1, 3).reshape(32, 32)
    copied = torch.cat((torch.arange(752, 768), torch.arange(1008, 1024)))
    assert torch.equal(tile[31], copied)
    visited = [(run // 8) * 16 * 8 + h * 8 + run % 8 for run in range(32) for h in range(16)]
    assert sorted(visited) == list(range(512))
    dependencies = [
        Path(__file__),
        args.snapshot,
        Path(sys.modules[setup_ordered.__module__].__file__),
        REPO
        / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp",
        REPO / "ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp",
        REPO / "tt_metal/hw/inc/api/compute/cumsum.h",
        REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h",
    ]
    result = dict(
        status="cpu_source_setup_pass",
        device_executed=False,
        classification="component diagnostic; no decoder acceptance or speedup claim",
        dependencies=[dict(path=str(p), sha256=sha(p)) for p in dependencies],
        captures=captures,
        revision=REVISION,
        checkpoint_weight_file=str(weight_path),
        random_seed=603635,
        changed_ast="six cumsum/crop statements replaced with terminal native scan and del products; exact AST restoration checked",
        raw_row31_to_row0_mapping_pass=True,
        tile_visit_bijection_pass=True,
        expected_saved_prefix_bytes=2097152,
        candidate_output_tile_bytes=131072,
        source_only_limit="CPU does not emulate native SFPU scan arithmetic; device bit comparison is required",
        cases=[],
        timing=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for label, source in dict(reader=READER, compute=COMPUTE, writer=WRITER, **sources).items():
        path = Path(str(args.output) + f".{label}.snapshot")
        path.write_text(source)
    fixture_path = Path(str(args.output) + ".fixtures.pt")
    torch.save(dict(weight=weight, inputs=inputs), fixture_path)
    result["fixtures"] = dict(path=str(fixture_path), sha256=sha(fixture_path))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SOURCE_SETUP_PASS", len(inputs))
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") in (None, "", "0")
    import ttnn

    mesh = None
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=67108864
        )
        result.update(status="running", device_executed=True, device_closed=False)
        obj = types.SimpleNamespace(terminal_cumsum=TerminalCumsum(ttnn, mesh), mesh_device=mesh)
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

        setup_ordered(obj, args.snapshot, weight, upload, ttnn, mesh)
        from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

        def audited(method):
            def wrapped(*a, **kw):
                with device_only():
                    return method(*a, **kw)

            return wrapped

        calls = {}
        for label, source in sources.items():
            namespace = dict(ttnn=ttnn, DRAM=ttnn.DRAM_MEMORY_CONFIG)
            exec(compile(source, str(args.output) + f":{label}", "exec"), namespace)
            calls[label] = audited(types.MethodType(namespace["linear"], obj))
        payload = {}
        for name, x in inputs.items():
            dx = upload(x)
            product = calls["products"](dx, "mlp.gate.weight")
            prefixes = ttnn.cumsum(
                product, -2, dtype=ttnn.float32, disable_compensated_sum=True, memory_config=ttnn.L1_MEMORY_CONFIG
            )
            tail = prefixes[:, :, 480:512, :]
            native_partial = tail[:, :, 31:32, :]
            fused_partial = obj.terminal_cumsum(product)
            host_native, host_fused = ttnn.to_torch(native_partial), ttnn.to_torch(fused_partial)
            row = dict(name=name, partial_fp32=bits(host_native, host_fused), outputs={})
            payload[name] = dict(native_partial=host_native, fused_partial=host_fused)
            del product, prefixes, tail, native_partial, fused_partial
            for dtype, label in [(ttnn.float32, "fp32"), (ttnn.bfloat16, "bf16")]:
                native = calls["baseline"](dx, "mlp.gate.weight", dtype)
                fused = calls["candidate"](dx, "mlp.gate.weight", dtype)
                hn, hf = ttnn.to_torch(native), ttnn.to_torch(fused)
                row["outputs"][label] = bits(hn, hf)
                payload[name][label] = dict(native=hn, fused=hf)
                del native, fused
            result["cases"].append(row)
            save()
            assert row["partial_fp32"]["exact"] and all(v["exact"] for v in row["outputs"].values()), name
            if name == "actual_227" and args.timing:
                for variant in ("baseline", "candidate"):
                    call = lambda: calls[variant](dx, "mlp.gate.weight", ttnn.bfloat16)
                    for _ in range(3):
                        output = call()
                    reference = ttnn.to_torch(output)
                    output = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = call()
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
                                variant=variant,
                                median_ms=statistics.median(samples),
                                samples_ms=samples,
                                after_100_replays=repeated,
                            )
                        )
                    finally:
                        ttnn.release_trace(mesh, trace)
                    del output
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
    parser.add_argument("--random-cases", type=int, default=8)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--timing", action="store_true", help="Opt-in warmed traced before/after component timing")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
