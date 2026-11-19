# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact tiny norm terminal scans; CPU setup by default, device/timing opt-in."""

import argparse
import ast
import copy
import hashlib
import json
import os
import statistics
import time
import traceback
import types
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
REPO = ROOT.parents[2]
SNAPSHOT = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
SNAPSHOT_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
NATIVE = [
    "ttnn/cpp/ttnn/operations/reduction/accumulation/device/accumulation_program_factory.cpp",
    "ttnn/cpp/ttnn/operations/reduction/accumulation/device/kernels/compute/accumulation_tile_axis_compute.cpp",
    "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h",
    "tt_metal/hw/inc/api/compute/cumsum.h",
]
READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
 const auto input=TensorAccessor(TensorAccessorArgs<0>(),get_arg_val<uint32_t>(0));
 const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);
 Noc noc;DataflowBuffer in(0);
 for(uint32_t row=start;row<start+count;++row) {
  in.reserve_back(1);
  noc.async_read(input,in,4096,{.page_id=row},{});
  noc.async_read_barrier();in.push_back(1);
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
 const uint32_t count=get_arg_val<uint32_t>(0);
 DataflowBuffer groups(0), transposed_lanes(1), height_scan(16), width_scan(17);
 compute_kernel_hw_startup(0,16);
 for(uint32_t row=0;row<count;++row) {
  groups.wait_front(1);tile_regs_acquire();
  copy_init(0);copy_tile(0,0,0);
  cumsum_tile_init<CumsumMode::Default>();
  cumsum_tile<CumsumMode::Default>(0,true);
  tile_regs_commit();groups.pop_front(1);tile_regs_wait();
  height_scan.reserve_back(1);pack_tile(0,16);height_scan.push_back(1);
  tile_regs_release();
  // BRISC copies the exact packed row3 bits into a transposed lane tile.
  // Native width scan also transposes before scanning; no arithmetic is moved.
  transposed_lanes.wait_front(1);tile_regs_acquire();
  copy_init(1);copy_tile(1,0,0);
  cumsum_tile_init<CumsumMode::Default>();
  cumsum_tile<CumsumMode::Default>(0,true);
  tile_regs_commit();transposed_lanes.pop_front(1);tile_regs_wait();
  width_scan.reserve_back(1);pack_tile(0,17);width_scan.push_back(1);
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
 constexpr bool debug=get_compile_time_arg_val(0)!=0;
 auto output_args=TensorAccessorArgs<1,0>();
 auto lanes_args=TensorAccessorArgs<output_args.next_compile_time_args_offset(),output_args.next_common_runtime_args_offset()>();
 const auto output=TensorAccessor(output_args,get_arg_val<uint32_t>(0));
 const auto lanes_output=TensorAccessor(lanes_args,get_arg_val<uint32_t>(1));
 const uint32_t start=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3);
 Noc noc;DataflowBuffer height_scan(16), transposed_lanes(1), width_scan(17), scalar(18), lanes(19);
 for(uint32_t row=start;row<start+count;++row) {
  height_scan.wait_front(1);transposed_lanes.reserve_back(1);
  const auto* h=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));
  auto* t=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
  for(uint32_t j=0;j<1024;++j)t[j]=0;
  // FP32 physical face0 row3 colj -> transposed physical rowj col0.
  // Later padded columns/rows cannot affect either logical terminal.
  for(uint32_t j=0;j<8;++j)t[j*16]=h[3*16+j];
  if constexpr(debug) {
   lanes.reserve_back(1);
   auto* l=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(19));
   for(uint32_t j=0;j<1024;++j)l[j]=0;
   for(uint32_t j=0;j<8;++j)l[j]=h[3*16+j];
   lanes.push_back(1);lanes.wait_front(1);
   noc.async_write(lanes,lanes_output,4096,{}, {.page_id=row});
   noc.async_write_barrier();lanes.pop_front(1);
  }
  transposed_lanes.push_back(1);height_scan.pop_front(1);
  width_scan.wait_front(1);scalar.reserve_back(1);
  const auto* w=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(17));
  auto* o=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(18));
  for(uint32_t j=0;j<1024;++j)o[j]=0;
  o[0]=w[7*16];
  scalar.push_back(1);scalar.wait_front(1);
  noc.async_write(scalar,output,4096,{}, {.page_id=row});
  noc.async_write_barrier();scalar.pop_front(1);width_scan.pop_front(1);
 }
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bitdiff(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    count = int((a.contiguous().view(dtype) != b.contiguous().view(dtype)).sum())
    return dict(exact=count == 0, differing_elements=count, elements=a.numel())


class NormTerminalScan:
    """Two native Default FP32 scans and exact packed-bit selection in one dispatch."""

    def __init__(self, ttnn, mesh):
        self.t, self.mesh = ttnn, mesh
        self.grid = mesh.compute_with_storage_grid_size()

    def __call__(self, groups, *, debug=False):
        t = self.t
        shape = tuple(groups.shape)
        assert len(shape) == 4 and shape[1:] == (1, 4, 8)
        assert groups.dtype == t.float32 and groups.layout == t.TILE_LAYOUT
        assert groups.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        rows = shape[0]
        active = min(rows, self.grid.x * self.grid.y)
        cores = t.CoreRangeSet(
            [
                t.CoreRange(
                    t.CoreCoord(i % self.grid.x, i // self.grid.x), t.CoreCoord(i % self.grid.x, i // self.grid.x)
                )
                for i in range(active)
            ]
        )
        output = t.empty(
            (rows, 1, 1, 1),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=groups.memory_config(),
        )
        lanes = (
            t.empty(
                (rows, 1, 1, 8),
                device=self.mesh,
                dtype=t.float32,
                layout=t.TILE_LAYOUT,
                memory_config=groups.memory_config(),
            )
            if debug
            else output
        )
        rd, wr, cr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        start = 0
        for i in range(active):
            count = rows // active + (i < rows % active)
            x, y = i % self.grid.x, i // self.grid.x
            rd[x][y] = [groups.buffer_address(), start, count]
            wr[x][y] = [output.buffer_address(), lanes.buffer_address(), start, count]
            cr[x][y] = [count]
            start += count
        assert start == rows
        config = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=False
        )
        modes = [t.UnpackToDestMode.Default] * 64
        modes[0] = modes[1] = t.UnpackToDestMode.UnpackToDestFp32
        config.unpack_to_dest_mode = modes

        def kernel(source, conf, compile_args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=conf,
                compile_time_args=compile_args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), t.TensorAccessorArgs(groups).get_compile_time_args(), rd),
                kernel(COMPUTE, config, [], cr),
                kernel(
                    WRITER,
                    t.WriterConfigDescriptor(),
                    [int(debug)]
                    + t.TensorAccessorArgs(output).get_compile_time_args()
                    + t.TensorAccessorArgs(lanes).get_compile_time_args(),
                    wr,
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=4096 * pages,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(index, t.float32, 4096)],
                )
                for index, pages in [(0, 2), (1, 1), (16, 1), (17, 1), (18, 1), (19, 1)]
            ],
        )
        t.generic_op([groups, output, lanes] if debug else [groups, output], program)
        return (output, lanes) if debug else output


def methods():
    assert sha(SNAPSHOT) == SNAPSHOT_SHA
    tree = ast.parse(SNAPSHOT.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    original = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "norm")
    branch = next(n for n in original.body if isinstance(n, ast.If) and ast.unparse(n.test).startswith("key in ("))
    baseline = copy.deepcopy(original)
    baseline.body = copy.deepcopy(branch.body)
    i = next(
        i for i, n in enumerate(baseline.body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "lanes"
    )
    removed = copy.deepcopy(baseline.body[i : i + 2])
    assert [ast.unparse(n.targets[0]) for n in removed] == ["lanes", "total"]
    assert all("disable_compensated_sum=True" in ast.unparse(n) for n in removed)
    candidate = copy.deepcopy(baseline)
    candidate.body[i : i + 2] = ast.parse("total = self.terminal_norm_scan(groups)").body
    restored = copy.deepcopy(candidate)
    restored.body[i : i + 1] = removed
    assert ast.dump(restored) == ast.dump(baseline)
    groups = copy.deepcopy(baseline)
    groups.body = groups.body[:i] + [ast.Return(ast.Name("groups", ast.Load()))]
    return {
        name: ast.unparse(ast.fix_missing_locations(node))
        for name, node in [("baseline", baseline), ("candidate", candidate), ("groups", groups)]
    }


def fixtures(args):
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for key in ("input_layernorm.weight", "post_attention_layernorm.weight"):
        full = "model.language_model.layers.34." + key
        with safe_open(args.checkpoint / index[full], framework="pt", device="cpu") as f:
            weights[key] = f.get_tensor(full).float() + 1
    assert json.loads((args.checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"] == 1e-6
    captures, cases = [], []
    for pos in (227, 738, 5799):
        path = E / f"v6_inline_precision_stream_probe.json.pos{pos}.pt"
        capture = torch.load(path, map_location="cpu", weights_only=False)
        assert capture["layer"] == 34 and capture["position"] == pos
        captures.append(dict(path=str(path), sha256=sha(path)))
        for key, name in [("input_layernorm.weight", "input"), ("post_attention_layernorm.weight", "post")]:
            cases.append(
                dict(
                    name=f"actual_{pos}/{name}",
                    key=key,
                    x=capture["tt_values"][name + "_norm_input"].bfloat16().clone(),
                )
            )
    for case in list(cases):
        if case["key"] == "post_attention_layernorm.weight":
            cases.append(dict(case, name=case["name"] + "_width_sharded", placement="production_width_sharded"))
    gen = torch.Generator().manual_seed(613611)
    for rows in (1, 3, 33, 65):
        for key in weights:
            cases.append(
                dict(
                    name=f"random_rows{rows}/" + key,
                    key=key,
                    x=(torch.randn((1, rows, 2048), generator=gen) * 0.2).bfloat16(),
                )
            )
    for label, x in [
        ("zero", torch.zeros(1, 1, 2048)),
        ("alternating_sign", torch.ones(1, 1, 2048).mul(torch.arange(2048).remainder(2).mul(2).sub(1))),
    ]:
        cases.append(dict(name=label, key="input_layernorm.weight", x=x.bfloat16()))
    # Exact BF16 midpoint totals and their adjacent FP32 values. Native typecast is retained.
    codes = torch.arange(0x3F00, 0x4000, dtype=torch.int64)
    raw = ((codes << 16) + 0x8000).repeat_interleave(3) + torch.tensor([-1, 0, 1]).repeat(len(codes))
    midpoint = torch.zeros((len(raw), 1, 4, 8), dtype=torch.float32)
    midpoint[:, 0, 0, 0] = raw.to(torch.int32).view(torch.float32)
    controls = torch.zeros((5, 1, 4, 8), dtype=torch.float32)
    controls[0] = -0.0
    controls[1, 0, 0, 0] = torch.tensor(1, dtype=torch.int32).view(torch.float32)
    controls[2, 0, 0, 0] = torch.tensor(0x007FFFFF, dtype=torch.int32).view(torch.float32)
    controls[3, 0, :, 0] = torch.tensor([1.0, 2.0**-24, -1.0, 2.0**-24])
    controls[4, 0, :, 0] = torch.tensor([1.0, -1.0, -0.0, 0.0])
    return (
        weights,
        cases,
        captures,
        dict(bf16_midpoints_and_adjacent_fp32=midpoint, zero_subnormal_cancellation_controls=controls),
    )


def native_terminal(t, groups):
    lanes = t.cumsum(groups, -2, dtype=t.float32, disable_compensated_sum=True)[:, :, 3:4, :]
    total = t.cumsum(lanes, -1, dtype=t.float32, disable_compensated_sum=True)[:, :, :, 7:8]
    return total, lanes


def timing(t, mesh, call, expected, args):
    for _ in range(3):
        out = call()
    assert bitdiff(expected, t.to_torch(out))["exact"]
    eager = []
    for _ in range(args.samples):
        t.synchronize_device(mesh)
        start = time.perf_counter()
        out = call()
        t.synchronize_device(mesh)
        eager.append((time.perf_counter() - start) * 1000)
    assert bitdiff(expected, t.to_torch(out))["exact"]
    out = None
    trace = t.begin_trace_capture(mesh, cq_id=0)
    out = call()
    t.end_trace_capture(mesh, trace, cq_id=0)
    try:
        samples = []
        for _ in range(args.samples):
            t.synchronize_device(mesh)
            start = time.perf_counter()
            for _ in range(args.replays):
                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
            t.synchronize_device(mesh)
            samples.append((time.perf_counter() - start) * 1000 / args.replays)
        check = bitdiff(expected, t.to_torch(out))
        assert check["exact"]
        return dict(
            warmed_eager_ms=statistics.median(eager),
            eager_samples_ms=eager,
            traced_warmed_ms=statistics.median(samples),
            trace_samples_ms=samples,
            replays=args.samples * args.replays,
            after_replay_bitdiff=check,
        )
    finally:
        t.release_trace(mesh, trace)


def run(args):
    assert not args.output.exists(), "Preserve earlier evidence"
    torch.set_num_threads(4)
    sources = methods()
    weights, cases, captures, synthetic = fixtures(args)
    # Packed-tile raw-bit map: all 8 lanes stay in face0 and never cross a face.
    physical = torch.arange(1024, dtype=torch.int32)
    lane_bits = physical[48:56].clone()
    transposed = torch.zeros(1024, dtype=torch.int32)
    transposed[torch.arange(8) * 16] = lane_bits
    assert torch.equal(transposed[torch.arange(8) * 16], lane_bits) and int(physical[112]) == 112
    deps = {str(SNAPSHOT): sha(SNAPSHOT), **{str(REPO / name): sha(REPO / name) for name in NATIVE}}
    result = dict(
        status="cpu_setup_pass",
        source_sha256=sha(__file__),
        snapshot_sha256=SNAPSHOT_SHA,
        dependencies=deps,
        captures=captures,
        revision=REVISION,
        device_executed=False,
        device_closed=False,
        cases=[],
        synthetic_cases=[],
        timing={},
        raw_bit_mapping_pass=True,
        ast_restoration_exact=True,
        norm_weight_setup="Checkpoint layer34 weight.float()+1 uploaded FP32, exactly from_state_dict; epsilon1e-6 verified against config.json",
        scope="Only two tiny norm terminal cumsums/crops replaced; all preceding outer reduction and following reciprocal refinement/BF16 arithmetic unchanged",
        kernel_sha256={
            name: hashlib.sha256(code.encode()).hexdigest()
            for name, code in [("reader", READER), ("compute", COMPUTE), ("writer", WRITER)]
        },
        fixture_shapes=[dict(name=c["name"], shape=list(c["x"].shape)) for c in cases],
        synthetic_shapes={name: list(x.shape) for name, x in synthetic.items()},
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Path(str(args.output) + ".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    for name, source in sources.items():
        Path(str(args.output) + f".{name}.py.snapshot").write_text(source + "\n")
    for name, source in [("reader", READER), ("compute", COMPUTE), ("writer", WRITER)]:
        Path(str(args.output) + f".{name}.cpp.snapshot").write_text(source)
    torch.save(dict(weights=weights, cases=cases, synthetic=synthetic), Path(str(args.output) + ".fixtures.pt"))

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        print("NORM_TERMINAL_SCAN_CPU_READY", result["source_sha256"])
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER") in (None, "", "0")
    assert not os.environ.get("TT_VISIBLE_DEVICES"), "Host uses explicit physical_device_ids=[0]"
    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=67108864, l1_small_size=32768
        )
        result.update(status="running", device_executed=True)
        adapter = NormTerminalScan(t, mesh)

        def upload(x, dtype=t.bfloat16):
            return t.from_torch(
                x.contiguous(), dtype=dtype, layout=t.TILE_LAYOUT, device=mesh, memory_config=t.DRAM_MEMORY_CONFIG
            )

        obj = types.SimpleNamespace(
            mesh_device=mesh,
            config=types.SimpleNamespace(rms_norm_eps=1e-6),
            w={key: upload(value.reshape(1, 1, -1), t.float32) for key, value in weights.items()},
            terminal_norm_scan=adapter,
        )
        obj.compute = t.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=t.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        norm_cores = t.num_cores_to_corerangeset(32, mesh.compute_with_storage_grid_size(), row_wise=True)
        norm_memory = t.create_sharded_memory_config(
            (32, 64),
            norm_cores,
            t.ShardStrategy.WIDTH,
            t.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )
        obj.round_widen_chain = [
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.float32.value), float(t.bfloat16.value)),
            t.UnaryWithParam(t.UnaryOpType.TYPECAST, float(t.bfloat16.value), float(t.float32.value)),
        ]
        calls = {}
        for name, source in sources.items():
            ns = dict(ttnn=t, DRAM=t.DRAM_MEMORY_CONFIG)
            exec(compile(source, str(args.output) + ":" + name, "exec"), ns)
            calls[name] = types.MethodType(ns["norm"], obj)
        for name, fixture in synthetic.items():
            groups = upload(fixture, t.float32)
            with device_only():
                total, lanes = native_terminal(t, groups)
                actual, actual_lanes = adapter(groups, debug=True)
                expected_bf16 = t.typecast(total, t.bfloat16)
                actual_bf16 = t.typecast(actual, t.bfloat16)
            row = dict(
                name=name,
                lanes_fp32=bitdiff(t.to_torch(lanes), t.to_torch(actual_lanes)),
                total_fp32=bitdiff(t.to_torch(total), t.to_torch(actual)),
                total_bf16=bitdiff(t.to_torch(expected_bf16), t.to_torch(actual_bf16)),
            )
            result["synthetic_cases"].append(row)
            save()
            assert all(row[k]["exact"] for k in ("lanes_fp32", "total_fp32", "total_bf16")), row
            del groups, total, lanes, actual, actual_lanes, expected_bf16, actual_bf16
        for case in cases:
            dx = upload(case["x"])
            if case.get("placement") == "production_width_sharded":
                dx = t.to_memory_config(dx, norm_memory)
            key = case["key"]
            with device_only():
                groups = calls["groups"](dx, key)
                total, lanes = native_terminal(t, groups)
                actual, actual_lanes = adapter(groups, debug=True)
                baseline = calls["baseline"](dx, key)
                candidate = calls["candidate"](dx, key)
            expected = t.to_torch(baseline)
            row = dict(
                name=case["name"],
                input_memory=str(dx.memory_config()),
                lanes_fp32=bitdiff(t.to_torch(lanes), t.to_torch(actual_lanes)),
                total_fp32=bitdiff(t.to_torch(total), t.to_torch(actual)),
                complete_norm_bf16=bitdiff(expected, t.to_torch(candidate)),
            )
            result["cases"].append(row)
            save()
            assert all(row[k]["exact"] for k in ("lanes_fp32", "total_fp32", "complete_norm_bf16")), row
            del groups, total, lanes, actual, actual_lanes, baseline, candidate
            if args.timing and case["name"].startswith("actual_227/"):
                result["timing"][case["name"]] = {}
                for label in ("baseline", "candidate"):

                    def call():
                        with device_only():
                            return calls[label](dx, key)

                    result["timing"][case["name"]][label] = timing(t, mesh, call, expected, args)
                    save()
            del dx
        result["status"] = "all_exact_component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        assert all(sha(path) == expected for path, expected in deps.items()), "Source changed during run"
        assert sha(__file__) == result["source_sha256"]
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
    args = parser.parse_args()
    assert args.samples > 0 and args.replays > 0
    run(args)


if __name__ == "__main__":
    main()
