# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Single-dispatch corrected FP32 GEMM. Standard-library source checks by default.

Device tests require --run-device and are owned by the parent hardware agent.
The first implementation retains each exact local FP32 pack/unpack boundary.
"""

import argparse
import ast
import copy
import hashlib
import itertools
import json
import math
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_35b_a3b.tests.frozen_source_compatibility import raw_sha256, verify_source

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / "doc/fused_decoder/endurance"
GRAPH_SOURCE = ROOT / "tests/v6_fma_sfpu_corrected_probe.py"
GRAPH_SHA = "185217ed00091eddcdf15db9494071901edaa472d3935b986b4b512834f6832f"
ADAPTER_SOURCE = ROOT / "tests/v6_corrected_fma_matmul_probe.py"
ADAPTER_SHA = "facef89f3f760fcd86c54e31e63ac938657f2ce02c639715699eb3c94ab90442"

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face_offset(uint32_t r,uint32_t c) {
 return (r/16)*512+(c/16)*256+(r%16)*16+c%16;
}
void kernel_main() {
 constexpr uint32_t b_accessor_offset=get_compile_time_arg_val(0);
 const auto at=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto bt=TensorAccessor(TensorAccessorArgs<b_accessor_offset>(),get_arg_val<uint32_t>(1));
 const uint32_t start=get_arg_val<uint32_t>(2), count=get_arg_val<uint32_t>(3);
 const uint32_t M=get_arg_val<uint32_t>(4), N=get_arg_val<uint32_t>(5), K=get_arg_val<uint32_t>(6);
 const uint32_t rank=get_arg_val<uint32_t>(7), Mt=(M+31)/32, Nt=(N+31)/32, Kt=(K+31)/32;
 DataflowBuffer a(0), b(1), zero(2), ascratch(3), bscratch(4);
 Noc noc;
 ascratch.reserve_back(1); bscratch.reserve_back(1);
 auto* sa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));
 auto* sb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(4));
 for(uint32_t tile=start;tile<start+count;++tile) {
  const uint32_t matrix_tile=tile%(Mt*Nt), mr=matrix_tile/Nt, nc=matrix_tile%Nt;
  uint32_t batch=tile/(Mt*Nt), abatch=0, bbatch=0;
  for(uint32_t i=rank;i>0;--i) {
   const uint32_t j=i-1, d=get_arg_val<uint32_t>(8+j), coordinate=batch%d;
   batch/=d;
   abatch+=coordinate*get_arg_val<uint32_t>(8+rank+j);
   bbatch+=coordinate*get_arg_val<uint32_t>(8+2*rank+j);
  }
  zero.reserve_back(1);
  auto* z=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
  for(uint32_t i=0;i<1024;++i) z[i]=0;
  zero.push_back(1);
  for(uint32_t k=0;k<K;++k) {
   if(k%32==0) {
    noc.async_read(at,ascratch,4096,{.page_id=(abatch*Mt+mr)*Kt+k/32},{.offset_bytes=0});
    noc.async_read(bt,bscratch,4096,{.page_id=(bbatch*Kt+k/32)*Nt+nc},{.offset_bytes=0});
    noc.async_read_barrier();
   }
   a.reserve_back(1); b.reserve_back(1);
   auto* aa=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
   auto* bb=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
   for(uint32_t r=0;r<32;++r) {
    const uint32_t av=mr*32+r<M ? sa[face_offset(r,k%32)] : 0;
    for(uint32_t c=0;c<32;++c) {
     const uint32_t offset=face_offset(r,c);
     aa[offset]=av;
     bb[offset]=nc*32+c<N ? sb[face_offset(k%32,c)] : 0;
    }
   }
   a.push_back(1); b.push_back(1);
  }
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
 const uint32_t start=get_arg_val<uint32_t>(1), count=get_arg_val<uint32_t>(2);
 DataflowBuffer result(16); Noc noc;
 for(uint32_t tile=start;tile<start+count;++tile) {
  result.wait_front(1);
  noc.async_write(result,output,4096,{}, {.page_id=tile});
  noc.async_write_barrier(); result.pop_front(1);
 }
}
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arithmetic():
    """Extract frozen pure source generator without importing its Torch module."""
    assert verify_source(GRAPH_SOURCE, GRAPH_SHA) and verify_source(ADAPTER_SOURCE, ADAPTER_SHA)
    tree = ast.parse(GRAPH_SOURCE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Graph")
    scope = {}
    exec(compile(ast.Module([copy.deepcopy(cls)], []), "<frozen-graph-source>", "exec"), scope)
    graph = scope["Graph"]().build()
    assert graph.dst(len(graph.nodes) - 1) + 31 < 224
    return graph.cpp(), graph.dst(graph.result), hashlib.sha256(ast.dump(cls).encode()).hexdigest()


def compute_source():
    graph, result, _ = arithmetic()
    return r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
#endif
void kernel_main() {
 CircularBuffer a(0),b(1),zero(2),feedback(24),output(16);
 compute_kernel_hw_startup(0,16); copy_init(0);
 constexpr uint32_t K=get_compile_time_arg_val(0);
 const uint32_t count=get_arg_val<uint32_t>(0);
 for(uint32_t tile=0;tile<count;++tile) {
  for(uint32_t k=0;k<K;++k) {
   tile_regs_acquire_math_clear();
   a.wait_front(1); b.wait_front(1);
   if(k==0) { zero.wait_front(1); } else { feedback.wait_front(1); }
   copy_tile(0,0,0); copy_tile(1,0,1);
   if(k==0) { copy_tile(2,0,2); } else { copy_tile(24,0,2); }
#ifdef TRISC_MATH
   sfpu::_init_sfpu_config_reg();
   addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
   math::reset_counters(p_setrwc::SET_ABD_F);
   _llk_math_eltwise_sfpu_start_(0);
   for(uint32_t lane=0;lane<32;++lane) {
    GRAPH
    { sfpi::vFloat result=sfpi::dst_reg[RESULT_SLOT]; sfpi::dst_reg[224]=result; }
    sfpi::dst_reg++;
   }
   _llk_math_eltwise_sfpu_done_();
#endif
   // Release the consumed feedback page before PACK reserves its successor.
   a.pop_front(1); b.pop_front(1);
   if(k==0) { zero.pop_front(1); } else { feedback.pop_front(1); }
   tile_regs_commit(); tile_regs_wait();
   if(k+1==K) { output.reserve_back(1); pack_tile(7,16); output.push_back(1); }
   else { feedback.reserve_back(1); pack_tile(7,24); feedback.push_back(1); }
   tile_regs_release_math_clear();
  }
 }
}
""".replace(
        "GRAPH", graph
    ).replace(
        "RESULT_SLOT", str(result)
    )


def shape_plan(ashape, bshape):
    ashape, bshape = tuple(ashape), tuple(bshape)
    assert len(ashape) >= 2 and len(bshape) >= 2
    assert all(x > 0 for x in ashape + bshape)
    assert ashape[-1] == bshape[-2]
    rank = max(len(ashape), len(bshape)) - 2
    al = (1,) * (rank - len(ashape) + 2) + ashape[:-2]
    bl = (1,) * (rank - len(bshape) + 2) + bshape[:-2]
    leading = []
    for a, b in zip(al, bl):
        assert a == b or a == 1 or b == 1, (ashape, bshape)
        leading.append(max(a, b))

    def strides(shape):
        return [math.prod(shape[i + 1 :]) if d != 1 else 0 for i, d in enumerate(shape)]

    m, k, n = ashape[-2], ashape[-1], bshape[-1]
    return dict(
        shape=tuple(leading) + (m, n),
        leading=tuple(leading),
        a_strides=strides(al),
        b_strides=strides(bl),
        M=m,
        N=n,
        K=k,
        Mt=(m + 31) // 32,
        Nt=(n + 31) // 32,
        Kt=(k + 31) // 32,
        total_tiles=math.prod(leading) * ((m + 31) // 32) * ((n + 31) // 32),
    )


def schedule(total, gx, gy, max_cores):
    active = min(total, gx * gy, max_cores)
    assert active > 0
    base, extra = divmod(total, active)
    result = []
    start = 0
    for i in range(active):
        count = base + (i < extra)
        result.append((i % gx, i // gx, start, count))
        start += count
    assert start == total
    return result


class FusedCorrectedMatmul:
    """Device-only streaming matrix recurrence; arbitrary broadcast leading axes."""

    def __init__(self, t, mesh, max_cores=64):
        self.t, self.mesh, self.max_cores = t, mesh, max_cores
        self.source = compute_source()
        self.grid = mesh.compute_with_storage_grid_size()
        self.last_plan = None

    def __call__(self, a, b):
        t = self.t
        assert a.dtype == b.dtype == t.float32
        assert a.layout == b.layout == t.TILE_LAYOUT
        assert a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        assert b.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        plan = shape_plan(tuple(a.shape), tuple(b.shape))
        partitions = schedule(plan["total_tiles"], self.grid.x, self.grid.y, self.max_cores)
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        out = t.empty(
            plan["shape"], device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        rd, wr, cr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        for x, y, start, count in partitions:
            rd[x][y] = [
                a.buffer_address(),
                b.buffer_address(),
                start,
                count,
                plan["M"],
                plan["N"],
                plan["K"],
                len(plan["leading"]),
                *plan["leading"],
                *plan["a_strides"],
                *plan["b_strides"],
            ]
            wr[x][y] = [out.buffer_address(), start, count]
            cr[x][y] = [count]
        ac = list(t.TensorAccessorArgs(a).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(b).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2, 24):
            modes[i] = t.UnpackToDestMode.UnpackToDestFp32
        cfg.unpack_to_dest_mode = modes

        def kernel(source, config, args, runtime):
            return t.KernelDescriptor(
                kernel_source=source,
                source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=cores,
                config=config,
                compile_time_args=args,
                runtime_args=runtime,
            )

        program = t.ProgramDescriptor(
            kernels=[
                kernel(READER, t.ReaderConfigDescriptor(), [1 + len(ac)] + ac + bc, rd),
                kernel(self.source, cfg, [plan["K"]], cr),
                kernel(WRITER, t.WriterConfigDescriptor(), t.TensorAccessorArgs(out).get_compile_time_args(), wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, t.float32, 4096)],
                )
                for i, pages in ((0, 2), (1, 2), (2, 1), (3, 1), (4, 1), (16, 1), (24, 2))
            ],
        )
        t.generic_op([a, b, out], program)
        self.last_plan = {
            **plan,
            "core_count": len(partitions),
            "tiles_per_core": [p[3] for p in partitions],
            "cb_bytes_per_core": 40960,
        }
        return out


def source_checks():
    graph, _, ast_sha = arithmetic()
    code = compute_source()
    assert code.count(graph) == 1
    assert "for(uint32_t k=0;k<K;++k)" in code
    tree = ast.parse(Path(__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedCorrectedMatmul")
    runtime = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    calls = [ast.unparse(n.func) for n in ast.walk(runtime) if isinstance(n, ast.Call)]
    assert calls.count("t.generic_op") == 1
    assert not any(c.startswith("torch.") or c.endswith((".from_torch", ".to_torch")) for c in calls)
    fixtures = [
        ((1, 1, 64, 128), (1, 1, 128, 64)),
        ((1, 32, 64, 64), (1, 32, 64, 128)),
        ((1, 32, 128, 64), (1, 32, 64, 128)),
        ((2, 1, 33, 65), (1, 3, 65, 17)),
        ((65, 1), (2, 3, 1, 1)),
        ((1, 1), (1, 1)),
        ((2, 1, 4, 1, 32, 64), (1, 3, 1, 5, 64, 32)),
    ]
    results = []
    lane_offsets = {(r // 16) * 512 + (c // 16) * 256 + (r % 16) * 16 + c % 16 for r in range(32) for c in range(32)}
    assert lane_offsets == set(range(1024))
    for ashape, bshape in fixtures:
        p = shape_plan(ashape, bshape)
        counts = []
        for maxcores in (1, 4, 64, 110):
            partition = schedule(p["total_tiles"], 11, 10, maxcores)
            covered = [i for _, _, start, count in partition for i in range(start, start + count)]
            assert covered == list(range(p["total_tiles"]))
            counts.append(len(partition))
        al = (1,) * (len(p["leading"]) - len(ashape) + 2) + ashape[:-2]
        bl = (1,) * (len(p["leading"]) - len(bshape) + 2) + bshape[:-2]
        checked = 0
        for batch, coords in enumerate(itertools.product(*(range(d) for d in p["leading"]))):
            quotient = batch
            gota = gotb = 0
            for j in reversed(range(len(coords))):
                coordinate = quotient % p["leading"][j]
                quotient //= p["leading"][j]
                gota += coordinate * p["a_strides"][j]
                gotb += coordinate * p["b_strides"][j]
            wanta = wantb = 0
            for c, a, b in zip(coords, al, bl):
                wanta = wanta * a + (0 if a == 1 else c)
                wantb = wantb * b + (0 if b == 1 else c)
            assert (gota, gotb) == (wanta, wantb)
            # Check every K source tile and selected lane for boundary/interior output tiles.
            for mr, nc in {(0, 0), (p["Mt"] - 1, p["Nt"] - 1)}:
                for k in range(p["K"]):
                    for r, c in ((0, 0), (15, 16), (16, 15), (31, 31)):
                        apos = (
                            ((gota * p["Mt"] + mr) * p["Kt"] + k // 32) * 1024
                            + (r // 16) * 512
                            + ((k % 32) // 16) * 256
                            + (r % 16) * 16
                            + k % 16
                        )
                        bpos = (
                            ((gotb * p["Kt"] + k // 32) * p["Nt"] + nc) * 1024
                            + ((k % 32) // 16) * 512
                            + (c // 16) * 256
                            + (k % 16) * 16
                            + c % 16
                        )
                        assert 0 <= apos < math.prod(ashape[:-2]) * p["Mt"] * p["Kt"] * 1024
                        assert 0 <= bpos < math.prod(bshape[:-2]) * p["Kt"] * p["Nt"] * 1024
                        checked += 1
        results.append(dict(a=ashape, b=bshape, plan=p, mapping_checks=checked, schedule_core_counts=counts))
    return dict(
        arithmetic_cpp_sha256=hashlib.sha256(graph.encode()).hexdigest(),
        frozen_graph_ast_sha256=ast_sha,
        runtime_single_generic_op=True,
        runtime_host_tensor_conversions=False,
        shape_checks=results,
        per_k_boundary="exact frozen graph plus FP32 pack/unpack via local CB24; positive-zero initial CB2",
        source_only=True,
    )


def run_device(args, result, save):
    import torch

    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_corrected_fma_matmul_probe import (
        CorrectedFma,
        CorrectedMatmul,
        compare,
        matrix_fixtures,
    )

    fixtures = matrix_fixtures(args.heads, set(args.matrices.split(",")), None)
    if args.broadcast_case:
        gen = torch.Generator().manual_seed(613647)
        a = torch.randn((2, 1, 32, 64), generator=gen) * 0.03
        b = torch.randn((1, 3, 64, 32), generator=gen) * 0.01
        fixtures.append(dict(name="leading_broadcast_2x3", a=a, b=b, expected=a @ b))
    torch.save(fixtures, args.output.with_suffix(".fixtures.pt"))
    result["fixtures_sha256"] = sha(args.output.with_suffix(".fixtures.pt"))
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        fused = FusedCorrectedMatmul(ttnn, mesh, args.max_cores)
        reference = CorrectedMatmul(CorrectedFma(ttnn, mesh))
        upload = lambda x: ttnn.from_torch(
            x.contiguous(),
            device=mesh,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        measured = {}
        for case in fixtures:
            a, b = case["a"], case["b"]
            da, db = upload(a), upload(b)
            shape = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            ra = upload(a.expand(*shape, *a.shape[-2:]))
            rb = upload(b.expand(*shape, *b.shape[-2:]))
            with device_only():
                base = reference(ra, rb)
            ttnn.synchronize_device(mesh)
            baseline = ttnn.to_torch(base)
            start = time.perf_counter()
            with device_only():
                out = fused(da, db)
            ttnn.synchronize_device(mesh)
            elapsed = (time.perf_counter() - start) * 1000
            actual = ttnn.to_torch(out)
            row = dict(
                name=case["name"],
                vs_frozen_adapter=compare(baseline, actual),
                vs_cpu=compare(case["expected"], actual),
                plan=fused.last_plan,
                diagnostic_cold_ms=elapsed,
            )
            result["cases"].append(row)
            measured[case["name"]] = actual
            save()
            print(json.dumps(row), flush=True)
        torch.save(measured, args.output.with_suffix(".device.pt"))
        result["status"] = (
            "component_adapter_exact"
            if all(c["vs_frozen_adapter"]["fp32_bits_different"] == 0 for c in result["cases"])
            else "component_measured_differences"
        )
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
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--matrices", default="chunk0_gram")
    p.add_argument("--max-cores", type=int, default=64)
    p.add_argument("--broadcast-case", action="store_true")
    args = p.parse_args()
    assert not args.output.exists(), "Use a fresh evidence path"
    checked = source_checks()
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    result = dict(
        status="source_setup_pass",
        classification="Single-dispatch corrected GEMM diagnostic; no integrated accuracy/latency/capacity claim",
        source_sha256=sha(__file__),
        dependencies={str(GRAPH_SOURCE): raw_sha256(GRAPH_SOURCE), str(ADAPTER_SOURCE): raw_sha256(ADAPTER_SOURCE)},
        dependencies_historical_sha256={str(GRAPH_SOURCE): GRAPH_SHA, str(ADAPTER_SOURCE): ADAPTER_SHA},
        checks=checked,
        cases=[],
        device_executed=args.run_device,
        device_closed=False,
        limits="Full-FP32 bounded correction inherited from measured helper, not arbitrary subnormal IEEE proof. Device code is uncompiled until opt-in run. Reader scalar bit expansion may bottleneck. No public context claim.",
    )
    for name, source in [("reader", READER), ("compute", compute_source()), ("writer", WRITER)]:
        path = args.output.with_suffix(f".{name}.cpp.snapshot")
        path.write_text(source)
        result[name + "_sha256"] = sha(path)
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if args.run_device:
        run_device(args, result, save)
    else:
        print("FUSED_CORRECTED_MATMUL_SOURCE_PASS", sha(__file__))


if __name__ == "__main__":
    main()
