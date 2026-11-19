# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact-order chunk64 triangular recurrence component; hardware only with opt-in.

One mutable local matrix per head. The existing native implementation is the
primary device oracle. CPU evidence establishes the order and signed FTZ rule,
not device correctness. This file does not patch any decoder or frozen probe.
"""
import argparse
import ast
import copy
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_streaming_scan_norm_tail_fused_matmul.py.snapshot"
BASE_SHA = "8cda999c18f9245c872e40d7f67e027838b56396056ede43e162b7a8386f3faa"

READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c) {
 return (r/16)*512+(c/16)*256+(r%16)*16+c%16;
}
constexpr uint32_t index(uint32_t r,uint32_t c) {
 return ((r/32)*2+c/32)*1024+face(r%32,c%32);
}
void kernel_main() {
 constexpr uint32_t output_offset=get_compile_time_arg_val(0);
 const auto input=TensorAccessor(TensorAccessorArgs<1>(),get_arg_val<uint32_t>(0));
 const auto output=TensorAccessor(TensorAccessorArgs<output_offset>(),get_arg_val<uint32_t>(1));
 const uint32_t head=get_arg_val<uint32_t>(2),stop=get_arg_val<uint32_t>(3);
 DataflowBuffer coeff(0),previous(1),oldrow(2),matrix(3),feedback(16); Noc noc;
 matrix.reserve_back(4);
 auto* a=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(3));
 for(uint32_t tile=0;tile<4;++tile)
  noc.async_read(input,matrix,4096,{.page_id=head*4+tile},{.offset_bytes=tile*4096});
 noc.async_read_barrier();
 // Each producer ring has two contiguous pages; all unused tile rows remain zero.
 coeff.reserve_back(2);previous.reserve_back(2);oldrow.reserve_back(1);
 auto* cp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
 auto* pp=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
 auto* op=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
 for(uint32_t n=0;n<2048;++n) {cp[n]=0;pp[n]=0;}
 for(uint32_t n=0;n<1024;++n) op[n]=0;
 for(uint32_t i=1;i<=stop;++i) {
  uint32_t saved[64];
  for(uint32_t k=0;k<64;++k) saved[k]=a[index(i,k)];
  const uint32_t passes=(i>=4&&i<8)?2:(i+31)/32;
  for(uint32_t pass=0;pass<passes;++pass) {
   const bool mixed=i>=4&&i<8;
   const uint32_t column=mixed?0:pass*32;
   const bool cascade=mixed?(pass==0):(i>=32&&pass==0);
   const uint32_t begin=mixed?(pass==0?0:4):column;
   const uint32_t end=mixed?(pass==0?4:i):((column+32<i)?column+32:i);
   oldrow.reserve_back(1);
   auto* old=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(2));
   for(uint32_t c=0;c<32;++c) old[face(0,c)]=saved[column+c];
   oldrow.push_back(1);
   for(uint32_t k=0;k<i;++k) {
    coeff.reserve_back(1);previous.reserve_back(1);
    auto* cc=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0));
    auto* pv=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(1));
    for(uint32_t c=0;c<32;++c) {
     cc[face(0,c)]=saved[k];pv[face(0,c)]=a[index(k,column+c)];
    }
    coeff.push_back(1);previous.push_back(1);
   }
   feedback.wait_front(1);
   auto* f=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(16));
   for(uint32_t j=begin;j<end;++j) a[index(i,j)]=f[face(0,j-column)];
   feedback.pop_front(1);
  }
 }
 for(uint32_t tile=0;tile<4;++tile)
  noc.async_write(matrix,output,4096,{.offset_bytes=tile*4096},{.page_id=head*4+tile});
 noc.async_write_barrier();
}
"""

COMPUTE = r"""
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/experimental/reg_api.h"
#include "api/compute/pack.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#ifdef TRISC_MATH
#include "llk_math_eltwise_ternary_sfpu.h"
// Native FP32 product/add output boundary: preserve the sign when flushing.
inline sfpi::vFloat flush_result(sfpi::vFloat value) {
 sfpi::vUInt bits=sfpi::as<sfpi::vUInt>(value);
 v_if((bits & 0x7f800000u)==0u) { bits=bits & 0x80000000u; } v_endif;
 return sfpi::as<sfpi::vFloat>(bits);
}
inline sfpi::vFloat add_round(sfpi::vFloat a,sfpi::vFloat b) {
 sfpi::vFloat sum=a+b;return flush_result(sum);
}
inline void begin_sfpu() {
 sfpu::_init_sfpu_config_reg();
 addr_mod_t{.srca={.incr=0},.srcb={.incr=0},.dest={.incr=0}}.set(ADDR_MOD_3);
 math::reset_counters(p_setrwc::SET_ABD_F);_llk_math_eltwise_sfpu_start_(0);
}
template<uint32_t SLOT> inline void product_add() {
 begin_sfpu();
 for(uint32_t lane=0;lane<32;++lane) {
  sfpi::vFloat a=sfpi::dst_reg[128],b=sfpi::dst_reg[160];
  sfpi::dst_reg[128]=flush_result(a*b);
  sfpi::vFloat product=sfpi::dst_reg[128],acc=sfpi::dst_reg[SLOT];
  sfpi::dst_reg[SLOT]=add_round(acc,product);
  sfpi::dst_reg++;
 }
 _llk_math_eltwise_sfpu_done_();
}
#endif
void kernel_main() {
 CircularBuffer coeff(0),previous(1),oldrow(2),feedback(16);
 compute_kernel_hw_startup(0,16);copy_init(0);
 const uint32_t stop=get_arg_val<uint32_t>(0);
 for(uint32_t i=1;i<=stop;++i) {
  const bool mixed=i>=4&&i<8;
  const uint32_t passes=mixed?2:(i+31)/32;
  for(uint32_t pass=0;pass<passes;++pass) {
   const bool cascade=mixed?(pass==0):(i>=32&&pass==0);
   tile_regs_acquire_math_clear();oldrow.wait_front(1);copy_tile(2,0,6);oldrow.pop_front(1);
#ifdef TRISC_MATH
   begin_sfpu();
   for(uint32_t lane=0;lane<32;++lane) {
    sfpi::dst_reg[0]=0.0f;sfpi::dst_reg[32]=0.0f;
    sfpi::dst_reg[64]=0.0f;sfpi::dst_reg[96]=0.0f;sfpi::dst_reg++;
   }
   _llk_math_eltwise_sfpu_done_();
#endif
   for(uint32_t k=0;k<i;++k) {
    coeff.wait_front(1);previous.wait_front(1);copy_tile(0,0,4);copy_tile(1,0,5);
#ifdef TRISC_MATH
    const uint32_t slot=cascade?0:((k<(i/4)*4)?k%4:0);
    if(slot==0) product_add<0>();
    else if(slot==1) product_add<32>();
    else if(slot==2) product_add<64>();
    else product_add<96>();
    if(cascade&&(k+1)%16==0) {
     begin_sfpu();
     for(uint32_t lane=0;lane<32;++lane) {
      sfpi::vFloat done=sfpi::dst_reg[32],part=sfpi::dst_reg[0];
      sfpi::dst_reg[32]=add_round(done,part);sfpi::dst_reg[0]=0.0f;sfpi::dst_reg++;
     }
     _llk_math_eltwise_sfpu_done_();
    }
#endif
    coeff.pop_front(1);previous.pop_front(1);
   }
#ifdef TRISC_MATH
   begin_sfpu();
   for(uint32_t lane=0;lane<32;++lane) {
    sfpi::vFloat a=sfpi::dst_reg[0],b=sfpi::dst_reg[32];
    sfpi::dst_reg[0]=add_round(a,b);
    if(!cascade) {
     sfpi::vFloat p=sfpi::dst_reg[0],c=sfpi::dst_reg[64];
     sfpi::dst_reg[0]=add_round(p,c);
     sfpi::vFloat q=sfpi::dst_reg[0],d=sfpi::dst_reg[96];
     sfpi::dst_reg[0]=add_round(q,d);
    }
    sfpi::vFloat z=0.0f,correction=sfpi::dst_reg[0];
    sfpi::dst_reg[0]=add_round(z,correction);
    sfpi::vFloat corrected=sfpi::dst_reg[0],old=sfpi::dst_reg[192];
    sfpi::dst_reg[224]=add_round(old,corrected);sfpi::dst_reg++;
   }
   _llk_math_eltwise_sfpu_done_();
#endif
   tile_regs_commit();tile_regs_wait();feedback.reserve_back(1);
   pack_tile(7,16);feedback.push_back(1);tile_regs_release_math_clear();
  }
 }
}
"""
EMPTY_WRITER = "void kernel_main() {}"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TriangularRecurrence:
    def __init__(self, t, mesh):
        self.t, self.mesh = t, mesh

    def __call__(self, matrix, stop=63):
        t = self.t
        shape = tuple(matrix.shape)
        assert shape[0] == 1 and shape[-2:] == (64, 64)
        assert matrix.dtype == t.float32 and matrix.layout == t.TILE_LAYOUT
        assert matrix.memory_config() == t.DRAM_MEMORY_CONFIG
        assert 1 <= stop <= 63
        heads = shape[1]
        grid = self.mesh.compute_with_storage_grid_size()
        assert heads <= grid.x * grid.y
        positions = [(h % grid.x, h // grid.x, h) for h in range(heads)]
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _ in positions])
        output = t.empty(
            shape, device=self.mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        rd, cr, wr = t.RuntimeArgs(), t.RuntimeArgs(), t.RuntimeArgs()
        for x, y, h in positions:
            rd[x][y] = [matrix.buffer_address(), output.buffer_address(), h, stop]
            cr[x][y] = [stop]
            wr[x][y] = []
        ac = list(t.TensorAccessorArgs(matrix).get_compile_time_args())
        bc = list(t.TensorAccessorArgs(output).get_compile_time_args())
        cfg = t.ComputeConfigDescriptor(
            math_fidelity=t.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, dst_full_sync_en=True
        )
        modes = [t.UnpackToDestMode.Default] * 64
        for i in (0, 1, 2):
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
                kernel(COMPUTE, cfg, [], cr),
                kernel(EMPTY_WRITER, t.WriterConfigDescriptor(), [], wr),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=pages * 4096,
                    core_ranges=cores,
                    format_descriptors=[t.CBFormatDescriptor(i, t.float32, 4096)],
                )
                for i, pages in ((0, 2), (1, 2), (2, 1), (3, 4), (16, 1))
            ],
        )
        t.generic_op([matrix, output], program)
        return output


def native_class():
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Chunk64StateGraph")
    reduction = copy.deepcopy(
        next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "triangular_sum")
    )
    call = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    loops = [
        n
        for n in ast.walk(call)
        if isinstance(n, ast.For) and ast.unparse(n.target) == "i" and ast.unparse(n.iter) == "range(1, 64)"
    ]
    assert len(loops) == 1
    original = ast.dump(loops[0])
    loop = copy.deepcopy(loops[0])
    loop.iter = ast.parse("range(1,stop+1)", mode="eval").body
    native = ast.parse("def run(self,a,stop):\n t=self.t\n return a\n").body[0]
    native.body.insert(1, loop)
    module = ast.fix_missing_locations(ast.Module([reduction, native], type_ignores=[]))
    scope = {"observer": None}
    exec(compile(module, "<unchanged-native-triangular-loop>", "exec"), scope)

    class Native:
        triangular_sum = scope["triangular_sum"]
        __call__ = scope["run"]

        def __init__(self, t, mesh):
            self.t, self.mesh, self.mem, self.sum_order = t, mesh, t.DRAM_MEMORY_CONFIG, "canonical"

        def add(self, a, b):
            return self.t.add(a, b, dtype=self.t.float32, memory_config=self.mem)

        def mul(self, a, b):
            return self.t.multiply(a, b, dtype=self.t.float32, memory_config=self.mem, fast_and_approximate_mode=False)

        def observe(self, *args):
            pass

    return Native, hashlib.sha256(original.encode()).hexdigest()


def source_checks():
    _, native_sha = native_class()
    for i in range(1, 64):
        passes = 2 if 4 <= i < 8 else (i + 31) // 32
        covered = []
        for p in range(passes):
            begin, end = (0, 4) if 4 <= i < 8 and p == 0 else ((4, i) if 4 <= i < 8 else (p * 32, min(i, p * 32 + 32)))
            covered.extend(range(begin, end))
        assert covered == list(range(i))
    offsets = [
        ((r // 32) * 2 + c // 32) * 1024 + (r % 32 // 16) * 512 + (c % 32 // 16) * 256 + (r % 16) * 16 + c % 16
        for r in range(64)
        for c in range(64)
    ]
    assert sorted(offsets) == list(range(4096))
    return dict(
        native_loop_ast_sha256=native_sha,
        coverage="all63rows/all64x64physicaladdresses",
        cb_bytes_per_core=40960,
        persistent_l1_bytes=0,
    )


def run_device(args, result, save):
    import torch

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    torch.set_num_threads(4)
    fixtures = []
    for tag in args.cases.split(","):
        if tag.startswith("actual"):
            chunk = int(tag.removeprefix("actual"))
            metadata = E / "v6_chunk64_fused_corrected_output_zero_device.json"
            meta = json.loads(metadata.read_text())
            path = Path(meta["device_matrix_snapshot"])
            assert sha(path) == meta["device_matrix_snapshot_sha256"]
            saved = torch.load(path, map_location="cpu", weights_only=False)[128]["intermediates"]
            initial = saved[f"chunk{chunk}/A_initial"][:, : args.heads].contiguous()
        elif tag == "zeros":
            initial = torch.randn(1, args.heads, 64, 64, generator=torch.Generator().manual_seed(11)) * 0.0
        else:
            assert tag == "random"
            initial = torch.randn(1, args.heads, 64, 64, generator=torch.Generator().manual_seed(20261113)) * 0.02
            initial = -initial.masked_fill(torch.ones(64, 64, dtype=torch.bool).triu(), 0.0)
        fixtures.append((tag, initial))
    mesh = None
    try:
        mesh = t.open_mesh_device(t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768)
        candidate = TriangularRecurrence(t, mesh)
        Native, _ = native_class()
        native = Native(t, mesh)
        upload = lambda x: t.from_torch(
            x, device=mesh, dtype=t.float32, layout=t.TILE_LAYOUT, memory_config=t.DRAM_MEMORY_CONFIG
        )
        observed = {}
        for name, initial in fixtures:
            value = upload(initial)
            for stop in [int(v) for v in args.stops.split(",")]:
                begin = time.perf_counter()
                with device_only():
                    expected = native(value, stop)
                t.synchronize_device(mesh)
                native_cold_ms = (time.perf_counter() - begin) * 1000
                begin = time.perf_counter()
                with device_only():
                    actual = candidate(value, stop)
                t.synchronize_device(mesh)
                candidate_cold_ms = (time.perf_counter() - begin) * 1000
                a, b = t.to_torch(expected), t.to_torch(actual)
                count = int((a.view(torch.int32) != b.view(torch.int32)).sum())
                row = dict(
                    case=name,
                    stop=stop,
                    bits_different=count,
                    max_abs=float((a - b).abs().max()),
                    native_cold_ms=native_cold_ms,
                    candidate_cold_ms=candidate_cold_ms,
                )
                result["cases"].append(row)
                observed[f"{name}_{stop}"] = {"expected": a, "actual": b}
                save()
                print(json.dumps(row), flush=True)
        path = args.output.with_suffix(".device.pt")
        torch.save(observed, path)
        result["device_tensors"] = str(path)
        result["device_tensors_sha256"] = sha(path)
        result["status"] = (
            "component_exact" if all(r["bits_different"] == 0 for r in result["cases"]) else "component_differences"
        )
        assert result["status"] == "component_exact", result["cases"]
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--run-device", action="store_true")
    p.add_argument("--heads", type=int, default=1)
    p.add_argument("--cases", default="actual0")
    p.add_argument("--stops", default="1,2,31,63")
    args = p.parse_args()
    assert not args.output.exists()
    assert 1 <= args.heads <= 32
    checked = source_checks()
    assert "ttnn" not in sys.modules
    result = dict(
        status="source_setup_pass",
        source_sha256=sha(__file__),
        base_sha256=BASE_SHA,
        checks=checked,
        cases=[],
        device_executed=args.run_device,
        device_closed=False,
        acceptance_eligible=False,
        classification="Isolated exact-order triangular recurrence component; no decoder integration or performance claim",
    )
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    for name, code in [("reader", READER), ("compute", COMPUTE)]:
        path = args.output.with_suffix(f".{name}.cpp.snapshot")
        path.write_text(code)
        result[name + "_sha256"] = sha(path)

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if args.run_device:
        run_device(args, result, save)
    else:
        print("TRIANGULAR_SOURCE_PASS", sha(__file__))


if __name__ == "__main__":
    main()
