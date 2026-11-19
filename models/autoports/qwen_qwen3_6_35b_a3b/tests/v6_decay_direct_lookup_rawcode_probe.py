# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Direct per-head decay lookup, exact native adapter gate and opt-in timing.

Uses the existing FP32 DRAM table. No persistent table copy or transcendental
approximation is introduced. Exhaustive cases cover every BF16 code per head.
"""
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
from pathlib import Path
from types import SimpleNamespace

E = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
BASE = E / "v6_corrected_fma_full_v5_triangular_compact_silu_formatted_integration.py.snapshot"
BASE_SHA = "ab28a91bb3cb5e8b88e5371da0defe3b0d731e86a131c8523483e4b6b65b0334"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c){return (r/16)*512+(c/16)*256+(r%16)*16+c%16;}
void kernel_main(){
 constexpr uint32_t ta=get_compile_time_arg_val(0),oa=get_compile_time_arg_val(1);
 constexpr bool compact=get_compile_time_arg_val(2)!=0;
 constexpr uint32_t lo=get_compile_time_arg_val(3),hi=get_compile_time_arg_val(4);
 const auto input=TensorAccessor(TensorAccessorArgs<5>(),get_arg_val<uint32_t>(0));
 const auto table=TensorAccessor(TensorAccessorArgs<ta>(),get_arg_val<uint32_t>(1));
 const auto output=TensorAccessor(TensorAccessorArgs<oa>(),get_arg_val<uint32_t>(2));
 const uint32_t start=get_arg_val<uint32_t>(3),count=get_arg_val<uint32_t>(4),rows=get_arg_val<uint32_t>(5);
 DataflowBuffer scratch(0);Noc noc;scratch.reserve_back(3);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* codes=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(0)+bias);
 auto* blocks=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+2048);
 auto* result=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);
 for(uint32_t tile=start;tile<start+count;++tile){
  noc.async_read(input,scratch,2048,{.page_id=tile},{.offset_bytes=bias});noc.async_read_barrier();
  for(uint32_t i=0;i<1024;++i)result[i]=0;
  const uint32_t valid=rows-tile*32<32?rows-tile*32:32;
  for(uint32_t r=0;r<valid;++r){
   uint32_t mapped[32];
   for(uint32_t h=0;h<32;++h){
    const uint32_t code=codes[face(r,h)],mag=code&32767u;
    if constexpr(compact){
     const uint32_t clamped=mag<lo?lo:(mag>hi?hi:mag);
     mapped[h]=clamped-lo+(mag>32640?mag-32640:0)+((code&32768u)>>2);
    }else mapped[h]=code;
    noc.async_read(table,scratch,64,{.page_id=h,.offset_bytes=(mapped[h]>>4)*64},
      {.offset_bytes=bias+2048+h*64});
   }
   noc.async_read_barrier();
   for(uint32_t h=0;h<32;++h)result[face(r,h)]=blocks[h*16+(mapped[h]&15)];
  }
  noc.async_write(scratch,output,4096,{.offset_bytes=bias+4096},{.page_id=tile});noc.async_write_barrier();
 }
}
"""
EMPTY = "void kernel_main() {}\n"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(node):
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()


class DirectDecayLookup:
    def __init__(self, t, mesh, table, lo=None, hi=None, max_cores=32):
        self.t, self.mesh, self.table = t, mesh, table
        self.lo, self.hi, self.max_cores = lo, hi, max_cores
        assert tuple(table.shape) in ((32, 16384), (32, 65536))
        assert table.dtype == t.float32 and table.layout == t.ROW_MAJOR_LAYOUT
        assert table.memory_config() == t.DRAM_MEMORY_CONFIG
        assert (lo is not None and hi is not None) == (table.shape[1] == 16384)

    def __call__(self, projection_a):
        t, table = self.t, self.table
        assert projection_a.dtype == t.bfloat16 and projection_a.layout == t.TILE_LAYOUT
        assert len(projection_a.shape) == 3 and projection_a.shape[0] == 1 and projection_a.shape[2] == 32
        assert projection_a.memory_config() in (t.DRAM_MEMORY_CONFIG, t.L1_MEMORY_CONFIG)
        rows = projection_a.shape[1]
        tiles = (rows + 31) // 32
        grid = self.mesh.compute_with_storage_grid_size()
        active = min(tiles, grid.x * grid.y, self.max_cores)
        partitions, start = [], 0
        for i in range(active):
            count = tiles // active + (i < tiles % active)
            partitions.append((i % grid.x, i // grid.x, start, count))
            start += count
        cores = t.CoreRangeSet([t.CoreRange(t.CoreCoord(x, y), t.CoreCoord(x, y)) for x, y, _, _ in partitions])
        output = t.empty(
            tuple(projection_a.shape),
            device=self.mesh,
            dtype=t.float32,
            layout=t.TILE_LAYOUT,
            memory_config=t.DRAM_MEMORY_CONFIG,
        )
        ia, ta, oa = [list(t.TensorAccessorArgs(x).get_compile_time_args()) for x in (projection_a, table, output)]
        args = [5 + len(ia), 5 + len(ia) + len(ta), int(self.lo is not None), self.lo or 0, self.hi or 0] + ia + ta + oa
        runtime = t.RuntimeArgs()
        for x, y, start, count in partitions:
            runtime[x][y] = [
                projection_a.buffer_address(),
                table.buffer_address(),
                output.buffer_address(),
                start,
                count,
                rows,
            ]
        program = t.ProgramDescriptor(
            kernels=[
                t.KernelDescriptor(
                    kernel_source=READER,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ReaderConfigDescriptor(),
                    compile_time_args=args,
                    runtime_args=runtime,
                ),
                t.KernelDescriptor(
                    kernel_source=EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.WriterConfigDescriptor(),
                ),
                t.KernelDescriptor(
                    kernel_source=EMPTY,
                    source_type=t.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=cores,
                    config=t.ComputeConfigDescriptor(),
                ),
            ],
            cbs=[
                t.CBDescriptor(
                    total_size=12288, core_ranges=cores, format_descriptors=[t.CBFormatDescriptor(0, t.float32, 4096)]
                )
            ],
        )
        t.generic_op([projection_a, table, output], program)
        return output


def base_methods():
    assert sha(BASE) == BASE_SHA
    tree = ast.parse(BASE.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    return {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def native_function(t):
    method = base_methods()["_linear_attention"]
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "raw_indices" for x in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(method.body[start:], start)
        if isinstance(n, ast.If)
        and any(
            isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == "decay_value" for y in x.targets)
            for x in n.body
        )
    )
    block = copy.deepcopy(method.body[start : end + 1])
    fn = ast.parse(
        "def lookup(self,projection_a):\n s=projection_a.shape[1]\n decode=True\n return decay_value\n"
    ).body[0]
    fn.body = fn.body[:2] + block + fn.body[-1:]
    module = ast.fix_missing_locations(ast.Module([fn], type_ignores=[]))
    scope = {"ttnn": t}
    exec(compile(module, str(BASE) + ":decay_lookup", "exec"), scope)
    return scope["lookup"], digest(ast.Module(block, type_ignores=[]))


def cpu_tables(state_dict):
    import torch

    method = base_methods()["from_state_dict"]
    # The top-level linear-kind block contains the actual selected table setup;
    # the earlier full-kind branch contains an unreachable historical copy.
    scope_block = next(
        n
        for n in method.body
        if isinstance(n, ast.If)
        and ast.unparse(n.test) == "self.kind == 'linear_attention'"
        and any(
            isinstance(x, ast.Assign) and any(isinstance(y, ast.Name) and y.id == "gate_values" for y in x.targets)
            for x in n.body
        )
    )
    names = ("codes", "gate_values", "decay_values")
    math_nodes = [
        copy.deepcopy(n)
        for n in scope_block.body
        if isinstance(n, ast.Assign)
        and len(n.targets) == 1
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in names
    ]
    assert len(math_nodes) == 3
    namespace = {"torch": torch, "state_dict": state_dict}
    exec(
        compile(ast.fix_missing_locations(ast.Module(math_nodes, type_ignores=[])), str(BASE) + ":tables", "exec"),
        namespace,
    )
    full = namespace["decay_values"].contiguous()
    bits = full.view(torch.int32).reshape(32, 2, 32768)
    changes = (bits[:, :, 1:32641] != bits[:, :, :32640]).any(dim=(0, 1)).nonzero().flatten() + 1
    lo, hi = (int(changes.min()) - 1, int(changes.max())) if changes.numel() else (0, 0)
    window = hi - lo + 1
    assert window + 127 <= 8192
    compact = torch.zeros((32, 16384), dtype=torch.float32)
    for sign in (0, 1):
        compact[:, sign * 8192 : sign * 8192 + window] = full[:, sign * 32768 + lo : sign * 32768 + hi + 1]
        compact[:, sign * 8192 + window : sign * 8192 + window + 127] = full[
            :, sign * 32768 + 32641 : (sign + 1) * 32768
        ]
    raw = torch.arange(65536, dtype=torch.int64)
    mag = raw & 32767
    mapped = mag.clamp(lo, hi) - lo + mag.clamp(min=32640) - 32640 + ((raw & 32768) >> 2)
    assert torch.equal(compact[:, mapped].contiguous().view(torch.int32), full.view(torch.int32))
    return full, compact, lo, hi, digest(ast.Module(math_nodes, type_ignores=[]))


def compare(a, b):
    import torch

    assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
    bad = a.contiguous().view(torch.int32) != b.contiguous().view(torch.int32)
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        bits_different=int(bad.sum()),
        elements=a.numel(),
        max_finite_abs=float((a[finite] - b[finite]).abs().max()) if bool(finite.any()) else 0,
        first_indices=bad.nonzero()[:8].tolist(),
    )


def prepare(args):
    import torch
    from safetensors import safe_open

    torch.set_num_threads(4)
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    state_dict = {}
    for suffix in ("A_log", "dt_bias"):
        key = f"model.language_model.layers.{args.layer}.linear_attn.{suffix}"
        with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as opened:
            state_dict["linear_attn." + suffix] = opened.get_tensor(key)
    full, compact, lo, hi, table_ast = cpu_tables(state_dict)
    fixtures, captures = [], {}
    generator = torch.Generator().manual_seed(2026111305)
    for group in args.cases.split(","):
        if group == "actual":
            for pos in (3271, 227, 738, 5799):
                name = "v6_prefix_control_stream_probe" if pos == 3271 else "v6_inline_precision_stream_probe"
                path = E / f"{name}.json.pos{pos}.pt"
                capture = torch.load(path, weights_only=False, map_location="cpu")
                assert capture["layer"] == args.layer
                fixtures.append((f"actual{pos}", capture["tt_values"]["projection_a"].contiguous()))
                captures[str(path)] = sha(path)
        elif group == "boundaries":
            values = [0, 1, 127, 128, lo - 1, lo, lo + 1, hi - 1, hi, hi + 1, 32639, 32640, 32641, 32703, 32704, 32767]
            raw = (
                torch.tensor(values + [v + 32768 for v in values], dtype=torch.int32)
                .to(torch.int16)
                .view(torch.bfloat16)
            )
            for shift in (0, 7, 16, 31):
                fixtures.append((f"boundaries{shift}", raw.roll(shift).reshape(1, 1, 32)))
        elif group == "random":
            for s in (1, 33, 65):
                fixtures.append((f"random{s}", torch.randn((1, s, 32), generator=generator).bfloat16()))
        elif group == "exhaustive":
            # Every head receives every code; chunking keeps native intermediates bounded.
            for part in range(32):
                codes = (
                    torch.arange(part * 2048, (part + 1) * 2048, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
                )
                fixtures.append((f"allcodes{part:02d}", codes.reshape(1, 2048, 1).expand(1, 2048, 32).contiguous()))
        else:
            raise ValueError(group)
    _, native_ast = native_function(None)
    report = dict(
        status="cpu_setup_pass",
        source_sha256=sha(__file__),
        base_sha256=BASE_SHA,
        device_executed=args.run_device,
        device_closed=False,
        acceptance_eligible=False,
        oracle_assisted=False,
        classification="Same-operand exact decay lookup adapter; no decoder acceptance claim",
        native_adapter_ast_sha256=native_ast,
        table_math_ast_sha256=table_ast,
        lo=lo,
        hi=hi,
        full_table_sha256=hashlib.sha256(full.view(torch.uint8).numpy().tobytes()).hexdigest(),
        compact_table_sha256=hashlib.sha256(compact.view(torch.uint8).numpy().tobytes()).hexdigest(),
        compact_reconstruction_all_65536_codes_all_32_heads_bitexact=True,
        fixture_names=[n for n, _ in fixtures],
        captures=captures,
        cases=[],
        timing=[],
        persistent_added_bytes=0,
        temporary_cb_bytes_per_core=12288,
        timing_scope="Existing BF16 TILE projection_a plus existing per-head FP32 DRAM table to DRAM FP32 TILE decay; all native indexing, transpose, untilize, gather and tilize costs included. Setup/readback excluded.",
    )
    return full, compact, lo, hi, fixtures, report


def run(args):
    assert not args.output.exists()
    full, compact, lo, hi, fixtures, result = prepare(args)
    args.output.with_suffix(".source.py.snapshot").write_bytes(Path(__file__).read_bytes())
    args.output.with_suffix(".reader.cpp.snapshot").write_text(READER)
    result["reader_sha256"] = hashlib.sha256(READER.encode()).hexdigest()

    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("DECAY_LOOKUP_CPU_SETUP_PASS", result["source_sha256"], lo, hi)
        return
    for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER"):
        assert os.environ.get(name) in (None, "", "0"), name
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    try:
        mesh = t.open_mesh_device(
            t.MeshShape(1, 1), physical_device_ids=[0], l1_small_size=32768, trace_region_size=32 * 1024 * 1024
        )

        def upload(x, dtype=t.float32, layout=t.ROW_MAJOR_LAYOUT, memory=t.DRAM_MEMORY_CONFIG):
            if dtype == t.bfloat16 and layout == t.TILE_LAYOUT:
                # Construct TILE layout on the host, preserving all BF16 bit patterns
                # without a device BF16 tilize before the operation under test.
                host_tile = t.from_torch(x.contiguous(), dtype=dtype, layout=layout)
                uploaded = t.to_device(host_tile, mesh, memory_config=memory)
                raw = t.to_torch(t.bitcast(uploaded, t.uint16)).to(torch.int32)
                intended = x.contiguous().view(torch.int16).to(torch.int32) & 65535
                differences = int((raw != intended).sum())
                result.setdefault("rawcode_uploads", []).append(
                    dict(
                        shape=list(x.shape),
                        intended_sha256=hashlib.sha256(intended.numpy().tobytes()).hexdigest(),
                        raw_uint16_bit_differences=differences,
                        elements=x.numel(),
                    )
                )
                assert differences == 0, "Input code-domain coverage changed during setup upload"
                return uploaded
            return t.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=memory)

        dfull, dcompact = upload(full), upload(compact)
        U = t.UnaryOpType

        def chain(values):
            return [t.UnaryWithParam(op, float(value)) for op, value in values]

        obj = SimpleNamespace(
            compact_decay_table=dcompact,
            decay_table=dfull,
            decay_finite_index_chain=chain(
                ((U.BITWISE_AND, 32767), (U.MAXIMUM, lo), (U.MINIMUM, hi), (U.SUB_UNARY_SFPU, lo))
            ),
            decay_nan_index_chain=chain(((U.BITWISE_AND, 32767), (U.MAXIMUM, 32640), (U.SUB_UNARY_SFPU, 32640))),
            decay_sign_index_chain=chain(((U.BITWISE_AND, 32768), (U.RIGHT_SHIFT, 2))),
        )
        native, _ = native_function(t)
        candidate = DirectDecayLookup(t, mesh, dcompact, lo, hi, args.max_cores)
        raw_candidate = DirectDecayLookup(t, mesh, dfull, max_cores=args.max_cores)
        saved = {}
        for name, host in fixtures:
            device = upload(host, t.bfloat16, t.TILE_LAYOUT, t.L1_MEMORY_CONFIG)
            with device_only():
                old = native(obj, device)
                new = candidate(device)
                raw = raw_candidate(device)
            a, b, c = (t.to_torch(x) for x in (old, new, raw))
            codes = host.view(torch.int16).to(torch.int64) & 65535
            cpu = full[torch.arange(32).reshape(1, 1, 32), codes]
            row = dict(
                name=name,
                native_vs_compact=compare(a, b),
                native_vs_full=compare(a, c),
                native_vs_stored_table=compare(cpu, a),
                compact_vs_stored_table=compare(cpu, b),
                native_memory=str(old.memory_config()),
                candidate_memory=str(new.memory_config()),
            )
            result["cases"].append(row)
            if row["native_vs_compact"]["bits_different"] or row["native_vs_full"]["bits_different"]:
                saved[name] = dict(input=host, native=a, compact=b, full=c, cpu=cpu)
            save()
            print(json.dumps(row), flush=True)
            del device, old, new, raw
        if saved:
            path = args.output.with_suffix(".differences.pt")
            torch.save(saved, path)
            result.update(differences=str(path), differences_sha256=sha(path))
        exact = all(
            r["native_vs_compact"]["bits_different"] == r["native_vs_full"]["bits_different"] == 0
            for r in result["cases"]
        )
        result["status"] = "component_exact" if exact else "component_differences"
        save()
        assert exact, "Native adapter bit differences; preserve all fixtures before failure"
        if args.time:
            assert set(args.cases.split(",")) == {"actual", "boundaries", "random", "exhaustive"}
            host = next(x for name, x in fixtures if name == "actual3271")
            device = upload(host, t.bfloat16, t.TILE_LAYOUT, t.L1_MEMORY_CONFIG)
            with device_only():
                expected_tensor = native(obj, device)
            expected = t.to_torch(expected_tensor).clone()
            del expected_tensor
            for name, function in (("native", lambda x: native(obj, x)), ("direct_compact", candidate)):
                for _ in range(5):
                    with device_only():
                        output = function(device)
                    t.synchronize_device(mesh)
                warm_check = compare(expected, t.to_torch(output))
                assert warm_check["bits_different"] == 0
                del output
                eager = []
                for _ in range(3):
                    t.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(20):
                        with device_only():
                            output = function(device)
                    t.synchronize_device(mesh)
                    eager.append(1000 * (time.perf_counter() - start) / 20)
                assert compare(expected, t.to_torch(output))["bits_different"] == 0
                del output
                trace, capturing = None, False
                try:
                    trace = t.begin_trace_capture(mesh, cq_id=0)
                    capturing = True
                    with device_only():
                        output = function(device)
                    t.end_trace_capture(mesh, trace, cq_id=0)
                    capturing = False
                    trace_allocation_tracker.acknowledge_corruptible(output)
                    for _ in range(5):
                        t.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = t.to_torch(output).clone()
                    assert compare(expected, first)["bits_different"] == 0
                    samples = []
                    for _ in range(3):
                        t.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(100):
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        t.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / 100)
                    last = t.to_torch(output).clone()
                    checks = dict(
                        first=compare(expected, first), last=compare(expected, last), deterministic=compare(first, last)
                    )
                    assert all(c["bits_different"] == 0 for c in checks.values())
                    result["timing"].append(
                        dict(
                            variant=name,
                            warmed_ms=eager,
                            warmed_median_ms=statistics.median(eager),
                            traced_ms=samples,
                            traced_median_ms=statistics.median(samples),
                            replays_per_sample=100,
                            checks=checks,
                        )
                    )
                    save()
                finally:
                    if trace is not None:
                        if capturing:
                            t.end_trace_capture(mesh, trace, cq_id=0)
                        t.synchronize_device(mesh)
                        t.release_trace(mesh, trace)
                    del output
            result["status"] = "completed_microbenchmark"
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
    p.add_argument("--time", action="store_true")
    p.add_argument("--cases", default="actual,boundaries,random,exhaustive")
    p.add_argument("--layer", type=int, default=34)
    p.add_argument("--max-cores", type=int, default=32)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    run(p.parse_args())


if __name__ == "__main__":
    main()
