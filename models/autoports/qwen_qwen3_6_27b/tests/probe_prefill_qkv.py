# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact TP4 direct compact prefill QKVG movement; native split/reshape oracle."""

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

from models.autoports.qwen_qwen3_6_27b.tt import prefill_qkv_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]


def cpu_check():
    row_checks = plans = tiles = 0
    sequences = list(range(1, 66)) + [127, 255, 511, 512, 513]
    for batch in range(1, 33):
        for seq in sequences:
            mt, flat_mt, counts = candidate.geometry(batch, seq)
            for cores in (1, 8, 48, 110):
                spans = candidate.partition(batch, seq, cores)
                cursor = 0
                for start, count in spans:
                    assert start == cursor and count > 0
                    cursor += count
                assert cursor == sum(counts)
                plans += 1
            offset = 0
            for output, count in enumerate(counts):
                # Exhaustive tile-level addresses; four live/dead row boundaries per tile.
                for local in range(count):
                    o, page, first, column, live = candidate.work_item(batch, seq, offset + local)
                    assert (o, page) == (output, local) and 1 <= live <= 32
                    assert 0 <= column < 112 and first + live <= batch * seq
                    first_page = first // 32 * 112 + column
                    assert first_page < flat_mt * 112
                    if first % 32 + live > 32:
                        assert first_page + 112 < flat_mt * 112
                    if output == 0:
                        owner_head, tr = divmod(local // 8, mt)
                        user, head = divmod(owner_head, 6)
                        assert first == user * seq + tr * 32 and column == head * 8 + local % 8
                    elif output in (1, 2):
                        user, tr = divmod(local // 8, mt)
                        assert first == user * seq + tr * 32 and column == (48 if output == 1 else 56) + local % 8
                    else:
                        assert first == local // 48 * 32 and column == 64 + local % 48
                    for row in {0, live - 1, live, 31}:
                        if row < live:
                            phase = first % 32 + row
                            source_page = first_page + phase // 32 * 112
                            assert source_page == (first + row) // 32 * 112 + column
                            assert phase % 32 == (first + row) % 32
                        else:
                            assert row >= live  # writer must emit zero without reading dead source rows
                        row_checks += 1
                    tiles += 1
                offset += count
    integer_copies = 0
    ti = lambda row, col: (row // 16) * 512 + (col // 16) * 256 + row % 16 * 16 + col % 16
    source = [0] * 2048
    for row in range(64):
        for col in range(32):
            source[row // 32 * 1024 + ti(row % 32, col)] = 1 + row * 32 + col
    for phase in range(32):
        for live in range(1, 33):
            destination = [-1] * 1024
            for row in range(32):
                for col in range(32):
                    logical = phase + row
                    destination[ti(row, col)] = (
                        source[logical // 32 * 1024 + ti(logical % 32, col)] if row < live else 0
                    )
                    assert destination[ti(row, col)] == (1 + (phase + row) * 32 + col if row < live else 0)
                    integer_copies += 1
    for residue in range(64):
        assert (-residue) % 64 + 8 * 2048 <= 16448
    assert "copy_tile" not in candidate.KERNEL and "ComputeConfig" not in Path(candidate.__file__).read_text()
    negatives = 0
    for fn in (
        lambda: candidate.geometry(0, 33),
        lambda: candidate.geometry(33, 33),
        lambda: candidate.geometry(1, 0),
        lambda: candidate.partition(1, 33, 0),
        lambda: candidate.partition(1, 33, True),
        lambda: candidate.work_item(1, 33, -1),
        lambda: candidate.geometry(32, 2**32),
    ):
        try:
            fn()
        except ValueError:
            negatives += 1
        else:
            raise AssertionError("Invalid geometry admitted")
    # The diagnostic append must stay inside the per-user loop, even on a nonfinite failure.
    tree = ast.parse(Path(__file__).read_text())
    verify = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "verify")
    loop = next(
        n
        for n in ast.walk(verify)
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "user"
    )
    assert any(
        isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "append"
        for n in loop.body
    )
    return dict(
        pass_cpu=True,
        plans=plans,
        tiles=tiles,
        row_boundaries=row_checks,
        negative_cases=negatives,
        scratch_bytes_per_worker=16448,
        integer_copy_checks=integer_copies,
        sequences=sequences,
        hardware_verified=False,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--seqs", nargs="+", type=int, default=[33, 512])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--all-residues", action="store_true")
    ap.add_argument("--cores", nargs="+", type=int, default=[110])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    if args.all_residues:
        args.seqs = sorted(set(args.seqs + list(range(33, 64))))
    assert args.output and not args.output.exists() and args.steps >= 2
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert min(args.queued, args.samples, args.replays) > 0
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    if tracking:
        assert os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    timed = not (watcher or profiler or tracking or args.no_timing)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = dict(
        status="running",
        mesh=[1, 4],
        cases=[],
        watcher=watcher,
        profiler=profiler,
        tracking=tracking,
        timed=timed,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "tt/prefill_qkv_candidate.py", ROOT / "tt/multichip_decoder.py")
        },
        reference="Native gate/QKV slices -> material reshape[B,S,2048] -> split heads; independent Torch movement oracle",
        timing_scope="Complete packed projection output -> final Q/K/V heads and compact gate, norm/RoPE/cache unchanged and excluded",
        allocation_scope="Candidate inputs/banks/outputs preallocated; original native outputs/intermediates allocated during capture",
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)

        def empty(shape):
            return ttnn.empty(
                shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def upload(values):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                preserve_nan_values=True,
            )

        def read(t):
            return [ttnn.to_torch(s).contiguous() for s in ttnn.get_device_tensors(t)]

        def physical(t):
            return [s.cpu().to_torch_with_padded_shape().contiguous() for s in ttnn.get_device_tensors(t)]

        def bits(a, b):
            return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())

        def raw_copy(src, dst):
            assert tuple(src.padded_shape) == tuple(dst.padded_shape)
            pages = src.padded_shape[-2] // 32 * 112
            assert pages % 8 == 0 and src.buffer_address() != dst.buffer_address()
            core = ttnn.CoreCoord(0, 0)
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [src.buffer_address(), dst.buffer_address(), pages]
            ct = list(ttnn.TensorAccessorArgs(src).get_compile_time_args()) + list(
                ttnn.TensorAccessorArgs(dst).get_compile_time_args()
            )
            code = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"
void kernel_main(){
 constexpr auto sa=TensorAccessorArgs<0>();constexpr auto da=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
 const auto s=TensorAccessor(sa,get_arg_val<uint32_t>(0),2048);const auto d=TensorAccessor(da,get_arg_val<uint32_t>(1),2048);
 const uint32_t pages=get_arg_val<uint32_t>(2);Noc noc;CircularBuffer cb(0);
 for(uint32_t base=0;base<pages;base+=8){
 cb.reserve_back(8);for(uint32_t j=0;j<8;j++)noc.async_read(s,cb,2048,{.page_id=base+j},{.offset_bytes=j*2048});
 noc.async_read_barrier();cb.push_back(8);cb.wait_front(8);
 for(uint32_t j=0;j<8;j++)noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cb),d,2048,{.offset_bytes=j*2048},{.page_id=base+j});
 noc.async_write_barrier();cb.pop_front(8);}}
"""
            kernel = ttnn.KernelDescriptor(
                kernel_source=code,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
            cb = ttnn.CBDescriptor(
                total_size=16384,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
            ttnn.generic_op([src, dst], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))

        positive = False
        for batch in args.batches:
            for seq in args.seqs:
                _, flat_mt, _ = candidate.geometry(batch, seq)
                entry = dict(batch=batch, seq=seq, checks=[], poison=[], trace_us={})
                report["cases"].append(entry)
                save()
                banks = []
                host = []
                expected = []
                for step in range(args.steps):
                    values = []
                    reference = [[], [], [], []]
                    for rank in range(4):
                        gen = torch.Generator().manual_seed(731 + rank * 997 + batch * 47 + seq + step * 7907)
                        value = torch.full((1, flat_mt * 32, 3584), float("nan"), dtype=torch.bfloat16)
                        rows = (torch.randn((batch, seq, 3584), generator=gen) * (0.25 + rank * 0.125)).bfloat16()
                        rows[:, :, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0], dtype=torch.bfloat16)
                        for user in range(batch):
                            rows[user, 0, 4:8] = torch.tensor([user + 1, rank + 1, step + 1, seq], dtype=torch.bfloat16)
                            rows[user, -1, -4:] = torch.tensor(
                                [-user - 1, -rank - 1, -step - 1, -seq], dtype=torch.bfloat16
                            )
                        value[:, : batch * seq] = rows.reshape(1, batch * seq, 3584)
                        values.append(value)
                        reference[0].append(
                            rows[:, :, :1536].reshape(batch, seq, 6, 256).permute(0, 2, 1, 3).contiguous()
                        )
                        reference[1].append(rows[:, :, 1536:1792].reshape(batch, 1, seq, 256).contiguous())
                        reference[2].append(rows[:, :, 1792:2048].reshape(batch, 1, seq, 256).contiguous())
                        reference[3].append(rows[:, :, 2048:].reshape(1, batch * seq, 1536).contiguous())
                    full = upload(values)
                    logical = empty((1, batch * seq, 3584))
                    raw_copy(full, logical)
                    counts = [bits(a, b) for a, b in zip(physical(logical), values)]
                    entry["poison"].append(
                        dict(
                            step=step,
                            bit_errors=counts,
                            expected_nan_rows=flat_mt * 32 - batch * seq,
                            nan_counts=[int(torch.isnan(t).sum()) for t in physical(logical)],
                        )
                    )
                    assert counts == [0] * 4
                    banks.append(logical)
                    host.append(values)
                    expected.append(reference)
                    ttnn.deallocate(full)
                live = empty((1, batch * seq, 3584))
                rebound = empty((1, batch * seq, 3584))
                shapes = ((batch, 6, seq, 256), (batch, 1, seq, 256), (batch, 1, seq, 256), (1, batch * seq, 1536))
                paths = [dict(name="native", cores=None, outputs=None)] + [
                    dict(name=f"direct_c{c}", cores=c, outputs=tuple(empty(s) for s in shapes)) for c in args.cores
                ]

                def set_input(step, dst=live):
                    raw_copy(banks[step], dst)

                def invoke(path, inp=live):
                    if path["cores"] is None:
                        gate = inp[:, :, 2048:]
                        qkv = ttnn.reshape(inp[:, :, :2048], (batch, seq, 2048))
                        q, k, v = ttnn.transformer.split_query_key_value_and_split_heads(
                            qkv, num_heads=6, num_kv_heads=1, transpose_key=False
                        )
                        path["outputs"] = (q, k, v, gate)
                    else:
                        candidate.prefill_qkv(inp, batch=batch, seq=seq, outputs=path["outputs"], cores=path["cores"])

                def snapshot(path):
                    return [read(t) for t in path["outputs"]]

                def verify(path, reference, label):
                    got = snapshot(path)
                    checks = []
                    for index, (actual, refs) in enumerate(zip(got, reference)):
                        for rank, (a, b) in enumerate(zip(actual, refs)):
                            assert a.shape == b.shape
                            if index == 3:
                                a = a.reshape(batch, seq, 1536)
                                b = b.reshape(batch, seq, 1536)
                            for user in range(batch):
                                aa = a[user].contiguous()
                                bb = b[user].contiguous()
                                errors = bits(aa, bb)
                                finite = bool(torch.isfinite(aa).all() and torch.isfinite(bb).all())
                                score = None
                                if finite:
                                    if errors == 0:
                                        score = 1.0
                                    elif bool(torch.var(aa.double(), correction=0) > 0) and bool(
                                        torch.var(bb.double(), correction=0) > 0
                                    ):
                                        try:
                                            value = float(pcc(aa.float(), bb.float()))
                                            score = value if math.isfinite(value) else 0.0
                                        except AssertionError:
                                            score = 0.0
                                    else:
                                        score = 0.0
                                checks.append(
                                    dict(
                                        output=("q", "k", "v", "gate")[index],
                                        rank=rank,
                                        user=user,
                                        bit_errors=errors,
                                        finite=finite,
                                        pcc=score,
                                    )
                                )
                    assert len(checks) == 16 * batch
                    entry["checks"].append(dict(path=path["name"], label=label, values=checks))
                    save()
                    passing = all(
                        c["bit_errors"] == 0 and c["finite"] and c["pcc"] is not None and c["pcc"] >= 0.995
                        for c in checks
                    )
                    if not passing and "first_failure" not in report:
                        dump = args.output.with_suffix(".first_failure.pt")
                        assert not dump.exists()
                        torch.save(
                            dict(path=path["name"], label=label, actual=got, expected=reference, input=read(live)), dump
                        )
                        report["first_failure"] = dict(path=path["name"], label=label, artifact=str(dump))
                        save()
                    assert passing, (path["name"], label)

                for step in range(args.steps):
                    set_input(step)
                    for path in paths:
                        invoke(path)
                        verify(path, expected[step], f"eager{step}_host_and_native_movement")
                    assert all(bits(a, b) == 0 for a, b in zip(physical(live), host[step]))
                    for path in paths[1:]:
                        for index, t in enumerate(path["outputs"]):
                            vals = physical(t)
                            assert all(
                                torch.count_nonzero(x[..., seq:, :] if index < 3 else x[:, batch * seq :, :]) == 0
                                for x in vals
                            )
                set_input(1, rebound)
                for path in paths:
                    previous = path["outputs"]
                    if path["cores"] is not None:
                        path["outputs"] = tuple(empty(s) for s in shapes)
                    invoke(path, rebound)
                    verify(path, expected[1], "rebound_input_outputs")
                    path["outputs"] = previous
                traces = []
                stable = [
                    [t.buffer_address() for t in path["outputs"]] if path["cores"] is not None else None
                    for path in paths
                ]
                try:
                    set_input(0)
                    for path in paths:
                        invoke(path)
                    for path in paths:
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        invoke(path)
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        traces.append(trace)
                    if tracking and not positive:
                        unsafe = empty((1, 32, 32))
                        detected = False
                        try:
                            ttnn.execute_trace(mesh, traces[-1], cq_id=0, blocking=True)
                        except RuntimeError as exc:
                            if not any(w in str(exc).lower() for w in ("unsafe", "allocation")):
                                raise
                            detected = True
                        finally:
                            ttnn.deallocate(unsafe)
                        entry["allocation_positive_control"] = detected
                        save()
                        assert detected
                        positive = True
                    for step in range(args.steps):
                        set_input(step)
                        for path, trace in reversed(list(zip(paths, traces))):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            verify(path, expected[step], f"trace{step}")
                    for step in range(args.queued):
                        set_input(step % args.steps)
                        for trace in traces:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    for path in paths:
                        verify(path, expected[(args.queued - 1) % args.steps], "queued50")
                    if timed:
                        samples = [[] for _ in paths]
                        for sample in range(args.samples):
                            for off in range(len(paths)):
                                i = (sample + off) % len(paths)
                                ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=True)
                                start = time.perf_counter_ns()
                                for _ in range(args.replays):
                                    ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                samples[i].append((time.perf_counter_ns() - start) / 1000 / args.replays)
                        for path, values in zip(paths, samples):
                            entry["trace_us"][path["name"]] = dict(samples=values, median=statistics.median(values))
                            verify(path, expected[(args.queued - 1) % args.steps], "timed_final")
                    assert stable == [
                        [t.buffer_address() for t in p["outputs"]] if p["cores"] is not None else None for p in paths
                    ]
                finally:
                    for trace in traces:
                        ttnn.release_trace(mesh, trace)
                entry.update(pass_all=True, queued=args.queued)
                save()
        report.update(status="pass", pass_all=True)
    except BaseException:
        report.update(status="fail", exception=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()


if __name__ == "__main__":
    main()
