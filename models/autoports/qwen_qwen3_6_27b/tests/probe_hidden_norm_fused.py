# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""TP4 exact-tree hidden norm oracle, queue and trace probe. CPU mode has no TTNN imports."""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import struct
import sys
import time
import traceback
from pathlib import Path
from models.autoports.qwen_qwen3_6_27b.tt.hidden_norm_fused_candidate import fused_sources, hidden_norm_fused

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22"


def source_proof(admission="software", defer_receiver_reserve=False):
    path = ROOT / "tt/optimized_decoder.py"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA, "Frozen oracle drift"
    constants = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.startswith("_norm_zero_mean"):
                    constants[target.id] = ast.literal_eval(node.value)
    args = tuple(constants["_norm_zero_mean" + key] for key in ("FOLD_COMPUTE", "INVERSE_COMPUTE", "INVERSE_WRITER"))
    software = fused_sources(*args)
    # Pinned before admission adaptation: preserve all three default kernels.
    assert [hashlib.sha256(v.encode()).hexdigest() for v in software] == [
        "c012327e3d272059c9b0809fab23d5dd88ad4516d02a9f76f4876f58c7c00d6f",
        "9c76c4b3fc8829aa8cffed8eb3e6f3442721a244c48072630b20b21da923c7f5",
        "8327682c24b7c9b623a93fc179324b7d407e4e39534ff7631dc51c14a255a2fd",
    ], "Default software-admission kernels changed"
    sources = fused_sources(*args, admission=admission, defer_receiver_reserve=defer_receiver_reserve)
    assert sources[2] == software[2]
    # Data writes, DONE publication and inverse stores remain literal.
    boundary = "    folded.wait_front(1);"
    assert sources[1].split(boundary, 1)[1] == software[1].split(boundary, 1)[1]
    # Arithmetic spans are byte-identical apart from the explicit CB indices.
    fold = constants["_norm_zero_meanFOLD_COMPUTE"].split("    CircularBuffer x(0),out(3);", 1)[1].rsplit("}", 1)[0]
    inv = (
        constants["_norm_zero_meanINVERSE_COMPUTE"]
        .split("    CircularBuffer dense(0),inverse(2);", 1)[1]
        .rsplit("}", 1)[0]
    )
    assert fold in sources[2]
    assert inv.replace("copy_tile(0,0,0)", "copy_tile(4,0,0)").replace("pack_tile(0,2)", "pack_tile(0,5)") in sources[2]
    wr = (
        constants["_norm_zero_meanINVERSE_WRITER"]
        .split("    Noc noc; CircularBuffer data(2),zero(4);", 1)[1]
        .rsplit("}", 1)[0]
    )
    assert wr in sources[1]
    repo = ROOT.parents[2]
    protocol_paths = [
        repo / p
        for p in (
            "tt_metal/impl/program/dispatch.cpp",
            "tt_metal/impl/program/program.cpp",
            "tt_metal/api/tt-metalium/program_descriptors.hpp",
            "tt_metal/hw/inc/api/dataflow/dataflow_api.h",
            "tt_metal/hw/inc/internal/circular_buffer_init.h",
            "tt_metal/impl/dispatch/kernels/cq_dispatch.cpp",
            "tt_metal/impl/dispatch/kernels/cq_dispatch_subordinate.cpp",
            "tt_metal/distributed/fd_mesh_command_queue.cpp",
        )
    ]
    return constants, {
        "frozen_sha256": FROZEN_SHA,
        "admission": admission,
        "defer_receiver_reserve": defer_receiver_reserve,
        "software_default_kernels_unchanged": True,
        "fused_kernel_sha256": [hashlib.sha256(s.encode()).hexdigest() for s in sources],
        "source_sha256": {
            str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), ROOT / "tt/hidden_norm_fused_candidate.py", path, *protocol_paths]
        },
    }


def cpu_check():
    from collections import defaultdict
    from types import SimpleNamespace as NS

    constants, proof = source_proof()
    fake = NS(bfloat16="bf16", float32="fp32", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram", L1_MEMORY_CONFIG="l1")
    fake.CoreCoord = lambda x, y: NS(x=x, y=y)
    fake.CoreRange = lambda a, b: (a, b)
    fake.CoreRangeSet = list
    fake.RuntimeArgs = lambda: defaultdict(dict)
    for name in (
        "ReaderConfigDescriptor",
        "WriterConfigDescriptor",
        "ComputeConfigDescriptor",
        "CBDescriptor",
        "CBFormatDescriptor",
        "ProgramDescriptor",
        "SemaphoreDescriptor",
    ):
        setattr(fake, name, lambda **kw: NS(**kw))
    fake.KernelDescriptor = lambda **kw: NS(**kw)
    fake.KernelDescriptor.SourceType = NS(SOURCE_CODE="source")
    fake.MathFidelity = NS(HiFi4="HiFi4")
    fake.UnpackToDestMode = NS(Default="default", UnpackToDestFp32="fp32")
    fake.TensorAccessorArgs = lambda t: NS(get_compile_time_args=lambda: [t.buffer_address()])
    captured = []
    fake.generic_op = lambda tensors, desc: captured.append((tensors, desc))
    mesh = NS(
        compute_with_storage_grid_size=lambda: NS(x=11, y=10),
        worker_core_from_logical_core=lambda c: NS(x=c.x + 1, y=c.y + 2),
    )

    def tensor(shape, address, dtype, memory="dram"):
        return NS(
            shape=shape,
            padded_shape=(*shape[:-2], math.ceil(shape[-2] / 32) * 32, math.ceil(shape[-1] / 32) * 32),
            dtype=dtype,
            layout="tile",
            device=lambda: mesh,
            memory_config=lambda: memory,
            buffer_address=lambda: address,
        )

    node = next(
        n
        for n in ast.parse((ROOT / "tt/hidden_norm_fused_candidate.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "hidden_norm_fused"
    )
    node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = constants | {"ttnn": fake, "fused_sources": fused_sources, "struct": struct}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<hidden-norm-descriptor>", "exec"), ns)
    for batch in range(1, 33):
        x, gamma, out, inverse = (
            tensor((1, batch, 5120), 100, "bf16"),
            tensor((1, 1, 5120), 200, "fp32"),
            tensor((1, batch, 5120), 300, "bf16"),
            tensor((1, 32, 32), 400, "fp32", "l1"),
        )
        assert ns["hidden_norm_fused"](x, gamma, output=out, inverse=inverse) is out
        assert len(captured) == 2
        (_, fused), (_, final) = captured
        captured.clear()
        assert sum(cb.total_size for cb in fused.cbs) == 26624
        assert sum(cb.total_size for cb in final.cbs) == 20480
        assert [(s.id, s.initial_value) for s in fused.semaphores] == [(0, 0), (1, 0), (2, 0)]
        assert not final.semaphores
        assert [i for i, v in enumerate(fused.kernels[2].config.unpack_to_dest_mode) if v == "fp32"] == [3, 4, 5, 6]
        for i, key in enumerate(("OUTPUT_READER", "OUTPUT_WRITER", "OUTPUT_COMPUTE")):
            assert final.kernels[i].kernel_source == constants["_norm_zero_mean" + key]
        rt = [v for row in fused.kernels[0].runtime_args.values() for v in row.values()]
        assert sorted(v[1] for v in rt) == list(range(32))
        assert all(v[2] == batch and len(v) == 67 and v[3:5] == [1, 2] for v in rt)
        ranges = [cb.core_ranges for cb in fused.cbs]
        assert all(r == ranges[0] for r in ranges)
        pages = []
        for row in final.kernels[1].runtime_args.values():
            for _, start, b, count in row.values():
                assert b == batch and count == 2
                pages.extend(range(start, start + count))
        assert sorted(pages) == list(range(160))
        for admission, deferred in (("dispatch", False), ("dispatch", True)):
            assert (
                ns["hidden_norm_fused"](
                    x, gamma, output=out, inverse=inverse, admission=admission, defer_receiver_reserve=deferred
                )
                is out
            )
            (_, variant), (_, variant_final) = captured
            captured.clear()
            assert variant_final == final
            assert variant.cbs == fused.cbs and variant.semaphores == fused.semaphores
            assert variant.kernels[2] == fused.kernels[2]
            variant_sources = fused_sources(
                *(constants["_norm_zero_mean" + key] for key in ("FOLD_COMPUTE", "INVERSE_COMPUTE", "INVERSE_WRITER")),
                admission=admission,
                defer_receiver_reserve=deferred,
            )
            for baseline_kernel, actual_kernel, source in zip(fused.kernels, variant.kernels, variant_sources):
                assert actual_kernel.kernel_source == source
                assert baseline_kernel.runtime_args == actual_kernel.runtime_args
                assert baseline_kernel.compile_time_args == actual_kernel.compile_time_args
                assert baseline_kernel.config == actual_kernel.config
            reader, writer, _ = variant_sources
            assert "ready=" not in reader and "go=" not in reader + writer
            assert "noc_semaphore_inc" not in reader
            assert writer.count("noc_async_atomic_barrier()") == 1
            assert reader.index("data.push_back(5)") < reader.index("dense.push_back(1)")
            assert "noc_semaphore_set" not in reader + writer
            if deferred:
                assert reader.index("get_semaphore(2)),31)") < reader.index("dense.reserve_back(1)")
                # Abstract schedule: peers1..31 can publish all their bytes and
                # DONE without any action by receiver0. Writer0 needs raw0 and
                # cannot overshoot the equality wait until reader0 proceeds.
                arrived = set()
                for peer in range(1, 32):
                    arrived.add(peer)
                assert len(arrived) == 31 and 0 not in arrived
                arrived.add(0)
                assert len(arrived) == 32
        for user in range(batch):
            owner = [-1] * 4096
            for block in range(32):
                for face in range(2):
                    src = ((user // 16) * 512 + face * 256 + (user % 16) * 16) * 4
                    dst = ((block // 16) * 512 + face * 256 + (block % 16) * 16) * 4
                    assert src % 64 == dst % 64 == 0 and 0 <= src <= 4032 and 0 <= dst <= 4032
                    for byte in range(dst, dst + 64):
                        assert owner[byte] == -1
                        owner[byte] = block
                    # Tiled source row user, channel face*16+j becomes dense[block,channel].
                    for j in range(16):
                        assert src + 4 * j == ((user // 16) * 512 + face * 256 + (user % 16) * 16 + j) * 4
            assert -1 not in owner
        # User rows plus last-user tail padding cover the inverse exactly once.
        intervals = []
        for user in range(batch):
            for face in range(2):
                start = ((user // 16) * 512 + face * 256 + (user % 16) * 16) * 4
                intervals.extend(range(start, start + 64))
        for user in range(batch, 32):
            for face in range(2):
                start = ((user // 16) * 512 + face * 256 + (user % 16) * 16) * 4
                intervals.extend(range(start, start + 64))
        assert sorted(intervals) == list(range(4096))
        assert 32 + 32 + 32 * batch in range(96, 1089, 32)
    for invalid in (0, 33):
        try:
            ns["hidden_norm_fused"](tensor((1, invalid, 5120), 100, "bf16"), gamma, output=out, inverse=inverse)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid batch accepted")
    for options in ({"admission": "unknown"}, {"admission": "software", "defer_receiver_reserve": True}):
        try:
            ns["hidden_norm_fused"](x, gamma, output=out, inverse=inverse, **options)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Invalid admission control accepted: {options}")
    reader, writer, compute = fused_sources(
        *(constants["_norm_zero_mean" + k] for k in ("FOLD_COMPUTE", "INVERSE_COMPUTE", "INVERSE_WRITER"))
    )
    assert reader.index("data.push_back(5)") < reader.index("noc_semaphore_inc") < reader.index("dense.push_back(1)")
    assert (
        writer.index("noc.async_write_barrier()")
        < writer.index("noc_semaphore_inc")
        < writer.index("noc_async_atomic_barrier()")
        < writer.index("data.wait_front(1)")
    )
    assert compute.index("out.push_back(1)") < compute.index("dense.wait_front(1)")
    assert "noc_semaphore_set" not in reader + writer and "ttnn" not in sys.modules and "torch" not in sys.modules
    return {
        "status": "pass",
        "scope": "CPU arithmetic-source/geometry/descriptor/protocol-order checks only",
        "cases": 96,
        "batch_cases": 32,
        "variants": ["software", "dispatch", "dispatch_deferred_receiver"],
        "cb_bytes_per_fused_core": 26624,
        "cb_bytes_per_output_core": 20480,
        **proof,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--admission", choices=["software", "dispatch"], default="software")
    ap.add_argument(
        "--defer-receiver-reserve",
        action="store_true",
        help="Correctness only: dispatch reader0 reserves dense only after31 peer write completions; skips timings",
    )
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--modes", nargs="+", choices=["normal", "zero", "scale"], default=["normal", "zero", "scale"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--require-exact", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    assert not args.defer_receiver_reserve or args.admission == "dispatch"
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert args.output and not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert (
        all(1 <= b <= 32 for b in args.batches)
        and args.steps >= 2
        and min(args.warmup, args.replays, args.queued, args.samples) > 0
    )
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    assert not (watcher and profiler)
    assert not tracking or os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "1", "No tracking exemptions"
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import _norm_zero_meanhidden_rmsnorm_sfpu as native
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "arguments": vars(args) | {"output": str(args.output)},
        "source_proof": source_proof(args.admission, args.defer_receiver_reserve)[1],
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "timings_skipped": watcher or profiler or tracking or args.defer_receiver_reserve,
        "cases": [],
        "trace_contract": "All retained inputs, outputs, scratch and warmups precede first capture; native three-program and candidate two-program complete chains. Tracker positive control deliberately allocates one live buffer after capture, verifies rejection without replay, frees it and verifies both traces clean; no exemptions.",
        "receiver_skew_contract": "When enabled, reader0 waits for31 peer DONE credits before reserving dense/raw; peer data writes complete before those credits, and own writer cannot send the32nd credit until raw is published.",
        "input_contract": "Rank/user distinct BF16 activations and actual FP32 gamma (no BF16 staging); synthetic parameters, not a real-model PCC gate.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)

        def upload(values, dtype):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def allocate(shape, dtype, memory=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.empty(shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=memory)

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def diagnostics(actual, expected, *, inverse=False, batch=None):
            details = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype
                if inverse:
                    a, b = a[:, :batch, :1], b[:, :batch, :1]
                for user in range(a.shape[1]):
                    x, y = a[:, user].contiguous().reshape(-1), b[:, user].contiguous().reshape(-1)
                    integer = torch.int32 if inverse else torch.int16
                    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    mismatches = int((x.view(integer) != y.view(integer)).sum())
                    score = (1.0 if torch.equal(x, y) else pcc(x, y)) if finite else -1.0
                    details.append(
                        {
                            "rank": rank,
                            "user": user,
                            "pcc": score,
                            "bits": mismatches,
                            "finite": finite,
                            "max_abs": float((x.float() - y.float()).abs().max()) if finite else None,
                        }
                    )
            return {
                "minimum_pcc": min(d["pcc"] for d in details),
                "bit_mismatches": sum(d["bits"] for d in details),
                "finite": all(d["finite"] for d in details),
                "maximum_abs": max(d["max_abs"] or 0 for d in details),
                "details": [d for d in details if d["bits"] or not d["finite"]],
            }

        def check(d, exact=False):
            assert d["finite"] and d["minimum_pcc"] >= 0.995, d
            if exact:
                assert d["bit_mismatches"] == 0, d

        for batch in args.batches:
            for mode in args.modes:
                entry = {
                    "batch": batch,
                    "mode": mode,
                    "admission": args.admission,
                    "defer_receiver_reserve": args.defer_receiver_reserve,
                    "eager": [],
                    "queued": [],
                    "trace": [],
                }
                report["cases"].append(entry)
                save()
                print(f"HIDDEN_NORM B{batch} {mode} {args.admission} defer={args.defer_receiver_reserve}", flush=True)
                bank = []
                for step in range(args.steps):
                    hosts = []
                    gammas = []
                    for rank in range(4):
                        gen = torch.Generator().manual_seed(81573 + batch * 31 + step * 613 + rank)
                        x = torch.randn(1, batch, 5120, generator=gen) + step * 0.125 + rank * 0.03125
                        gamma = (
                            1
                            + torch.randn(1, 1, 5120, generator=gen) * 0.125
                            + torch.arange(5120).reshape(1, 1, -1) * 0.000013
                        )
                        if mode == "zero":
                            x.zero_()
                            x[:, :, ::2] = -0.0
                        if mode == "scale":
                            scale = torch.tensor([2.0**-10, 2.0**10, 1.0])[torch.arange(batch) % 3].reshape(1, batch, 1)
                            x *= scale
                            # Include isolated signals near every five-fold boundary.
                            x[:, :, ::1024] += scale * (step + 1)
                        hosts.append(x.bfloat16())
                        gammas.append(gamma.contiguous())
                    bank.append([upload(hosts, ttnn.bfloat16), upload(gammas, ttnn.float32)])
                inputs = [ttnn.empty_like(t) for t in bank[0]]
                outputs = [allocate((1, batch, 5120), ttnn.bfloat16) for _ in range(2)]
                inverses = [allocate((1, 32, 32), ttnn.float32, ttnn.L1_MEMORY_CONFIG) for _ in range(2)]
                folded = allocate((1, 32, 1024), ttnn.float32, ttnn.L1_MEMORY_CONFIG)
                queued_outputs = [
                    [allocate((1, batch, 5120), ttnn.bfloat16) for _ in range(args.queued)] for _ in range(2)
                ]
                retained = [
                    *inputs,
                    *outputs,
                    *inverses,
                    folded,
                    *(t for pair in bank for t in pair),
                    *(t for row in queued_outputs for t in row),
                ]
                traces = []

                def update(step):
                    for src, dst in zip(bank[step], inputs):
                        ttnn.copy(src, dst)

                def invoke(index, source=None, output=None):
                    source = inputs if source is None else source
                    output = outputs[index] if output is None else output
                    if index:
                        return hidden_norm_fused(
                            *source,
                            output=output,
                            inverse=inverses[1],
                            admission=args.admission,
                            defer_receiver_reserve=args.defer_receiver_reserve,
                        )
                    return native(*source, output=output, scratch=(folded, inverses[0]))

                eager = []
                try:
                    for step in range(args.steps):
                        update(step)
                        for index in range(2):
                            invoke(index)
                        values = [read(t) for t in outputs]
                        d = diagnostics(values[1], values[0])
                        inv = diagnostics(read(inverses[1]), read(inverses[0]), inverse=True, batch=batch)
                        entry["eager"].append({"step": step, "output": d, "inverse_col0": inv})
                        save()
                        check(d, args.require_exact)
                        check(inv, args.require_exact)
                        eager.append(values)
                    for index in range(2):
                        invoke(index, source=bank[0])
                        d = diagnostics(read(outputs[index]), eager[0][index])
                        entry.setdefault("rebind", []).append({"path": index, **d})
                        save()
                        check(d, True)
                    # Queue every generation before synchronizing: retained unique outputs
                    # expose stale readout/epoch failures hidden by checking only the last.
                    for step in range(args.queued):
                        for index in range(2):
                            invoke(index, source=bank[step % args.steps], output=queued_outputs[index][step])
                    ttnn.synchronize_device(mesh)
                    for step in range(args.queued):
                        for index in range(2):
                            d = diagnostics(read(queued_outputs[index][step]), eager[step % args.steps][index])
                            entry["queued"].append({"step": step, "path": index, **d})
                            check(d, True)
                    save()
                    update(0)
                    for _ in range(args.warmup):
                        invoke(0)
                        invoke(1)
                    ttnn.synchronize_device(mesh)
                    cache_count = mesh.num_program_cache_entries()
                    for index in range(2):
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        invoke(index)
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        traces.append(trace)
                    assert mesh.num_program_cache_entries() == cache_count, "Program allocation during capture"
                    if tracking:
                        from ttnn.tools.trace_allocation_tracker import TraceAllocationTracker
                        from ttnn._ttnn.operations.trace import get_unsafe_tracked_ids

                        for trace in traces:
                            TraceAllocationTracker.verify_before_replay(mesh, trace)
                        positive = {"unsafe_buffers_before": [len(get_unsafe_tracked_ids(mesh, t)) for t in traces]}
                        unsafe = allocate((1, 32, 32), ttnn.bfloat16)
                        unsafe_id = unsafe.buffer_unique_id()
                        try:
                            assert unsafe_id in get_unsafe_tracked_ids(mesh, traces[1])
                            rejected = False
                            try:
                                TraceAllocationTracker.verify_before_replay(mesh, traces[1])
                            except RuntimeError as error:
                                assert "device buffer(s) still alive before trace replay" in str(error), str(error)
                                assert f"Buffer {unsafe_id}" in str(error), str(error)
                                rejected = True
                            assert rejected, "Tracker did not reject deliberately retained live allocation"
                            positive["rejected_before_replay"] = True
                        finally:
                            ttnn.deallocate(unsafe)
                            del unsafe
                        for trace in traces:
                            TraceAllocationTracker.verify_before_replay(mesh, trace)
                        positive["unsafe_buffers_after_release"] = [
                            len(get_unsafe_tracked_ids(mesh, t)) for t in traces
                        ]
                        positive["acknowledgements_or_exemptions"] = False
                        entry["allocation_positive_control"] = positive
                        save()
                    for step in range(args.steps):
                        update(step)
                        for index, trace in enumerate(traces):
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            d = diagnostics(read(outputs[index]), eager[step][index])
                            entry["trace"].append({"step": step, "path": index, **d})
                            save()
                            check(d, True)
                    # Alternate changed inputs on each replay, not just each 50-call group.
                    for step in range(args.replays):
                        update(step % args.steps)
                        for trace in traces:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    for index in range(2):
                        d = diagnostics(read(outputs[index]), eager[(args.replays - 1) % args.steps][index])
                        entry.setdefault("alternating_trace", []).append({"path": index, **d})
                        save()
                        check(d, True)
                    if not report["timings_skipped"]:
                        update(0)
                        times = [[], []]
                        for sample in range(args.samples):
                            for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                                ttnn.synchronize_device(mesh)
                                start = time.perf_counter_ns()
                                for _ in range(args.replays):
                                    ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                times[index].append((time.perf_counter_ns() - start) / 1000 / args.replays)
                        entry["warmed_traced_us"] = {
                            name: {"samples": v, "median": statistics.median(v)}
                            for name, v in zip(("native", "candidate"), times)
                        }
                finally:
                    for trace in traces:
                        ttnn.release_trace(mesh, trace)
                assert retained
                save()
        if profiler:
            ttnn.ReadDeviceProfiler(mesh)
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
