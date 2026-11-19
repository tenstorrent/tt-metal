# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Coherent TP4 head-major recurrence/norm-gate probe. CPU mode imports no device libraries."""

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
from models.autoports.qwen_qwen3_6_27b.tt import head_major_recurrent_norm_candidate as candidate
from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import sources, VARIANTS

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
FROZEN_SHA = "315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22"


def sources_and_proof():
    constants = {}
    texts = {name: (ROOT / f"tt/{name}.py").read_text() for name in ("optimized_decoder", "multichip_decoder")}
    assert hashlib.sha256(texts["optimized_decoder"].encode()).hexdigest() == FROZEN_SHA
    for text in texts.values():
        for n in ast.parse(text).body:
            if isinstance(n, ast.Assign):
                for target in n.targets:
                    if isinstance(target, ast.Name) and (
                        target.id.startswith("_fused_delta_row_GDN_") or target.id == "_LOCAL12_READER"
                    ):
                        constants[target.id] = ast.literal_eval(n.value)
    old = constants["_fused_delta_row_GDN_COMPACT_WRITER"]
    new = candidate.head_major_writer_source(old)
    start = "        for(uint32_t i=0;i<4;i++) for(uint32_t face=0;face<2;face++) {"
    tail = "        noc.async_write_barrier(); updated.pop_front(16); out.pop_front(4);"
    assert new.startswith(old.split(start, 1)[0]) and new.endswith(tail + old.split(tail, 1)[1])
    reader = candidate.head_major_norm_reader_source(candidate.READER)
    assert "noc.async_read(r,raw,4096" in reader and "noc.async_read(r,raw,64" not in reader
    assert "noc.async_write_zeros(raw" not in reader
    paths = [
        Path(__file__),
        ROOT / "tt/head_major_recurrent_norm_candidate.py",
        ROOT / "tt/local_norm_gate_candidate.py",
        ROOT / "tt/local_affine_gate_candidate.py",
        ROOT / "tt/recurrent_variants.py",
        ROOT / "tt/optimized_decoder.py",
        ROOT / "tt/multichip_decoder.py",
        REPO / "tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_pack_common.h",
    ]
    proof = {
        "frozen_sha256": FROZEN_SHA,
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "writer_sha256": hashlib.sha256(new.encode()).hexdigest(),
        "norm_reader_sha256": hashlib.sha256(reader.encode()).hexdigest(),
        "norm_compute_sha256": hashlib.sha256(candidate.COMPUTE.encode()).hexdigest(),
        "proof": "Recurrence reader/compute/state-write prefix/CBs/partition unchanged; only readout row/page and owned padding change. Norm compute/writer unchanged. REDUCE_COL default pack mask zeroes rows1..31 and faces2/3; B1 full writes and B>1 tail copies use those zeros. CPU source comparison is not hardware exactness proof.",
    }
    return texts, constants, proof


def cpu_check():
    from collections import defaultdict
    from types import SimpleNamespace as NS

    texts, constants, proof = sources_and_proof()
    fake = NS(bfloat16="bf16", float32="fp32", TILE_LAYOUT="tile", ROW_MAJOR_LAYOUT="row", DRAM_MEMORY_CONFIG="dram")
    fake.CoreCoord = lambda x, y: NS(x=x, y=y)
    fake.CoreRange = lambda a, b: (a, b)
    fake.CoreRangeSet = list
    fake.RuntimeArgs = lambda: defaultdict(dict)
    for name in (
        "ComputeConfigDescriptor",
        "ReaderConfigDescriptor",
        "WriterConfigDescriptor",
        "CBDescriptor",
        "CBFormatDescriptor",
        "ProgramDescriptor",
    ):
        setattr(fake, name, lambda **kw: NS(**kw))
    fake.KernelDescriptor = lambda **kw: NS(**kw)
    fake.KernelDescriptor.SourceType = NS(SOURCE_CODE="source")
    fake.MathFidelity = NS(HiFi4="HiFi4")
    fake.UnpackToDestMode = NS(Default="default", UnpackToDestFp32="fp32")
    fake.TensorAccessorArgs = lambda t: NS(get_compile_time_args=lambda: [t.buffer_address(), t.shape[-1]])
    captured = []
    fake.generic_op = lambda tensors, desc: captured.append((tensors, desc))
    mesh = NS(compute_with_storage_grid_size=lambda: NS(x=11, y=10))

    def tensor(shape, address, dtype="bf16", layout="tile"):
        return NS(
            shape=shape,
            padded_shape=(
                shape
                if layout == "row"
                else (*shape[:-2], math.ceil(shape[-2] / 32) * 32, math.ceil(shape[-1] / 32) * 32)
            ),
            dtype=dtype,
            layout=layout,
            device=lambda: mesh,
            memory_config=lambda: "dram",
            buffer_address=lambda: address,
        )

    def node(text, name):
        return next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)

    ns = vars(candidate).copy() | constants | {"ttnn": fake, "sources": sources}
    nodes = [
        node(texts["multichip_decoder"], "_local12_head_partition"),
        node(texts["optimized_decoder"], "_fused_delta_row_compact_fp32_compute_config"),
        node(texts["multichip_decoder"], "_local12_recurrent"),
    ]
    ct = (ROOT / "tt/head_major_recurrent_norm_candidate.py").read_text()
    nodes += [node(ct, "local12_recurrent_head_major"), node(ct, "local_norm_gate_head_major")]

    class StripImports(ast.NodeTransformer):
        def visit_Import(self, n):
            return None

        def visit_ImportFrom(self, n):
            return None

    nodes = [StripImports().visit(n) for n in nodes]
    exec(
        compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), "<head-major-descriptors>", "exec"),
        ns,
    )
    checks = 0
    for b in range(1, 33):
        original_gate_zero = set()
        grouped_gate_zero = set()
        grouped_requests = 0
        for tile in range(4):
            for user in range(b, 32):
                for face in range(2):
                    start = tile * 2048 + (user // 16) * 1024 + face * 512 + (user % 16) * 32
                    original_gate_zero.update(range(start, start + 32))
            for half in range(2):
                first, end = max(b, half * 16), (half + 1) * 16
                if first < end:
                    for face in range(2):
                        start = tile * 2048 + half * 1024 + face * 512 + (first % 16) * 32
                        span = set(range(start, start + (end - first) * 32))
                        assert not grouped_gate_zero.intersection(span)
                        grouped_gate_zero.update(span)
                        grouped_requests += 1
        assert original_gate_zero == grouped_gate_zero
        assert grouped_requests <= 16
        # Every byte of the48 head-major pages has exactly one producer.
        ownership = [-1] * (48 * 4096)
        for h in range(b * 12):
            user, head = divmod(h, 12)
            spans = []
            if b == 1:
                spans = [(head * 4 + i, 0, 4096) for i in range(4)]
            else:
                spans = [
                    (head * 4 + i, ((user // 16) * 512 + face * 256 + (user % 16) * 16) * 4, 64)
                    for i in range(4)
                    for face in range(2)
                ]
                if user + 1 == b:
                    row = b
                    while row < 32:
                        limit = (row // 16 + 1) * 16
                        spans += [
                            (head * 4 + i, ((row // 16) * 512 + face * 256 + (row % 16) * 16) * 4, (limit - row) * 64)
                            for i in range(4)
                            for face in range(2)
                        ]
                        assert (limit - row) * 64 <= 1024  # entirely inside masked face2
                        row = limit
            for page, start, length in spans:
                assert start % 64 == length % 64 == 0 and 0 <= start < 4096 and start + length <= 4096
                for offset in range(page * 4096 + start, page * 4096 + start + length):
                    assert ownership[offset] == -1
                    ownership[offset] = h
        assert -1 not in ownership
        for state_dtype in ("bf16", "fp32"):
            for cores in (1, 7, 110):
                qkv = tensor((b, 1, 2560), 100, layout="row")
                g, beta = tensor((b, 1, 12), 200, "fp32"), tensor((b, 1, 12), 300)
                state = tensor((b * 12, 128, 128), 400, state_dtype)
                oldout = tensor((1, b * 12, 128), 500, "fp32")
                newout = tensor((12, b, 128), 500, "fp32")
                ns["_local12_recurrent"](qkv, qkv, qkv, g, beta, state, packed_qkv=True, output=oldout, cores=cores)
                ns["local12_recurrent_head_major"](
                    qkv, qkv, qkv, g, beta, state, packed_qkv=True, output=newout, cores=cores
                )
                (_, old), (_, new) = captured
                captured.clear()
                assert old.cbs == new.cbs and old.semaphores == new.semaphores
                assert old.kernels[0] == new.kernels[0] and old.kernels[2] == new.kernels[2]
                ow, nw = old.kernels[1], new.kernels[1]
                assert ow.runtime_args == nw.runtime_args and ow.compile_time_args == nw.compile_time_args
                assert nw.kernel_source == candidate.head_major_writer_source(ow.kernel_source)
                for mode in ("head", "tile"):
                    ns["local_norm_gate_head_major"](
                        newout,
                        tensor((1, 1, 128), 600),
                        tensor((1, b, 4160), 700),
                        output=tensor((1, b, 1536), 800),
                        ownership=mode,
                        cores=cores,
                    )
                    _, norm = captured.pop()
                    assert (
                        norm.kernels[2].kernel_source == candidate.COMPUTE
                        and norm.kernels[1].kernel_source == candidate.WRITER
                    )
                    assert sum(cb.total_size for cb in norm.cbs) == 86016
                    pages = []
                    for start, count in candidate.task_partition(mode, 110, cores):
                        for task in range(start, start + count):
                            head = task // 4 if mode == "tile" else task
                            assert 0 <= head < 12
                            pages.extend([task] if mode == "tile" else range(head * 4, head * 4 + 4))
                    assert sorted(pages) == list(range(48))
                checks += 1
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return {
        "status": "pass",
        "scope": "CPU source/descriptor/ownership only",
        "recurrence_descriptor_cases": checks,
        "norm_descriptor_cases": checks * 2,
        "scatter_batches": 32,
        "norm_cb_bytes_per_core": 86016,
        "readout_bytes": {"head_major": 196608, "compact_B1": 16384, "compact_B32": 196608},
        **proof,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--state-dtypes", nargs="+", choices=["bf16", "fp32"], default=["bf16", "fp32"])
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=["normal", "zero_qkv", "zero_state", "scale"],
        default=["normal", "zero_qkv", "zero_state", "scale"],
    )
    ap.add_argument("--ownership", nargs="+", choices=["head", "tile"], default=["head", "tile"])
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--cores", type=int)
    ap.add_argument("--norm-cores", type=int)
    ap.add_argument("--low-cores", type=int, default=1)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--epsilon", type=float, default=1e-6)
    ap.add_argument("--require-exact", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert (
        args.output
        and not args.output.exists()
        and args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    )
    assert args.steps >= 2 and min(args.queued, args.replays, args.warmup, args.samples) > 0 and args.low_cores >= 0
    assert all(1 <= b <= 32 for b in args.batches) and math.isfinite(args.epsilon) and args.epsilon > 0
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    assert not (watcher and profiler)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import _local12_recurrent
    from models.autoports.qwen_qwen3_6_27b.tt.local_affine_gate_candidate import local_affine_gate
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "arguments": vars(args) | {"output": str(args.output)},
        "mesh": [1, 4],
        "source_proof": sources_and_proof()[2],
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "timings_skipped": watcher or profiler or tracking,
        "cases": [],
        "comparison": "Full native recurrence+typecast+RMS+local affine gate vs head-major recurrence+direct fused norm/gate; state update counts matched. No restore, copies or reset inside timed traces.",
        "trace_contract": "All external allocations and all program/copy warmups before first capture; native transient RMS allocation inside capture; changed-state trace replay and input updates use retained buffers.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )

        def upload(values, dtype, layout=None):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=layout or ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def allocate(shape, dtype):
            return ttnn.empty(
                shape, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def diag(actual, expected, batch):
            rows = []
            for rank, (a, b) in enumerate(zip(actual, expected)):
                assert a.shape == b.shape and a.dtype == b.dtype
                a, b = a.reshape(batch, -1).contiguous(), b.reshape(batch, -1).contiguous()
                integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
                for user in range(batch):
                    x, y = a[user], b[user]
                    finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                    bits = int((x.view(integer) != y.view(integer)).sum())
                    score = (1.0 if torch.equal(x, y) else pcc(x, y)) if finite else -1.0
                    rows.append(
                        {
                            "rank": rank,
                            "user": user,
                            "pcc": score,
                            "bits": bits,
                            "finite": finite,
                            "max_abs": float((x.float() - y.float()).abs().max()) if finite else None,
                        }
                    )
            return {
                "minimum_pcc": min(x["pcc"] for x in rows),
                "bit_mismatches": sum(x["bits"] for x in rows),
                "finite": all(x["finite"] for x in rows),
                "maximum_abs": max(x["max_abs"] or 0 for x in rows),
                "details": [x for x in rows if x["bits"] or not x["finite"]],
            }

        def check(d, exact=False):
            assert d["finite"] and d["minimum_pcc"] >= 0.995, d
            if exact:
                assert d["bit_mismatches"] == 0, d

        def run_case(batch, state_name, mode, cores, norm_cores):
            names = ["native", *args.ownership]
            npaths = len(names)
            entry = {
                "batch": batch,
                "state_dtype": state_name,
                "mode": mode,
                "cores": cores,
                "norm_cores": norm_cores,
                "eager": [],
                "trace": [],
                "readout_bytes": {"native": math.ceil(batch * 12 / 32) * 4 * 4096, "head_major": 48 * 4096},
            }
            report["cases"].append(entry)
            save()
            print(f"HEAD_MAJOR B{batch} {state_name} {mode} cores={cores} norm_cores={norm_cores}", flush=True)
            dtype = ttnn.bfloat16 if state_name == "bf16" else ttnn.float32
            tdtype = torch.bfloat16 if state_name == "bf16" else torch.float32
            gs = [torch.Generator().manual_seed(92921 + batch * 73 + r) for r in range(4)]
            initial = [
                (torch.randn(batch * 12, 128, 128, generator=g) * 0.015625 + r * 0.001).to(tdtype)
                for r, g in enumerate(gs)
            ]
            if mode == "zero_state":
                initial = [t.zero_() for t in initial]
            initial_t = upload(initial, dtype)
            bank = []
            users = torch.arange(batch).reshape(batch, 1, 1).float()
            heads = torch.arange(12).reshape(1, 1, 12).float()
            for step in range(args.steps):
                qkvs = []
                decays = []
                betas = []
                weights = []
                packed = []
                for rank, gen in enumerate(gs):
                    qkv = torch.randn(batch, 1, 2560, generator=gen) * 0.125 + users * 0.003 + rank * 0.01 + step * 0.02
                    if mode == "zero_qkv":
                        qkv.zero_()
                    if mode == "scale":
                        qkv *= torch.tensor([0.01, 1.0, 100.0])[torch.arange(batch) % 3].reshape(batch, 1, 1)
                    qkvs.append(qkv.bfloat16())
                    decays.append(-0.02 - users * 0.001 - heads * 0.002 - rank * 0.003 - step * 0.004)
                    betas.append((0.125 + users * 0.005 + heads * 0.015 + rank * 0.007 + step * 0.01).bfloat16())
                    weights.append((1 + torch.randn(1, 1, 128, generator=gen) * 0.125).bfloat16())
                    packed.append(
                        (torch.randn(1, batch, 4160, generator=gen) * 0.5 + rank * 0.03125 + step * 0.125).bfloat16()
                    )
                bank.append(
                    [
                        upload(qkvs, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                        upload(decays, ttnn.float32),
                        upload(betas, ttnn.bfloat16),
                        upload(weights, ttnn.bfloat16),
                        upload(packed, ttnn.bfloat16),
                    ]
                )
            inputs = [ttnn.empty_like(t) for t in bank[0]]
            states = [ttnn.empty_like(initial_t) for _ in names]
            readouts = [
                allocate((1, batch * 12, 128) if i == 0 else (12, batch, 128), ttnn.float32) for i in range(npaths)
            ]
            outputs = [allocate((1, batch, 1536), ttnn.bfloat16) for _ in names]
            rounded = allocate((1, batch * 12, 128), ttnn.bfloat16)
            rebind_states = [ttnn.empty_like(initial_t) for _ in names]
            rebind_readouts = [ttnn.empty_like(t) for t in readouts]
            rebind_outputs = [ttnn.empty_like(t) for t in outputs]
            retained = [
                initial_t,
                *inputs,
                *states,
                *readouts,
                *outputs,
                rounded,
                *rebind_states,
                *rebind_readouts,
                *rebind_outputs,
                *(t for row in bank for t in row),
            ]
            traces = []

            def update(step):
                for src, dst in zip(bank[step], inputs):
                    ttnn.copy(src, dst)

            def reset(targets=states):
                for state in targets:
                    ttnn.copy(initial_t, state)

            def invoke(index, source=None, state=None, readout=None, output=None):
                qkv, g, beta, weight, packed = inputs if source is None else source
                state = states[index] if state is None else state
                readout = readouts[index] if readout is None else readout
                output = outputs[index] if output is None else output
                recurrence = _local12_recurrent if index == 0 else candidate.local12_recurrent_head_major
                recurrence(
                    qkv, qkv, qkv, g, beta, state, packed_qkv=True, output=readout, cores=cores, variant=args.variant
                )
                if index == 0:
                    ttnn.typecast(readout, ttnn.bfloat16, output_tensor=rounded)
                    norm = ttnn.rms_norm(rounded, epsilon=args.epsilon, compute_kernel_config=ckc)
                    return local_affine_gate(norm, weight, packed, output=output)
                return candidate.local_norm_gate_head_major(
                    readout,
                    weight,
                    packed,
                    epsilon=args.epsilon,
                    ownership=names[index],
                    output=output,
                    cores=norm_cores,
                )

            def snapshot():
                ro = []
                for i, t in enumerate(readouts):
                    values = read(t)
                    ro.append(
                        values
                        if i == 0
                        else [v.permute(1, 0, 2).contiguous().reshape(1, batch * 12, 128) for v in values]
                    )
                return [[read(t) for t in outputs], ro, [read(t) for t in states]]

            def compare(values, label, records, expected=None):
                row = {"label": label, "paths": []}
                records.append(row)
                for i in range(npaths):
                    if expected is None and i == 0:
                        continue
                    d = {"path": names[i]}
                    for branch, kind in enumerate(("output", "readout", "state")):
                        ref = values[branch][0] if expected is None else expected[branch][i]
                        d[kind] = diag(values[branch][i], ref, batch)
                        # Recurrence state/readout must be exact; norm output floor unchanged.
                        check(d[kind], True if expected is not None or kind != "output" else args.require_exact)
                    row["paths"].append(d)
                save()

            eager = []
            try:
                reset()
                for step in range(args.steps):
                    update(step)
                    for i in range(npaths):
                        invoke(i)
                    values = snapshot()
                    compare(values, step, entry["eager"])
                    eager.append(values)
                reset(rebind_states)
                for i in range(npaths):
                    invoke(
                        i, source=bank[0], state=rebind_states[i], readout=rebind_readouts[i], output=rebind_outputs[i]
                    )
                    d = {
                        "path": names[i],
                        "output": diag(read(rebind_outputs[i]), eager[0][0][i], batch),
                        "state": diag(read(rebind_states[i]), eager[0][2][i], batch),
                    }
                    entry.setdefault("rebind", []).append(d)
                    save()
                    check(d["output"], True)
                    check(d["state"], True)
                reset()
                for step in range(args.queued):
                    for i in range(npaths):
                        invoke(i, source=bank[step % args.steps])
                compare(snapshot(), args.queued, entry.setdefault("queued", []))
                reset()
                update(0)
                # Warm ALL paths and copies before capturing ANY path.
                for _ in range(args.warmup):
                    for i in range(npaths):
                        invoke(i)
                ttnn.synchronize_device(mesh)
                count = mesh.num_program_cache_entries()
                for i in range(npaths):
                    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                    invoke(i)
                    ttnn.end_trace_capture(mesh, tid, cq_id=0)
                    traces.append(tid)
                assert mesh.num_program_cache_entries() == count, "Program allocated during capture"
                reset()
                for step in range(args.steps):
                    update(step)
                    for tid in traces:
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    compare(snapshot(), step, entry["trace"], eager[step])
                reset()
                for step in range(args.replays):
                    update(step % args.steps)
                    for tid in traces:
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                compare(snapshot(), args.replays, entry.setdefault("changed_trace", []))
                if not report["timings_skipped"]:
                    reset()
                    update(0)
                    times = [[] for _ in names]
                    for sample in range(args.samples):
                        for i in (range(npaths) if sample % 2 == 0 else reversed(range(npaths))):
                            ttnn.synchronize_device(mesh)
                            start = time.perf_counter_ns()
                            for _ in range(args.replays):
                                ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=False)
                            ttnn.synchronize_device(mesh)
                            times[i].append((time.perf_counter_ns() - start) / 1000 / args.replays)
                    entry["warmed_traced_chain_us"] = {
                        name: {"samples": v, "median": statistics.median(v)} for name, v in zip(names, times)
                    }
                    compare(snapshot(), "timed", entry.setdefault("post_timing", []))
            finally:
                for tid in traces:
                    ttnn.release_trace(mesh, tid)
                for t in retained:
                    ttnn.deallocate(t)
            save()

        for b in args.batches:
            for dtype in args.state_dtypes:
                for mode in args.modes:
                    run_case(b, dtype, mode, args.cores, args.norm_cores)
        if args.low_cores:
            for dtype in args.state_dtypes:
                run_case(3, dtype, "normal", args.low_cores, args.low_cores)
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
