# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Serialized TP4 local beta/decay oracle and trace probe. CPU mode imports no TTNN."""

import argparse
import ast
import hashlib
import json
import math
import os
import statistics
import sys
import time
import traceback
from pathlib import Path
from models.autoports.qwen_qwen3_6_27b.tt.local_beta_decay_candidate import local_beta_decay, local_reader_source

ROOT = Path(__file__).resolve().parents[1]
FROZEN_SHA = "315187bc6315168797a38922abca3718c36fd9a5d040eae87b33c25a8f6a9d22"


def source_proof():
    path = ROOT / "tt/optimized_decoder.py"
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == FROZEN_SHA, "Frozen oracle drift"
    constants = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in (
                    "_gate178_READER",
                    "_gate178_COMPUTE",
                    "_gate178_WRITER",
                ):
                    constants[target.id] = ast.literal_eval(node.value)
    assert len(constants) == 3
    adapted = local_reader_source(constants["_gate178_READER"])
    restored = adapted.replace(
        "task=get_arg_val<uint32_t>(3),user=task,col=0", "task=get_arg_val<uint32_t>(3),user=task/2,col=task%2", 1
    )
    restored = restored.replace(".page_id=129+col", ".page_id=514+col", 1).replace(
        ".page_id=128+col", ".page_id=512+col", 1
    )
    assert restored == constants["_gate178_READER"]
    return constants, {
        "frozen_sha256": FROZEN_SHA,
        "kernel_sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in constants.items()},
        "adapted_reader_sha256": hashlib.sha256(adapted.encode()).hexdigest(),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), ROOT / "tt/local_beta_decay_candidate.py", ROOT / "tt/optimized_decoder.py")
        },
    }


def cpu_check():
    from collections import defaultdict
    from types import SimpleNamespace as NS

    constants, proof = source_proof()
    fake = NS(bfloat16="bf16", float32="fp32", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram")
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
    ):
        setattr(fake, name, lambda **kw: NS(**kw))
    fake.KernelDescriptor = lambda **kw: NS(**kw)
    fake.KernelDescriptor.SourceType = NS(SOURCE_CODE="source")
    fake.MathFidelity = NS(HiFi4="HiFi4")
    fake.UnpackToDestMode = NS(Default="default", UnpackToDestFp32="fp32")
    fake.TensorAccessorArgs = lambda t: NS(get_compile_time_args=lambda: [t.buffer_address()])
    captured = []
    fake.generic_op = lambda tensors, desc: captured.append((tensors, desc))
    mesh = NS(compute_with_storage_grid_size=lambda: NS(x=11, y=10))

    def tensor(shape, address, dtype):
        return NS(
            shape=shape,
            padded_shape=(*shape[:-2], math.ceil(shape[-2] / 32) * 32, math.ceil(shape[-1] / 32) * 32),
            dtype=dtype,
            layout="tile",
            device=lambda: mesh,
            memory_config=lambda: "dram",
            buffer_address=lambda: address,
        )

    node = next(
        n
        for n in ast.parse((ROOT / "tt/local_beta_decay_candidate.py").read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "local_beta_decay"
    )
    node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    ns = constants | {"ttnn": fake, "local_reader_source": local_reader_source}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<beta-decay-descriptor>", "exec"), ns)
    cases = 0
    for b in range(1, 33):
        for stage in (0, 1, 2):
            inputs = [
                tensor((1, b, 4160), 100, "bf16"),
                tensor((1, 1, 12), 200, "fp32"),
                tensor((1, 1, 12), 300, "fp32"),
            ]
            outputs = (tensor((b, 1, 12), 400, "bf16"), tensor((b, 1, 12), 500, "fp32"))
            assert ns["local_beta_decay"](*inputs, outputs=outputs, diagnostic_stage=stage) == outputs
            tensors, desc = captured.pop()
            reader, writer, compute = desc.kernels
            assert (
                compute.kernel_source == constants["_gate178_COMPUTE"]
                and writer.kernel_source == constants["_gate178_WRITER"]
            )
            assert compute.compile_time_args == [stage]
            assert [i for i, v in enumerate(compute.config.unpack_to_dest_mode) if v == "fp32"] == [1, 3]
            assert sum(cb.total_size for cb in desc.cbs) == 24576
            tasks = [task for row in writer.runtime_args.values() for _, _, task in row.values()]
            assert sorted(tasks) == list(range(b))
            for user in range(b):
                for face in range(2):
                    src = (user // 16) * 1024 + face * 512 + (user % 16) * 32
                    assert (src & ~63) + 64 <= 2048
                    assert (src & ~63) + (user % 2) * 32 == src
                for channel in range(12):
                    face, local = divmod(channel, 16)
                    assert face == 0
                    src = (user // 16) * 1024 + face * 512 + (user % 16) * 32
                    selected = (src & ~63) + (user % 2) * 32 + local * 2
                    assert selected == src + channel * 2
                    # Pages128/129 hold B/A, and one output page belongs to user.
                    assert 128 < 130 and 129 < 130 and user < b
            cases += 1
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    return {
        "status": "pass",
        "scope": "CPU source/indexing/descriptor only",
        "cases": cases,
        "cb_bytes_per_core": 24576,
        **proof,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--stages", type=int, nargs="+", choices=[0, 1, 2], default=[0, 1, 2])
    ap.add_argument("--modes", nargs="+", choices=["normal", "extreme"], default=["normal", "extreme"])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--replays", type=int, default=50)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--require-exact", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert args.output and not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert (
        all(1 <= b <= 32 for b in args.batches) and args.steps >= 2 and min(args.warmup, args.replays, args.samples) > 0
    )
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracking = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1"
    assert not (watcher and profiler)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "arguments": vars(args) | {"output": str(args.output)},
        "source_proof": source_proof()[1],
        "watcher": watcher,
        "profiler": profiler,
        "allocation_tracking": tracking,
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "timings_skipped": watcher or profiler or tracking,
        "cases": [],
        "trace_contract": "All retained inputs/outputs and both program warmups precede first capture; native transient intermediates are scoped inside capture; no allocation exemptions.",
        "input_contract": "Synthetic BF16 packed projections; independently generated actual FP32 dt_bias/negativeA, no BF16 staging of parameters.",
        "timing_limitation": "Native reference copies its final reshaped beta/g into preallocated outputs (two extra copy ops); candidate writes outputs directly. Microtimings are provisional; whole-layer comparison is required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)

        def upload(host, dtype):
            return ttnn.from_torch(
                torch.cat(host, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def diagnostics(actual, expected, batch):
            details = []
            for branch in range(2):
                for rank, (a, b) in enumerate(zip(actual[branch], expected[branch])):
                    assert a.dtype == b.dtype and a.shape == b.shape
                    integer = torch.int16 if branch == 0 else torch.int32
                    for user in range(batch):
                        x, y = a[user].reshape(-1), b[user].reshape(-1)
                        finite = bool(torch.isfinite(x).all() and torch.isfinite(y).all())
                        bits = int((x.view(integer) != y.view(integer)).sum())
                        score = pcc(x, y) if finite else -1.0
                        details.append(
                            {
                                "branch": "beta" if branch == 0 else "g",
                                "rank": rank,
                                "user": user,
                                "pcc": score,
                                "bits": bits,
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
                for stage in args.stages:
                    entry = {"batch": batch, "mode": mode, "stage": stage, "eager": [], "trace": []}
                    report["cases"].append(entry)
                    save()
                    print(f"BETA_DECAY B{batch} {mode} stage{stage}", flush=True)
                    generators = [torch.Generator().manual_seed(51932 + batch * 19 + r) for r in range(4)]
                    bank = []
                    for step in range(args.steps):
                        packed_host, bias_host, neg_host = [], [], []
                        for rank, generator in enumerate(generators):
                            p = (
                                torch.randn(1, batch, 4160, generator=generator) * 0.5 + step * 0.125 + rank * 0.03125
                            ).bfloat16()
                            d = (
                                torch.randn(1, 1, 12, generator=generator) * 0.125
                                + torch.arange(12).reshape(1, 1, 12) * 0.00123
                            )
                            n = -torch.exp(torch.randn(1, 1, 12, generator=generator) * 0.25)
                            if mode == "extreme":
                                grid = torch.tensor(
                                    [-80.0, -40.0, -20.0, -2.0, -0.01, 0.0, 0.01, 2.0, 19.5, 20.0, 40.0, 80.0]
                                )
                                p[:, :, 4096:4108] = (grid + step * 0.03125).bfloat16()
                                p[:, :, 4128:4140] = (grid.flip(0) + rank * 0.0625 + step * 0.125).bfloat16()
                            packed_host.append(p)
                            bias_host.append(d.contiguous())
                            neg_host.append(n.contiguous())
                        bank.append(
                            [
                                upload(packed_host, ttnn.bfloat16),
                                upload(bias_host, ttnn.float32),
                                upload(neg_host, ttnn.float32),
                            ]
                        )
                    inputs = [ttnn.empty_like(t) for t in bank[0]]
                    outputs = [
                        tuple(
                            ttnn.empty(
                                (batch, 1, 12),
                                dtype=dt,
                                layout=ttnn.TILE_LAYOUT,
                                device=mesh,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                            for dt in (ttnn.bfloat16, ttnn.float32)
                        )
                        for _ in range(2)
                    ]
                    allocations = [
                        *inputs,
                        *(t for pair in outputs for t in pair),
                        *(t for fixture in bank for t in fixture),
                    ]
                    traces = []

                    def update(step):
                        for src, dst in zip(bank[step], inputs):
                            ttnn.copy(src, dst)

                    def invoke(index):
                        if index:
                            return local_beta_decay(*inputs, outputs=outputs[1], diagnostic_stage=stage)
                        p, d, n = inputs
                        beta = ttnn.sigmoid(p[:, :, 4096:4108])
                        value = ttnn.add(p[:, :, 4128:4140], d, dtype=ttnn.float32)
                        if stage != 1:
                            value = ttnn.softplus(value)
                        if stage == 0:
                            value = ttnn.multiply(n, value)
                        for src, dst in zip((beta, value), outputs[0]):
                            ttnn.copy(ttnn.reshape(src, (batch, 1, 12)), dst)
                        return outputs[0]

                    eager = []
                    try:
                        for step in range(args.steps):
                            update(step)
                            for index in range(2):
                                invoke(index)
                            values = [[read(t) for t in pair] for pair in outputs]
                            d = diagnostics(values[1], values[0], batch)
                            entry["eager"].append({"step": step, **d})
                            save()
                            check(d, args.require_exact)
                            eager.append(values)
                            assert all(
                                torch.equal(a, b) for t, s in zip(inputs, bank[step]) for a, b in zip(read(t), read(s))
                            )
                        local_beta_decay(*bank[0], outputs=outputs[1], diagnostic_stage=stage)
                        d = diagnostics([read(t) for t in outputs[1]], eager[0][1], batch)
                        entry["rebind"] = d
                        save()
                        check(d, True)
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
                        for step in range(args.steps):
                            update(step)
                            for index, trace in enumerate(traces):
                                for _ in range(args.replays):
                                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                d = diagnostics([read(t) for t in outputs[index]], eager[step][index], batch)
                                entry["trace"].append({"step": step, "path": "candidate" if index else "native", **d})
                                save()
                                check(d, True)
                        if not report["timings_skipped"]:
                            update(0)
                            times = [[], []]
                            for sample in range(args.samples):
                                for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                                    ttnn.synchronize_device(mesh)
                                    before = time.perf_counter_ns()
                                    for _ in range(args.replays):
                                        ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                                    ttnn.synchronize_device(mesh)
                                    times[index].append((time.perf_counter_ns() - before) / 1000 / args.replays)
                            entry["warmed_traced_us"] = {
                                name: {"samples": values, "median": statistics.median(values)}
                                for name, values in zip(("native", "candidate"), times)
                            }
                    finally:
                        for trace in traces:
                            ttnn.release_trace(mesh, trace)
                    assert allocations
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
