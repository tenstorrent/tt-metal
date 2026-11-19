# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact TP4 affine/gate/repack probe; --cpu-check never imports TTNN/torch."""

import argparse
import ast
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

from models.autoports.qwen_qwen3_6_27b.tt.local_affine_gate_candidate import local_affine_gate, local_affine_sources

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "8502f42bca20dcad1507fbc8cebf7032220b7d8e"
PREFIX = "_affinev3__GDN_AFFINE_GATE_"


def constant(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise AssertionError(name)


def source_proof():
    path = ROOT / "tt/optimized_decoder.py"
    current = path.read_text()
    relative = "models/autoports/qwen_qwen3_6_27b/tt/optimized_decoder.py"
    frozen = subprocess.check_output(["git", "show", f"{FROZEN}:{relative}"], text=True)
    result = {}
    for kind in ("READER", "COMPUTE", "WRITER"):
        source = constant(current, PREFIX + kind)
        assert source == constant(frozen, PREFIX + kind), f"Frozen {kind} drift"
        result[kind.lower() + "_sha256"] = hashlib.sha256(source.encode()).hexdigest()
        if kind == "READER":
            adapted = local_affine_sources(source)
            restored = adapted.replace("source_head=user*12+head", "source_head=user*48+head", 1)
            restored = restored.replace("row=(user*12+head)%32", "row=(user*48+head)%32", 1)
            restored = restored.replace(".page_id=80+tile", ".page_id=320+tile", 1)
            assert restored == source
            result["local_reader_sha256"] = hashlib.sha256(adapted.encode()).hexdigest()
    return result


def hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (
            Path(__file__).resolve(),
            ROOT / "tt/local_affine_gate_candidate.py",
            ROOT / "tt/optimized_decoder.py",
        )
    }


def cpu_check():
    proof = source_proof()
    for batch in range(1, 33):
        seen = set()
        for tile in range(48):
            head, column = divmod(tile, 4)
            assert 80 + tile < 4160 // 32
            for user in range(batch):
                source_head = user * 12 + head
                page, row = (source_head // 32) * 4 + column, source_head % 32
                assert page < ((batch * 12 + 31) // 32) * 4
                for face in range(2):
                    src = (row // 16) * 1024 + face * 512 + (row % 16) * 32
                    dst = (user // 16) * 1024 + face * 512 + (user % 16) * 32
                    assert src + 32 <= 2048 and dst + 32 <= 2048 and (src & ~63) + 64 <= 2048
                    assert user * 128 + face * 64 + 64 <= 4096
                    for lane in range(16):
                        key = user, head, column * 32 + face * 16 + lane
                        assert key not in seen
                        seen.add(key)
        assert len(seen) == batch * 1536
        for cores in (1, 3, 8, 32, 48, 110):
            workers = min(48, cores)
            counts = [48 // workers + (i < 48 % workers) for i in range(workers)]
            assert sum(counts) == 48 and min(counts) >= 1
    # Execute the actual wrapper body with descriptor-only stubs, stripping
    # only its two imports. This checks host partition/config construction.
    from collections import defaultdict
    from types import SimpleNamespace

    text = (ROOT / "tt/local_affine_gate_candidate.py").read_text()
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "local_affine_gate")
    node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    fake = SimpleNamespace(bfloat16="bf16", float32="fp32", TILE_LAYOUT="tile", DRAM_MEMORY_CONFIG="dram")
    fake.CoreCoord = lambda x, y: SimpleNamespace(x=x, y=y)
    fake.CoreRange = lambda start, end: (start, end)
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
        setattr(fake, name, lambda **kwargs: SimpleNamespace(**kwargs))
    fake.KernelDescriptor = lambda **kwargs: SimpleNamespace(**kwargs)
    fake.KernelDescriptor.SourceType = SimpleNamespace(SOURCE_CODE="source")
    fake.MathFidelity = SimpleNamespace(HiFi4="HiFi4")
    fake.TensorAccessorArgs = lambda tensor: SimpleNamespace(get_compile_time_args=lambda: [tensor.buffer_address()])
    captured = []
    fake.generic_op = lambda tensors, descriptor: captured.append((tensors, descriptor))
    mesh = SimpleNamespace(compute_with_storage_grid_size=lambda: SimpleNamespace(x=11, y=10))

    def tensor(shape, address):
        return SimpleNamespace(
            shape=shape,
            padded_shape=(1, ((shape[1] + 31) // 32) * 32, shape[2]),
            dtype="bf16",
            layout="tile",
            memory_config=lambda: "dram",
            buffer_address=lambda: address,
            device=lambda: mesh,
        )

    frozen_text = (ROOT / "tt/optimized_decoder.py").read_text()
    namespace = {"ttnn": fake, "local_affine_sources": local_affine_sources}
    namespace.update({PREFIX + kind: constant(frozen_text, PREFIX + kind) for kind in ("READER", "COMPUTE", "WRITER")})
    exec(compile(ast.Module(body=[node], type_ignores=[]), "<descriptor-only-affine>", "exec"), namespace)
    for batch in range(1, 33):
        inputs = [tensor((1, batch * 12, 128), 100), tensor((1, 1, 128), 200), tensor((1, batch, 4160), 300)]
        output = tensor((1, batch, 1536), 400)
        for cores in (1, 3, 8, 32, 48, 110):
            assert namespace["local_affine_gate"](*inputs, output=output, cores=cores) is output
            _, descriptor = captured.pop()
            reader, writer, compute = descriptor.kernels
            counts = [row for rows in writer.runtime_args.values() for row in rows.values()]
            assert sorted(i for _, start, count in counts for i in range(start, start + count)) == list(range(48))
            assert compute.kernel_source == namespace[PREFIX + "COMPUTE"]
            assert writer.kernel_source == namespace[PREFIX + "WRITER"]
            assert [cb.total_size for cb in descriptor.cbs] == [2048, 2048, 2048, 8192, 2048]
            assert compute.config.fp32_dest_acc_en and not compute.config.math_approx_mode
    assert "ttnn" not in sys.modules and "torch" not in sys.modules
    print(json.dumps({"status": "pass", "batches": 32, "scope": "frozen source/CPU indexing only", "sources": proof}))


def reference(normalized, weight, packed, *, output):
    """The original six-op region, with only its final output preallocated."""
    import ttnn

    batch = packed.shape[1]
    affine = ttnn.multiply(normalized, weight, dtype=ttnn.bfloat16)
    flat = ttnn.reshape(affine, (1, batch, 1536))
    gate = ttnn.typecast(packed[:, :, 2560:4096], ttnn.float32)
    flat32 = ttnn.typecast(flat, ttnn.float32)
    return ttnn.multiply(
        gate, flat32, dtype=ttnn.bfloat16, input_tensor_a_activations=[ttnn.UnaryOpType.SILU], output_tensor=output
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batches", nargs="+", type=int, default=list(range(1, 33)))
    ap.add_argument(
        "--modes", nargs="+", choices=["normal", "zero_norm", "zero_gate", "extreme_gate"], default=["normal"]
    )
    ap.add_argument("--cores", type=int)
    ap.add_argument("--low-cores", type=int, default=1, help="Extra B3 wrap case; 0 disables")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timing-samples", type=int, default=7)
    ap.add_argument("--replays-per-sample", type=int, default=10)
    ap.add_argument("--physical-device-ids", nargs=4, type=int)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.cores is None or args.cores >= 1
    assert args.steps >= 2 and min(args.warmup, args.timing_samples, args.replays_per_sample) >= 1
    assert args.low_cores >= 0
    assert not args.output.exists(), "Use a fresh output path"
    proof = source_proof()
    import torch
    import ttnn

    torch.set_num_threads(8)
    report = {
        "status": "running",
        "arguments": vars(args) | {"output": str(args.output)},
        "source_proof": proof,
        "source_sha256": hashes(),
        "cases": [],
        "scope": "BF16 affine + compact tile repack + FP32 gate SiLU/multiply; RMSNorm and recurrence excluded",
        "trace_lifetime": "all inputs/outputs allocated and copies/both programs warmed before first capture",
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def upload(values):
        return ttnn.from_torch(
            torch.cat(values, -1).contiguous(),
            device=mesh,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def read(tensor):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

    def check(actual, expected, batch, label, entry):
        checks = []
        assert len(actual) == len(expected) == 4
        for rank, (a, b) in enumerate(zip(actual, expected)):
            assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
            a, b = a.reshape(batch, 12, 128), b.reshape(batch, 12, 128)
            for user in range(batch):
                checks.append(
                    {
                        "rank": rank,
                        "user": user,
                        "head_bit_mismatches": [
                            int((a[user, h].view(torch.int16) != b[user, h].view(torch.int16)).sum()) for h in range(12)
                        ],
                        "finite": bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                    }
                )
        entry[label] = checks
        save()
        assert all(c["finite"] and not any(c["head_bit_mismatches"]) for c in checks), label

    def run_case(batch, cores, mode):
        entry = {"batch": batch, "cores": cores, "mode": mode, "eager": [], "trace": []}
        report["cases"].append(entry)
        save()
        print(f"AFFINE begin B{batch} cores={cores} {mode}", flush=True)
        generators = [torch.Generator().manual_seed(20261113 + batch * 97 + rank) for rank in range(4)]
        bank = []
        for step in range(args.steps):
            norms, weights, packed = [], [], []
            for rank, generator in enumerate(generators):
                n = (
                    torch.randn(1, batch * 12, 128, generator=generator) * 0.25
                    + torch.arange(batch * 12).reshape(1, -1, 1) * 0.001
                    + rank * 0.03125
                    + step * 0.0625
                ).bfloat16()
                w = (torch.randn(1, 1, 128, generator=generator) * 0.125 + 1).bfloat16()
                p = (
                    torch.randn(1, batch, 4160, generator=generator) * 0.5
                    + torch.arange(batch).reshape(1, -1, 1) * 0.03125
                    + rank * 0.0625
                    + step * 0.125
                ).bfloat16()
                if mode == "zero_norm":
                    n.zero_()
                if mode == "zero_gate":
                    p[:, :, 2560:4096].zero_()
                if mode == "extreme_gate":
                    p[:, :, 2560:4096] = ((torch.arange(1536) % 2) * 64 - 32).bfloat16()
                norms.append(n)
                weights.append(w)
                packed.append(p)
            bank.append([upload(values) for values in (norms, weights, packed)])
        inputs = [ttnn.empty_like(t) for t in bank[0]]
        outputs = [
            upload([torch.full((1, batch, 1536), float("nan"), dtype=torch.bfloat16) for _ in range(4)])
            for _ in range(2)
        ]
        allocations = [*inputs, *outputs, *(tensor for step in bank for tensor in step)]
        traces = []

        def set_input(step):
            for src, dst in zip(bank[step], inputs):
                ttnn.copy(src, dst)

        def invoke(index):
            result = (
                reference(*inputs, output=outputs[index])
                if index == 0
                else local_affine_gate(*inputs, output=outputs[index], cores=cores)
            )
            assert result.buffer_address() == outputs[index].buffer_address()

        try:
            eager = []
            for step in range(args.steps):
                set_input(step)
                invoke(0)
                invoke(1)
                values = [read(t) for t in outputs]
                record = {"step": step}
                entry["eager"].append(record)
                check(values[1], values[0], batch, "output", record)
                eager.append(values[0])
            set_input(0)
            for index in range(2):
                for _ in range(args.warmup):
                    invoke(index)
            ttnn.synchronize_device(mesh)
            cache_entries = mesh.num_program_cache_entries()
            for index in range(2):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                invoke(index)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
            assert mesh.num_program_cache_entries() == cache_entries, "Unexpected program allocation during capture"
            for step in range(args.steps):
                set_input(step)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                record = {"step": step}
                entry["trace"].append(record)
                for index, name in enumerate(("baseline", "candidate")):
                    check(read(outputs[index]), eager[step], batch, name + "_vs_eager", record)
            set_input(0)
            times = [[], []]
            for sample in range(args.timing_samples):
                for index in ((0, 1) if sample % 2 == 0 else (1, 0)):
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter_ns()
                    for _ in range(args.replays_per_sample):
                        ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times[index].append((time.perf_counter_ns() - begin) / 1000 / args.replays_per_sample)
            entry["trace_us"] = {
                name: {"samples": values, "median": statistics.median(values)}
                for name, values in zip(("baseline", "candidate"), times)
            }
            for index, name in enumerate(("baseline", "candidate")):
                check(read(outputs[index]), eager[0], batch, name + "_timed_output", entry)
            assert mesh.num_program_cache_entries() == cache_entries, "Unexpected program allocation during replay"
            entry["pass"] = True
            save()
            print(f"AFFINE pass B{batch} cores={cores} {mode}", flush=True)
        finally:
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            for tensor in allocations:
                ttnn.deallocate(tensor)

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        kwargs = {"trace_region_size": 16 * 1024 * 1024}
        if args.physical_device_ids:
            kwargs["physical_device_ids"] = args.physical_device_ids
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), **kwargs)
        cases = [(batch, args.cores) for batch in args.batches]
        if args.low_cores and (3, args.low_cores) not in cases:
            cases.append((3, args.low_cores))
        for batch, cores in cases:
            for mode in args.modes:
                run_case(batch, cores, mode)
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
    if sys.argv[1:] == ["--cpu-check"]:
        cpu_check()
    else:
        main()
