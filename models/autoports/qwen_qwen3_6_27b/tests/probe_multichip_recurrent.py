# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized device probe of frozen local12 baseline and source variants.

Run under a bounded timeout. Defaults cover tail batches, one-core user/ring
wraps, BF16/FP32 state, changed-input traces, zero controls and matched timings.
No device reset or trace-allocation exemptions are performed.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

from models.autoports.qwen_qwen3_6_27b.tt.recurrent_variants import VARIANTS, sources

ROOT = Path(__file__).resolve().parents[1]
FROZEN = "8502f42bca20dcad1507fbc8cebf7032220b7d8e"


def constant(text, name):
    for node in ast.parse(text).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing source constant {name}")


def frozen_sources(commit):
    files = {}
    for name in ("multichip_decoder", "optimized_decoder"):
        path = f"models/autoports/qwen_qwen3_6_27b/tt/{name}.py"
        files[name] = subprocess.check_output(["git", "show", f"{commit}:{path}"], text=True)
    return (
        constant(files["multichip_decoder"], "_LOCAL12_READER"),
        constant(files["optimized_decoder"], "_fused_delta_row_GDN_COMPACT_COMPUTE"),
        files,
    )


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS[1:7]))
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 2, 3, 10, 17, 31, 32])
    ap.add_argument("--state-dtypes", nargs="+", choices=["bf16", "fp32"], default=["bf16", "fp32"])
    ap.add_argument(
        "--modes", nargs="+", choices=["normal", "zero_qkv", "zero_state"], default=["normal", "zero_qkv", "zero_state"]
    )
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--cores", type=int, help="Override regular-case core count")
    ap.add_argument("--low-cores", type=int, default=1, help="Additional B3 ring/user-wrap case; 0 disables")
    ap.add_argument("--grouped-qk", choices=["on", "off"], default="on")
    ap.add_argument("--poison-dead-padding", action="store_true", help="Correctness only; disables timings")
    ap.add_argument("--mesh-cols", type=int, choices=[1, 4], default=4)
    ap.add_argument("--physical-device-ids", nargs="+", type=int)
    ap.add_argument("--reference-commit", default=FROZEN)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--timing-samples", type=int, default=7)
    ap.add_argument("--replays-per-sample", type=int, default=10)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.steps >= 2 and args.warmup >= 1 and args.timing_samples >= 1 and args.replays_per_sample >= 1
    assert args.cores is None or args.cores >= 1
    assert args.low_cores >= 0
    assert not args.physical_device_ids or len(args.physical_device_ids) == args.mesh_cols
    reader, compute, frozen = frozen_sources(args.reference_commit)
    # Importing TTNN is deliberately deferred to the actual device entrypoint.
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt import multichip_decoder as local

    current_optimized = (ROOT / "tt/optimized_decoder.py").read_text()
    for name in (
        "_fused_delta_row_GDN_COMPACT_WRITER",
        "_fused_delta_row_GDN_FUSE_ACCUMULATOR",
        "_fused_delta_row_GDN_HOIST_DELTA",
        "_fused_delta_row_GDN_HOIST_SCALAR_INIT",
        "_fused_delta_row_GDN_HOIST_COL_INIT",
        "_fused_delta_row_GDN_FUSE_DECAY_PREDICT",
    ):
        assert constant(current_optimized, name) == constant(frozen["optimized_decoder"], name), name
    torch.set_num_threads(8)
    report = {
        "status": "running",
        "arguments": vars(args) | {"output": str(args.output)},
        "frozen_reader_sha256": digest(reader),
        "frozen_compute_sha256": digest(compute),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p.read_bytes())
            for p in (
                Path(__file__),
                ROOT / "tt/recurrent_variants.py",
                ROOT / "tt/multichip_decoder.py",
                ROOT / "tt/optimized_decoder.py",
            )
        },
        "environment": {
            k: os.environ.get(k)
            for k in ("TT_METAL_WATCHER", "TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_DEVICE_PROFILER")
        },
        "interface": "raw packed QKV only",
        "timing_kind": "serialized synchronized trace wall time",
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def upload(values, dtype, layout=None):
        return ttnn.from_torch(
            torch.cat(values, dim=-1).contiguous(),
            device=mesh,
            dtype=dtype,
            layout=layout or ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
        )

    def read(tensor):
        return [ttnn.to_torch(t).contiguous() for t in ttnn.get_device_tensors(tensor)]

    def check(left, right, batch, label, record):
        checks = []
        for rank, (a, b) in enumerate(zip(left, right)):
            assert a.shape == b.shape and a.dtype == b.dtype
            integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
            a, b = a.reshape(batch, -1), b.reshape(batch, -1)
            for user in range(batch):
                checks.append(
                    {
                        "rank": rank,
                        "user": user,
                        "bit_mismatches": int((a[user].view(integer) != b[user].view(integer)).sum()),
                        "finite": bool(torch.isfinite(a[user]).all() and torch.isfinite(b[user]).all()),
                    }
                )
        record[label] = checks
        save()
        assert all(c["bit_mismatches"] == 0 and c["finite"] for c in checks), label

    def run_case(batch, state_name, mode, cores, variant):
        group, partition = local._local12_head_partition(batch, grid.x * grid.y, cores, args.grouped_qk == "on")
        candidate = sources(reader, compute, variant, grouped=group, poison_dead_padding=args.poison_dead_padding)
        entry = {
            "batch": batch,
            "state_dtype": state_name,
            "mode": mode,
            "cores": cores,
            "variant": variant,
            "grouped_qk": group,
            "partition": partition,
            "candidate_sha256": [digest(s) for s in candidate],
            "eager": [],
            "trace": [],
        }
        report["cases"].append(entry)
        save()
        print(f"RECURRENT begin B{batch} {state_name} {mode} cores={cores} {variant}", flush=True)
        dtype = ttnn.bfloat16 if state_name == "bf16" else ttnn.float32
        torch_dtype = torch.bfloat16 if state_name == "bf16" else torch.float32
        generators = [torch.Generator().manual_seed(20261113 + batch * 97 + r) for r in range(args.mesh_cols)]
        users = torch.arange(batch).reshape(batch, 1, 1).float()
        channels = torch.arange(2560).reshape(1, 1, 2560).float() % 131 / 1024
        heads = torch.arange(12).reshape(1, 1, 12).float()
        initial = [
            (torch.randn(batch * 12, 128, 128, generator=g) * 0.015625 + r * 0.001).to(torch_dtype)
            for r, g in enumerate(generators)
        ]
        if mode == "zero_state":
            initial = [x.zero_() for x in initial]
        initial_t = upload(initial, dtype)
        states = [upload(initial, dtype) for _ in range(2)]
        outputs = [upload([torch.zeros(1, batch * 12, 128) for _ in generators], ttnn.float32) for _ in range(2)]
        bank = []
        for step in range(args.steps):
            packed = [
                (
                    torch.randn(batch, 1, 2560, generator=g) * 0.125 + users * 0.003 + channels + r * 0.01 + step * 0.02
                ).to(torch.bfloat16)
                for r, g in enumerate(generators)
            ]
            if mode == "zero_qkv":
                packed = [x.zero_() for x in packed]
            gates = [-0.02 - users * 0.001 - heads * 0.002 - r * 0.003 - step * 0.004 for r in range(args.mesh_cols)]
            betas = [
                (0.125 + users * 0.005 + heads * 0.015 + r * 0.007 + step * 0.01).to(torch.bfloat16)
                for r in range(args.mesh_cols)
            ]
            bank.append(
                (
                    upload(packed, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
                    upload(gates, ttnn.float32),
                    upload(betas, ttnn.bfloat16),
                )
            )
        inputs = [ttnn.empty_like(t) for t in bank[0]]
        assert tuple(inputs[0].padded_shape) == (batch, 1, 2560)
        assert tuple(inputs[1].padded_shape) == tuple(inputs[2].padded_shape) == (batch, 32, 32)
        all_allocations = [initial_t, *states, *outputs, *inputs, *(t for row in bank for t in row)]
        traces = []

        def set_input(step):
            for src, dst in zip(bank[step], inputs):
                ttnn.copy(src, dst)

        def reset():
            for state in states:
                ttnn.copy(initial_t, state)

        def invoke(index):
            qkv, g, beta = inputs
            return local._local12_recurrent(
                qkv,
                qkv,
                qkv,
                g,
                beta,
                states[index],
                packed_qkv=True,
                output=outputs[index],
                cores=cores,
                grouped_qk=args.grouped_qk == "on",
                source_overrides=(reader, compute) if index == 0 else candidate,
            )

        try:
            eager = []
            for step in range(args.steps):
                set_input(step)
                invoke(0)
                invoke(1)
                ttnn.synchronize_device(mesh)
                yr, yc, sr, sc = map(read, (*outputs, *states))
                evidence = {"step": step}
                entry["eager"].append(evidence)
                check(yc, yr, batch, "output", evidence)
                check(sc, sr, batch, "state", evidence)
                eager.append((yr, sr))
            # All tensor allocations precede both captures. No tracker allowlist.
            reset()
            set_input(0)
            for index in range(2):
                for _ in range(args.warmup):
                    invoke(index)
                ttnn.synchronize_device(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                invoke(index)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                traces.append(tid)
            reset()
            for step in range(args.steps):
                set_input(step)
                for tid in traces:
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                evidence = {"step": step}
                entry["trace"].append(evidence)
                for index, name in enumerate(("baseline", "candidate")):
                    check(read(outputs[index]), eager[step][0], batch, name + "_output_vs_eager", evidence)
                    check(read(states[index]), eager[step][1], batch, name + "_state_vs_eager", evidence)
            if not args.poison_dead_padding:
                reset()
                set_input(0)
                times = [[], []]
                for tid in traces:
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                # Alternate order to reduce ordering bias; identical replay counts.
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
                check(read(outputs[1]), read(outputs[0]), batch, "timed_output", entry)
                check(read(states[1]), read(states[0]), batch, "timed_state", entry)
            entry["pass"] = True
            save()
            print(f"RECURRENT pass B{batch} {state_name} {mode} cores={cores} {variant}", flush=True)
        finally:
            for tid in traces:
                ttnn.release_trace(mesh, tid)
            # Avoid retaining hundreds of device buffers during the default sweep.
            for tensor in all_allocations:
                ttnn.deallocate(tensor)

    mesh = None
    save()
    try:
        if args.mesh_cols == 4:
            ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        kwargs = {"trace_region_size": 8 * 1024 * 1024}
        if args.physical_device_ids:
            kwargs["physical_device_ids"] = args.physical_device_ids
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, args.mesh_cols), **kwargs)
        grid = mesh.compute_with_storage_grid_size()
        report["worker_grid"] = [grid.x, grid.y]
        cases = [(batch, args.cores) for batch in args.batches]
        if args.low_cores and (3, args.low_cores) not in cases:
            cases.append((3, args.low_cores))
        for variant in args.variants:
            for batch, cores in cases:
                for dtype in args.state_dtypes:
                    for mode in args.modes:
                        run_case(batch, dtype, mode, cores, variant)
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
