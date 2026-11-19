# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic actual norm: final four-group/eight-lane scans versus outer FastReduce.

CPU fixtures/mapping are available with --oracle-only and import no TTNN. Device
execution belongs to the serialized hardware owner. Complete norm timings include
all row-major rearrangement, tilization and output allocation costs.
"""

import argparse
import json
import os
import runpy
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.exact_next_fusion_probe import DOC, compare, digest, fixtures
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record


def cpu_mapping(x):
    rows = x.shape[1]
    square = x.float().square().reshape(rows, 4, 16, 32)
    partial = square[:, :, 0].clone()
    for j in range(1, 16):
        partial = partial + square[:, :, j]
    total = partial[:, 0].clone()
    for j in range(1, 4):
        total = total + partial[:, j]
    groups = total.reshape(rows, 4, 8)
    old_lanes = torch.zeros((rows, 8), dtype=torch.float32)
    for j in range(4):
        old_lanes = old_lanes + groups[:, j]
    old = torch.zeros(rows, dtype=torch.float32)
    for j in range(8):
        old = old + old_lanes[:, j]
    packed = total.reshape(rows, 4, 8, 1).permute(2, 1, 0, 3).contiguous()
    new_lanes = packed[:, 0].clone()
    for j in range(1, 4):
        new_lanes = new_lanes + packed[:, j]
    new = new_lanes[0].clone()
    for j in range(1, 8):
        new = new + new_lanes[j]
    return compare(old.flatten(), new.flatten())


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    assert not output.exists(), f"Preserve evidence: {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    base = DOC / "selected_fused_decoder_v3.py.snapshot"
    candidate = Path(args.candidate_snapshot)
    proof = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert digest(base) == proof["base_sha256"] and digest(candidate) == proof["candidate_sha256"]
    rows, _, gamma, eps = fixtures(args)
    oracle = torch.load(args.oracle, map_location="cpu", weights_only=False)
    rows.append(("prefill_first128", oracle["prefix"][:, :128].contiguous()))
    rows.append(("zero_input", torch.zeros((1, 1, 2048), dtype=torch.bfloat16)))
    result = dict(
        status="cpu_mapping_pass",
        diagnostic_only=True,
        acceptance_eligible=False,
        base_sha256=digest(base),
        candidate_sha256=digest(candidate),
        probe_sha256=digest(__file__),
        oracle_sha256=digest(args.oracle),
        fixtures=[dict(name=n, shape=list(x.shape)) for n, x in rows],
        cpu_mapping={name: cpu_mapping(x) for name, x in rows},
        checks=[],
        timing=[],
        device_closed=None,
        timing_scope="Complete actual norm, including conversions, permutation, tilization, both FastReduce stages and output allocation",
    )
    assert all(x["passed"] for x in result["cpu_mapping"].values())

    def save():
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    record(output)
    save()
    if args.oracle_only:
        print("CPU_NORM_LAST_REDUCE_MAPPING_PASS", len(rows))
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), f"Clean timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    mesh = None
    stage = "open_device"
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768
        )
        result.update(status="running", device_closed=False)
        classes = {
            name: runpy.run_path(str(path))["FusedDecoder"]
            for name, path in [("baseline", base), ("candidate", candidate)]
        }
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        rw = [
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value)),
            ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value)),
        ]

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        obj = SimpleNamespace(
            kind="linear_attention",
            mesh_device=mesh,
            compute=compute,
            round_widen_chain=rw,
            config=SimpleNamespace(rms_norm_eps=eps),
            w={"input_layernorm.weight": upload(gamma, ttnn.float32)},
        )

        def call(variant, x):
            with device_only():
                return classes[variant].norm(obj, x, "input_layernorm.weight")

        def cuts(variant, x):
            original_add = ttnn.add
            seen = {}

            def add(a, b, *pos, **kw):
                out = original_add(a, b, *pos, **kw)
                if isinstance(b, (int, float)) and b == eps and kw.get("input_tensor_a_activations"):
                    seen.update(sum32=a, shifted_mean=out)
                return out

            ttnn.add = add
            try:
                out = call(variant, x)
            finally:
                ttnn.add = original_add
            assert set(seen) == {"sum32", "shifted_mean"}
            ttnn.synchronize_device(mesh)
            return {**{k: ttnn.to_torch(v).flatten() for k, v in seen.items()}, "complete_norm": ttnn.to_torch(out)}

        for name, row in rows:
            stage = name
            x = upload(row)
            old, new = cuts("baseline", x), cuts("candidate", x)
            checks = {key: compare(old[key], new[key]) for key in old}
            result["checks"].append(dict(name=name, rows=row.shape[1], values=checks))
            save()
            if not all(c["passed"] for c in checks.values()):
                continue
            for variant in classes:
                trace, capture_open, out = None, False, None
                try:
                    for _ in range(args.warmup):
                        out = call(variant, x)
                        ttnn.synchronize_device(mesh)
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    out = call(variant, x)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(old["complete_norm"], ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(old["complete_norm"], ttnn.to_torch(out))
                    result["timing"].append(
                        dict(
                            name=name,
                            rows=row.shape[1],
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first=first,
                            last=last,
                        )
                    )
                    assert first["passed"] and last["passed"]
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            del x
            print("NORM_LAST_REDUCE_CASE_COMPLETE", name, flush=True)
        result["status"] = (
            "passed" if all(c["passed"] for row in result["checks"] for c in row["values"].values()) else "different"
        )
        assert digest(base) == proof["base_sha256"] and digest(candidate) == proof["candidate_sha256"]
    except Exception:
        result.update(status="error", failed_stage=stage, traceback=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result.update(status="error", close_error=traceback.format_exc())
        save()
    print(result["status"], output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--candidate-snapshot", default=str(DOC / "endurance/v3_norm_last_reduce.py.snapshot"))
    p.add_argument("--oracle", default=str(DOC / "oracle_cache/stress_8192_0.pt"))
    p.add_argument("--positions", nargs="+", type=int, default=[128, 130, 430, 3979, 5991, 7703])
    p.add_argument("--capture-positions", nargs="*", type=int, default=[])
    p.add_argument("--capture-base", default=str(DOC / "endurance_precise_probe.json"))
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--replays", type=int, default=5)
    p.add_argument("--output", default=str(DOC / "endurance/v3_norm_last_reduce_device.json"))
    args = p.parse_args()
    assert min(args.warmup, args.samples, args.replays) > 0
    run(args)
