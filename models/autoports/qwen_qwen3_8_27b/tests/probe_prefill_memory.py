# SPDX-License-Identifier: Apache-2.0
"""Local-only real optimized projection probe; L1 copy cost included, no model gain claim."""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_27b.tt.generator import build_generator

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="/data/mvasiljev/weights/Qwen3.8-27B")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--physical", type=int, choices=[256, 512], default=512)
p.add_argument("--candidate-memory", choices=["l1", "dram"], default="l1")
p.add_argument("--grid-y", type=int, choices=[8, 10])
p.add_argument("--case-indices", type=int, nargs="+")
p.add_argument("--case-ids", nargs="+", help="Stable layer:role IDs, e.g. 3:gate")
p.add_argument("--l1-h-cap", type=int, default=0)
a = p.parse_args()
assert a.l1_h_cap >= 0
assert os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1" and os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
assert not os.getenv("TT_METAL_WATCHER") and not os.getenv("TT_METAL_DEVICE_PROFILER")
for name in ["QWEN38_GROUPED_RECURRENCE", "QWEN38_MASKED_RECURRENCE", "QWEN38_FLAT_PREFILL"]:
    os.environ[name] = "1"
torch.set_num_threads(8)
ttnn.CONFIG.throw_exception_on_fallback = True
root = Path(__file__).resolve().parents[1]
result = {
    "status": "running",
    "scope": "Real layers0/3 native flat projection calls at B16, active8; candidate placement/copy and projection included. Not full-model/SWE timing or a global layout-family rejection.",
    "checkpoint": a.checkpoint,
    "physical": a.physical,
    "candidate_memory": a.candidate_memory,
    "grid_y_override": a.grid_y,
    "batch": 16,
    "active_requests": 8,
    "fixture_cache_context": 8192,
    "native_context": 262144,
    "allocation_tracking": 1,
    "program_cache_skip": 0,
    "cases": [],
}


def save():
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")


original = ttnn.linear
captured = {}
traces = []
role_map = {}


def observed(x, w, *args, **kw):
    pc = kw.get("program_config")
    if (
        len(x.shape) == 3
        and list(x.shape)[:2] == [1, 16 * a.physical]
        and type(pc).__name__ == "MatmulMultiCoreReuseMultiCastProgramConfig"
    ):
        layer, role = role_map.get(id(w), (-1, "unknown"))
        desc = {
            "layer": layer,
            "role": role,
            "input": list(x.shape),
            "weight": list(w.shape),
            "input_dtype": str(x.dtype),
            "weight_dtype": str(w.dtype),
            "program": str(pc),
            "compute": {
                key: str(getattr(kw["compute_kernel_config"], key))
                for key in ("math_fidelity", "math_approx_mode", "fp32_dest_acc_en", "packer_l1_acc")
            },
        }
        key = json.dumps(desc, sort_keys=True)
        if key not in captured:
            captured[key] = (x, w, args, kw, desc)
    return original(x, w, *args, **kw)


ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024**2)
try:
    gen = build_generator(
        root,
        mesh,
        checkpoint=a.checkpoint,
        layer_indices=[0, 3],
        batch_size=16,
        context_length=8192,
        prefill_chunk_size=512,
        prepare=False,
    )
    gen.prepare_decode()
    gen.reset()
    result["precision"] = gen.model.precision_config
    role_map = {
        ident: (layer.layer_idx, role) for layer in gen.model.layers for ident, role in layer.weight_roles.items()
    }
    values = torch.zeros(16, a.physical, dtype=torch.int64)
    values[:8] = 100
    ttnn.linear = observed
    try:
        gen.prefill_forward(
            values,
            page_table=gen.page_table,
            kv_cache=gen.kv_cache,
            prompt_lens=[a.physical] * 8 + [0] * 8,
            return_device=True,
        )
    finally:
        ttnn.linear = original
    ttnn.synchronize_device(mesh)
    assert captured
    assert all(v[-1]["role"] != "unknown" for v in captured.values())
    ordered = sorted(captured, key=lambda k: (captured[k][-1]["layer"], captured[k][-1]["role"]))
    result["inventory"] = [json.loads(k) for k in ordered]
    ids = {f"{d['layer']}:{d['role']}" for d in result["inventory"]}
    assert len(ids) == len(captured), "Expected one real call per layer/role"
    if a.case_ids is not None:
        assert set(a.case_ids) <= ids, a.case_ids
    if a.case_indices is not None:
        assert set(a.case_indices) <= set(range(len(captured))), a.case_indices
    result["inventory_sha256"] = hashlib.sha256(json.dumps(result["inventory"], sort_keys=True).encode()).hexdigest()
    for index, key in enumerate(ordered):
        if a.case_indices is not None and index not in a.case_indices:
            continue
        x, w, args, kw, desc = captured[key]
        case_id = f"{desc['layer']}:{desc['role']}"
        if a.case_ids is not None and case_id not in a.case_ids:
            continue
        row = {
            "case_id": case_id,
            "case_index": index,
            "geometry": desc,
            "input_memory": str(x.memory_config()),
            "status": "running",
        }
        result["cases"].append(row)
        save()
        if x.memory_config().buffer_type == ttnn.BufferType.L1 and a.candidate_memory == "l1" and not a.grid_y:
            row.update(
                status="already_l1", decision="Selected production input is already L1; no new transfer is required."
            )
            save()
            continue
        assert x.memory_config().buffer_type in (ttnn.BufferType.DRAM, ttnn.BufferType.L1)
        l1_kw = dict(kw)
        if a.l1_h_cap or a.grid_y:
            pc = kw["program_config"]
            grid = pc.compute_with_storage_grid_size
            gy = a.grid_y or grid.y
            pm = math.ceil(x.shape[-2] / (32 * gy))
            h = min(a.l1_h_cap or pc.out_block_h, pm)
            while pm % h:
                h -= 1
            sh = min(pc.out_subblock_h, h)
            while h % sh or sh * pc.out_subblock_w > 8:
                sh -= 1
            grid = pc.compute_with_storage_grid_size
            l1_kw["program_config"] = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(grid.x, gy),
                in0_block_w=pc.in0_block_w,
                out_subblock_h=sh,
                out_subblock_w=pc.out_subblock_w,
                out_block_h=h,
                out_block_w=pc.out_block_w,
                per_core_M=pm,
                per_core_N=pc.per_core_N,
                transpose_mcast=pc.transpose_mcast,
                fused_activation=pc.fused_activation,
                fuse_batch=pc.fuse_batch,
            )
        row["l1_program"] = str(l1_kw["program_config"])
        save()

        def variant(l1):
            inp = (
                ttnn.to_memory_config(
                    x, ttnn.L1_MEMORY_CONFIG if a.candidate_memory == "l1" else ttnn.DRAM_MEMORY_CONFIG
                )
                if l1
                else x
            )
            return original(inp, w, *args, **(l1_kw if l1 else kw))

        try:
            ys = [variant(False), variant(True)]
            outputs = [ttnn.zeros_like(y) for y in ys]
            del ys

            def run(i):
                ttnn.copy(variant(bool(i)), outputs[i])

            for i in [0, 1]:
                run(i)
            ttnn.synchronize_device(mesh)

            def checks():
                rows = []
                for l, r in zip(ttnn.get_device_tensors(outputs[0]), ttnn.get_device_tensors(outputs[1])):
                    ll, rr = ttnn.to_torch(l), ttnn.to_torch(r)
                    rows.append(
                        {
                            "exact": torch.equal(ll, rr),
                            "finite": bool(torch.isfinite(rr).all()),
                            "max_abs": float((ll.float() - rr.float()).abs().max()),
                        }
                    )
                return rows

            row["before"] = checks()
            assert all(z["exact"] and z["finite"] for z in row["before"])
            for i in [0, 1]:
                tr = ttnn.begin_trace_capture(mesh, cq_id=0)
                traces.append(tr)
                try:
                    run(i)
                finally:
                    ttnn.end_trace_capture(mesh, tr, cq_id=0)
            times = [[], []]
            for order in [[0, 1], [1, 0], [0, 1]]:
                for i in order:
                    ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=True)
                    started = time.perf_counter()
                    for _ in range(30):
                        ttnn.execute_trace(mesh, traces[i], cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    times[i].append((time.perf_counter() - started) * 1e6 / 30)
            row["after90_replays_each"] = checks()
            assert all(z["exact"] and z["finite"] for z in row["after90_replays_each"])
            row.update(
                status="pass",
                windows_us={"baseline": times[0], "candidate_placement_matmul": times[1]},
                median_us=[statistics.median(v) for v in times],
                speedup=statistics.median(times[0]) / statistics.median(times[1]),
            )
        finally:
            for tr in traces:
                ttnn.release_trace(mesh, tr)
            traces = []
            save()
    result["status"] = "pass"
except BaseException as error:
    result.update(status="fail", error=repr(error))
    if result["cases"] and result["cases"][-1]["status"] == "running":
        result["cases"][-1].update(status="fail", error=repr(error))
    raise
finally:
    ttnn.linear = original
    result["probe_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result["source_sha256"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((root / "tt").glob("*.py"))
    }
    save()
    ttnn.close_mesh_device(mesh)
