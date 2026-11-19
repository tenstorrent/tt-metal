# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same captured HF/TT logits through native and alternative decode routers.

Includes complete router and sparse-consumer adapters in eager/trace timings.
No hardware access or TTNN import in --oracle-only mode. Diagnostic, not acceptance.
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

import torch

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare
from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

DOC = Path(__file__).resolve().parents[1] / "doc/fused_decoder/endurance"
VARIANTS = ("native", "bf16_topk", "fp32_topk", "fp32_softmax_topk")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def router_function(source, ttnn):
    """Extract actual decode producer plus actual routing/ID consumer adapters."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_moe")
    branches = [n for n in method.body if isinstance(n, ast.If)]
    assignments = {}
    for node in branches[1].body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            assignments[node.targets[0].id] = node
    fn = ast.parse("def route(self, logits):\n    return routing, ids\n").body[0]
    fn.body = copy.deepcopy(branches[0].body) + [copy.deepcopy(assignments[k]) for k in ("ids", "routing")] + fn.body
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reshape")
    module = ast.Module(body=[copy.deepcopy(helper), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    code_hash = hashlib.sha256(ast.dump(module).encode()).hexdigest()
    if ttnn is None:
        return None, code_hash
    scope = {"ttnn": ttnn, "DRAM": ttnn.DRAM_MEMORY_CONFIG}
    exec(compile(module, str(source) + ":router", "exec"), scope)
    return scope["route"], code_hash


def canonical(logits):
    probability = torch.softmax(logits.reshape(1, 256).float(), dim=-1)
    scores, ids = torch.topk(probability, 8, dim=-1)
    scores = (scores / scores.sum(-1, keepdim=True)).bfloat16()
    return scores.flatten(), ids.flatten()


def assess(values, expected, actual_hf_ids):
    scores, ids = values[0].flatten().float(), values[1].flatten().long()
    es, ei = expected[0].flatten().float(), expected[1].flatten().long()
    got_set, expected_set = set(ids.tolist()), set(ei.tolist())
    shared = sorted(got_set & expected_set)
    by_id = {int(i): float(s) for i, s in zip(ids, scores)}
    expected_by_id = {int(i): float(s) for i, s in zip(ei, es)}
    return dict(
        ids=ids.tolist(),
        scores=scores.tolist(),
        finite=bool(torch.isfinite(scores).all()),
        unique_ids=len(got_set) == 8,
        same_input_cpu_membership=got_set == expected_set,
        missing_same_input_cpu_ids=sorted(expected_set - got_set),
        extra_same_input_cpu_ids=sorted(got_set - expected_set),
        captured_hf_membership=got_set == set(actual_hf_ids),
        common_id_max_score_error=max(abs(by_id[i] - expected_by_id[i]) for i in shared) if shared else None,
        score_sum=float(scores.sum()),
    )


def run(args):
    torch.set_num_threads(4)
    assert not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    paths = {"native": args.baseline_snapshot}
    paths.update({v: DOC / f"v5_router_{v}.py.snapshot" for v in VARIANTS[1:]})
    for path in list(paths.values())[1:]:
        proof = json.loads(path.with_suffix(".provenance.json").read_text())
        assert proof["base_sha256"] == digest(args.baseline_snapshot)
        assert proof["candidate_sha256"] == digest(path)
    fixtures, metadata = [], []
    for capture in args.captures:
        data = torch.load(capture, map_location="cpu", weights_only=False)
        h, t = data["hf_values"], data["tt_values"]
        hf_ids = h["router"][2].flatten().tolist()
        for name, logit in (("hf", h["router"][0]), ("tt", t["router"])):
            logit = logit.reshape(1, 256).bfloat16().contiguous()
            expected = canonical(logit)
            tag = f"pos{data['position']}_{name}"
            fixtures.append((tag, logit, expected, hf_ids))
            metadata.append(
                dict(
                    name=tag,
                    capture=str(capture),
                    capture_sha256=digest(capture),
                    measured_source=data.get("measured_decoder_source"),
                    measured_source_sha256=data.get("measured_decoder_source_sha256"),
                    logits_sha256=hashlib.sha256(logit.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    cpu_softmax_topk_ids=expected[1].tolist(),
                    cpu_softmax_topk_scores=expected[0].float().tolist(),
                    cpu_bf16_direct_topk_ids=torch.topk(logit, 8).indices.flatten().tolist(),
                    cpu_fp32_direct_topk_ids=torch.topk(logit.float(), 8).indices.flatten().tolist(),
                    captured_hf_ids=hf_ids,
                    captured_tt_ids=t["router_ids"].flatten().tolist(),
                    cpu_same_input_route_matches_hf=set(expected[1].tolist()) == set(hf_ids),
                )
            )
    result = dict(
        classification="same-logit diagnostic; not model acceptance",
        status="cpu_setup_pass",
        probe_sha256=digest(__file__),
        snapshots={
            k: dict(path=str(p), sha256=digest(p), router_ast_sha256=router_function(p, None)[1])
            for k, p in paths.items()
        },
        fixtures=metadata,
        timing_scope="Complete decode router from existing BF16 logits through normalization/selection plus BF16 TILE routing [1,8,1,1] and UINT16 RM sparse IDs [1,1,1,8]. All reshape, sharding, casts, topk/softmax, sum/divide and output adapters included; logits projection excluded identically.",
        compute="HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False",
        device_open_parameters=dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768),
        cases=[],
        timing=[],
        eager_timing=[],
        device_closed=None,
        limitations="CPU topk8 tie choices are reference observations, not a general stable-order contract. Equal logits do not make experts interchangeable. No quality threshold waiver or model-selection claim.",
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    record(args.output)
    if args.oracle_only:
        result["ttnn_imported"] = "ttnn" in sys.modules
        assert not result["ttnn_imported"]
        save()
        print("CPU_ROUTER_TIE_SETUP_PASS", [f[0] for f in fixtures])
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
    try:
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768
        )
        obj = SimpleNamespace()
        obj.compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        obj.router_shard = ttnn.create_sharded_memory_config(
            (32, 32),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def upload(value, dtype, layout, memory):
            return ttnn.from_torch(value.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=memory)

        obj.router_bias = upload(torch.zeros(1, 16, 16), ttnn.bfloat16, ttnn.TILE_LAYOUT, obj.router_shard)
        obj.router_indices = upload(
            torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, ttnn.TILE_LAYOUT, obj.router_shard
        )
        obj.router_out = upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, obj.router_shard)
        obj.router_out_ids = upload(
            torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT, obj.router_shard
        )
        routers = {k: router_function(p, ttnn)[0] for k, p in paths.items()}
        result.update(status="running", device_closed=False)
        save()

        def host(values):
            return [ttnn.to_torch(v) for v in values]

        def checks(a, b):
            comparisons = [compare(x, y) for x, y in zip(a, b)]
            return dict(exact=all(c["exact"] for c in comparisons), tensors=comparisons)

        for name, logits, expected, hf_ids in fixtures:
            dx = upload(logits, ttnn.bfloat16, ttnn.TILE_LAYOUT, ttnn.DRAM_MEMORY_CONFIG)
            original = ttnn.to_torch(dx)
            for variant in VARIANTS:
                trace, out, capture_open = None, None, False

                def call():
                    with device_only():
                        values = routers[variant](obj, dx)
                    routing, ids = values
                    assert (
                        tuple(routing.shape) == (1, 8, 1, 1)
                        and routing.dtype == ttnn.bfloat16
                        and routing.layout == ttnn.TILE_LAYOUT
                    )
                    assert (
                        tuple(ids.shape) == (1, 1, 1, 8)
                        and ids.dtype == ttnn.uint16
                        and ids.layout == ttnn.ROW_MAJOR_LAYOUT
                    )
                    assert routing.memory_config() == ids.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                    return values

                try:
                    out = call()
                    reference = host(out)
                    row = dict(name=name, variant=variant, output=assess(reference, expected, hf_ids))
                    result["cases"].append(row)
                    assert row["output"]["finite"] and row["output"]["unique_ids"]
                    for _ in range(args.warmup):
                        out = call()
                        ttnn.synchronize_device(mesh)
                    samples = []
                    for _ in range(args.eager_samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        out = call()
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start))
                    eager_check = checks(reference, host(out))
                    assert eager_check["exact"]
                    result["eager_timing"].append(
                        dict(
                            name=name,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            own_eager_repeated=eager_check,
                        )
                    )
                    out = None
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    out = call()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = checks(reference, host(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = checks(reference, host(out))
                    immutable = compare(original, ttnn.to_torch(dx))
                    assert first["exact"] and last["exact"] and immutable["exact"]
                    result["timing"].append(
                        dict(
                            name=name,
                            variant=variant,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            first_vs_own_eager=first,
                            last_vs_own_eager=last,
                            input_immutable=immutable,
                        )
                    )
                    save()
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            del dx
            print("ROUTER_TIE_CASE_COMPLETE", name, flush=True)
        result["status"] = "diagnostic_complete"
        assert all(digest(p) == result["snapshots"][k]["sha256"] for k, p in paths.items())
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result.update(status="error", close_error=traceback.format_exc())
        save()
    print(result["status"], args.output)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-snapshot", type=Path, default=DOC / "v5_combined_accurate_residual.py.snapshot")
    p.add_argument(
        "--captures",
        nargs="+",
        type=Path,
        default=[
            DOC / "v5_combined_layer34_stream_probe.json.pos181.pt",
            DOC / "v5_accurate_residual_layer34_worst_probe.json.pos7944.pt",
        ],
    )
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--eager-samples", type=int, default=5)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--replays", type=int, default=10)
    p.add_argument("--output", type=Path, default=DOC / "v5_router_tie_device.json")
    args = p.parse_args()
    assert min(args.warmup, args.eager_samples, args.samples, args.replays) > 0
    run(args)
