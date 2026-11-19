# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: v3 shared RM convolution carry and delayed router score tilization.

Uses AST-extracted actual convolution blocks and actual post-gate router adapters.
All adapters, allocations and convolution carry writes remain inside timing.
No device operations occur in --oracle-only mode.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def conv_function(source, ttnn):
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_linear_attention")
    start = next(
        i
        for i, n in enumerate(method.body)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "history" for t in n.targets)
    )
    end = next(
        i
        for i, n in enumerate(method.body[start:], start)
        if isinstance(n, ast.Expr)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "copy"
    )
    block = copy.deepcopy(method.body[start : end + 1])
    fn = ast.parse("def convolution(self, qkv, state):\n    s = qkv.shape[1]\n    return conv\n").body[0]
    fn.body = fn.body[:1] + block + fn.body[-1:]
    module = ast.Module(body=[fn], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"ttnn": ttnn}
    exec(compile(module, str(source) + ":convolution", "exec"), scope)
    return scope["convolution"], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def router_function(source, ttnn):
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_moe")
    branches = [n for n in method.body if isinstance(n, ast.If)]
    assignments = {}
    for branch in branches[:2]:
        for node in branch.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                assignments[node.targets[0].id] = node
    fn = ast.parse("def router(gate_scores, gate_ids):\n    return routing, ids\n").body[0]
    fn.body = [copy.deepcopy(assignments[k]) for k in ("scores", "indices", "ids", "routing")] + fn.body
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reshape")
    module = ast.Module(body=[copy.deepcopy(helper), fn], type_ignores=[])
    ast.fix_missing_locations(module)
    scope = {"ttnn": ttnn, "DRAM": getattr(ttnn, "DRAM_MEMORY_CONFIG", None)}
    exec(compile(module, str(source) + ":router", "exec"), scope)
    return scope["router"], hashlib.sha256(ast.dump(module).encode()).hexdigest()


def load_weights(checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in ("mlp.gate.weight", "linear_attn.conv1d.weight"):
        key = "model.language_model.layers.0." + name
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            result[name] = f.get_tensor(key)
    return result


def run(args):
    torch.set_num_threads(4)
    assert args.overwrite or not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights = load_weights(args.checkpoint)
    paths = {
        "baseline": DOC / "v3_movement_baseline.py.snapshot",
        "conv": DOC / "v3_conv_rm_carry.py.snapshot",
        "router": args.router_snapshot or DOC / "v3_router_rm_scores.py.snapshot",
        "combined": DOC / "v3_rm_carry_router.py.snapshot",
    }
    generator = torch.Generator().manual_seed(args.seed)
    conv_history = (3 * torch.randn((1, max(args.lengths) + 3, 8192), generator=generator)).bfloat16()
    cases = []
    if args.kind in ("both", "conv"):
        cases.extend(
            ("conv", f"s{s}", conv_history[:, 3 : s + 3].contiguous(), conv_history[:, :3].contiguous())
            for s in args.lengths
        )
    if args.kind in ("both", "router"):
        for seed in (42, 154, 913):
            hidden = torch.randn((1, 2048), generator=torch.Generator().manual_seed(seed)).bfloat16()
            logits = torch.nn.functional.linear(hidden, weights["mlp.gate.weight"].bfloat16()).reshape(1, 16, 16)
            cases.append(("router", f"seed{seed}", logits, None))
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=digest(Path(__file__)),
        snapshots={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        seed=args.seed,
        dependencies={
            str(Path(__file__).with_name("conv_first_tap_mixed_probe.py")): digest(
                Path(__file__).with_name("conv_first_tap_mixed_probe.py")
            )
        },
        input_description="Conv: seeded BF16 history with checkpoint taps. Router: checkpoint-derived CPU BF16 logits from synthetic hidden rows, actual native gate outputs on device.",
        timing_scope="Complete extracted conv including concat, slices, widening, products, activation and carry write; or all post-native-gate routing score/id adapters up to sparse-matmul and weighted-multiply consumers. Full decoder timing required separately.",
        cases=[],
        timing=[],
    )
    for key, path in paths.items():
        for name, extractor in (("conv", conv_function), ("router", router_function)):
            _, code_hash = extractor(path, None)
            result["snapshots"][key][name + "_extracted_ast_sha256"] = code_hash
    assert (
        result["snapshots"]["conv"]["router_extracted_ast_sha256"]
        == result["snapshots"]["baseline"]["router_extracted_ast_sha256"]
    )
    assert (
        result["snapshots"]["router"]["conv_extracted_ast_sha256"]
        == result["snapshots"]["baseline"]["conv_extracted_ast_sha256"]
    )
    for kind in ("conv", "router"):
        if kind == "router" and args.router_snapshot:
            continue
        assert (
            result["snapshots"][kind][kind + "_extracted_ast_sha256"]
            == result["snapshots"]["combined"][kind + "_extracted_ast_sha256"]
        )
    result["cpu_carry_mapping"] = {
        str(s): compare(
            conv_history[:, s : s + 3], torch.cat([conv_history[:, :3], conv_history[:, 3 : s + 3]], dim=1)[:, -3:]
        )
        for s in args.lengths
    }

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        save()
        print("CPU_ADAPTER_SETUP_PASS", len(cases))
        return
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_DEVICE_PROFILER",
        "TT_METAL_TRACE_ALLOC_TRACKING",
        "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
    ):
        assert os.environ.get(key) in (None, "", "0"), f"Clean timing requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(args.output)
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024)

        def upload(x, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        narrow = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.float32.value), float(ttnn.bfloat16.value))
        widen = ttnn.UnaryWithParam(ttnn.UnaryOpType.TYPECAST, float(ttnn.bfloat16.value), float(ttnn.float32.value))
        obj = SimpleNamespace(
            kind="linear_attention",
            mesh_device=mesh,
            compute=compute,
            round_widen_chain=[narrow, widen],
            conv_post_chain=[narrow, widen, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), narrow],
            taps_fp32=[
                upload(weights["linear_attn.conv1d.weight"][:, 0, j].reshape(1, 1, 8192), ttnn.float32)
                for j in range(4)
            ],
        )
        routers = {k: router_function(p, ttnn)[0] for k, p in paths.items()}
        convs = {k: conv_function(p, ttnn)[0] for k, p in paths.items()}
        router_shard = ttnn.create_sharded_memory_config(
            (32, 32),
            ttnn.CoreGrid(y=1, x=1),
            ttnn.ShardStrategy.HEIGHT,
            ttnn.ShardOrientation.ROW_MAJOR,
            use_height_and_width_as_shard_shape=True,
        )

        def router_upload(x, dtype, layout):
            return ttnn.from_torch(x.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=router_shard)

        gate_kw = dict(
            bias_tensor=router_upload(torch.zeros(1, 16, 16), ttnn.bfloat16, ttnn.TILE_LAYOUT),
            input_indices_tensor=router_upload(
                torch.arange(256).reshape(1, 16, 16).transpose(1, 2), ttnn.uint16, ttnn.TILE_LAYOUT
            ),
            output_tensor=router_upload(torch.zeros(1, 32, 32), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT),
            output_indices_tensor=router_upload(
                torch.zeros(1, 32, 32, dtype=torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT
            ),
            scaling_factor=1.0,
            enable_sigmoid=False,
            topk=8,
            output_softmax=True,
        )
        result["status"] = "running"
        save()
        for kind, name, host_x, host_carry in cases:
            variants = (
                ("baseline", kind) if kind == "router" and args.router_snapshot else ("baseline", kind, "combined")
            )
            dx = upload(host_x)
            states = {v: {"conv": upload(host_carry)} for v in variants} if kind == "conv" else {}
            if kind == "router":
                face = ttnn.to_memory_config(dx, router_shard)
                gate_scores, gate_ids = ttnn.experimental.deepseek.moe.generalized_moe_gate(face, **gate_kw)
                assert gate_scores.dtype == ttnn.bfloat16 and gate_ids.dtype == ttnn.uint16

            def call(variant):
                if kind == "router":
                    routing, ids = routers[variant](gate_scores, gate_ids)
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
                    return routing, ids
                return convs[variant](obj, dx, states[variant])

            def host_output(value):
                return [ttnn.to_torch(v) for v in value] if kind == "router" else [ttnn.to_torch(value)]

            def compare_output(a, b):
                checks = [compare(x, y) for x, y in zip(a, b)]
                return dict(exact=all(c["exact"] for c in checks), tensors=checks)

            captured = {}
            outputs = {}
            original_chain = ttnn.unary_chain
            for variant in variants:
                seen = []

                def chain(value, ops, **kw):
                    if ops is obj.conv_post_chain:
                        seen.append(value)
                    return original_chain(value, ops, **kw)

                try:
                    ttnn.unary_chain = chain
                    with device_only():
                        output = call(variant)
                finally:
                    ttnn.unary_chain = original_chain
                assert len(seen) == (1 if kind == "conv" else 0), (kind, variant, len(seen))
                outputs[variant] = host_output(output)
                captured[variant] = ttnn.to_torch(seen[0]) if seen else outputs[variant][1]
                del output, seen
            row = dict(
                kind=kind,
                name=name,
                rows=host_x.shape[1],
                input_sha256=hashlib.sha256(host_x.view(torch.uint8).numpy().tobytes()).hexdigest(),
                output={v: compare_output(outputs["baseline"], outputs[v]) for v in variants[1:]},
                intermediate={v: compare(captured["baseline"], captured[v]) for v in variants[1:]},
                intermediate_name="conv_preactivation_fp32" if kind == "conv" else "expert_ids",
            )
            if kind == "conv":
                expected_carry = torch.cat([host_carry, host_x], dim=1)[:, -3:]
                row["carry"] = {v: compare(expected_carry, ttnn.to_torch(states[v]["conv"])) for v in states}
            result["cases"].append(row)
            save()
            exact = (
                all(c["exact"] for c in row["output"].values())
                and all(c["exact"] for c in row["intermediate"].values())
                and all(c["exact"] for c in row.get("carry", {}).values())
            )
            if not exact or name == "all_finite_bf16_codes":
                row["timing_skipped"] = "Code difference" if not exact else "Exhaustive correctness case only"
                continue
            stable_reference = None
            stable_carry = None
            for variant in variants:
                trace, out, capture_open = None, None, False
                try:
                    # Four complete calls make S1 history stable; all larger S need at most two.
                    for _ in range(max(4, args.warmup)):
                        with device_only():
                            out = call(variant)
                        ttnn.synchronize_device(mesh)
                    if variant == "baseline":
                        stable_reference = host_output(out)
                        stable_carry = ttnn.to_torch(states[variant]["conv"]) if kind == "conv" else None
                    del out
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        out = call(variant)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare_output(stable_reference, host_output(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare_output(stable_reference, host_output(out))
                    timing = dict(
                        kind=kind,
                        name=name,
                        variant=variant,
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        replays_per_sample=args.replays,
                        first_vs_baseline=first,
                        last_vs_baseline=last,
                    )
                    if kind == "conv":
                        timing["carry_vs_baseline"] = compare(stable_carry, ttnn.to_torch(states[variant]["conv"]))
                        assert timing["carry_vs_baseline"]["exact"]
                    result["timing"].append(timing)
                    assert first["exact"] and last["exact"]
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            print("ADAPTER_CASE_COMPLETE", kind, name, flush=True)
            del dx, states
        result["device_exact"] = all(
            all(v["exact"] for v in r["output"].values())
            and all(v["exact"] for v in r["intermediate"].values())
            and all(v["exact"] for v in r.get("carry", {}).values())
            for r in result["cases"]
        )
        result["status"] = "exact" if result["device_exact"] else "different"
        assert all(digest(paths[k]) == v["sha256"] for k, v in result["snapshots"].items())
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(result["status"], args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--kind", choices=("both", "router", "conv"), default="both")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 32, 33, 127, 128, 129, 256])
    parser.add_argument(
        "--router-snapshot",
        type=Path,
        help="Override router-only candidate; preserve measured reports with a new output path",
    )
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v3_data_movement_probe.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays) > 0
    if args.router_snapshot:
        assert args.output != DOC / "v3_data_movement_probe.json", "Use a new output name for an alternate snapshot"
    run(args)


if __name__ == "__main__":
    main()
