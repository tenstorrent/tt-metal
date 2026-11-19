# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: v2 shared convolution widening and prefill norm HC reduction.

Uses the frozen actual norm method and AST-extracted actual convolution block.
All adapters, allocations and convolution carry writes remain inside timing.
No device operations occur in --oracle-only mode.
"""

import argparse
import ast
import copy
import hashlib
import json
import os
import runpy
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


def load_weights(checkpoint):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    result = {}
    for name in ("input_layernorm.weight", "linear_attn.conv1d.weight"):
        key = "model.language_model.layers.0." + name
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            result[name] = f.get_tensor(key)
    eps = json.loads((checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    return result, eps


def norm_mapping(x):
    rows = x.shape[1]
    squares = x.float().square().reshape(rows, 4, 16, 32)
    partial = squares[:, :, 0].clone()
    for j in range(1, 16):
        partial = partial + squares[:, :, j]
    old = partial[:, 0].clone()
    for j in range(1, 4):
        old = old + partial[:, j]
    packed = x.float().square().reshape(1, rows, 64, 32).permute(0, 2, 1, 3).reshape(4, 16, rows, 32)
    partial = packed[:, 0].clone()
    for j in range(1, 16):
        partial = partial + packed[:, j]
    new = partial[0].clone()
    for j in range(1, 4):
        new = new + partial[j]
    return compare(old, new)


def run(args):
    torch.set_num_threads(4)
    assert args.overwrite or not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    weights, eps = load_weights(args.checkpoint)
    paths = {
        "baseline": DOC / "v2_adapter_baseline.py.snapshot",
        "conv": args.conv_snapshot or DOC / "v2_conv_shared_history.py.snapshot",
        "norm": DOC / "v2_prefill_norm_hc.py.snapshot",
    }
    generator = torch.Generator().manual_seed(args.seed)
    norm_input = torch.randn((1, max(args.lengths), 2048), generator=generator).bfloat16()
    if args.oracle_cache:
        oracle = torch.load(args.oracle_cache, map_location="cpu", weights_only=False)
        norm_input = torch.cat([oracle[k] for k in ("prefix", "tokens", "continuation")], dim=1)[
            :, : max(args.lengths)
        ].contiguous()
        assert norm_input.shape == (1, max(args.lengths), 2048)
    conv_history = (3 * torch.randn((1, max(args.lengths) + 3, 8192), generator=generator)).bfloat16()
    cases = []
    if args.kind in ("both", "norm"):
        cases.extend(("norm", f"s{s}", norm_input[:, :s].contiguous(), None) for s in args.lengths)
    if args.kind in ("both", "conv"):
        cases.extend(
            ("conv", f"s{s}", conv_history[:, 3 : s + 3].contiguous(), conv_history[:, :3].contiguous())
            for s in args.lengths
        )
        codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
        finite = codes[torch.isfinite(codes)]
        values = torch.cat([finite, torch.zeros(256, dtype=torch.bfloat16)]).reshape(1, 8, 8192)
        cases.append(("conv", "all_finite_bf16_codes", values[:, 3:].contiguous(), values[:, :3].contiguous()))
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=digest(Path(__file__)),
        snapshots={k: dict(path=str(p), sha256=digest(p)) for k, p in paths.items()},
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        seed=args.seed,
        oracle_cache=str(args.oracle_cache) if args.oracle_cache else None,
        oracle_sha256=digest(args.oracle_cache) if args.oracle_cache else None,
        input_description="Norm: oracle rows when supplied, otherwise synthetic BF16; conv: synthetic BF16 + exhaustive finite codes, real checkpoint taps",
        timing_scope="Actual norm method or complete extracted conv block including concat, slices, widening, products, activation and carry write",
        cpu_norm_mapping={str(s): norm_mapping(norm_input[:, :s]) for s in args.lengths},
        cases=[],
        timing=[],
    )
    assert all(c["exact"] for c in result["cpu_norm_mapping"].values())
    for path in paths.values():
        _, code_hash = conv_function(path, None)
        result["snapshots"][next(k for k, p in paths.items() if p == path)]["conv_extracted_ast_sha256"] = code_hash

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
            config=SimpleNamespace(rms_norm_eps=eps),
            round_widen_chain=[narrow, widen],
            conv_post_chain=[narrow, widen, ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU), narrow],
            w={
                "input_layernorm.weight": upload(
                    (weights["input_layernorm.weight"].float() + 1).reshape(1, 1, 2048), ttnn.float32
                )
            },
            taps_fp32=[
                upload(weights["linear_attn.conv1d.weight"][:, 0, j].reshape(1, 1, 8192), ttnn.float32)
                for j in range(4)
            ],
        )
        norms = {k: runpy.run_path(str(p))["FusedDecoder"].norm for k, p in paths.items()}
        convs = {k: conv_function(p, ttnn)[0] for k, p in paths.items()}
        result["status"] = "running"
        save()
        for kind, name, host_x, host_carry in cases:
            dx = upload(host_x)
            states = {v: {"conv": upload(host_carry)} for v in ("baseline", kind)} if kind == "conv" else {}

            def call(variant):
                if kind == "norm":
                    return norms[variant](obj, dx, "input_layernorm.weight")
                return convs[variant](obj, dx, states[variant])

            captured = {}
            outputs = {}
            original_add, original_chain = ttnn.add, ttnn.unary_chain
            for variant in ("baseline", kind):
                seen = []

                def add(*a, **kw):
                    value = original_add(*a, **kw)
                    if (
                        len(a) > 1
                        and isinstance(a[1], (int, float))
                        and a[1] == eps
                        and kw.get("input_tensor_a_activations")
                    ):
                        seen.append(value)
                    return value

                def chain(value, ops, **kw):
                    if ops is obj.conv_post_chain:
                        seen.append(value)
                    return original_chain(value, ops, **kw)

                try:
                    ttnn.add, ttnn.unary_chain = add, chain
                    with device_only():
                        output = call(variant)
                finally:
                    ttnn.add, ttnn.unary_chain = original_add, original_chain
                assert len(seen) == 1, (kind, variant, len(seen))
                outputs[variant] = ttnn.to_torch(output)
                captured[variant] = ttnn.to_torch(seen[0])
                del output, seen
            row = dict(
                kind=kind,
                name=name,
                rows=host_x.shape[1],
                input_sha256=hashlib.sha256(host_x.view(torch.uint8).numpy().tobytes()).hexdigest(),
                output=compare(outputs["baseline"], outputs[kind]),
                intermediate=compare(captured["baseline"], captured[kind]),
                intermediate_name="shifted_mean_fp32" if kind == "norm" else "conv_preactivation_fp32",
            )
            if kind == "conv":
                expected_carry = torch.cat([host_carry, host_x], dim=1)[:, -3:]
                row["carry"] = {v: compare(expected_carry, ttnn.to_torch(states[v]["conv"])) for v in states}
            result["cases"].append(row)
            save()
            exact = (
                row["output"]["exact"]
                and row["intermediate"]["exact"]
                and all(c["exact"] for c in row.get("carry", {}).values())
            )
            if not exact or name == "all_finite_bf16_codes":
                row["timing_skipped"] = "Code difference" if not exact else "Exhaustive correctness case only"
                continue
            stable_reference = None
            stable_carry = None
            for variant in ("baseline", kind):
                trace, out, capture_open = None, None, False
                try:
                    # Four complete calls make S1 history stable; all larger S need at most two.
                    for _ in range(max(4, args.warmup)):
                        with device_only():
                            out = call(variant)
                        ttnn.synchronize_device(mesh)
                    if variant == "baseline":
                        stable_reference = ttnn.to_torch(out)
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
                    first = compare(stable_reference, ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(stable_reference, ttnn.to_torch(out))
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
            r["output"]["exact"] and r["intermediate"]["exact"] and all(v["exact"] for v in r.get("carry", {}).values())
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
    parser.add_argument("--kind", choices=("both", "norm", "conv"), default="both")
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 2, 33, 128, 129, 256])
    parser.add_argument("--conv-snapshot", type=Path, help="Explicit isolated conv snapshot; use a new --output name")
    parser.add_argument("--oracle-cache", type=Path)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v2_adapter_fusion_probe.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
