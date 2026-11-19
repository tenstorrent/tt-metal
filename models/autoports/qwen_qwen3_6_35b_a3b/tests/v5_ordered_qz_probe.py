# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: native Q/Z projection versus ordered BF16-product FP32 scan.

Actual frozen linear methods, real checkpoint weights, both output dtypes,
K512/K1024 and complete eager/traced adapter timing. No model acceptance claim.
"""

import argparse
import ast
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

from models.autoports.qwen_qwen3_6_35b_a3b.tests.conv_first_tap_mixed_probe import compare

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


COLUMNS = [0, 1, 31, 32, 2047, 2048, 2049, 4095, 4096, 6143]


def weights(checkpoint, layer):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    values = []
    for name in ("qkv", "z"):
        key = f"model.language_model.layers.{layer}.linear_attn.in_proj_{name}.weight"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            values.append(f.get_tensor(key))
    qkv, z = values
    raw = torch.cat((qkv[:2048], z)).T.contiguous()
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    return raw[order].contiguous(), raw, qkv, z


def ordered_reference(w, x, k):
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    rows, blocks = x.shape[1], 2048 // k
    coeff = x[0, :, order].reshape(rows, blocks, k, 1).float()
    products = coeff * w[:, COLUMNS].reshape(1, blocks, k, len(COLUMNS)).float()
    running = torch.zeros_like(products[:, :, 0])
    for j in range(k):
        running = running + products[:, :, j]
    total = torch.zeros_like(running[:, 0])
    for block in range(blocks):
        total = total + running[:, block]
    return total.unsqueeze(0)


def metrics(expected, actual):
    report = compare(expected, actual)
    a, b = expected.double().flatten(), actual.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    report["pcc"] = float(torch.dot(a, b) / (torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)))
    return report


def load_snapshot(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def candidate_setup(obj, source, w, pair32, upload, ttnn, mesh):
    """Execute the actual setup-only ordered QZ weight allocation."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_state_dict")
    names = {"self.scan_output_weight"}
    statements = [
        n for n in ast.walk(setup) if isinstance(n, ast.Assign) and any(ast.unparse(t) in names for t in n.targets)
    ]
    assert len(statements) == 1
    module = ast.Module(body=copy.deepcopy(statements), type_ignores=[])
    ast.fix_missing_locations(module)
    code_hash = hashlib.sha256(ast.dump(module).encode()).hexdigest()
    if obj is not None:
        scope = dict(
            self=obj,
            pieces=[w[0], w[1]],
            order=torch.arange(2048).reshape(1024, 2).flip(1).flatten(),
            pair32=pair32,
            upload=upload,
            ttnn=ttnn,
            torch=torch,
            mesh_device=mesh,
            DRAM=ttnn.DRAM_MEMORY_CONFIG,
        )
        exec(compile(module, str(source) + ":projection_setup", "exec"), scope)
    return code_hash


def source_methods(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    return tree, {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}


def check_source_scope(baseline, candidate):
    old_tree, old = source_methods(baseline)
    tree, new = source_methods(candidate)
    changed = [key for key in old if ast.dump(old[key]) != ast.dump(new[key])]
    assert set(changed) == {"linear", "from_state_dict"}
    new["linear"].body[0].test = copy.deepcopy(old["linear"].body[0].test)
    new["linear"].body[1].test = copy.deepcopy(old["linear"].body[1].test)
    body = new["linear"].body[0].body
    body[:] = [n for n in body if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "width")]

    class Restore(ast.NodeTransformer):
        def visit_Name(self, node):
            return ast.copy_location(ast.Constant(6208), node) if node.id == "width" else node

        def visit_IfExp(self, node):
            return (
                copy.deepcopy(node.body)
                if ast.unparse(node.body) == "self.scan_attention_weight"
                else self.generic_visit(node)
            )

    Restore().visit(new["linear"])
    for node in ast.walk(tree):
        if hasattr(node, "body") and isinstance(node.body, list):
            node.body[:] = [
                n
                for n in node.body
                if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.scan_output_weight")
            ]
    assert ast.dump(old_tree) == ast.dump(tree)
    return dict(
        full_module_ast_restoration_pass=True,
        kvab_arithmetic_unchanged=True,
        setup_ast_sha256=candidate_setup(None, candidate, None, None, None, None, None),
    )


def native_setup(obj, source, ttnn, mesh):
    # Execute V5's actual native QZ sharding/config construction for only this role.
    _, methods = source_methods(source)
    body = methods["from_state_dict"].body
    start = next(
        i for i, n in enumerate(body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "self.decode_linear"
    )
    end = next(
        i
        for i in range(start, len(body))
        if isinstance(body[i], ast.For) and ast.unparse(body[i].iter) == "projection_keys"
    )
    selected = [
        n
        for n in body[start : end + 1]
        if not (isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "projection_keys")
    ]
    module = ast.Module(body=copy.deepcopy(selected), type_ignores=[])
    ast.fix_missing_locations(module)
    scope = dict(self=obj, ttnn=ttnn, mesh_device=mesh, projection_keys=("output_only_packed",))
    exec(compile(module, str(source) + ":native_QZ_setup", "exec"), scope)
    return hashlib.sha256(ast.dump(module).encode()).hexdigest()


def run(args):
    torch.set_num_threads(4)
    assert args.overwrite or not args.output.exists(), f"Preserve evidence: {args.output}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_snapshot
    candidate = args.candidate_snapshot
    provenance = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert sha(Path(provenance["base"])) == provenance["base_sha256"]
    assert sha(baseline) == provenance["comparison_baseline_sha256"]
    assert sha(candidate) == provenance["candidate_sha256"]
    source_scope = check_source_scope(baseline, candidate)
    w, raw, qkv, z = weights(args.checkpoint, args.layer)
    assert w.shape == (2048, 6144) and w.dtype == torch.bfloat16
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn((1, max(args.lengths), 2048), generator=generator)
    x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    cases = [(1, 512), (1, 1024)] + [(s, 1024) for s in args.lengths if s > 1]
    canonical_bf16, canonical_fp32 = {}, {}
    for rows, k in cases:
        xx = x[:, :rows]
        canonical_bf16[(rows, k)] = torch.cat(
            (torch.nn.functional.linear(xx, qkv)[..., :2048], torch.nn.functional.linear(xx, z)), dim=-1
        )
        canonical_fp32[(rows, k)] = torch.cat(
            (
                torch.nn.functional.linear(xx.float(), qkv.float())[..., :2048],
                torch.nn.functional.linear(xx.float(), z.float()),
            ),
            dim=-1,
        )
    ordered = {(rows, k): ordered_reference(w, x[:, :rows], k) for rows, k in cases}
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=sha(Path(__file__)),
        baseline_snapshot=str(baseline),
        baseline_sha256=sha(baseline),
        candidate_base=provenance["base"],
        candidate_base_sha256=provenance["base_sha256"],
        candidate_snapshot=str(candidate),
        candidate_sha256=sha(candidate),
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        layer=args.layer,
        seed=args.seed,
        ordered_qz_weight_sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_description="Synthetic normalized BF16 rows, real checkpoint weights; not recorded model activations",
        timing_scope="Complete actual QZ linear: baseline native input/output sharding and matmul; candidate input pair gather, ordered BF16 products/FP32 scan, terminal extraction, block reduce and cast; serial prefill includes all per-row adapters, concat and final tilize. BF16 production-output timing; FP32 also correctness checked.",
        setup_scope="Real native QZ weights and actual baseline AST sharded matmul config; candidate adds actual setup AST ordered BF16 QZ copy. Projection-only fixture excludes unrelated decoder weights and is not a capacity test.",
        source_scope=source_scope,
        dependencies={
            str(Path(__file__).with_name("conv_first_tap_mixed_probe.py")): sha(
                Path(__file__).with_name("conv_first_tap_mixed_probe.py")
            )
        },
        compute="HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False",
        cpu_reference_scope="Selected columns use explicit sequential FP32 products/sums in the candidate K policy. Full Q/Z HF-style torch linear outputs are reported separately; their kernel arithmetic may differ, especially the FP32 diagnostic output.",
        selected_columns=COLUMNS,
        cpu_ordered_vs_hf={
            f"s{rows}_k{k}": metrics(canonical_bf16[(rows, k)][..., COLUMNS], value.bfloat16())
            for (rows, k), value in ordered.items()
        },
        cases=[],
        timing=[],
        eager_timing=[],
        device_open_parameters=dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768),
        additional_resident_dram_weight_bytes=25165824,
        product_or_prefix_bytes=100663296,
        setup=[],
        comparison_baseline_is_selected_v5=True,
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

        result["ttnn_imported"] = "ttnn" in sys.modules
        assert not result["ttnn_imported"]
        record(args.output)
        save()
        print("CPU_ORDERED_QZ_PASS", cases)
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
        mesh = ttnn.open_mesh_device(
            ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768
        )

        def upload(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                v.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        models = {}
        for name, source in (("baseline", baseline), ("candidate", candidate)):
            setup_start = time.perf_counter()
            obj = load_snapshot(source, "probe_ordered_qz_" + name).FusedDecoder()
            obj.kind = "linear_attention"
            obj.mesh_device = mesh
            obj.compute = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=ttnn.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            )
            order = torch.arange(2048).reshape(1024, 2).flip(1).reshape(1, 1, 2048)
            pair32 = torch.arange(32).reshape(16, 2).flip(1).reshape(1, 1, 32).expand(1, 64, 32)
            obj.attention_k_indices = upload(order.to(torch.int32), ttnn.uint16)
            obj.attention_pair32_indices = upload(pair32.to(torch.int32), ttnn.uint16, ttnn.ROW_MAJOR_LAYOUT)
            obj.w = {"output_only_packed": upload(raw)}
            setup_hash = native_setup(obj, baseline, ttnn, mesh)
            if name == "candidate":
                assert (
                    candidate_setup(obj, source, (qkv, z), pair32, upload, ttnn, mesh)
                    == source_scope["setup_ast_sha256"]
                )
                assert compare(w.reshape(1, 1, 2048, 6144), ttnn.to_torch(obj.scan_output_weight))["exact"]
            result.setdefault("native_setup_ast_hashes", {})[name] = setup_hash
            ttnn.synchronize_device(mesh)
            result["setup"].append(
                dict(
                    name=name,
                    elapsed_ms=1000 * (time.perf_counter() - setup_start),
                    scope="Lightweight checkpoint projection setup plus candidate weight/index readback verification; not runtime latency",
                )
            )
            models[name] = obj

        result["status"] = "running"
        save()
        for rows, k in cases:
            dx = upload(x[:, :rows])
            row = dict(rows=rows, projection_k=k, comparisons={})
            result["cases"].append(row)
            references = {}
            for label, dtype in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
                with device_only():
                    old = models["baseline"].linear(dx, "output_only_packed", dtype, projection_k=k)
                    new = models["candidate"].linear(dx, "output_only_packed", dtype, projection_k=k)
                assert tuple(old.shape) == tuple(new.shape) == (1, rows, 6144)
                assert old.dtype == new.dtype == dtype
                assert old.layout == new.layout == ttnn.TILE_LAYOUT
                a, b = ttnn.to_torch(old), ttnn.to_torch(new)
                hf = (canonical_bf16 if label == "bf16" else canonical_fp32)[(rows, k)]
                expected_columns = ordered[(rows, k)].bfloat16() if label == "bf16" else ordered[(rows, k)]
                row["comparisons"][label] = {
                    "old_vs_new": metrics(a, b),
                    "baseline_vs_hf": metrics(hf, a),
                    "candidate_vs_hf": metrics(hf, b),
                    "ordered_selected_columns": metrics(expected_columns, b[..., COLUMNS]),
                    "candidate_q_vs_hf": metrics(hf[..., :2048], b[..., :2048]),
                    "candidate_z_vs_hf": metrics(hf[..., 2048:], b[..., 2048:]),
                }
                if label == "bf16":
                    references = {"baseline": a, "candidate": b}
                del old, new
                save()
            for name, obj in models.items():
                reference = references[name]
                trace, out, capture_open = None, None, False
                try:
                    for _ in range(args.warmup):
                        with device_only():
                            out = obj.linear(dx, "output_only_packed", projection_k=k)
                        ttnn.synchronize_device(mesh)
                        del out
                    # Complete Python/eager path: includes per-row dispatch and final concat.
                    eager_samples = []
                    for _ in range(args.eager_samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        with device_only():
                            out = obj.linear(dx, "output_only_packed", projection_k=k)
                        ttnn.synchronize_device(mesh)
                        eager_samples.append(1000 * (time.perf_counter() - start))
                        del out
                    with device_only():
                        out = obj.linear(dx, "output_only_packed", projection_k=k)
                    ttnn.synchronize_device(mesh)
                    eager_last = compare(reference, ttnn.to_torch(out))
                    del out
                    result["eager_timing"].append(
                        dict(
                            name=name,
                            rows=rows,
                            projection_k=k,
                            median_ms=statistics.median(eager_samples),
                            samples_ms=eager_samples,
                            last_vs_own_eager=eager_last,
                            scope="Complete eager linear adapter, host dispatch plus synchronized device completion",
                        )
                    )
                    assert eager_last["exact"]
                    save()
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        out = obj.linear(dx, "output_only_packed", projection_k=k)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(reference, ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(reference, ttnn.to_torch(out))
                    result["timing"].append(
                        dict(
                            name=name,
                            rows=rows,
                            projection_k=k,
                            median_ms=statistics.median(samples),
                            samples_ms=samples,
                            replays_per_sample=args.replays,
                            first_vs_own_eager=first,
                            last_vs_own_eager=last,
                        )
                    )
                    assert first["exact"] and last["exact"]
                finally:
                    if trace is not None:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        ttnn.synchronize_device(mesh)
                        ttnn.release_trace(mesh, trace)
                    out = None
                    save()
            del dx
            print("ORDERED_QZ_CASE_COMPLETE", rows, k, flush=True)
        result["ordered_selected_columns_exact"] = all(
            c["ordered_selected_columns"]["exact"] for row in result["cases"] for c in row["comparisons"].values()
        )
        result["status"] = "diagnostic_complete"
        assert sha(baseline) == result["baseline_sha256"] and sha(candidate) == result["candidate_sha256"]
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
                result["device_closed"] = True
            except Exception:
                result["status"] = "failed"
                result["close_error"] = traceback.format_exc()
        save()
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
    parser.add_argument("--candidate-snapshot", type=Path, default=DOC / "v5_ordered_qz.py.snapshot")
    parser.add_argument("--baseline-snapshot", type=Path, default=DOC.parent / "selected_fused_decoder_v5.py.snapshot")
    parser.add_argument("--eager-samples", type=int, default=3)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 33, 128])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v5_ordered_qz_device.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays, args.eager_samples) > 0
    run(args)


if __name__ == "__main__":
    main()
