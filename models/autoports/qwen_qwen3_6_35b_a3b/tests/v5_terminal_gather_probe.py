# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: complete ordered projection with FP32 TILE terminal-row gather.

Real checkpoint weights, both output dtypes and K512/K1024. All transpose,
gather, crop, reduction, serial-prefill and final layout costs are included.
CPU-only mode never imports TTNN. Device execution belongs to the parent agent.
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


def weights(checkpoint, layer):
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    values = []
    for name in ("qkv", "a", "b"):
        key = f"model.language_model.layers.{layer}.linear_attn.in_proj_{name}.weight"
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            value = f.get_tensor(key)
        values.append(value[2048:] if name == "qkv" else value)
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    return torch.cat(values).T[order].contiguous()


def cpu_mapping_check():
    reports = []
    for blocks in (2, 4):
        # Distinct finite bit payloads make accidental row/face/column selection visible.
        bits = torch.arange(blocks * 32 * 6208, dtype=torch.int32).reshape(blocks, 1, 32, 6208)
        bits = bits + 0x3E000000
        tail = bits.view(torch.float32)
        transposed = tail.transpose(-2, -1)
        indices = torch.full((blocks, 1, 6208, 32), 31, dtype=torch.int64)
        gathered = torch.gather(transposed, -1, indices).transpose(-2, -1)
        reports.append(
            dict(
                blocks=blocks,
                logical_row=compare(tail[:, :, 31:32], gathered[:, :, :1]),
                all_physical_rows=compare(tail[:, :, 31:32].expand_as(gathered), gathered),
            )
        )
    assert all(r["logical_row"]["exact"] and r["all_physical_rows"]["exact"] for r in reports)
    return reports


def load_snapshot(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


def candidate_setup(obj, source, w, pair32, upload, ttnn, mesh):
    """Execute the actual setup-only index allocation from the candidate AST."""
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
    setup = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "from_state_dict")
    names = {"self.terminal_gather_indices"}
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
            state_weight=w,
            pair32=pair32,
            upload=upload,
            ttnn=ttnn,
            torch=torch,
            mesh_device=mesh,
            DRAM=ttnn.DRAM_MEMORY_CONFIG,
        )
        exec(compile(module, str(source) + ":projection_setup", "exec"), scope)
    return code_hash


def check_source_scope(baseline, candidate):
    methods = {}
    for key, path in (("baseline", baseline), ("candidate", candidate)):
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "FusedDecoder")
        methods[key] = {n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)}
    assert methods["baseline"].keys() == methods["candidate"].keys()
    changed = {k for k in methods["baseline"] if ast.dump(methods["baseline"][k]) != ast.dump(methods["candidate"][k])}
    assert changed == {"linear", "from_state_dict"}
    old = copy.deepcopy(methods["baseline"]["linear"])
    new = copy.deepcopy(methods["candidate"]["linear"])
    old_body, new_body = old.body[0].body, new.body[0].body
    old_i = next(
        i for i, n in enumerate(old_body) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "partial"
    )
    new_i = next(
        i
        for i, n in enumerate(new_body)
        if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "transposed_tail"
    )
    assert old_i == new_i
    assert ast.unparse(old_body[old_i].value) == "tail[:, :, 31:32, :]"
    new_body[new_i : new_i + 5] = [copy.deepcopy(old_body[old_i])]
    assert ast.dump(old) == ast.dump(new), "Only terminal extraction may change in linear"
    return dict(
        changed_methods=sorted(changed),
        only_terminal_extraction_changed=True,
        setup_ast_sha256=candidate_setup(None, candidate, None, None, None, None, None),
    )


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
    w = weights(args.checkpoint, args.layer)
    assert w.shape == (2048, 6208) and w.dtype == torch.bfloat16
    generator = torch.Generator().manual_seed(args.seed)
    x = torch.randn((1, max(args.lengths), 2048), generator=generator)
    x = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)).bfloat16()
    cases = [(1, 512), (1, 1024)] + [(s, 1024) for s in args.lengths if s > 1]
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
        state_weight_sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_description="Synthetic normalized BF16 rows, real checkpoint weights; not recorded model activations",
        timing_scope="Complete actual linear: input adapters, ordered products/cumsum, terminal band, both FP32 transposes, TILE gather, logical crop, unchanged block reduction, output cast; serial prefill includes every row plus concat/final tilize.",
        setup_scope="Compact checkpoint weights and actual candidate AST allocation of fully tiled UINT16 index tensors in DRAM, verified by readback; no runtime index creation.",
        source_scope=source_scope,
        dependencies={
            str(Path(__file__).with_name("conv_first_tap_mixed_probe.py")): sha(
                Path(__file__).with_name("conv_first_tap_mixed_probe.py")
            )
        },
        compute="HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False",
        cpu_mapping_checks=cpu_mapping_check(),
        cases=[],
        timing=[],
        eager_timing=[],
        device_open_parameters=dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768),
        additional_resident_dram_uint16_index_bytes=2383872,
        additional_resident_weight_bytes=0,
        transient_tail_payload_bytes={"K512": 3178496, "K1024": 1589248},
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
        print("CPU_TERMINAL_GATHER_PASS", cases)
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
            obj = load_snapshot(source, "probe_terminal_gather_" + name).FusedDecoder()
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
            obj.scan_attention_weight = upload(w.reshape(1, 1, 2048, 6208))
            if name == "candidate":
                assert candidate_setup(obj, source, w, pair32, upload, ttnn, mesh) == source_scope["setup_ast_sha256"]
                for blocks, indices in obj.terminal_gather_indices.items():
                    assert indices.memory_config() == ttnn.DRAM_MEMORY_CONFIG
                    assert tuple(indices.shape) == (blocks, 1, 6208, 32)
                    assert indices.layout == ttnn.TILE_LAYOUT
                    assert torch.all(ttnn.to_torch(indices) == 31)
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
            reference = None
            for label, dtype in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
                with device_only():
                    old = models["baseline"].linear(dx, "attention_packed", dtype, projection_k=k)
                    new = models["candidate"].linear(dx, "attention_packed", dtype, projection_k=k)
                assert tuple(old.shape) == tuple(new.shape) == (1, rows, 6208)
                assert old.dtype == new.dtype == dtype
                assert old.layout == new.layout == ttnn.TILE_LAYOUT
                a, b = ttnn.to_torch(old), ttnn.to_torch(new)
                row["comparisons"][label] = {
                    "all": compare(a, b),
                    "kv": compare(a[..., :6144], b[..., :6144]),
                    "ab": compare(a[..., 6144:], b[..., 6144:]),
                }
                if label == "bf16":
                    reference = a
                del old, new
                save()
            if not all(c["all"]["exact"] for c in row["comparisons"].values()):
                row["timing_skipped"] = "Exact old/new projection mismatch"
                continue
            for name, obj in models.items():
                trace, out, capture_open = None, None, False
                try:
                    for _ in range(args.warmup):
                        with device_only():
                            out = obj.linear(dx, "attention_packed", projection_k=k)
                        ttnn.synchronize_device(mesh)
                        del out
                    # Complete Python/eager path: includes per-row dispatch and final concat.
                    eager_samples = []
                    for _ in range(args.eager_samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        with device_only():
                            out = obj.linear(dx, "attention_packed", projection_k=k)
                        ttnn.synchronize_device(mesh)
                        eager_samples.append(1000 * (time.perf_counter() - start))
                        del out
                    with device_only():
                        out = obj.linear(dx, "attention_packed", projection_k=k)
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
                            last_vs_baseline=eager_last,
                            scope="Complete eager linear adapter, host dispatch plus synchronized device completion",
                        )
                    )
                    assert eager_last["exact"]
                    save()
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        out = obj.linear(dx, "attention_packed", projection_k=k)
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
                            first_vs_baseline=first,
                            last_vs_baseline=last,
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
            print("TERMINAL_GATHER_CASE_COMPLETE", rows, k, flush=True)
        result["device_exact"] = all(c["all"]["exact"] for row in result["cases"] for c in row["comparisons"].values())
        result["status"] = "exact" if result["device_exact"] else "different"
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
    parser.add_argument("--candidate-snapshot", type=Path, default=DOC / "v5_terminal_gather.py.snapshot")
    parser.add_argument("--baseline-snapshot", type=Path, default=DOC.parent / "selected_fused_decoder_v5.py.snapshot")
    parser.add_argument("--eager-samples", type=int, default=3)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 32, 33, 128, 129])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=3)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v5_terminal_gather_device.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays, args.eager_samples) > 0
    run(args)


if __name__ == "__main__":
    main()
