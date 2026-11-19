# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: transient BF16 L1 weights and sequential K1024 PF blocks, complete eager and traced timing.

Loads checkpoint K/V/A/B weights only. Calls each frozen decoder's real linear
method, including pair gather, coefficients, layout changes, reductions and concat.
No production candidate selection or model acceptance is performed here.
"""

import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import statistics
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


def cpu_mapping_check(w, x, chunk):
    columns = [0, 1, 31, 32, 2047, 2048, 4095, 6143, 6144, 6175, 6176, 6207]
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    rows = min(x.shape[1], 33)
    coeff = x[0, :rows, order].reshape(rows, 2, 1024, 1).float()
    selected = w[:, columns].reshape(1, 2, 1024, len(columns)).float()
    products = coeff * selected
    partial = products[:, :, 0].clone()
    for j in range(1, 1024):
        partial = partial + products[:, :, j]
    expected = partial[:, 0] + partial[:, 1]
    outputs = []
    partial_checks = []
    for start in range(0, rows, chunk):
        count = min(chunk, rows - start)
        p = coeff[start : start + count] * selected
        running = torch.zeros_like(p[:, :, 0])
        for j in range(1024):
            running = running + p[:, :, j]
        partial_checks.append(compare(partial[start : start + count], running))
        outputs.append(running[:, 0] + running[:, 1])
    return dict(
        **compare(expected, torch.cat(outputs)),
        rows=rows,
        chunk_rows=chunk,
        partial_checks=partial_checks,
        selected_columns=columns,
        note="Copy-first reference versus +0 cumsum on real selected weights; no all-input signed-zero equivalence claim",
    )


def load_snapshot(path, name):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


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
        timing_scope="Actual snapshot linear method: pair gather, coefficient construction, all dtype/layout changes, transient DRAM-to-L1 BF16 weight staging, serial row slicing, sequential K1024 L1 product/full-prefix scans, terminal slices, FP32 partial add, cast, RM row output and concat",
        compute="HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False",
        cpu_mapping_checks={
            str(provenance["prefill_projection_microbatch_rows"]): cpu_mapping_check(
                w, x, provenance["prefill_projection_microbatch_rows"]
            )
        },
        cases=[],
        timing=[],
        eager_timing=[],
        device_open_parameters=dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768),
        candidate_scan_weight_bytes=25427968,
        transient_l1_bf16_weight_bytes=25427968,
        one_block_product_prefix_bytes=50855936,
        prefill_microbatch_rows=provenance["prefill_projection_microbatch_rows"],
        singleton_combined_base_unchanged=True,
        comparison_baseline_is_v4_combined=True,
    )
    assert all(
        c["exact"] and all(p["exact"] for p in c["partial_checks"]) for c in result["cpu_mapping_checks"].values()
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        save()
        print("CPU_PREFILL_CUMSUM_PASS", cases)
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
            obj = load_snapshot(source, "probe_prefill_cumsum_" + name).FusedDecoder()
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
            print("PREFILL_CUMSUM_CASE_COMPLETE", rows, k, flush=True)
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
    parser.add_argument("--candidate-snapshot", type=Path, default=DOC / "v4_prefill_weight_l1.py.snapshot")
    parser.add_argument("--baseline-snapshot", type=Path, default=DOC / "v4_combined_adapters.py.snapshot")
    parser.add_argument("--eager-samples", type=int, default=5)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 32, 33, 128, 129])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v4_prefill_weight_l1_probe.json")
    args = parser.parse_args()
    assert min(args.lengths) > 0 and min(args.warmup, args.samples, args.replays, args.eager_samples) > 0
    run(args)


if __name__ == "__main__":
    main()
