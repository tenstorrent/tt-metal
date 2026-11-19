# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: compact BF16 singleton L1 cumsum projection, complete adapter timing.

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


def cpu_mapping_check(w, x, k):
    columns = [0, 1, 31, 32, 2047, 2048, 4095, 6143, 6144, 6175, 6176, 6207]
    order = torch.arange(2048).reshape(1024, 2).flip(1).flatten()
    blocks = 2048 // k
    products = x[0, :, order].reshape(-1, blocks, k, 1).float() * w[:, columns].reshape(blocks, k, -1).float()
    old = products[:, :, 0].clone()
    scan = torch.zeros_like(old)
    for j in range(k):
        if j:
            old = old + products[:, :, j]
        scan = scan + products[:, :, j]
    partials = compare(old, scan)
    expected, actual = old[:, 0].clone(), scan[:, 0].clone()
    for j in range(1, blocks):
        expected = expected + old[:, j]
        actual = actual + scan[:, j]
    return dict(
        **compare(expected, actual),
        partials=partials,
        selected_columns=columns,
        note="Copies first product versus starts from +0; exact real-fixture check, not all-input equivalence",
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
    baseline = DOC.parent / "selected_fused_decoder_v3.py.snapshot"
    candidate = args.candidate_snapshot
    provenance = json.loads(candidate.with_suffix(".provenance.json").read_text())
    assert sha(baseline) == provenance["base_sha256"]
    assert sha(candidate) == provenance["candidate_sha256"]
    w = weights(args.checkpoint, args.layer)
    assert w.shape == (2048, 6208) and w.dtype == torch.bfloat16
    capture_paths = [Path(f"{args.capture_base}.pos{position}.pt") for position in args.positions]
    x = torch.cat(
        [torch.load(path, map_location="cpu", weights_only=False)["tt_values"]["input_norm"] for path in capture_paths],
        dim=1,
    )
    assert x.dtype == torch.bfloat16 and x.shape == (1, len(args.positions), 2048)
    cases = [(i, k) for i in range(len(args.positions)) for k in (512, 1024)]
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        source_sha256=sha(Path(__file__)),
        baseline_snapshot=str(baseline),
        baseline_sha256=sha(baseline),
        candidate_snapshot=str(candidate),
        candidate_sha256=sha(candidate),
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        layer=args.layer,
        seed=args.seed,
        state_weight_sha256=hashlib.sha256(w.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_sha256=hashlib.sha256(x.view(torch.uint8).numpy().tobytes()).hexdigest(),
        input_description="Saved real BF16 inputnorm rows and real checkpoint state projection weights",
        capture_provenance=[dict(path=str(p), sha256=sha(p)) for p in capture_paths],
        timing_scope="Actual snapshot linear method: pair gather, coefficient construction, all dtype/layout changes, accurate product, full cumsum, terminal-band and terminal-row slices, block reduction and BF16 output",
        compute="HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False",
        cpu_mapping_checks={str(k): cpu_mapping_check(w, x, k) for k in (512, 1024)},
        cases=[],
        timing=[],
        device_open_parameters=dict(physical_device_ids=[0], trace_region_size=32 * 1024 * 1024, l1_small_size=32768),
        candidate_extra_weight_bytes=25427968,
        singleton_only=True,
    )
    assert all(c["exact"] for c in result["cpu_mapping_checks"].values())

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        save()
        print("CPU_SINGLETON_CUMSUM_PASS", cases)
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
            obj = load_snapshot(source, "probe_singleton_cumsum_" + name).FusedDecoder()
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
            padded = torch.nn.functional.pad(w, (0, 960))
            obj.reduce_attention_weight = upload(padded.reshape(4, 512, 32, 224))
            obj.reduce_prefill_attention_weight = upload(padded.reshape(2, 1024, 32, 224))
            if name == "candidate":
                obj.scan_attention_weight = upload(w.reshape(1, 1, 2048, 6208))
            models[name] = obj

        result["status"] = "running"
        save()
        grid = mesh.compute_with_storage_grid_size()
        result["observed_compute_storage_grid"] = dict(x=grid.x, y=grid.y, cores=grid.x * grid.y)
        result["two_l1_tensors_plus_scan_cb_nominal_bytes_per_core"] = 2 * 50855936 / (grid.x * grid.y) + 32768
        for input_index, k in cases:
            rows = 1
            dx = upload(x[:, input_index : input_index + 1])
            row = dict(rows=rows, position=args.positions[input_index], projection_k=k, comparisons={})
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
                            position=args.positions[input_index],
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
            print("SINGLETON_CUMSUM_CASE_COMPLETE", rows, k, flush=True)
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
    parser.add_argument("--candidate-snapshot", type=Path, default=DOC / "v3_singleton_cumsum_l1.py.snapshot")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--positions", nargs="+", type=int, default=[128, 130, 7703])
    parser.add_argument("--capture-base", default=str(DOC.parent / "endurance_precise_probe.json"))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--replays", type=int, default=5)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "v3_singleton_cumsum_l1_probe.json")
    args = parser.parse_args()
    assert min(args.positions) >= 0 and min(args.warmup, args.samples, args.replays) > 0
    run(args)


if __name__ == "__main__":
    main()
