# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic only: first convolution tap cast versus mixed-input SFPU multiply.

Enumerates every finite BF16 code for representative checkpoint tap values.
Timings include slicing and the original cast, at real convolution dimensions.
No decoder acceptance claim follows from this isolated product comparison.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder/endurance"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def compare(expected, actual):
    a, b = expected.float().contiguous().flatten(), actual.float().contiguous().flatten()
    assert a.shape == b.shape
    ac, bc = a.view(torch.int32), b.view(torch.int32)
    bad = torch.nonzero(ac != bc).flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    return dict(
        elements=a.numel(),
        exact=bad.numel() == 0,
        code_differences=bad.numel(),
        finite_pair_elements=int(finite.sum()),
        finite_pair_code_differences=int(((ac != bc) & finite).sum()),
        nonfinite_pair_elements=int((~finite).sum()),
        max_abs_finite=float((a[finite].double() - b[finite].double()).abs().max()) if finite.any() else None,
        first_differences=[
            dict(
                index=int(i),
                expected_code=f"0x{int(ac[i]) & 0xffffffff:08x}",
                actual_code=f"0x{int(bc[i]) & 0xffffffff:08x}",
            )
            for i in bad[:8]
        ],
    )


def setup(args):
    index = json.loads((args.checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    key = f"model.language_model.layers.{args.layer}.linear_attn.conv1d.weight"
    with safe_open(args.checkpoint / index[key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key)
    assert weight.shape == (8192, 1, 4) and torch.isfinite(weight).all()
    tap = weight[:, 0, 0].float().reshape(1, 1, 8192)
    unique = torch.unique(tap)
    selected = unique[torch.linspace(0, len(unique) - 1, args.tap_values).round().long()].unique()
    codes = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(torch.bfloat16)
    finite = codes[torch.isfinite(codes)].reshape(1, -1, 32)
    assert finite.numel() == 65280
    cases = [
        (f"exhaustive_tap_{i}", finite, torch.full((1, 1, 32), float(t)), finite.shape[1])
        for i, t in enumerate(selected)
    ]
    generator = torch.Generator().manual_seed(args.seed)
    # These are synthetic finite BF16 histories at the actual channel dimensions,
    # not claimed to be a capture of the running model.
    for s in args.lengths:
        history = (torch.randn((1, s + 3, 8192), generator=generator) * 3).bfloat16()
        cases.append((f"shape_s{s}", history, tap, s))
    return weight, selected, cases


def run(args):
    torch.set_num_threads(4)
    path = args.output
    assert args.overwrite or not path.exists(), f"Preserve existing evidence: {path}"
    path.parent.mkdir(parents=True, exist_ok=True)
    weight, selected, cases = setup(args)
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="cpu_setup_pass",
        checkpoint=str(args.checkpoint),
        checkpoint_revision=REVISION,
        layer=args.layer,
        source_sha256=digest(Path(__file__).read_bytes()),
        checkpoint_conv_dtype=str(weight.dtype),
        checkpoint_conv_sha256=digest(weight.view(torch.uint8).numpy().tobytes()),
        representative_first_tap_values=selected.tolist(),
        finite_bf16_codes_per_tap=65280,
        excluded_input_codes=256,
        seed=args.seed,
        baseline_snapshot_sha256=digest((DOC / "conv_first_tap_baseline.py.snapshot").read_bytes()),
        candidate_snapshot_sha256=digest((DOC / "conv_first_tap_mixed.py.snapshot").read_bytes()),
        timing_scope="First history slice + optional FP32 cast + multiply only; not full convolution or decoder",
        cases=[],
        timing=[],
    )

    def save():
        path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    if args.oracle_only:
        for name, x, tap, s in cases:
            ref = x[:, :s].float() * tap
            assert compare(ref, ref)["exact"]
            result["cases"].append(dict(name=name, shape=list(x.shape), output_elements=ref.numel()))
        save()
        print("CPU_SETUP_PASS", len(cases))
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

    record(path)
    mesh = None
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=16 * 1024 * 1024)

        def upload(x, dtype):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def adapter(x, tap, s, mixed):
            sliced = x[:, :s, :]
            if mixed:
                return ttnn.multiply(sliced, tap, dtype=ttnn.float32, fast_and_approximate_mode=False)
            return ttnn.multiply(ttnn.typecast(sliced, ttnn.float32), tap)

        result["status"] = "running"
        save()
        for name, x, tap, s in cases:
            dx, dt = upload(x, ttnn.bfloat16), upload(tap, ttnn.float32)
            with device_only():
                baseline = adapter(dx, dt, s, False)
                candidate = adapter(dx, dt, s, True)
            for out in (baseline, candidate):
                assert tuple(out.shape) == (1, s, x.shape[-1]) and out.dtype == ttnn.float32
                assert out.layout == ttnn.TILE_LAYOUT and out.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            old, new = ttnn.to_torch(baseline), ttnn.to_torch(candidate)
            ref = x[:, :s].float() * tap
            row = dict(
                name=name,
                shape=list(x.shape),
                input_sha256=digest(x.view(torch.uint8).numpy().tobytes()),
                candidate_vs_baseline=compare(old, new),
                baseline_vs_cpu=compare(ref, old),
                candidate_vs_cpu=compare(ref, new),
            )
            result["cases"].append(row)
            del baseline, candidate
            save()
            if not name.startswith("shape_") or not row["candidate_vs_baseline"]["exact"]:
                continue
            for mixed in (False, True):
                trace = None
                capture_open = False
                out = None
                try:
                    for _ in range(args.warmup):
                        with device_only():
                            out = adapter(dx, dt, s, mixed)
                        ttnn.synchronize_device(mesh)
                        del out
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    capture_open = True
                    with device_only():
                        out = adapter(dx, dt, s, mixed)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    capture_open = False
                    for _ in range(args.warmup):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    first = compare(old, ttnn.to_torch(out))
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        for _ in range(args.replays):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append(1000 * (time.perf_counter() - start) / args.replays)
                    last = compare(old, ttnn.to_torch(out))
                    result["timing"].append(
                        dict(
                            name=name,
                            mixed=mixed,
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
            del dx, dt
        result["device_exact"] = all(r["candidate_vs_baseline"]["exact"] for r in result["cases"])
        result["status"] = "exact" if result["device_exact"] else "different"
    except Exception:
        result.update(status="failed", error=traceback.format_exc())
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
    print(result["status"], path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--tap-values", type=int, default=8)
    parser.add_argument("--lengths", type=int, nargs="+", default=[1, 33, 128])
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output", type=Path, default=DOC / "conv_first_tap_mixed_probe.json")
    args = parser.parse_args()
    assert args.tap_values > 0 and all(s > 0 for s in args.lengths)
    assert args.warmup > 0 and args.samples > 0 and args.replays > 0
    run(args)


if __name__ == "__main__":
    main()
