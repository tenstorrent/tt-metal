# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Compare generic BF16 expert sum with the same native reduction sans padding fill.

Diagnostic only. Starts after BF16 routing multiplication; compares complete
sum/crop/reshape adapters, including generic sum's clone and padding preparation.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"


def code_check(a, b):
    a, b = a.bfloat16().contiguous(), b.bfloat16().contiguous()
    assert a.shape == b.shape
    aa, bb = a.view(torch.int16).flatten(), b.view(torch.int16).flatten()
    bad = torch.nonzero(aa != bb).flatten()
    return dict(
        elements=aa.numel(),
        code_differences=bad.numel(),
        exact=bad.numel() == 0,
        examples=[dict(index=int(i), before_code=int(aa[i]) & 65535, after_code=int(bb[i]) & 65535) for i in bad[:8]],
    )


def padding_report(padding, fill):
    value = padding.bfloat16().contiguous()
    codes, counts = torch.unique(value.view(torch.int16).flatten(), return_counts=True)
    valid = torch.isnan(value) if math.isnan(fill) else value == fill
    return dict(
        elements=value.numel(),
        requested=repr(fill),
        injection_verified=bool(valid.all()),
        matching_elements=int(valid.sum()),
        finite_elements=int(torch.isfinite(value).sum()),
        nan_elements=int(torch.isnan(value).sum()),
        positive_infinity_elements=int(torch.isposinf(value).sum()),
        negative_infinity_elements=int(torch.isneginf(value).sum()),
        unique_codes=codes.numel(),
        code_counts=[
            dict(code=f"0x{int(code) & 65535:04x}", count=int(count)) for code, count in zip(codes[:32], counts[:32])
        ],
    )


def run(args):
    torch.set_num_threads(4)
    output = Path(args.output)
    assert args.overwrite or not output.exists(), f"Preserve existing evidence: {output}"
    output.parent.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator().manual_seed(args.seed)
    cases = []
    for _ in range(args.random_cases):
        down = torch.randn((1, 8, 1, 2048), dtype=torch.bfloat16, generator=generator)
        routing = torch.softmax(torch.randn((1, 8, 1, 1), generator=generator), dim=1).bfloat16()
        cases.append((down.float() * routing.float()).bfloat16())
    if args.weighted_capture:
        data = torch.load(args.weighted_capture, map_location="cpu", weights_only=False)
        captured = data if isinstance(data, torch.Tensor) else data[args.weighted_key]
        assert captured.dtype == torch.bfloat16 and tuple(captured.shape) == (1, 8, 1, 2048)
        cases.insert(0, captured.contiguous())
    assert all(torch.isfinite(x).all() for x in cases)
    result = dict(
        classification="diagnostic_only_not_acceptance",
        status="setup",
        cases=len(cases),
        seed=args.seed,
        capture=args.weighted_capture,
        capture_key=args.weighted_key,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        compute_config=dict(math_fidelity="HiFi4", math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False),
        config_evidence="profile_projection_l1/tracy/layer_0/ops.csv BF16 FastReduceNC(dim1) attributes",
        contract="BF16 TILE DRAM [1,8,1,2048] -> BF16 TILE DRAM [1,1,2048]",
        timing_scope="Complete reduction/crop/reshape, beginning after the existing BF16 weighted product",
        checks=[],
        injection_attempts=[],
        unverified_padding_cases=[],
        timing=[],
    )

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    # Address-dependency proof only; native arithmetic is checked against native baseline below.
    for value in cases:
        expanded = value.expand(1, 8, 32, 2048).clone()
        zero, poisoned = expanded.clone(), expanded.clone()
        zero[:, :, 1:] = 0
        poisoned[:, :, 1:] = float("nan")
        assert code_check(zero.float().sum(1)[:, :1].bfloat16(), poisoned.float().sum(1)[:, :1].bfloat16())["exact"]
    if args.oracle_only:
        result["status"] = "cpu_padding_independence_pass"
        save()
        print(result["status"])
        return

    for key in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER", "TT_METAL_TRACE_ALLOC_TRACKING"):
        assert os.environ.get(key) in (None, "", "0"), f"Clean microbenchmark requires {key} disabled"
    import ttnn
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only

    record(output)
    mesh = None
    save()
    try:
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=16 * 1024 * 1024)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def adapter(x, native):
            if native:
                total = ttnn.experimental.fast_reduce_nc(
                    x, dims=[1], compute_kernel_config=compute, memory_config=ttnn.DRAM_MEMORY_CONFIG
                )
                total = total[:, :, :1, :2048]
            else:
                total = ttnn.sum(x, dim=1)  # preserve current generic-sum defaults
            return ttnn.to_layout(ttnn.reshape(total, (1, 1, 2048)), ttnn.TILE_LAYOUT)

        def host(x):
            assert tuple(x.shape) == (1, 1, 2048) and x.dtype == ttnn.bfloat16
            assert x.layout == ttnn.TILE_LAYOUT and x.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            return ttnn.to_torch(x).clone()

        def physical(x):
            return ttnn.from_device(x).to_torch_with_padded_shape().clone()

        timing_input, timing_reference = None, None
        for case, value in enumerate(cases):
            baseline_zero = None
            for label, fill in (
                ("zero", 0.0),
                ("large_positive", 32768.0),
                ("large_negative", -32768.0),
                ("positive_infinity", float("inf")),
                ("negative_infinity", -float("inf")),
                ("nan", float("nan")),
            ):
                x = ttnn.from_torch(
                    value,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                x = ttnn.fill_implicit_tile_padding(x, fill)
                before = physical(x)
                assert tuple(before.shape) == (1, 8, 32, 2048)
                observed = padding_report(before[:, :, 1:], fill)
                method = "fill_implicit_tile_padding"
                result["injection_attempts"].append(dict(case=case, padding=label, method=method, observed=observed))
                save()
                if not observed["injection_verified"]:
                    # Setup-only host tilization avoids the SFPU fill/store path.
                    # Verify host bytes and transferred device bytes independently.
                    del x
                    host_input = ttnn.from_torch(
                        value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, pad_value=fill, preserve_nan_values=True
                    )
                    host_observed = padding_report(host_input.to_torch_with_padded_shape()[:, :, 1:], fill)
                    method = "host_tilize_pad_value_then_to_device"
                    x = ttnn.to_device(host_input, mesh, memory_config=ttnn.DRAM_MEMORY_CONFIG)
                    before = physical(x)
                    observed = padding_report(before[:, :, 1:], fill)
                    result["injection_attempts"].append(
                        dict(case=case, padding=label, method=method, host_observed=host_observed, observed=observed)
                    )
                    del host_input
                    save()
                if not observed["injection_verified"]:
                    result["unverified_padding_cases"].append(
                        dict(
                            case=case,
                            padding=label,
                            reason="Neither device fill nor host pad upload produced the requested physical padding; reductions not tested for this case",
                        )
                    )
                    del x
                    save()
                    continue
                assert code_check(value, before[:, :, :1])["exact"]
                with device_only():
                    old = adapter(x, False)
                old_value = host(old)
                old_unchanged = code_check(before, physical(x))
                with device_only():
                    new = adapter(x, True)
                new_value = host(new)
                if baseline_zero is None:
                    baseline_zero = old_value
                row = dict(
                    case=case,
                    padding=label,
                    injection_method=method,
                    physical_padding=observed,
                    old_vs_new=code_check(old_value, new_value),
                    generic_padding_independence=code_check(baseline_zero, old_value),
                    native_padding_independence=code_check(baseline_zero, new_value),
                    original_input_unchanged=old_unchanged,
                    native_input_unchanged=code_check(before, physical(x)),
                )
                result["checks"].append(row)
                save()
                assert all(v["exact"] for v in row.values() if isinstance(v, dict) and "exact" in v), row
                if case == 0 and label == "zero":
                    timing_input, timing_reference = x, old_value
                del old, new, x
        for native in (False, True):
            trace_id, capture_open, out = None, False, None
            try:
                for _ in range(args.warmup):
                    with device_only():
                        out = adapter(timing_input, native)
                    ttnn.synchronize_device(mesh)
                    del out
                    out = None
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                with device_only():
                    out = adapter(timing_input, native)
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                capture_open = False
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                first = code_check(timing_reference, host(out))
                samples = []
                for _ in range(args.samples):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.replays_per_sample):
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append(1000 * (time.perf_counter() - start) / args.replays_per_sample)
                last = code_check(timing_reference, host(out))
                assert first["exact"] and last["exact"]
                result["timing"].append(
                    dict(
                        name="direct_fast_reduce" if native else "generic_sum",
                        median_ms=statistics.median(samples),
                        samples_ms=samples,
                        replays_per_sample=args.replays_per_sample,
                        first_vs_original=first,
                        last_vs_original=last,
                    )
                )
            finally:
                if trace_id is not None:
                    if capture_open:
                        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace_id)
                del out
                save()
        result["status"] = "exact" if not result["unverified_padding_cases"] else "exact_for_verified_padding_only"
        result["nan_padding_tested_all_cases"] = sum(r["padding"] == "nan" for r in result["checks"]) == len(cases)
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
        save()
    print("MOE_PADDING_ELISION", result["status"], output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DOC / "endurance/moe_padding_elision_probe_v2.json"))
    parser.add_argument("--seed", type=int, default=154)
    parser.add_argument("--random-cases", type=int, default=3)
    parser.add_argument("--weighted-capture")
    parser.add_argument("--weighted-key", default="weighted")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--replays-per-sample", type=int, default=20)
    parser.add_argument("--oracle-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    assert min(args.random_cases, args.warmup, args.samples, args.replays_per_sample) > 0
    run(args)
