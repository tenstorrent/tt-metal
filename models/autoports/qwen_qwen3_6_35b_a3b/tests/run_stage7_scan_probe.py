# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Opt-in TP4 exact scan probe on real layer34 weights/recorded activations.

CPU setup is default. Every oracle is the CURRENT production helper/consumer.
Historical fixture loading is reused only for real inputs and checkpoint tensors.
No full decoder, cache, dtype policy or shared runtime implementation is changed.
"""

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"
VARIANTS = {
    "read32": (32, "baseline"),
    "read64": (64, "baseline"),
    "nop": (2048, "nop"),
    "elide": (2048, "elide"),
    "read32_nop": (32, "nop"),
    "read64_nop": (64, "nop"),
    "read32_elide": (32, "elide"),
    "read64_elide": (64, "elide"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact(expected, actual):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    dtype = torch.int32 if expected.dtype == torch.float32 else torch.int16
    mismatch = expected.contiguous().view(dtype) != actual.contiguous().view(dtype)
    return {
        "exact": not bool(mismatch.any()),
        "bit_different": int(mismatch.sum()),
        "signed_zero_differences": int((mismatch & (expected == 0) & (actual == 0)).sum()),
        "max_abs": float((expected.float() - actual.float()).abs().max()),
    }


def fixtures(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.v6_terminal_projection_scan_probe import fixtures as real_fixtures

    state, previous, captures = real_fixtures(args.checkpoint, args.random_cases)
    q, k, v = state["linear_attn.in_proj_qkv.weight"].split((2048, 2048, 4096), dim=0)
    z, a, b = (state[f"linear_attn.in_proj_{x}.weight"] for x in ("z", "a", "b"))
    pair = torch.arange(2048).reshape(-1, 2).flip(1).flatten()
    combined = []
    for rank in range(4):
        qs, vs, hs = (
            slice(rank * 512, (rank + 1) * 512),
            slice(rank * 1024, (rank + 1) * 1024),
            slice(rank * 8, (rank + 1) * 8),
        )
        state_part = torch.cat(
            (
                k[qs],
                v[vs],
                torch.nn.functional.pad(a[hs], (0, 0, 0, 24)),
                torch.nn.functional.pad(b[hs], (0, 0, 0, 24)),
            ),
            dim=0,
        ).T[pair]
        output_part = torch.cat((q[qs], z[vs]), dim=0).T[pair]
        combined.append(torch.cat((state_part, output_part), dim=1))
    output_pair = torch.arange(4096).reshape(-1, 2).flip(1).flatten()
    weights = {
        "input": torch.cat(combined, dim=1).reshape(1, 1, 2048, 12544).bfloat16(),
        "output": state["linear_attn.out_proj.weight"].T[output_pair].reshape(1, 1, 4096, 2048).bfloat16(),
        "router": state["mlp.gate.weight"].T[pair].reshape(1, 1, 2048, 256).bfloat16(),
    }
    cases = []
    for case in previous:
        prefix, suffix = case["name"].split("/", 1)
        if prefix.startswith("actual_") and int(prefix.split("_")[1]) not in args.positions:
            continue
        role = {"router": "router", "out": "output", "attention_k512": "input", "attention_k1024": "input1024"}.get(
            suffix
        )
        if role not in args.roles:
            continue
        cases.append({"name": f"{prefix}/{role}", "role": role, "x": case["x"].reshape(1, 1, -1).contiguous()})
    assert cases
    return weights, cases, captures


def cpu_mapping():
    # Exhaust every BF16 code at all valid original-row lanes; same proof as
    # v6_direct_coefficient_stream_probe, extended to both narrow read widths.
    codes = torch.arange(65536, dtype=torch.int32).to(torch.int16).reshape(-1, 32)
    tile = torch.zeros((len(codes), 1024), dtype=torch.int16)
    tile[:, :16], tile[:, 256:272] = codes[:, :16], codes[:, 16:]
    for transfer in (32, 64):
        narrow = torch.full_like(tile, 123)
        narrow[:, : transfer // 2] = tile[:, : transfer // 2]
        narrow[:, 256 : 256 + transfer // 2] = tile[:, 256 : 256 + transfer // 2]
        got = torch.stack([narrow[:, ((r ^ 1) // 16) * 256 + (r ^ 1) % 16] for r in range(32)], dim=1)
        assert torch.equal(got, codes[:, torch.arange(32) ^ 1])
    addresses = [{(p % 4) * 4 + (p // 4) * 32 + j for j in (0, 2, 16, 18)} for p in range(8)]
    assert all(not addresses[p] & addresses[q] for p in range(8) for q in range(p + 1, 8))
    return {
        "all_65536_bf16_codes_narrow32_narrow64": True,
        "scan_pass_dst_sets_disjoint": True,
        "hardware_hazard_proof": False,
    }


def run(args):
    torch.set_num_threads(8)
    assert not args.output.exists(), "Preserve prior evidence"
    if args.run_device:
        assert args.timing, "Device acceptance requires --timing (trace replay checks)"
        assert 227 in args.positions, "Device acceptance requires real position227 trace fixtures"
        assert args.repeats > 0 and args.steps > 0
        assert os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") == ("1" if args.mode == "acceptance" else "0")
        assert os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
        assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", "")
        if args.mode == "timing":
            assert os.environ.get("TT_METAL_WATCHER", "0") in ("0", "")
    weights, cases, captures = fixtures(args)
    if args.run_device:
        assert {c["role"] for c in cases if c["name"].startswith("actual_227/")} == set(args.roles)
    result = {
        "status": "cpu_setup_pass",
        "device_executed": False,
        "variant": args.variant,
        "mode": args.mode,
        "timing_eligible": args.run_device and args.mode == "timing",
        "output_activation_layout": "Production TP4LocalSiluMultiply: 8 row-major width shards [32,128]",
        "prior_evidence_limitation": "Earlier scan probe artifacts used 32 width shards [32,32] for output activation; those output results do not qualify the production layout and are preserved unchanged",
        "mesh": [1, 4],
        "checkpoint": str(args.checkpoint),
        "revision": REVISION,
        "argv": sys.argv,
        "cpu_mapping": cpu_mapping(),
        "captures": captures,
        "fixture_scope": "Real layer34 weights and saved layer34 activations from positions227/738/5799 as selected; synthetic cases are extra controls",
        "oracle": "Current production DirectCoefficientStreamingScan and current LinearTP4Mixin/FusedDecoder.linear consumers; no historical snapshot runtime",
        "precision": "BF16 operands; unchanged HiFi4 FP32 FPU product/Default ordered cumsum/FP32 block reduction; consumer casts unchanged",
        "source_sha256": {
            str(p.relative_to(REPO)): sha(p)
            for p in [
                Path(__file__),
                ROOT / "tt/scan_terminal_probe.py",
                ROOT / "tt/fused_decoder.py",
                ROOT / "tt/multichip_decoder.py",
                REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_cumsum.h",
            ]
        },
        "environment": {
            k: os.environ.get(k)
            for k in (
                "TT_METAL_TRACE_ALLOC_TRACKING",
                "TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE",
                "TT_METAL_WATCHER",
                "TT_METAL_DEVICE_PROFILER",
            )
        },
        "fixtures": [{"name": c["name"], "shape": list(c["x"].shape)} for c in cases],
        "cases": [],
        "timing": [],
        "trace_lifetime": "One variant/case trace at a time; exact final output backing acknowledged only while its producer trace alone is live; consumed before release; later preparation follows release",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if not args.run_device:
        assert "ttnn" not in sys.modules
        print("CPU_SETUP_PASS", len(cases), args.variant)
        return
    assert os.environ.get("TT_METAL_DEVICE_PROFILER", "0") in ("0", "")
    import ttnn as t
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.fused_decoder import DirectCoefficientStreamingScan, FusedDecoder
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import LinearTP4Mixin
    from models.autoports.qwen_qwen3_6_35b_a3b.tt.scan_terminal_probe import ScanTerminalProbe

    mesh = None
    try:
        t.set_fabric_config(t.FabricConfig.FABRIC_1D)
        mesh = t.open_mesh_device(t.MeshShape(1, 4), l1_small_size=32768, trace_region_size=67108864)
        result.update(status="running", device_executed=True, device_closed=False)

        def upload(value, dim=None, dtype=t.bfloat16):
            return t.from_torch(
                value.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=t.TILE_LAYOUT,
                memory_config=t.DRAM_MEMORY_CONFIG,
                mesh_mapper=t.ReplicateTensorToMesh(mesh) if dim is None else t.ShardTensorToMesh(mesh, dim=dim),
            )

        dw = {
            "input": upload(weights["input"], 3),
            "output": upload(weights["output"], 2),
            "router": upload(weights["router"]),
        }
        baseline = DirectCoefficientStreamingScan(t, mesh)
        read_bytes, stores = VARIANTS[args.variant]
        candidate = ScanTerminalProbe(t, mesh, read_bytes=read_bytes, stores=stores)
        for name, source in (("reader", candidate.reader_source), ("compute", candidate.compute_source)):
            path = Path(str(args.output) + f".{name}.cpp")
            path.write_text(source)
        owner = types.SimpleNamespace(
            kind="linear_attention",
            mesh_device=mesh,
            scan_combined_weight=dw["input"],
            scan_attention_out_weight=dw["output"],
            scan_router_weight=dw["router"],
            compute=t.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=t.MathFidelity.HiFi4,
                math_approx_mode=False,
                fp32_dest_acc_en=True,
                packer_l1_acc=False,
            ),
        )

        def host(value):
            return [t.to_torch(x) for x in t.get_device_tensors(value)]

        def compare(left, right):
            values = [exact(x, y) for x, y in zip(left, right)]
            assert len(values) == 4
            return {"exact": all(x["exact"] for x in values), "ranks": values}

        def final(helper, dx, role, dtype):
            owner.direct_coefficient_scan = helper
            if role == "router":
                return FusedDecoder.linear(owner, dx, "mlp.gate.weight", dtype, ordered_router=True, terminal_scan=True)
            return LinearTP4Mixin.linear(
                owner,
                dx,
                "linear_attn.out_proj.weight" if role == "output" else "attention_combined",
                dtype,
                projection_k=1024 if role == "input1024" else 512,
                ordered_out=role == "output",
                terminal_scan=True,
            )

        for case in cases:
            role = case["role"]
            dx = upload(case["x"], 2 if role == "output" else None)
            if role == "output":
                cores = t.num_cores_to_corerangeset(8, mesh.compute_with_storage_grid_size(), row_wise=True)
                dx = t.to_memory_config(
                    dx,
                    t.MemoryConfig(
                        t.TensorMemoryLayout.WIDTH_SHARDED,
                        t.BufferType.L1,
                        t.ShardSpec(cores, (32, 128), t.ShardOrientation.ROW_MAJOR),
                    ),
                )
            key = "input" if role == "input1024" else role
            pk = 1024 if role == "input1024" else 512
            total_k = 1024 if role == "output" else 2048
            width = {"input": 3136, "output": 2048, "router": 256}[key]
            weight = t.reshape(dw[key], (total_k // pk, 1, pk, width))
            row = {
                "name": case["name"],
                "weight_shape": list(weight.shape),
                "activation_memory": str(dx.memory_config()),
                "checks": {},
            }
            for label, kwargs in (
                ("coefficients", {"coefficients_only": True}),
                ("products_fp32", {"products_only": True}),
                ("partial_fp32", {}),
            ):
                with device_only():
                    old = baseline(weight, dx, **kwargs)
                expected = host(old)
                del old
                with device_only():
                    new = candidate(weight, dx, **kwargs)
                row["checks"][label] = compare(expected, host(new))
                del new
            for dtype, label in ((t.float32, "final_fp32"), (t.bfloat16, "final_bf16")):
                with device_only():
                    old = final(baseline, dx, role, dtype)
                expected, memory = host(old), old.memory_config()
                del old
                with device_only():
                    new = final(candidate, dx, role, dtype)
                assert new.memory_config() == memory
                row["checks"][label] = compare(expected, host(new))
                del new
            result["cases"].append(row)
            save()
            assert all(x["exact"] for x in row["checks"].values()), row
            if args.timing and case["name"].startswith("actual_227/"):
                # Stable alternate inputs and every copy signature are prepared
                # before capture; they survive until their sole trace is released.
                alternate_inputs, alternate_references = [], []
                for value in (case["x"], (case["x"].float() * 0.5).bfloat16()):
                    alt = upload(value, 2 if role == "output" else None)
                    alt = t.to_memory_config(alt, dx.memory_config())
                    alternate_inputs.append(alt)
                    t.copy(alt, dx)
                    with device_only():
                        check = final(baseline, dx, role, t.bfloat16)
                    alternate_references.append(host(check))
                    del check
                t.copy(alternate_inputs[0], dx)
                # All new signatures are warmed while no trace exists.
                for label, helper in (("baseline", baseline), (args.variant, candidate)):
                    with device_only():
                        output = final(helper, dx, role, t.bfloat16)
                    reference = host(output)
                    del output
                    t.synchronize_device(mesh)
                    trace = t.begin_trace_capture(mesh, cq_id=0)
                    try:
                        with device_only():
                            output = final(helper, dx, role, t.bfloat16)
                    finally:
                        t.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        from ttnn.tools import trace_allocation_tracker

                        trace_allocation_tracker.acknowledge_corruptible(output)
                        samples = []
                        entries = mesh.num_program_cache_entries()
                        for _ in range(args.repeats):
                            t.synchronize_device(mesh)
                            started = time.perf_counter()
                            for _ in range(args.steps):
                                t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            t.synchronize_device(mesh)
                            samples.append((time.perf_counter() - started) * 1000 / args.steps)
                        replay = compare(reference, host(output))
                        assert replay["exact"] and entries == mesh.num_program_cache_entries()
                        changed = []
                        for index in (1, 0, 1, 0):
                            t.copy(alternate_inputs[index], dx)
                            t.execute_trace(mesh, trace, cq_id=0, blocking=False)
                            match = compare(alternate_references[index], host(output))
                            assert match["exact"]
                            changed.append({"input_variant": index, "output": match})
                        assert entries == mesh.num_program_cache_entries()
                        result["timing"].append(
                            {
                                "changed_input_replays": changed,
                                "name": case["name"],
                                "variant": label,
                                "samples_ms": samples,
                                "median_ms": statistics.median(samples),
                                "replays": args.repeats * args.steps,
                                "after_replay": replay,
                                "program_cache_entries": entries,
                                "scope": "complete current projection consumer including K512 block reduction and final cast/layout; boundary synchronization only",
                            }
                        )
                        save()
                    finally:
                        t.release_trace(mesh, trace)
                    del output
                del alt, alternate_inputs, alternate_references
            del dx, weight
        expected_replays = {
            (f"actual_227/{role}", variant) for role in args.roles for variant in ("baseline", args.variant)
        }
        observed_replays = {(row["name"], row["variant"]) for row in result["timing"]}
        assert observed_replays == expected_replays, (observed_replays, expected_replays)
        assert all(row["after_replay"]["exact"] and len(row["changed_input_replays"]) == 4 for row in result["timing"])
        result["required_trace_coverage"] = sorted([list(item) for item in expected_replays])
        result["status"] = "component_pass"
    except Exception:
        result.update(status="error", error=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            t.close_mesh_device(mesh)
            result["device_closed"] = True
        save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/home/ttuser/qwen36-35b-a3b/hf"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION,
    )
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--positions", nargs="+", type=int, default=[227, 738, 5799])
    parser.add_argument(
        "--roles",
        nargs="+",
        choices=("input", "output", "router", "input1024"),
        default=["input", "output", "router", "input1024"],
    )
    parser.add_argument("--random-cases", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-device", action="store_true")
    parser.add_argument("--mode", choices=("acceptance", "timing"), default="acceptance")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
