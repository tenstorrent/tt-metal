# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Same-input stock FP32 depthwise Conv1D diagnostic; no decoder modification.

Run serially on physical chip 0. All host conversion and weight preparation are
outside the timed closures. Timings include dispatch, adapter operations, and
device synchronization. Optional isolated trace replays time the same complete
adapters; these are not kernel-only or full-decoder latency measurements.
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

from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "doc/fused_decoder"
REVISION = "345d467b117d1fd5eda9b879a5988de9ae5f47a7"


def compare(expected, actual):
    expected = expected.contiguous().float()
    actual = actual.reshape(expected.shape).contiguous().float()
    a, b = expected.double().flatten(), actual.double().flatten()
    finite = torch.isfinite(a) & torch.isfinite(b)
    ac, bc = a - a.mean(), b - b.mean()
    denom = ac.norm() * bc.norm()
    bf_a, bf_b = expected.bfloat16(), actual.bfloat16()
    codes = bf_a.view(torch.int16) != bf_b.view(torch.int16)
    ordinary = torch.maximum(expected.abs(), actual.abs()) > 1e-30
    return {
        "elements": a.numel(),
        "nonfinite_pairs": int((~finite).sum()),
        "pcc": 1.0 if torch.equal(a, b) else (float(ac @ bc / denom) if denom else None),
        "relative_l2": float((a - b).norm() / a.norm()) if a.norm() else None,
        "max_abs": float((a - b).abs().max()),
        "different_fp32_codes": int((expected.view(torch.int32) != actual.view(torch.int32)).sum()),
        "different_bf16_codes": int(codes.sum()),
        "ordinary_bf16_code_differences": int((codes & ordinary).sum()),
        "different_values": int((expected != actual).sum()),
    }


def load_cases(args):
    checkpoint = (
        Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
        / "hub/models--Qwen--Qwen3.6-35B-A3B/snapshots"
        / REVISION
    )
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]
    weights = {}
    for name in ("input_layernorm.weight", "linear_attn.in_proj_qkv.weight", "linear_attn.conv1d.weight"):
        key = "model.language_model.layers.0." + name
        with safe_open(checkpoint / index[key], framework="pt", device="cpu") as f:
            weights[name] = f.get_tensor(key)
    eps = json.loads((checkpoint / "config.json").read_text())["text_config"]["rms_norm_eps"]
    torch.manual_seed(args.seed)
    prefix = torch.randn(1, 128, 2048, dtype=torch.bfloat16) * 0.05
    tokens = torch.randn(1, 8192, 2048, dtype=torch.bfloat16) * 0.05

    def projection(x):
        x = x.float()
        normalized = (
            x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * (1 + weights["input_layernorm.weight"].float())
        ).bfloat16()
        return torch.nn.functional.linear(normalized, weights["linear_attn.in_proj_qkv.weight"])

    projected_prefix = projection(prefix)
    cases = [
        {
            "name": "canonical_decode_history4",
            "history": torch.cat([projected_prefix[:, -3:], projection(tokens[:, :1])], dim=1),
        },
        {
            "name": "canonical_prefill_history131",
            "history": torch.cat([torch.zeros(1, 3, 8192, dtype=torch.bfloat16), projected_prefix], dim=1),
        },
    ]
    if args.capture:
        captured = torch.load(args.capture, map_location="cpu", weights_only=False)
        values = captured["tt_values"]
        cases.append(
            {
                "name": "captured_decode_history4",
                "history": torch.cat(
                    [
                        values["conv_input_state"].bfloat16().reshape(1, 3, 8192),
                        values["projection_packed"][..., :8192].bfloat16().reshape(1, 1, 8192),
                    ],
                    dim=1,
                ),
            }
        )
    taps = weights["linear_attn.conv1d.weight"].bfloat16().contiguous()
    for case in cases:
        history = case["history"].contiguous()
        s = history.shape[1] - 3
        acc = history[:, :s].float() * taps[:, 0, 0].float()
        for j in range(1, 4):
            acc = acc + history[:, j : j + s].float() * taps[:, 0, j].float()
        case.update(
            history=history,
            output_length=s,
            cpu_manual=acc,
            cpu_activated=torch.nn.functional.silu(acc.bfloat16().float()).bfloat16(),
        )
        cpu_conv = torch.nn.functional.conv1d(history.transpose(1, 2), taps, groups=8192).transpose(1, 2)
        case["cpu_bf16_conv_vs_manual"] = compare(acc.bfloat16(), cpu_conv)
    return cases, taps


def run(args):
    import ttnn

    torch.set_num_threads(8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record(output)
    device_parameters = dict(
        physical_device_ids=[0],
        trace_region_size=args.trace_region_size if args.trace else 0,
        l1_small_size=args.l1_small_size,
    )
    provenance_path = Path(str(output) + ".provenance.json")
    provenance = json.loads(provenance_path.read_text())
    provenance["device_open_parameters"] = device_parameters
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    result = {
        "purpose": "Native depthwise Conv1D keep/reject evidence; no full-decoder acceptance claim",
        "execution": "serial physical chip 0, eager and isolated traces"
        if args.trace
        else "serial physical chip 0, eager",
        "checkpoint_revision": REVISION,
        "seed": args.seed,
        "channel_chunks": args.channel_chunks,
        "device_open_parameters": device_parameters,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "trace_enabled": args.trace,
        "trace_replays_per_sample": args.trace_replays,
        "trace_allocation_tracking": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
        "timings_include_tracking_overhead": os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "timing_scope": "Warm synchronous wall time per complete 8192-channel adapter; setup and readback excluded",
        "native_contract": "FP32 input/weights/output; groups=C; kernel4; stride1; padding0; no activation; L1_FULL",
        "status": "running",
        "cases": [],
    }
    if args.capture:
        result["capture"] = str(Path(args.capture).resolve())
        result["capture_sha256"] = hashlib.sha256(Path(args.capture).read_bytes()).hexdigest()

    def save():
        output.write_text(json.dumps(result, indent=2) + "\n")

    save()
    mesh = None
    stage = "cpu_setup"
    try:
        cases, weights = load_cases(args)
        torch.save(
            {"weights_bf16": weights, "cases": [{"name": c["name"], "history_bf16": c["history"]} for c in cases]},
            str(output) + ".inputs.pt",
        )
        stage = "open_device"
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), **device_parameters)
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

        def upload(x, dtype):
            return ttnn.from_torch(
                x.contiguous(), device=mesh, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        def postprocess(acc):
            rounded = ttnn.typecast(ttnn.typecast(acc, ttnn.bfloat16), ttnn.float32)
            return ttnn.typecast(ttnn.silu(rounded), ttnn.bfloat16)

        def timing(fn):
            for _ in range(args.warmup):
                value = fn()
                ttnn.synchronize_device(mesh)
                del value
            samples = []
            for _ in range(args.iterations):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                value = fn()
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - start) * 1000)
                del value
            return {"median_ms": statistics.median(samples), "min_ms": min(samples), "samples_ms": samples}

        def trace_timing(fn, expected):
            # One live trace at a time. Its weights/input already exist and stay
            # alive; release it before another variant prepares or compiles.
            from ttnn.tools import trace_allocation_tracker

            for _ in range(args.warmup):
                warm = fn()
                ttnn.synchronize_device(mesh)
                del warm
            trace_id = None
            capture_open = False
            trace_output = None
            try:
                ttnn.synchronize_device(mesh)
                trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                capture_open = True
                trace_output = fn()
                ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                capture_open = False
                # Only this trace overwrites this exact output allocation. It
                # is read before release; no other trace can overlap its life.
                trace_allocation_tracker.acknowledge_corruptible(trace_output)
                for _ in range(args.warmup):
                    ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                first = ttnn.to_torch(trace_output).reshape(expected.shape).clone()
                samples = []
                for _ in range(args.iterations):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    for _ in range(args.trace_replays):
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - start) * 1000 / args.trace_replays)
                last = ttnn.to_torch(trace_output).reshape(expected.shape).clone()
                return {
                    "median_ms": statistics.median(samples),
                    "min_ms": min(samples),
                    "samples_ms": samples,
                    "scope": "Nonblocking replay groups, synchronized at group boundaries; ms per complete adapter",
                    "replays_per_sample": args.trace_replays,
                    "timed_replays": args.iterations * args.trace_replays,
                    "warm_replays": args.warmup,
                    "capture_count": 1,
                    "same_input_each_replay": True,
                    "input_refresh_in_timed_region": False,
                    "first_replay_vs_eager": compare(expected, first),
                    "last_replay_vs_eager": compare(expected, last),
                    "first_vs_last_replay": compare(first, last),
                }
            finally:
                if trace_id is not None:
                    try:
                        if capture_open:
                            ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                        ttnn.synchronize_device(mesh)
                    finally:
                        ttnn.release_trace(mesh, trace_id)
                del trace_output

        tt_taps = [upload(weights[:, 0, j].float().reshape(1, 1, 8192), ttnn.float32) for j in range(4)]
        for case in cases:
            row = {
                "name": case["name"],
                "history_shape": list(case["history"].shape),
                "cpu_bf16_conv_vs_manual": case["cpu_bf16_conv_vs_manual"],
                "native": [],
            }
            result["cases"].append(row)
            stage = case["name"] + ":upload"
            history = upload(case["history"], ttnn.bfloat16)
            s = case["output_length"]

            def manual():
                acc = ttnn.multiply(ttnn.typecast(history[:, :s, :], ttnn.float32), tt_taps[0])
                for j in range(1, 4):
                    acc = ttnn.addcmul(acc, ttnn.typecast(history[:, j : j + s, :], ttnn.float32), tt_taps[j])
                return acc

            stage = case["name"] + ":manual"
            reference = manual()
            manual_cpu = ttnn.to_torch(reference).reshape(1, s, 8192)
            manual_post_cpu = ttnn.to_torch(postprocess(reference)).reshape(1, s, 8192)
            row["manual_vs_cpu"] = compare(case["cpu_manual"], manual_cpu)
            row["manual_activated_vs_cpu"] = compare(case["cpu_activated"], manual_post_cpu)
            del reference
            row["manual_preactivation_timing"] = timing(manual)
            row["manual_full_timing"] = timing(lambda: postprocess(manual()))
            save()
            if args.trace:
                stage = case["name"] + ":manual_trace"
                row["manual_preactivation_trace_timing"] = trace_timing(manual, manual_cpu)
                row["manual_full_trace_timing"] = trace_timing(lambda: postprocess(manual()), manual_post_cpu)
                save()
            for channels in args.channel_chunks:
                candidate = {"channels_per_call": channels, "calls_per_adapter": 8192 // channels, "status": "running"}
                row["native"].append(candidate)
                prepared = []
                native_stage = "prepare_weights"
                try:
                    cfg = ttnn.Conv1dConfig(
                        weights_dtype=ttnn.float32,
                        shard_layout=ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                        act_block_h_override=32,
                        output_layout=ttnn.TILE_LAYOUT,
                        deallocate_activation=False,
                        activation=None,
                    )
                    slicing = ttnn.Conv2dL1FullSliceConfig
                    for lo in range(0, 8192, channels):
                        raw = ttnn.from_torch(
                            weights[lo : lo + channels].float().reshape(channels, 1, 1, 4),
                            dtype=ttnn.float32,
                            layout=ttnn.ROW_MAJOR_LAYOUT,
                        )
                        prepared.append(
                            ttnn.prepare_conv_weights(
                                weight_tensor=raw,
                                input_memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                input_layout=ttnn.ROW_MAJOR_LAYOUT,
                                weights_format="OIHW",
                                in_channels=channels,
                                out_channels=channels,
                                batch_size=1,
                                input_height=1,
                                input_width=s + 3,
                                kernel_size=(1, 4),
                                stride=(1, 1),
                                padding=(0, 0),
                                dilation=(1, 1),
                                has_bias=False,
                                groups=channels,
                                device=mesh,
                                input_dtype=ttnn.float32,
                                output_dtype=ttnn.float32,
                                conv_config=cfg,
                                compute_config=compute,
                                slice_config=slicing,
                            )
                        )

                    def native():
                        parts = []
                        for j, lo in enumerate(range(0, 8192, channels)):
                            inp = ttnn.to_memory_config(
                                ttnn.to_layout(
                                    ttnn.typecast(history[:, :, lo : lo + channels], ttnn.float32),
                                    ttnn.ROW_MAJOR_LAYOUT,
                                ),
                                ttnn.DRAM_MEMORY_CONFIG,
                            )
                            acc = ttnn.conv1d(
                                input_tensor=inp,
                                weight_tensor=prepared[j],
                                device=mesh,
                                in_channels=channels,
                                out_channels=channels,
                                batch_size=1,
                                input_length=s + 3,
                                kernel_size=4,
                                stride=1,
                                padding=0,
                                dilation=1,
                                groups=channels,
                                dtype=ttnn.float32,
                                conv_config=cfg,
                                compute_config=compute,
                                slice_config=slicing,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            )
                            parts.append(ttnn.reshape(acc, (1, s, channels)))
                        return ttnn.concat(parts, dim=-1)

                    native_stage = "native_first_call"
                    native_out = native()
                    actual = ttnn.to_torch(native_out).reshape(1, s, 8192)
                    activated = ttnn.to_torch(postprocess(native_out)).reshape(1, s, 8192)
                    candidate["preactivation_vs_cpu"] = compare(case["cpu_manual"], actual)
                    candidate["preactivation_vs_tt_manual"] = compare(manual_cpu, actual)
                    candidate["activated_vs_cpu"] = compare(case["cpu_activated"], activated)
                    candidate["activated_vs_tt_manual"] = compare(manual_post_cpu, activated)
                    del native_out
                    native_stage = "warm_timing"
                    candidate["preactivation_timing"] = timing(native)
                    candidate["full_timing"] = timing(lambda: postprocess(native()))
                    candidate["full_speedup_vs_manual"] = (
                        row["manual_full_timing"]["median_ms"] / candidate["full_timing"]["median_ms"]
                    )
                    if args.trace:
                        native_stage = "trace_timing"
                        candidate["preactivation_trace_timing"] = trace_timing(native, actual)
                        candidate["full_trace_timing"] = trace_timing(lambda: postprocess(native()), activated)
                        candidate["full_trace_speedup_vs_manual"] = (
                            row["manual_full_trace_timing"]["median_ms"] / candidate["full_trace_timing"]["median_ms"]
                        )
                    candidate["status"] = "completed"
                except Exception as exc:
                    candidate.update(
                        status="failed",
                        failed_stage=native_stage,
                        error_type=type(exc).__name__,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                finally:
                    prepared.clear()
                    save()
            del history
        result["status"] = (
            "completed_with_failures"
            if any(c["status"] == "failed" for r in result["cases"] for c in r["native"])
            else "completed"
        )
    except Exception as exc:
        result.update(
            status="failed",
            failed_stage=stage,
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        save()
        if mesh is not None:
            try:
                ttnn.close_mesh_device(mesh)
            except Exception as exc:
                result["close_error"] = str(exc)
                result["status"] = "failed"
                save()
                raise
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DOC / "endurance/stock_conv_precision.json"))
    parser.add_argument("--capture", help="Optional saved intermediate .pt with TT conv_input_state/projection_packed")
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--channel-chunks", type=int, nargs="+", default=[1024, 512])
    parser.add_argument(
        "--l1-small-size",
        type=int,
        default=32768,
        help="Bytes reserved per core for native convolution L1_SMALL metadata",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--trace", action="store_true", help="Also capture and time each prepared adapter in isolation")
    parser.add_argument(
        "--trace-region-size", type=int, default=32 * 1024 * 1024, help="Reserved trace bytes when --trace is enabled"
    )
    parser.add_argument(
        "--trace-replays", type=int, default=20, help="Nonblocking replays per synchronized timing sample"
    )
    args = parser.parse_args()
    if any(c <= 0 or c > 1024 or c % 32 or 8192 % c for c in args.channel_chunks):
        parser.error("channel chunks must be tile-aligned divisors of 8192, at most 1024")
    if args.warmup < 1 or args.iterations < 1:
        parser.error("warmup and iterations must be positive")
    if args.l1_small_size < 0 or args.l1_small_size % 32:
        parser.error("l1-small-size must be nonnegative and 32-byte aligned")
    if args.trace_replays < 1 or args.trace_region_size <= 0 or args.trace_region_size % 32:
        parser.error("trace-replays must be positive; trace-region-size must be positive and 32-byte aligned")
    run(args)


if __name__ == "__main__":
    main()
