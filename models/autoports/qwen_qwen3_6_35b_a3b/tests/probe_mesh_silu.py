# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact TP4 SiLU component gates; launch externally under a bounded timeout.

Optional --capture files contain {weighted: FP32, z: BF16}, or a list of such
records under "cases". Local shapes are [1,32,128] and [1,1,1024]; optionally
prepend a dimension of four for distinct rank inputs. Without captures, this
probe explicitly reports synthetic coverage, including every BF16 Z bit code.
All trace inputs, LUTs, expected controls, and output references survive until
every trace is released. Only exact replay-overwritten outputs are acknowledged.
"""

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_silu_probe import install
from models.autoports.qwen_qwen3_6_35b_a3b.tt.multichip_decoder import TP4LocalSiluMultiply


def _ranked(value, local_shape, dtype):
    if tuple(value.shape) == local_shape:
        value = value.unsqueeze(0).expand(4, *local_shape).clone()
    if tuple(value.shape) != (4, *local_shape) or value.dtype != dtype:
        raise ValueError(f"capture requires {dtype} shape {local_shape} or {(4, *local_shape)}")
    return value.contiguous()


def _cases(args):
    rng = torch.Generator().manual_seed(args.seed)
    result = []

    def weights(scale=1.0):
        return (torch.randn(4, 1, 32, 128, generator=rng) * scale).bfloat16().float()

    for path in args.capture:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        records = payload["cases"] if "cases" in payload else [payload]
        for i, row in enumerate(records):
            result.append(
                (
                    f"capture_{Path(path).stem}_{i}",
                    _ranked(row["weighted"], (1, 32, 128), torch.float32),
                    _ranked(row["z"], (1, 1, 1024), torch.bfloat16),
                )
            )
    for i in range(args.samples):
        result.append((f"random_realistic_{i}", weights(), (torch.randn(4, 1, 1, 1024, generator=rng) * 2).bfloat16()))
    for name, scale, sign in (("positive", 16.0, 1), ("negative", 16.0, -1), ("random_large", 1e8, 0)):
        z = torch.randn(4, 1, 1, 1024, generator=rng) * 24
        if sign:
            z = z.abs() * sign
        result.append((name, weights(scale), z.bfloat16()))
    tiny_codes = torch.tensor([0, 1, 2, 0x007F, 0x0080, 0x8000, 0x8001, 0x807F], dtype=torch.int32)
    tiny = tiny_codes.repeat(512).to(torch.int16).view(torch.bfloat16).reshape(4, 1, 1, 1024)
    result.append(("signed_zero_and_subnormal", weights(), tiny))
    result.append(("zero_weights", torch.zeros(4, 1, 32, 128), torch.randn(4, 1, 1, 1024, generator=rng).bfloat16()))
    # Sixteen mesh inputs cover all 65,536 BF16 bit patterns exactly once.
    # Each device gets a different 1024-code range, exposing rank/address bugs.
    for i in range(16):
        codes = torch.arange(i * 4096, (i + 1) * 4096, dtype=torch.int32)
        z = codes.to(torch.int16).view(torch.bfloat16).reshape(4, 1, 1, 1024)
        result.append((f"all_bf16_codes_{i:02d}", weights(3.0), z))
    return result


def _compare(actual, expected):
    if len(actual) != 4 or len(expected) != 4:
        return [dict(rank=None, passed=False, reason="requires all four physical rank outputs")]
    rows = []
    for rank, (a, b) in enumerate(zip(actual, expected)):
        if a.shape != b.shape or a.dtype != b.dtype:
            rows.append(dict(rank=rank, passed=False, reason="shape/dtype mismatch"))
            continue
        integer = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        ab, bb = a.contiguous().view(integer).flatten(), b.contiguous().view(integer).flatten()
        bad = torch.nonzero(ab != bb).flatten()
        rows.append(
            dict(
                rank=rank,
                passed=bad.numel() == 0,
                mismatches=bad.numel(),
                first_indices=bad[:8].tolist(),
                actual_bits=ab[bad[:8]].tolist(),
                expected_bits=bb[bad[:8]].tolist(),
            )
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--capture", action="append", default=[])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=154)
    args = parser.parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        raise ValueError("run with TT_METAL_TRACE_ALLOC_TRACKING=1")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") not in ("", "0"):
        raise ValueError("program-cache allocations must remain included in trace allocation tracking")
    if args.samples < 1 or args.rounds < 2:
        raise ValueError("require at least one random sample and two changing-input rounds")
    cases = _cases(args)
    report = dict(
        passed=False,
        arguments=vars(args),
        mesh=[1, 4],
        seed=args.seed,
        capture_paths=args.capture,
        input_source="saved captures plus synthetic"
        if args.capture
        else "synthetic only; no saved SiLU captures found",
        exhaustive_bf16_z_codes=65536,
        debug_gate=True,
        trace_allocation_tracking=True,
        program_cache_excluded=False,
        eager=[],
        replay=[],
        trace_ids={},
        limitations=["component equality does not establish full-decoder accuracy or performance"],
    )
    mesh = None
    traces = {}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16000000)
        baseline = TP4LocalSiluMultiply(ttnn, mesh, debug=True)
        helpers = {"baseline": baseline}
        for variant in ("once_per_page", "tile_workers"):
            dec = SimpleNamespace(kind="linear_attention", decode_local_silu=baseline, precision_policy={})
            install(dec, variant=variant)
            helpers[variant] = dec.decode_local_silu

        def upload(value, dtype):
            # Remove only the local singleton; shard the explicit rank axis.
            return ttnn.from_torch(
                value.reshape(4, *value.shape[2:]).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
            )

        def host(tensor):
            tensor = ttnn.to_memory_config(tensor, ttnn.DRAM_MEMORY_CONFIG)
            return [ttnn.to_torch(part).clone() for part in ttnn.get_device_tensors(tensor)]

        uploaded = [(name, upload(w, ttnn.float32), upload(z, ttnn.bfloat16)) for name, w, z in cases]
        stable_w = upload(cases[0][1], ttnn.float32)
        stable_z = upload(cases[0][2], ttnn.bfloat16)
        controls = []
        # All producer/consumer program families run before the first capture.
        # Eager controls are retained as host bit patterns for changing replays.
        for name, dw, dz in uploaded:
            expected = None
            for label, helper in helpers.items():
                with device_only():
                    out, gate = helper(dw, dz)
                actual = (host(out), host(gate))
                if expected is None:
                    expected = actual
                    controls.append(actual)
                report["eager"].append(
                    dict(
                        case=name,
                        variant=label,
                        output=_compare(actual[0], expected[0]),
                        gate=_compare(actual[1], expected[1]),
                    )
                )
                del out, gate
        captured = {}
        from ttnn.tools import trace_allocation_tracker

        # Refresh compiles its own CopyDeviceOperation programs for FP32 and
        # BF16. Warm every exact source/destination pair before any trace is
        # live; helper and readback warmup alone does not prepare these copies.
        for _, dw, dz in uploaded:
            with device_only():
                ttnn.copy(dw, stable_w)
                ttnn.copy(dz, stable_z)
        with device_only():
            ttnn.copy(uploaded[0][1], stable_w)
            ttnn.copy(uploaded[0][2], stable_z)
        ttnn.synchronize_device(mesh)
        report["copy_warmup"] = {
            "source_pairs": len(uploaded),
            "dtypes": ["float32", "bfloat16"],
            "before_any_capture": True,
            "initial_case_restored": uploaded[0][0],
        }
        for helper in helpers.values():
            for _ in range(2):
                with device_only():
                    out, gate = helper(stable_w, stable_z)
                host(out)
                host(gate)
                del out, gate
        for label, helper in helpers.items():
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                captured[label] = helper(stable_w, stable_z)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[label] = trace
            report["trace_ids"][label] = str(trace)
            # Both exact output allocations remain referenced through release.
            # This excludes neither the forward nor its program-cache storage.
            for tensor in captured[label]:
                trace_allocation_tracker.acknowledge_corruptible(tensor)
        for round_idx in range(args.rounds):
            ordering = list(range(len(uploaded)))
            if round_idx % 2:
                ordering.reverse()
            for index in ordering:
                name, dw, dz = uploaded[index]
                ttnn.copy(dw, stable_w)
                ttnn.copy(dz, stable_z)
                for label, trace in traces.items():
                    with device_only():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    out, gate = captured[label]
                    report["replay"].append(
                        dict(
                            round=round_idx,
                            case=name,
                            variant=label,
                            output=_compare(host(out), controls[index][0]),
                            gate=_compare(host(gate), controls[index][1]),
                        )
                    )
        report["passed"] = all(
            rank["passed"]
            for phase in ("eager", "replay")
            for row in report[phase]
            for kind in ("output", "gate")
            for rank in row[kind]
        )
        report["capture_count"] = len(traces)
        report["replay_count"] = len(report["replay"])
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("passed", "mesh", "input_source", "capture_count", "replay_count")}))
    assert report["passed"], "SiLU component bit equality failed; inspect per-rank output/gate mismatches"


if __name__ == "__main__":
    main()
