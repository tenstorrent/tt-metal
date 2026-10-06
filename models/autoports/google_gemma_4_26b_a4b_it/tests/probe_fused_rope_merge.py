# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-input sliding decode RoPE peer-merge equality and warmed trace timing."""

import argparse
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.fused_decoder import FusedAttention
from models.common.utility_functions import comp_pcc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--timing-samples", type=int, default=3)
    parser.add_argument("--warmup-replays", type=int, default=30)
    args = parser.parse_args()
    if min(args.repeats, args.timing_samples, args.warmup_replays) < 1:
        parser.error("repeats, timing-samples and warmup-replays must be positive")

    torch.manual_seed(args.seed)
    query_heads, key_heads, head_dim = 16, 8, 256

    def normalized_random(heads):
        value = torch.randn(1, 1, heads, head_dim, dtype=torch.float32)
        return value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + 1.0e-6)

    query_host, key_host = normalized_random(query_heads), normalized_random(key_heads)
    half_angles = torch.rand(1, 1, 1, head_dim // 2) * (2 * torch.pi)
    angles = torch.cat((half_angles, half_angles), dim=-1)
    cos_host, sin_host = angles.cos().bfloat16(), angles.sin().bfloat16()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        # Exercise the live method without loading projections or changing runtime.
        attention = SimpleNamespace(fuse_rope="decode", compute=compute)

        def upload(value, dtype):
            return ttnn.from_torch(
                value,
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        query, key = upload(query_host, ttnn.float32), upload(key_host, ttnn.float32)
        cos, sin = upload(cos_host, ttnn.bfloat16), upload(sin_host, ttnn.bfloat16)

        def rotate(value):
            return FusedAttention.rotary(attention, value, cos, sin, decode=True)

        def separate():
            return rotate(query), rotate(key)

        def merged():
            packed = ttnn.concat((query, key), dim=2)
            rotated = rotate(packed)
            return rotated[:, :, :query_heads, :], rotated[:, :, query_heads:, :]

        with device_only():
            reference_device = separate()
        reference = tuple(ttnn.to_torch(value).float() for value in reference_device)
        for value in reference_device:
            value.deallocate(True)

        report = dict(
            layer_type="sliding_attention",
            phase="decode",
            input_source="random FP32 rows normalized in host FP32; BF16 paired cos/sin tables",
            real_weights=False,
            query_shape=list(query_host.shape),
            key_shape=list(key_host.shape),
            seed=args.seed,
            pcc_threshold=0.995,
            repeats=args.repeats,
            timing_samples=args.timing_samples,
            warmup_replays=args.warmup_replays,
            results=[],
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        for name, fn in (("separate", separate), ("concat_rotary_split", merged)):
            with device_only():
                outputs = fn()
            actual = tuple(ttnn.to_torch(value).float() for value in outputs)
            comparisons = []
            for label, expected, observed in zip(("query", "key"), reference, actual):
                passing, pcc = comp_pcc(expected, observed, 0.995)
                comparisons.append(
                    dict(
                        tensor=label,
                        pcc=float(pcc),
                        passed=bool(passing),
                        exact_equal=torch.equal(expected, observed),
                        max_abs=float((expected - observed).abs().max()),
                    )
                )
            for value in outputs:
                value.deallocate(True)

            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                outputs = fn()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                for _ in range(args.warmup_replays):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples = []
                for _ in range(args.timing_samples):
                    start = time.perf_counter_ns()
                    for _ in range(args.repeats):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter_ns() - start) / (args.repeats * 1000))
                replay = tuple(ttnn.to_torch(value).float() for value in outputs)
                replay_equal = all(torch.equal(first, last) for first, last in zip(actual, replay))
            finally:
                ttnn.release_trace(mesh, trace)
                for value in outputs:
                    value.deallocate(True)

            row = dict(
                candidate=name,
                comparisons=comparisons,
                min_pcc=min(item["pcc"] for item in comparisons),
                passed=all(item["passed"] for item in comparisons) and replay_equal,
                exact_equal=all(item["exact_equal"] for item in comparisons),
                trace_replay_equal=replay_equal,
                traced_host_us=statistics.median(samples),
                traced_host_us_samples=samples,
            )
            report["results"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)
        assert all(row["passed"] for row in report["results"]), report["results"]
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
