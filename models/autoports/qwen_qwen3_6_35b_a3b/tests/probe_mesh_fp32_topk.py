# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bound externally with timeout; compare exact B1 selector and native topk."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from safetensors import safe_open

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import REVISION, SNAPSHOT
from models.autoports.qwen_qwen3_6_35b_a3b.tt.mesh_topk_probe import MeshFP32TopK


def _valid(logits, values, ids):
    logits, values, ids = logits.flatten(), values.flatten(), ids.flatten().long()
    if ids.numel() != 8 or ids.unique().numel() != 8 or not bool(((ids >= 0) & (ids < 256)).all()):
        return False
    if not torch.equal(values.contiguous().view(torch.int32), logits[ids].contiguous().view(torch.int32)):
        return False

    # NaNs rank first, numeric order next; equal-score IDs may be permuted.
    def key(value):
        return (bool(torch.isnan(value)), 0.0 if torch.isnan(value) else float(value))

    chosen = [key(v) for v in values]
    expected = sorted((key(v) for v in logits), reverse=True)[:8]
    return chosen == expected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.manual_seed(127)
    name = f"model.language_model.layers.{args.layer}.mlp.gate.weight"
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    with safe_open(SNAPSHOT / index[name], framework="pt", device="cpu") as handle:
        weight = handle.get_tensor(name).float().T.contiguous()
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=8000000)
    traces = []
    report = dict(revision=REVISION, layer=args.layer, mesh=[1, 4], cases=[], timing={}, argv=vars(args))
    try:
        mapper = ttnn.ReplicateTensorToMesh(mesh)

        def upload(x, dtype=ttnn.float32):
            return ttnn.from_torch(
                x.contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )

        def host(x):
            return [ttnn.to_torch(t).clone() for t in ttnn.get_device_tensors(x)]

        compute = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        dw = upload(weight)
        selector = MeshFP32TopK(mesh)
        stable = selector.input_template
        cases = []
        for i in range(args.samples):
            activation = upload(torch.randn(1, 1, 1, 2048).bfloat16() * 0.05, ttnn.bfloat16)
            logits = ttnn.linear(activation, dw, dtype=ttnn.float32, compute_kernel_config=compute)
            cases.append((f"real_router_{i}", host(logits)[0]))
        close = torch.arange(256, dtype=torch.int32) + 0x3F800000
        cases.append(("near_cutoff_one_ulp", close.view(torch.float32)[torch.randperm(256)].reshape(1, 1, 1, 256)))
        cases.append(("equal_keys", torch.ones(1, 1, 1, 256)))
        cases.append(("cutoff_ties", (torch.arange(256) % 9).float().reshape(1, 1, 1, 256)))
        signed = torch.zeros(256)
        signed[::2] = -0.0
        cases.append(("signed_zeros", signed.reshape(1, 1, 1, 256)))
        nonfinite = torch.randn(256)
        nonfinite[:4] = float("nan")
        nonfinite[4:8] = float("inf")
        nonfinite[8:12] = float("-inf")
        cases.append(("nonfinite", nonfinite[torch.randperm(256)].reshape(1, 1, 1, 256)))
        cases.append(("all_negative_infinity", torch.full((1, 1, 1, 256), float("-inf"))))
        uploaded = [(name, upload(value)) for name, value in cases]
        ttnn.copy(uploaded[0][1], stable)
        functions = {
            "candidate": lambda: selector(stable),
            "native": lambda: ttnn.topk(stable, k=8, dim=-1, sorted=True),
        }
        outputs = {}
        for name, function in functions.items():
            for _ in range(2):
                function()
        for name, function in functions.items():
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            outputs[name] = function()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces.append((name, trace))
            from ttnn.tools import trace_allocation_tracker

            # Exact outputs only; all backing allocations remain live until
            # both traces are released, and each replay rewrites all values.
            for output in outputs[name]:
                trace_allocation_tracker.acknowledge_corruptible(output)
        for (case_name, dx), (_, logical) in zip(uploaded, cases):
            ttnn.copy(dx, stable)
            actual_input = host(stable)
            result = {"name": case_name, "finite": bool(torch.isfinite(logical).all()), "rank_checks": {}}
            for name, trace in traces:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                vals, ids = map(host, outputs[name])
                checks = [_valid(x, v, i) for x, v, i in zip(actual_input, vals, ids)]
                result["rank_checks"][name] = checks
                result[name] = {"values": vals[0].flatten().tolist(), "ids": ids[0].flatten().tolist()}
                # Replay with the same input must preserve all output bits.
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                repeated = host(outputs[name][0])
                result["rank_checks"][name + "_repeat"] = [
                    torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))
                    for a, b in zip(vals, repeated)
                ]
            report["cases"].append(result)
        ttnn.copy(uploaded[0][1], stable)
        for name, trace in traces:
            durations = []
            for _ in range(args.repeat):
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                durations.append((time.perf_counter() - start) * 1000)
            report["timing"][name] = {
                "median_ms": statistics.median(durations),
                "mean_ms": statistics.mean(durations),
                "samples_ms": durations,
            }
        report["passed"] = all(all(check) for row in report["cases"] for check in row["rank_checks"].values())
    finally:
        for _, trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "timing": report["timing"], "output": args.output}))
    assert report["passed"], "selector/component contract failure; inspect per-case and per-rank evidence"


if __name__ == "__main__":
    main()
