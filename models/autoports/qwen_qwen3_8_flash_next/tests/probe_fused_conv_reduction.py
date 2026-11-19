# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-tap GDN convolution fusion preserving both original BF16 boundaries."""
import argparse
import json
import time
from pathlib import Path

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import fused_exact_conv_silu


def reduced_conv(history, packed_taps, count, compute):
    values = ttnn.typecast(history, ttnn.float32)
    windows = ttnn.concat([values[:, :, i : i + count, :] for i in range(4)], dim=0)
    products = ttnn.multiply(windows, packed_taps)
    total = ttnn.sum(products, dim=0, keepdim=True, compute_kernel_config=compute, fast_and_approximate_mode=False)
    rounded = ttnn.typecast(total, ttnn.bfloat16)
    return ttnn.typecast(ttnn.silu(ttnn.typecast(rounded, ttnn.float32)), ttnn.bfloat16)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(8)
    torch.manual_seed(1901)
    weight = load_weights(args.snapshot, 0, "linear_attn.conv1d.weight")["linear_attn.conv1d.weight"]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(rows=[], passed=False)
    trace = None
    try:
        loader = WeightLoader({}, mesh)
        taps = [loader.tensor(weight[:, 0, i].reshape(1, 1, 1, -1), ttnn.float32) for i in range(4)]
        packed = ttnn.concat(taps, dim=0)
        for n in (1, 2, 31, 32):
            history = loader.tensor((torch.randn(1, 1, n + 3, 10240) * 0.02).bfloat16())

            def original():
                values = ttnn.typecast(history, ttnn.float32)
                parts = [ttnn.multiply(values[:, :, i : i + n, :], taps[i]) for i in range(4)]
                total = ttnn.add(ttnn.add(parts[0], parts[1]), ttnn.add(parts[2], parts[3]))
                rounded = ttnn.typecast(total, ttnn.bfloat16)
                return ttnn.typecast(ttnn.silu(ttnn.typecast(rounded, ttnn.float32)), ttnn.bfloat16)

            expected = ttnn.to_torch(original())
            for name, operation in (
                ("original", original),
                ("reduced", lambda: reduced_conv(history, packed, n, loader.compute)),
                ("fused_exact", lambda: fused_exact_conv_silu(history, taps, n)),
            ):
                actual = ttnn.to_torch(operation())
                pcc = float(torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1])
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                output = operation()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                trace_allocation_tracker.acknowledge_corruptible(output)
                times = []
                for _ in range(12):
                    ttnn.synchronize_device(mesh)
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter() - start) * 1000)
                replay = ttnn.to_torch(output)
                exact_replay = torch.equal(actual, replay)
                ttnn.release_trace(mesh, trace)
                trace = None
                del output
                report["rows"].append(
                    dict(
                        tokens=n,
                        variant=name,
                        pcc=pcc,
                        exact_vs_original=torch.equal(expected, actual),
                        exact_replay=exact_replay,
                        traced_median_ms=float(torch.tensor(times[2:]).median()),
                        passed=pcc >= 0.995 and exact_replay,
                    )
                )
                print(report["rows"][-1], flush=True)
        report["rounding_controls"] = []
        for label, inputs, weights in (
            ("bf16_midpoint", [1.0, 1.0, 0.0, 0.0], [1.0, 0.00390625, 0.0, 0.0]),
            ("cancellation", [256.0, 1.0, -256.0, 1.0], [1.0, 1.0, 1.0, 1.0]),
        ):
            host_x = torch.tensor(inputs, dtype=torch.bfloat16).reshape(1, 1, 4, 1).expand(1, 1, 4, 10240).contiguous()
            history = loader.tensor(host_x)
            control_taps = [loader.tensor(torch.full((1, 1, 1, 10240), value), ttnn.float32) for value in weights]
            products = [host_x[:, :, i : i + 1].float() * weights[i] for i in range(4)]
            rounded = ((products[0] + products[1]) + (products[2] + products[3])).bfloat16()
            expected = torch.nn.functional.silu(rounded.float()).bfloat16()
            observed = ttnn.to_torch(fused_exact_conv_silu(history, control_taps, 1))
            report["rounding_controls"].append(
                dict(
                    case=label,
                    exact=torch.equal(expected, observed),
                    expected=float(expected.flatten()[0]),
                    observed=float(observed.flatten()[0]),
                )
            )
        report["passed"] = all(row["passed"] for row in report["rows"]) and all(
            row["exact"] for row in report["rounding_controls"]
        )
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
