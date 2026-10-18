# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serialized-device precision/trace probe for the experimental S1 recurrence."""

import argparse
import json
import runpy
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.precise_recurrence_candidate import PreciseDecodeGDN
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["identity", "decay", "random", "real"], default="identity")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(1978)
    b = args.batch
    q = torch.nn.functional.normalize(torch.randn(b, 1, 16, 128), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, 1, 16, 128), dim=-1)
    v = torch.randn(b, 1, 48, 128)
    g = -torch.rand(b, 1, 48) * 0.05
    beta = torch.rand(b, 1, 48)
    initial = torch.randn(b, 48, 128, 128) * 0.1
    real_series = None
    if args.case == "identity":
        g.zero_()
        beta.zero_()
        # Nonconstant low FP32 mantissa bits: an identity may not quantize them.
        initial = (torch.arange(initial.numel()).reshape(initial.shape) % 127).float() * 2**-19 + 1.0
    elif args.case == "decay":
        g.fill_(float(torch.tensor(1 - 2**-14).log()))
        beta.zero_()
    elif args.case == "real":
        data = runpy.run_path(str(ROOT / "doc/functional_decoder/gdn_debug/probe_cpu.py"))
        real_series = [
            data["q"][:, :, ::3].float() * 128**0.5,
            data["k"][:, :, ::3].float(),
            *[data[n].float() for n in ("v", "g", "beta")],
        ]
        real_series = [t.repeat(b, *([1] * (t.ndim - 1))) for t in real_series]
        q, k, v, g, beta = [t[:, :1].contiguous() for t in real_series]
        initial.zero_()
        assert args.steps <= real_series[0].shape[1]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=16777216)
    traces = []
    try:

        def upload(t):
            return ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)

        inputs = [upload(t) for t in (q, k, v, g, beta)]
        original = upload(initial)
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        models = []
        for cls in (GatedDeltaNet, PreciseDecodeGDN):
            model = cls()
            model.device = mesh
            model.compute = compute
            model.recurrent_state = upload(initial)
            models.append(model)

        def execute(model):
            output, state = model._recurrence(*inputs)
            ttnn.copy(state, model.recurrent_state)
            return output

        for step in range(args.steps):
            if real_series is not None:
                q, k, v, g, beta = [t[:, step : step + 1].contiguous() for t in real_series]
                for value, dest in zip((q, k, v, g, beta), inputs):
                    host = ttnn.from_torch(value, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                    ttnn.copy_host_to_device_tensor(host, dest)
            outputs = [execute(m) for m in models]
            for label, a, z in (
                ("output", outputs[0], outputs[1]),
                ("state", models[0].recurrent_state, models[1].recurrent_state),
            ):
                ah, zh = ttnn.to_torch(a), ttnn.to_torch(z)
                metric = pcc_metrics(ah, zh)
                print(json.dumps(dict(case=args.case, batch=b, step=step, kind=label, **metric)), flush=True)
                assert metric["pcc"] >= 0.99999
                if args.case == "identity" and label == "state":
                    torch.testing.assert_close(zh, initial, rtol=0, atol=0)
        if args.trace:
            # Prepare both signatures and persistent buffers before the first capture.
            for model in models:
                for _ in range(2):
                    ttnn.copy(original, model.recurrent_state)
                    execute(model)
            ttnn.synchronize_device(mesh)
            expected = []
            for repeat in range(4):
                multiplier = 1.0 if repeat in (0, 3) else 1.0 + repeat * 0.125
                newq = ttnn.from_torch(q * multiplier, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(newq, inputs[0])
                ttnn.copy(original, models[0].recurrent_state)
                expected_out = execute(models[0])
                expected.append((ttnn.to_torch(expected_out), ttnn.to_torch(models[0].recurrent_state)))
            del expected_out
            for model in models:
                ttnn.copy(original, model.recurrent_state)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = execute(model)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                traces.append(trace)
                # Each trace owns its output until release. Capture the next only
                # after release and dropping this output; no allocation exemptions.
                first_result = None
                for repeat in range(4):
                    multiplier = 1.0 if repeat in (0, 3) else 1.0 + repeat * 0.125
                    newq = ttnn.from_torch(q * multiplier, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                    ttnn.copy_host_to_device_tensor(newq, inputs[0])
                    ttnn.copy(original, model.recurrent_state)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    result = (ttnn.to_torch(out), ttnn.to_torch(model.recurrent_state))
                    for j, label in enumerate(("trace_output", "trace_state")):
                        metric = pcc_metrics(expected[repeat][j], result[j])
                        print(
                            json.dumps(
                                dict(
                                    case=args.case,
                                    batch=b,
                                    path=type(model).__name__,
                                    repeat=repeat,
                                    kind=label,
                                    **metric,
                                )
                            ),
                            flush=True,
                        )
                        assert metric["pcc"] >= 0.99999
                    if repeat == 0:
                        first_result = result
                    if repeat == 3:
                        for a, z in zip(first_result, result):
                            torch.testing.assert_close(a, z, rtol=0, atol=0)
                ttnn.copy(original, model.recurrent_state)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(args.repeats):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - begin) * 1000 / args.repeats
                print(json.dumps(dict(case=args.case, batch=b, path=type(model).__name__, traced_ms=ms)), flush=True)
                ttnn.release_trace(mesh, trace)
                traces.remove(trace)
                del out
    finally:
        for trace in traces:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
