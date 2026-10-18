# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Focused exact-prefill candidate comparison; requires serialized device lease."""

import argparse
import json
import runpy
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tests.precise_prefill_candidate import (
    PrecisePrefillAliasGDN,
    PrecisePrefillGDN,
    PrecisePrefillTiledGDN,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.gdn import GatedDeltaNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq", type=int, default=3)
    parser.add_argument("--case", choices=["random", "real", "identity", "decay"], default="random")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--variant", choices=["rm", "tiled", "both", "alias", "all"], default="rm")
    args = parser.parse_args()
    b, s = args.batch, args.seq
    torch.set_num_threads(8)
    torch.manual_seed(1984)
    q = torch.nn.functional.normalize(torch.randn(b, s, 16, 128), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, s, 16, 128), dim=-1)
    v = torch.randn(b, s, 48, 128)
    g = -torch.rand(b, s, 48) * 0.05
    beta = torch.rand(b, s, 48)
    initial = torch.randn(b, 48, 128, 128) * 0.1
    if args.case == "identity":
        g.zero_()
        beta.zero_()
        initial = (torch.arange(initial.numel()).reshape(initial.shape) % 127).float() * 2**-19 + 1
    if args.case == "decay":
        g.fill_(float(torch.tensor(1 - 2**-14).log()))
        beta.zero_()
    if args.case == "real":
        root = Path(__file__).resolve().parents[1]
        data = runpy.run_path(str(root / "doc/functional_decoder/gdn_debug/probe_cpu.py"))
        assert s <= 384
        values = [
            data["q"][:, :s, ::3].float() * 128**0.5,
            data["k"][:, :s, ::3].float(),
            *[data[n][:, :s].float() for n in ("v", "g", "beta")],
        ]
        q, k, v, g, beta = [t.repeat(b, *([1] * (t.ndim - 1))).contiguous() for t in values]
        initial.zero_()
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=33554432)
    trace = None
    try:

        def upload(t):
            return ttnn.from_torch(t.contiguous(), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=mesh)

        inputs = [upload(t) for t in (q, k, v, g, beta)]
        original = upload(initial)
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
        )
        models = []
        candidates = [PrecisePrefillGDN] if args.variant == "rm" else [PrecisePrefillTiledGDN]
        if args.variant == "both":
            candidates = [PrecisePrefillGDN, PrecisePrefillTiledGDN]
        if args.variant == "alias":
            candidates = [PrecisePrefillAliasGDN]
        if args.variant == "all":
            candidates = [PrecisePrefillGDN, PrecisePrefillTiledGDN, PrecisePrefillAliasGDN]
        for cls in [GatedDeltaNet, *candidates]:
            obj = cls()
            obj.device = mesh
            obj.compute = compute
            obj.recurrent_state = upload(initial)
            models.append(obj)

        def execute(obj):
            out, state = obj._recurrence(*inputs)
            ttnn.copy(state, obj.recurrent_state)
            return out

        def reset(obj):
            ttnn.copy(original, obj.recurrent_state)

        expected = None
        for obj in models:
            reset(obj)
            out = execute(obj)
            actual = (ttnn.to_torch(out), ttnn.to_torch(obj.recurrent_state))
            if expected is None:
                expected = actual
            else:
                for label, a, z in zip(("output", "state"), expected, actual):
                    metric = pcc_metrics(a, z)
                    print(
                        json.dumps(dict(case=args.case, batch=b, seq=s, path=type(obj).__name__, kind=label, **metric)),
                        flush=True,
                    )
                    assert metric["pcc"] >= 0.99999
                    if args.case == "identity" and label == "state":
                        torch.testing.assert_close(z, initial, rtol=0, atol=0)
            reset(obj)
            execute(obj)
            ttnn.synchronize_device(mesh)
            times = []
            for repeat in range(args.repeats):
                reset(obj)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                execute(obj)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - begin) * 1000)
            print(
                json.dumps(
                    dict(
                        case=args.case,
                        batch=b,
                        seq=s,
                        path=type(obj).__name__,
                        warmed_ms=statistics.median(times),
                        samples_ms=times,
                    )
                ),
                flush=True,
            )
        if args.trace:
            expected_variants = []
            for repeat in range(4):
                scale = 1.0 if repeat in (0, 3) else 1.0 + repeat * 0.125
                host = ttnn.from_torch(q * scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, inputs[0])
                reset(models[0])
                golden = execute(models[0])
                expected_variants.append((ttnn.to_torch(golden), ttnn.to_torch(models[0].recurrent_state)))
            del golden
            obj = models[-1]
            reset(obj)
            execute(obj)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = execute(obj)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for repeat in range(4):
                scale = 1.0 if repeat in (0, 3) else 1.0 + repeat * 0.125
                host = ttnn.from_torch(q * scale, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT)
                ttnn.copy_host_to_device_tensor(host, inputs[0])
                reset(obj)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = (ttnn.to_torch(out), ttnn.to_torch(obj.recurrent_state))
                for label, a, z in zip(("trace_output", "trace_state"), expected_variants[repeat], actual):
                    metric = pcc_metrics(a, z)
                    print(
                        json.dumps(dict(case=args.case, batch=b, seq=s, repeat=repeat, kind=label, **metric)),
                        flush=True,
                    )
                    assert metric["pcc"] >= 0.99999
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
