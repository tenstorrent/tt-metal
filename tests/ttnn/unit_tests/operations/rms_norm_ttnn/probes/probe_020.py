# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
#
# D41 -- the placement-conditional NoC read-barrier cadence, four-case family A/B.
#
# Four cases x {native, generated}, ONE dispatch each.  No trial loop: the metric has
# no warm-up transient and some cases accumulate a residual.  Shape, dtype, epsilon and
# GAMMA-BEFORE-X ALLOCATION ORDER are copied from
# tests/ttnn/nightly/unit_tests/operations/fused/test_rmsnorm.py -- the order is
# load-bearing, it is worth ~5.8 us of L1 placement luck on the RMSN_GB case.
#
#   BENCH_TAG=<label> scripts/tt-probe.sh rms_norm_ttnn < bench_family.py
#
# with TT_METAL_DEVICE_PROFILER=1 TT_METAL_PROFILER_MID_RUN_DUMP=1
#      TT_METAL_PROFILER_CPP_POST_PROCESS=1 set before device init.
import hashlib
import os

import ttnn

N, C, H, W = 1, 9, 384, 1024
EPS = 1e-2


def device_ns():
    per = ttnn.get_latest_programs_perf_data()
    return sum(
        float(pr.program_analyses_results["DEVICE KERNEL DURATION [ns]"].duration)
        for v in (per or {}).values()
        for pr in v
        if "DEVICE KERNEL DURATION [ns]" in (pr.program_analyses_results or {})
    )


def main():
    import torch
    from ttnn.operations import normalization as nrm

    dev = ttnn.open_device(device_id=0)
    torch.manual_seed(1234)
    L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    DR = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)

    x = torch.rand((N, C, H, W)) * 2 - 0.95
    gamma = torch.rand(1, 1, 1, W) * 2 - 1
    beta = torch.rand(1, 1, 1, W) * 2.0 - 1.1

    def run(op, label, in_mc, out_mc, gb):
        tg = tb = None
        if gb:
            tg = ttnn.from_torch(
                gamma.expand(1, 1, 32, W).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=in_mc,
            )
            tb = ttnn.from_torch(
                beta.expand(1, 1, 32, W).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=dev,
                memory_config=in_mc,
            )
        tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=in_mc)
        ttnn.ReadDeviceProfiler(dev)
        kw = dict(epsilon=EPS, memory_config=out_mc)
        if gb:
            kw["weight"] = tg
            kw["bias"] = tb
        out = op(tx, **kw)
        ttnn.synchronize_device(dev)
        ttnn.ReadDeviceProfiler(dev)
        ns = device_ns()

        t = ttnn.to_torch(out).to(torch.float32)
        ref = x.to(torch.float32)
        ref = ref * torch.rsqrt(ref.pow(2).mean(-1, keepdim=True) + EPS)
        if gb:
            ref = ref * gamma.to(torch.float32) + beta.to(torch.float32)
        pcc = float(torch.corrcoef(torch.stack([t.flatten(), ref.flatten()]))[0, 1])
        # t is the lossless f32 view of a bf16 output, so this sha is a bit-identity check.
        sha = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
        print(f"RESULT {label:34s} {ns:9.0f} ns  PCC {pcc:.6f}  sha {sha}", flush=True)
        for tt in (out, tx, tg, tb):
            if tt is not None:
                tt.deallocate()
        return ns

    cases = [
        ("RMSN     L1->L1  ", L1, L1, False),
        ("RMSN     L1->DRAM", L1, DR, False),
        ("RMSN_GB  L1->L1  ", L1, L1, True),
        ("RMSN     DRAM->L1", DR, L1, False),
    ]
    tag = os.environ.get("BENCH_TAG", "run")
    res = {}
    for name, imc, omc, gb in cases:
        nat = run(nrm._native_rms_norm, f"native    {name}", imc, omc, gb)
        gen = run(ttnn.rms_norm, f"{tag:9s} {name}", imc, omc, gb)
        res[name] = (nat, gen)

    print(flush=True)
    print(f"{'case':20s} {'native':>9s} {tag:>9s} {'ratio nat/gen':>14s}", flush=True)
    for name, (nat, gen) in res.items():
        print(f"{name:20s} {nat:9.0f} {gen:9.0f} {nat / gen:14.3f}", flush=True)
    ttnn.close_device(dev)
    print("PROBE_OK", flush=True)


main()
