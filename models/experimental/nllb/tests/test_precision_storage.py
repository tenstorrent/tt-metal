# SPDX-FileCopyrightText: © 2026 Abror Shopulatov

# SPDX-License-Identifier: Apache-2.0

"""Independent mixed BFP8_B weight/BF16 activation probe; no CHIA fixtures."""

import argparse, json, time, hashlib, resource
from pathlib import Path
import torch, ttnn


def check(device):
    torch.manual_seed(250921)
    kernel = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    def sync():
        ttnn.synchronize_device(device)

    def alloc():
        v = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return {k: getattr(v, k) for k in ("total_bytes_allocated_per_bank", "num_banks")}

    def up(v, dtype):
        sync()
        t = time.perf_counter()
        out = ttnn.from_torch(
            v.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        sync()
        return out, time.perf_counter() - t

    results = []
    for name, m, k, n in [
        ("projection", 32, 1024, 1024),
        ("ffn-up", 64, 1024, 4096),
        ("ffn-down", 32, 4096, 1024),
        ("lm-slice", 32, 1024, 8192),
    ]:
        x = torch.randn(1, 1, m, k).bfloat16().float()
        w = torch.randn(n, k) / k**0.5
        bias = (torch.randn(1, 1, 1, n) * 0.01).bfloat16().float()
        ref = x @ w.T + bias
        a, _ = up(x, ttnn.bfloat16)
        b, _ = up(bias, ttnn.bfloat16)
        before = alloc()
        wb, tb = up(w, ttnn.bfloat16)
        after_b = alloc()
        wc, tc = up(w, ttnn.bfloat8_b)
        after_c = alloc()

        def run(activation, weight, bias_tensor):
            sync()
            start = time.perf_counter()
            out = ttnn.linear(
                activation,
                weight,
                transpose_b=True,
                bias=bias_tensor,
                dtype=ttnn.bfloat16,
                compute_kernel_config=kernel,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            sync()
            op = time.perf_counter() - start
            start = time.perf_counter()
            y = ttnn.to_torch(out).float()
            sync()
            return y, op, time.perf_counter() - start

        ys = [run(a, wb, b)[0], run(a, wc, b)[0]]
        samples = {"bf16": [], "bfp8_b": []}
        transfer = {"bf16": [], "bfp8_b": []}
        for r in range(6):
            for label, weight in [("bf16", wb), ("bfp8_b", wc)] if r % 2 == 0 else [("bfp8_b", wc), ("bf16", wb)]:
                y, dt, dh = run(a, weight, b)
                samples[label].append(dt)
                transfer[label].append(dh)
                assert torch.equal(y, ys[label == "bfp8_b"])

        def error(y, reference):
            return float(
                ((y - reference).square().mean(-1).sqrt() / reference.square().mean(-1).sqrt().clamp_min(1e-8)).max()
            )

        row = dict(
            name=name,
            shape=[m, k, n],
            max_row_nrmse_fp32=[error(y, ref) for y in ys],
            candidate_vs_bf16_nrmse=error(ys[1], ys[0]),
            finite=all(bool(torch.isfinite(y).all()) for y in ys),
            op_seconds=samples,
            d2h_seconds=transfer,
            upload_seconds={"bf16": tb, "bfp8_b": tc},
            memory_before=before,
            memory_after_bf16=after_b,
            memory_after_bfp8=after_c,
            weight_dtypes=[str(wb.dtype), str(wc.dtype)],
        )
        print("PROBE " + json.dumps(row), flush=True)
        results.append(row)
        del a, b, wb, wc, y, ys
        sync()
    passed = all(r["finite"] and max(r["max_row_nrmse_fp32"]) <= 0.04 for r in results)
    print(
        "RESULT "
        + json.dumps(
            dict(
                passed=passed, results=results, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
            )
        ),
        flush=True,
    )
    assert passed, "mixed-storage component threshold failed"


def test_precision_storage(nllb_component_runner):
    nllb_component_runner("precision_storage")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print("SOURCE " + hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    device = ttnn.open_device(device_id=args.device)
    try:
        check(device)
    finally:
        ttnn.close_device(device)
