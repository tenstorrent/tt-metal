# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
A/B benchmark: transpose+reshape vs concatenate_heads for Gated Attention.

Compares the current implementation (ttnn.transpose + ttnn.reshape) against
the optimized path (ttnn.transformer.concatenate_heads) for converting
SDPA output from [B, H, T, D] back to [B, T, H*D].

Both paths are validated against the torch golden reference for PCC,
then benchmarked across a range of sequence lengths.
"""

import torch
import time
import sys
import os
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ttnn


def compute_pcc(torch_out, ttnn_out):
    if isinstance(ttnn_out, torch.Tensor):
        ttnn_torch = ttnn_out.to(torch.float32)
    else:
        ttnn_torch = ttnn.to_torch(ttnn_out).to(torch.float32)
    torch_flat = torch_out.to(torch.float32).flatten()
    ttnn_flat = ttnn_torch.flatten()
    mean_t, mean_n = torch_flat.mean(), ttnn_flat.mean()
    diff_t, diff_n = torch_flat - mean_t, ttnn_flat - mean_n
    return (
        (diff_t * diff_n).sum() / (torch.sqrt((diff_t**2).sum()) * torch.sqrt((diff_n**2).sum()) + 1e-12)
    ).item()


def prepare_ttnn_params(params, device):
    ttnn_params = {}
    skip_keys = {"attention_mask"}
    for key, val in params.items():
        if key in skip_keys:
            continue
        if isinstance(val, torch.Tensor):
            if key.endswith("_proj_weight"):
                val = val.T.contiguous()
            ttnn_params[key] = ttnn.from_torch(
                val,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
            )
        else:
            ttnn_params[key] = val
    ttnn_params["device"] = device
    return ttnn_params


def bench_single_config(device, seq_len, batch_size, warmup=3, iterations=10):
    from torch_functional.gated_attention import gated_attention_forward
    from tt.ttnn_gated_attention import gated_attention_forward_ttnn
    from tests.test_gated_attention import make_gated_attention_params

    params = make_gated_attention_params(seq_len=seq_len, batch_size=batch_size)
    torch_out, _, _ = gated_attention_forward(**params)

    ttnn_params = prepare_ttnn_params(params, device)

    # --- Baseline: transpose + reshape ---
    ttnn_params_base = {**ttnn_params, "use_optimized_concat": False}

    for _ in range(warmup):
        gated_attention_forward_ttnn(**ttnn_params_base)
        ttnn.synchronize_device(device)

    base_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out_base = gated_attention_forward_ttnn(**ttnn_params_base)
        ttnn.synchronize_device(device)
        base_times.append(time.perf_counter() - t0)

    pcc_base = compute_pcc(torch_out, out_base)
    base_min = min(base_times)
    base_avg = sum(base_times) / len(base_times)

    del out_base
    ttnn.synchronize_device(device)

    # --- Optimized: concatenate_heads ---
    ttnn_params_opt = {**ttnn_params, "use_optimized_concat": True}

    for _ in range(warmup):
        gated_attention_forward_ttnn(**ttnn_params_opt)
        ttnn.synchronize_device(device)

    opt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        out_opt = gated_attention_forward_ttnn(**ttnn_params_opt)
        ttnn.synchronize_device(device)
        opt_times.append(time.perf_counter() - t0)

    pcc_opt = compute_pcc(torch_out, out_opt)
    opt_min = min(opt_times)
    opt_avg = sum(opt_times) / len(opt_times)

    del out_opt, torch_out, ttnn_params, ttnn_params_base, ttnn_params_opt, params
    ttnn.synchronize_device(device)
    gc.collect()

    return {
        "T": seq_len,
        "B": batch_size,
        "base_min": base_min,
        "base_avg": base_avg,
        "opt_min": opt_min,
        "opt_avg": opt_avg,
        "pcc_base": pcc_base,
        "pcc_opt": pcc_opt,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="A/B benchmark: transpose+reshape vs concatenate_heads")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--max-t", type=int, default=4096, help="Max sequence length")
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Specific sequence lengths to test (overrides --max-t)",
    )
    args = parser.parse_args()

    if args.seq_lens:
        seq_lens = args.seq_lens
    else:
        all_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
        seq_lens = [t for t in all_lens if t <= args.max_t]

    device = ttnn.open_device(device_id=0)
    try:
        header = (
            f"{'T':>7} | {'B':>2} | "
            f"{'Base min':>10} | {'Opt min':>10} | {'Delta':>8} | "
            f"{'PCC base':>10} | {'PCC opt':>10}"
        )
        sep = "-" * len(header)

        print(f"\n{'='*len(header)}")
        print(f"  Gated Attention: transpose+reshape vs concatenate_heads")
        print(f"  H=8, H_kv=2, D=64, warmup={args.warmup}, iters={args.iterations}")
        print(f"{'='*len(header)}")
        print(header)
        print(sep)

        for T in seq_lens:
            B = 1 if T >= 8192 else 2
            n_warmup = 1 if T >= 8192 else args.warmup
            n_iter = max(1, args.iterations // 2) if T >= 8192 else args.iterations

            try:
                r = bench_single_config(device, T, B, warmup=n_warmup, iterations=n_iter)
                delta_pct = (r["opt_min"] - r["base_min"]) / r["base_min"] * 100
                sign = "+" if delta_pct >= 0 else ""
                print(
                    f"{r['T']:>7} | {r['B']:>2} | "
                    f"{r['base_min']*1000:>9.2f}ms | {r['opt_min']*1000:>9.2f}ms | "
                    f"{sign}{delta_pct:>6.1f}% | "
                    f"{r['pcc_base']:>10.6f} | {r['pcc_opt']:>10.6f}"
                )
            except Exception as e:
                print(f"{T:>7} | {B:>2} | ERROR: {str(e)[:60]}")
                gc.collect()

        print(sep)
        print("\nNegative delta% = optimized path is faster")

    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
