"""
Sweep sequence length comparing torch (CPU) vs TTNN (device).
Covers both Gated Attention and Gated DeltaNet (chunk mode).
Opens the device once and runs all benchmarks in a single session.
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import ttnn


def compute_pcc(torch_out, ttnn_out):
    ttnn_torch = ttnn.to_torch(ttnn_out).to(torch.float32)
    torch_flat = torch_out.to(torch.float32).flatten()
    ttnn_flat = ttnn_torch.flatten()
    mean_t, mean_n = torch_flat.mean(), ttnn_flat.mean()
    diff_t, diff_n = torch_flat - mean_t, ttnn_flat - mean_n
    return (
        (diff_t * diff_n).sum() / (torch.sqrt((diff_t**2).sum()) * torch.sqrt((diff_n**2).sum()) + 1e-12)
    ).item()


def sweep_gated_attention(device, seq_lens, warmup=2, iterations=3, batch_size=2):
    import gc
    from torch_functional.gated_attention import gated_attention_forward
    from tt.ttnn_gated_attention import gated_attention_forward_ttnn
    from tests.test_gated_attention import make_gated_attention_params

    header = f"{'T':>7} | {'B':>2} | {'Torch (ms)':>11} | {'TTNN (ms)':>11} | {'Speedup':>8} | {'PCC':>10}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Gated Attention Sweep: torch (CPU) vs TTNN (device)")
    print(f"  heads=8, kv_heads=2, head_dim=64")
    print(f"  warmup={warmup}, iterations={iterations}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for T in seq_lens:
        B = 1 if T >= 8192 else batch_size
        n_warmup = 1 if T >= 8192 else warmup
        n_iter = 1 if T >= 8192 else iterations
        try:
            params = make_gated_attention_params(seq_len=T, batch_size=B)

            for _ in range(n_warmup):
                gated_attention_forward(**params)
            torch_times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                torch_out, _, _ = gated_attention_forward(**params)
                torch_times.append(time.perf_counter() - t0)
            torch_min = min(torch_times)

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

            for _ in range(n_warmup):
                gated_attention_forward_ttnn(**ttnn_params)
                ttnn.synchronize_device(device)
            ttnn_times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                ttnn_out = gated_attention_forward_ttnn(**ttnn_params)
                ttnn.synchronize_device(device)
                ttnn_times.append(time.perf_counter() - t0)
            ttnn_min = min(ttnn_times)

            pcc = compute_pcc(torch_out, ttnn_out)
            speedup = torch_min / ttnn_min
            print(
                f"{T:>7} | {B:>2} | {torch_min*1000:>10.2f}ms | {ttnn_min*1000:>10.2f}ms | {speedup:>7.2f}x | {pcc:>10.6f}"
            )

            del torch_out, ttnn_out, ttnn_params, params
            ttnn.synchronize_device(device)
            gc.collect()

        except Exception as e:
            print(f"{T:>7} | {B:>2} | {'ERROR':>11} | {str(e)[:60]}")
            gc.collect()

    print(sep)


def sweep_gated_deltanet(device, seq_lens, chunk_size=64, warmup=2, iterations=3):
    from torch_functional.gated_deltanet import gated_deltanet_forward
    from tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn
    from tests.test_gated_deltanet import make_gated_deltanet_params

    header = f"{'T':>7} | {'Torch (ms)':>11} | {'TTNN (ms)':>11} | {'Speedup':>8} | {'PCC':>10}"
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(f"  Gated DeltaNet (chunk) Sweep: torch (CPU) vs TTNN (device)")
    print(f"  chunk_size={chunk_size}, B=2, H=4, K=128, V=256")
    print(f"  warmup={warmup}, iterations={iterations}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for T in seq_lens:
        try:
            params = make_gated_deltanet_params(seq_len=T)
            torch_mode = "chunk" if T > 64 else "fused_recurrent"

            for _ in range(warmup):
                gated_deltanet_forward(**params, mode=torch_mode, chunk_size=chunk_size)
            torch_times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                torch_out, _ = gated_deltanet_forward(**params, mode=torch_mode, chunk_size=chunk_size)
                torch_times.append(time.perf_counter() - t0)
            torch_min = min(torch_times)

            skip_keys = {
                "mode",
                "chunk_size",
                "conv_state_q",
                "conv_state_k",
                "conv_state_v",
                "recurrent_state",
                "output_final_state",
                "allow_neg_eigval",
            }
            ttnn_params = {}
            for key, val in params.items():
                if key in skip_keys:
                    continue
                if isinstance(val, torch.Tensor):
                    if key.endswith("_proj_weight"):
                        val = val.T.contiguous()
                    if key.endswith("_conv_weight"):
                        ttnn_params[key] = ttnn.from_torch(val, dtype=ttnn.bfloat16)
                    else:
                        ttnn_params[key] = ttnn.from_torch(
                            val,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                        )
                else:
                    ttnn_params[key] = val
            ttnn_params["device"] = device
            ttnn_params["mode"] = "chunk"
            ttnn_params["chunk_size"] = chunk_size

            for _ in range(warmup):
                gated_deltanet_forward_ttnn(**ttnn_params)
                ttnn.synchronize_device(device)
            ttnn_times = []
            for _ in range(iterations):
                t0 = time.perf_counter()
                ttnn_out, _ = gated_deltanet_forward_ttnn(**ttnn_params)
                ttnn.synchronize_device(device)
                ttnn_times.append(time.perf_counter() - t0)
            ttnn_min = min(ttnn_times)

            pcc = compute_pcc(torch_out, ttnn_out)
            speedup = torch_min / ttnn_min
            print(f"{T:>7} | {torch_min*1000:>10.2f}ms | {ttnn_min*1000:>10.2f}ms | {speedup:>7.2f}x | {pcc:>10.6f}")

            del ttnn_out, ttnn_params
            ttnn.synchronize_device(device)

        except Exception as e:
            print(f"{T:>7} | {'ERROR':>11} | {str(e)[:50]}")

    print(sep)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module", choices=["attention", "deltanet"], default=None, help="Run only one module (default: both)"
    )
    parser.add_argument(
        "--max-t", type=int, default=None, help="Max sequence length (default: 4096 for attention, 32768 for deltanet)"
    )
    args = parser.parse_args()

    all_seq_lens = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        if args.module in (None, "attention"):
            max_t = args.max_t or 4096
            attn_lens = [t for t in all_seq_lens if t <= max_t]
            sweep_gated_attention(device, attn_lens)

        if args.module in (None, "deltanet"):
            max_t = args.max_t or 32768
            delta_lens = [t for t in all_seq_lens if t <= max_t]
            sweep_gated_deltanet(device, delta_lens)
    finally:
        ttnn.close_device(device)

    print("\nAll sweeps complete.")
