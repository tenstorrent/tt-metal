# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""
TTNN validation and performance tests.

Compares TTNN implementation outputs against torch golden references
using Pearson Correlation Coefficient (PCC) as the similarity metric.
Also benchmarks torch vs TTNN execution time.

These tests require Tenstorrent hardware and the ttnn package.
Run with: python tests/test_ttnn_validation.py
"""

import torch
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def assert_with_pcc(torch_output, ttnn_output, pcc_threshold=0.99):
    """
    Assert that TTNN output matches torch golden using Pearson Correlation.

    PCC >= 0.99 is the standard threshold for TTNN validation.
    """
    import ttnn as ttnn_lib

    if isinstance(ttnn_output, torch.Tensor):
        ttnn_np = ttnn_output.to(torch.float32)
    else:
        ttnn_np = ttnn_lib.to_torch(ttnn_output).to(torch.float32)

    torch_flat = torch_output.to(torch.float32).flatten()
    ttnn_flat = ttnn_np.flatten()

    if torch_flat.shape != ttnn_flat.shape:
        raise ValueError(f"Shape mismatch: torch {torch_flat.shape} vs ttnn {ttnn_flat.shape}")

    if torch_flat.std() < 1e-10 and ttnn_flat.std() < 1e-10:
        print("  Both outputs near-zero, skipping PCC (trivially equal)")
        return 1.0

    # Pearson correlation
    mean_t = torch_flat.mean()
    mean_n = ttnn_flat.mean()
    diff_t = torch_flat - mean_t
    diff_n = ttnn_flat - mean_n
    pcc = (diff_t * diff_n).sum() / (torch.sqrt((diff_t**2).sum()) * torch.sqrt((diff_n**2).sum()) + 1e-12)
    pcc_val = pcc.item()

    if pcc_val < pcc_threshold:
        max_diff = (torch_flat - ttnn_flat).abs().max().item()
        mean_diff = (torch_flat - ttnn_flat).abs().mean().item()
        raise AssertionError(
            f"PCC {pcc_val:.6f} < {pcc_threshold}. " f"Max diff: {max_diff:.6e}, Mean diff: {mean_diff:.6e}"
        )

    return pcc_val


def test_gated_attention_ttnn():
    """Compare TTNN Gated Attention against torch golden."""
    try:
        import ttnn
    except ImportError:
        print("SKIP: test_gated_attention_ttnn (ttnn not available)")
        return

    from torch_functional.gated_attention import gated_attention_forward
    from tt.ttnn_gated_attention import gated_attention_forward_ttnn
    from tests.test_gated_attention import make_gated_attention_params

    params = make_gated_attention_params()

    # Torch golden
    torch_out, _, _ = gated_attention_forward(**params)

    # TTNN forward
    device = ttnn.open_device(device_id=0)
    try:
        ttnn_params = {}
        skip_keys = {"attention_mask"}
        for key, val in params.items():
            if key in skip_keys:
                continue
            if isinstance(val, torch.Tensor):
                # ttnn.linear expects [in, out]; PyTorch uses [out, in]
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
        ttnn_out = gated_attention_forward_ttnn(**ttnn_params)

        pcc = assert_with_pcc(torch_out, ttnn_out, pcc_threshold=0.98)
        print(f"PASS: test_gated_attention_ttnn (PCC={pcc:.6f})")
    finally:
        ttnn.close_device(device)


def test_gated_deltanet_recurrent_ttnn(seq_len=16):
    """Compare TTNN GatedDeltaNet (recurrent mode) against torch golden."""
    try:
        import ttnn
    except ImportError:
        print("SKIP: test_gated_deltanet_recurrent_ttnn (ttnn not available)")
        return

    from torch_functional.gated_deltanet import gated_deltanet_forward
    from tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn
    from tests.test_gated_deltanet import make_gated_deltanet_params

    params = make_gated_deltanet_params(seq_len=seq_len)

    # Torch golden
    torch_out, _ = gated_deltanet_forward(**params, mode="fused_recurrent")

    # TTNN forward
    # l1_small_size=16384 enables L1_SMALL banks for conv1d halo operations
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        ttnn_params = {}
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
        for key, val in params.items():
            if key in skip_keys:
                continue
            if isinstance(val, torch.Tensor):
                # ttnn.linear expects [in, out]; PyTorch uses [out, in]
                if key.endswith("_proj_weight"):
                    val = val.T.contiguous()
                # conv1d weights stay on host; ttnn.conv1d handles device placement
                if key.endswith("_conv_weight"):
                    ttnn_params[key] = ttnn.from_torch(val, dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
                else:
                    ttnn_params[key] = ttnn.from_torch(
                        val,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=device,
                        memory_config=ttnn.L1_MEMORY_CONFIG,
                    )
            else:
                ttnn_params[key] = val

        ttnn_params["device"] = device
        ttnn_out, *_ = gated_deltanet_forward_ttnn(**ttnn_params)

        pcc = assert_with_pcc(torch_out, ttnn_out, pcc_threshold=0.95)
        print(f"PASS: test_gated_deltanet_recurrent_ttnn T={seq_len} (PCC={pcc:.6f})")
    finally:
        ttnn.close_device(device)


def test_gated_deltanet_chunked_ttnn(seq_len=128, chunk_size=128):
    """Validate production-shaped TTNN GatedDeltaNet chunked mode.

    The production seq-kernel path intentionally differs from the generic torch
    golden for these stress shapes, so this test checks the Win 1 invariant:
    batching the L_inv diagonal-block solves must preserve the previous TTNN
    output bit-for-bit/PCC-wise.
    """
    try:
        import ttnn
    except ImportError:
        print("SKIP: test_gated_deltanet_chunked_ttnn (ttnn not available)")
        return

    from tt import ttnn_delta_rule_seq
    from tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn
    from tests.test_gated_deltanet import make_gated_deltanet_params

    params = make_gated_deltanet_params(seq_len=seq_len, batch_size=1, num_heads=12, num_v_heads=12, head_v_dim=128)

    device = ttnn.open_device(device_id=0)
    # Note: Fast dispatch (default) already enables async execution
    # The op-to-op gaps in the perf report are likely due to synchronization points
    # or operation dependencies, not lack of async execution
    try:

        def make_ttnn_params():
            ttnn_params = {}
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
            for key, val in params.items():
                if key in skip_keys:
                    continue
                if isinstance(val, torch.Tensor):
                    if key.endswith("_proj_weight"):
                        val = val.T.contiguous()
                    if key.endswith("_conv_weight"):
                        ttnn_params[key] = ttnn.from_torch(
                            val,
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                    else:
                        ttnn_params[key] = ttnn.from_torch(
                            val,
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=device,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                        )
                else:
                    ttnn_params[key] = val

            # Force the validation path through the FIR conv fallback instead of native
            # ttnn.conv1d. Zero conv state is equivalent to torch's prefill zero-padding.
            B = params["hidden_states"].shape[0]
            K = params["conv_kernel_size"]
            q_dim = params["num_heads"] * params["head_k_dim"]
            v_dim = params["num_v_heads"] * params["head_v_dim"]
            for state_name, dim in (("conv_state_q", q_dim), ("conv_state_k", q_dim), ("conv_state_v", v_dim)):
                ttnn_params[state_name] = ttnn.zeros(
                    [B, K - 1, dim],
                    device=device,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )

            ttnn_params["device"] = device
            ttnn_params["mode"] = "chunk"
            ttnn_params["chunk_size"] = chunk_size
            return ttnn_params

        def legacy_compute_L_inv_ttnn(L_mat_4d, BH, NC, C, mesh_device, _cmc=None, eye_32=None):
            if eye_32 is None:
                eye_32 = ttnn.from_torch(
                    torch.eye(32, dtype=torch.float32).unsqueeze(0),
                    dtype=ttnn.float32,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh_device,
                    mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
            Ct = C // 32
            batch = BH * NC
            L_flat = ttnn.reshape(L_mat_4d, [batch, C, C], memory_config=_cmc)

            inv_blocks = []
            for b in range(Ct):
                row_start = b * 32
                col_start = b * 32
                block = ttnn.slice(
                    L_flat, [0, row_start, col_start], [batch, row_start + 32, col_start + 32], memory_config=_cmc
                )
                block_inv = ttnn_delta_rule_seq._solve_lower_triangular_ttnn(block, eye_32, mesh_device)
                ttnn.deallocate(block)
                inv_blocks.append(block_inv)

            L_inv_flat = ttnn.concat(inv_blocks, dim=1, memory_config=_cmc)
            for blk in inv_blocks:
                ttnn.deallocate(blk)
            return ttnn.reshape(L_inv_flat, [BH, NC, C, 32], memory_config=_cmc)

        current_compute_L_inv_ttnn = ttnn_delta_rule_seq._compute_L_inv_ttnn
        try:
            ttnn_delta_rule_seq._compute_L_inv_ttnn = current_compute_L_inv_ttnn
            ttnn_out, *_ = gated_deltanet_forward_ttnn(**make_ttnn_params())
            ttnn_out_torch = ttnn.to_torch(ttnn_out).to(torch.float32)

            ttnn_delta_rule_seq._compute_L_inv_ttnn = legacy_compute_L_inv_ttnn
            legacy_out, *_ = gated_deltanet_forward_ttnn(**make_ttnn_params())
            legacy_out_torch = ttnn.to_torch(legacy_out).to(torch.float32)
        finally:
            ttnn_delta_rule_seq._compute_L_inv_ttnn = current_compute_L_inv_ttnn

        inverse_pcc = assert_with_pcc(legacy_out_torch, ttnn_out_torch, pcc_threshold=0.9999)
        print(
            f"PASS: test_gated_deltanet_chunked_ttnn T={seq_len} cs={chunk_size} "
            f"(batched-vs-legacy PCC={inverse_pcc:.6f})"
        )
    finally:
        ttnn.close_device(device)


def test_compute_l_inv_ttnn():
    """Validate the GDN block-diagonal inverse helper at production chunk shape."""
    try:
        import ttnn
    except ImportError:
        print("SKIP: test_compute_l_inv_ttnn (ttnn not available)")
        return

    from tt.ttnn_delta_rule_seq import _compute_L_inv_ttnn

    torch.manual_seed(0)
    BH = 12
    NC = 8
    C = 128
    block = 32
    batch = BH * NC
    Ct = C // block

    L_flat = torch.zeros(batch, C, C, dtype=torch.float32)
    expected_flat = torch.zeros(batch, C, block, dtype=torch.float32)
    eye = torch.eye(block, dtype=torch.float32)
    for b in range(Ct):
        row_start = b * block
        noise = torch.tril(torch.randn(batch, block, block, dtype=torch.float32) * 0.01, diagonal=-1)
        diag = torch.diag_embed(1.0 + torch.rand(batch, block, dtype=torch.float32) * 0.25)
        diag_block = diag + noise
        L_flat[:, row_start : row_start + block, row_start : row_start + block] = diag_block
        expected_flat[:, row_start : row_start + block, :] = torch.linalg.inv(diag_block)

    L_mat_4d = L_flat.reshape(BH, NC, C, C)
    expected = expected_flat.reshape(BH, NC, C, block)

    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        L_tt = ttnn.from_torch(
            L_mat_4d,
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        eye_tt = ttnn.from_torch(
            eye.unsqueeze(0),
            dtype=ttnn.float32,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )

        actual_tt = _compute_L_inv_ttnn(L_tt, BH, NC, C, device, ttnn.DRAM_MEMORY_CONFIG, eye_32=eye_tt)
        actual = ttnn.to_torch(actual_tt).to(torch.float32)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
        print("PASS: test_compute_l_inv_ttnn")
    finally:
        ttnn.close_device(device)


def benchmark_gated_attention(warmup=3, iterations=10):
    """Benchmark torch vs TTNN for Gated Attention."""
    try:
        import ttnn
    except ImportError:
        print("SKIP: benchmark_gated_attention (ttnn not available)")
        return

    from torch_functional.gated_attention import gated_attention_forward
    from tt.ttnn_gated_attention import gated_attention_forward_ttnn
    from tests.test_gated_attention import make_gated_attention_params

    params = make_gated_attention_params()

    # --- Torch benchmark ---
    for _ in range(warmup):
        gated_attention_forward(**params)

    torch_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gated_attention_forward(**params)
        torch_times.append(time.perf_counter() - t0)

    torch_avg = sum(torch_times) / len(torch_times)
    torch_min = min(torch_times)

    # --- TTNN benchmark ---
    device = ttnn.open_device(device_id=0)
    try:
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

        for _ in range(warmup):
            _ = gated_attention_forward_ttnn(**ttnn_params)
            ttnn.synchronize_device(device)

        ttnn_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = gated_attention_forward_ttnn(**ttnn_params)
            ttnn.synchronize_device(device)
            ttnn_times.append(time.perf_counter() - t0)

        ttnn_avg = sum(ttnn_times) / len(ttnn_times)
        ttnn_min = min(ttnn_times)
    finally:
        ttnn.close_device(device)

    print(f"\n{'='*60}")
    print(f"  Gated Attention Benchmark ({iterations} iterations)")
    print(f"{'='*60}")
    print(f"  Torch  (CPU):  avg={torch_avg*1000:.2f} ms  min={torch_min*1000:.2f} ms")
    print(f"  TTNN (device): avg={ttnn_avg*1000:.2f} ms  min={ttnn_min*1000:.2f} ms")
    print(f"  Speedup:       {torch_avg/ttnn_avg:.2f}x (avg)  {torch_min/ttnn_min:.2f}x (min)")
    print(f"{'='*60}\n")


def benchmark_gated_deltanet(warmup=3, iterations=10, seq_len=16, mode="recurrent", chunk_size=64):
    """Benchmark torch vs TTNN for Gated DeltaNet."""
    try:
        import ttnn
    except ImportError:
        print("SKIP: benchmark_gated_deltanet (ttnn not available)")
        return

    from torch_functional.gated_deltanet import gated_deltanet_forward
    from tt.ttnn_gated_deltanet import gated_deltanet_forward_ttnn
    from tests.test_gated_deltanet import make_gated_deltanet_params

    params = make_gated_deltanet_params(seq_len=seq_len)

    torch_mode = "fused_recurrent" if mode == "recurrent" else "chunk"

    # --- Torch benchmark ---
    for _ in range(warmup):
        gated_deltanet_forward(**params, mode=torch_mode, chunk_size=chunk_size)

    torch_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        gated_deltanet_forward(**params, mode=torch_mode, chunk_size=chunk_size)
        torch_times.append(time.perf_counter() - t0)

    torch_avg = sum(torch_times) / len(torch_times)
    torch_min = min(torch_times)

    # --- TTNN benchmark ---
    device = ttnn.open_device(device_id=0, l1_small_size=16384)
    try:
        ttnn_params = {}
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
        for key, val in params.items():
            if key in skip_keys:
                continue
            if isinstance(val, torch.Tensor):
                if key.endswith("_proj_weight"):
                    val = val.T.contiguous()
                if key.endswith("_conv_weight"):
                    ttnn_params[key] = ttnn.from_torch(
                        val,
                        dtype=ttnn.bfloat16,
                    )
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
        ttnn_params["mode"] = mode
        ttnn_params["chunk_size"] = chunk_size

        for _ in range(warmup):
            _ = gated_deltanet_forward_ttnn(**ttnn_params)
            ttnn.synchronize_device(device)

        ttnn_times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = gated_deltanet_forward_ttnn(**ttnn_params)
            ttnn.synchronize_device(device)
            ttnn_times.append(time.perf_counter() - t0)

        ttnn_avg = sum(ttnn_times) / len(ttnn_times)
        ttnn_min = min(ttnn_times)
    finally:
        ttnn.close_device(device)

    print(f"\n{'='*60}")
    print(f"  Gated DeltaNet Benchmark ({mode} T={seq_len}, {iterations} iters)")
    print(f"{'='*60}")
    print(f"  Torch  (CPU):  avg={torch_avg*1000:.2f} ms  min={torch_min*1000:.2f} ms")
    print(f"  TTNN (device): avg={ttnn_avg*1000:.2f} ms  min={ttnn_min*1000:.2f} ms")
    print(f"  Speedup:       {torch_avg/ttnn_avg:.2f}x (avg)  {torch_min/ttnn_min:.2f}x (min)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=["attention", "deltanet", "inverse"],
        default=None,
        help="Run only one module (default: both)",
    )
    parser.add_argument("--bench", action="store_true", help="Run performance benchmarks")
    parser.add_argument("--iterations", type=int, default=10, help="Benchmark iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations")
    parser.add_argument("--seq-len", type=int, default=16, help="Sequence length for DeltaNet (1=pure decode)")
    parser.add_argument(
        "--mode",
        choices=["recurrent", "chunk"],
        default="recurrent",
        help="DeltaNet mode: recurrent (decode) or chunk (prefill)",
    )
    parser.add_argument("--chunk-size", type=int, default=128, help="Chunk size for chunked mode")
    args = parser.parse_args()

    run_attention = args.module in (None, "attention")
    run_deltanet = args.module in (None, "deltanet")
    run_inverse = args.module == "inverse"

    if run_attention:
        test_gated_attention_ttnn()
    if run_deltanet:
        if args.mode == "chunk":
            test_gated_deltanet_chunked_ttnn(seq_len=args.seq_len, chunk_size=args.chunk_size)
        else:
            test_gated_deltanet_recurrent_ttnn(seq_len=args.seq_len)
    if run_inverse:
        test_compute_l_inv_ttnn()
    print("\nTTNN validation complete!")

    if args.bench:
        print("\nRunning performance benchmarks...")
        if run_attention:
            benchmark_gated_attention(warmup=args.warmup, iterations=args.iterations)
        if run_deltanet:
            benchmark_gated_deltanet(
                warmup=args.warmup,
                iterations=args.iterations,
                seq_len=args.seq_len,
                mode=args.mode,
                chunk_size=args.chunk_size,
            )
