# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)
import ttnn
from loguru import logger
import pytest


def fa_rand(*shape):
    normal_1 = torch.randn(shape)
    normal_2 = torch.randn(shape) * 10
    bernoulli = torch.bernoulli(torch.full(shape, 0.001))
    return normal_1 + normal_2 * bernoulli


def create_sliding_window_mask_prefill(b, nh, seq_len, sliding_window=0, is_causal=True):
    """
    Create attention mask for sliding window attention in prefill mode.

    Args:
        b: batch size
        nh: number of heads
        seq_len: sequence length
        sliding_window: sliding window size
        is_causal: whether to apply causal constraint

    Returns:
        attn_mask: [b, nh, seq_len, seq_len] mask with -inf for positions outside window
    """
    attn_mask = torch.zeros((b, nh, seq_len, seq_len))

    for i in range(b):
        for q_pos in range(seq_len):
            if is_causal:
                # Causal sliding window: spans from (q_pos - sliding_window + 1) to q_pos (inclusive)
                window_end = q_pos + 1  # exclusive (causal constraint)
                window_start = max(0, window_end - sliding_window) if sliding_window > 0 else 0

                # Mask positions before sliding window start
                if window_start > 0:
                    attn_mask[i, :, q_pos, :window_start] = torch.finfo(torch.float32).min

                # Mask positions after current position (causal constraint)
                if q_pos + 1 < seq_len:
                    attn_mask[i, :, q_pos, q_pos + 1 :] = torch.finfo(torch.float32).min
            else:
                # Non-causal sliding window: centered on diagonal with half before and half after
                half_window = sliding_window // 2 if sliding_window > 0 else seq_len // 2
                window_start = max(0, q_pos - half_window)
                window_end = min(seq_len, q_pos + half_window + 1)  # exclusive

                # Mask positions outside the sliding window
                if window_start > 0:
                    attn_mask[i, :, q_pos, :window_start] = torch.finfo(torch.float32).min
                if window_end < seq_len:
                    attn_mask[i, :, q_pos, window_end:] = torch.finfo(torch.float32).min

    return attn_mask


def reference_sdpa_with_attention_sinks(Q, K, V, S, is_causal=True, sliding_window=0):
    """
    Reference implementation of scaled dot product attention with attention sinks.

    Args:
        Q: Query tensor [b, nh, s, d]
        K: Key tensor [b, nh, s, d]
        V: Value tensor [b, nh, s, d]
        S: Attention sink tensor [b, nh] - one sink value per head (broadcast to all queries)
        is_causal: Whether to apply causal masking
        sliding_window: Sliding window size (0 = no sliding window)
    Returns:
        Output tensor [b, nh, s, d]
    """
    b, nh, s, d = Q.shape
    assert K.shape == (b, nh, s, d)
    assert V.shape == (b, nh, s, d)
    assert S.shape == (1, nh, 1, 1), f"Expected S shape {(1, nh, 1, 1)}, got {S.shape}"

    # Compute attention scores: QK = Q @ K^T
    # Q: [b, nh, s, d], K: [b, nh, s, d] -> QK: [b, nh, s, s]
    QK = torch.matmul(Q, K.transpose(-2, -1))

    # Scale
    sm_scale = 1.0 / math.sqrt(d)
    QK = QK * sm_scale

    # Apply causal mask if needed
    if is_causal or sliding_window > 0:
        mask = create_sliding_window_mask_prefill(b, nh, s, sliding_window, is_causal)
        QK = QK + mask

    if S is not None:
        # Broadcast attention sink values to all query positions
        # S: [b, nh] -> [b, nh, s, 1]
        S_broadcast = S.repeat_interleave(b, dim=0)
        S_broadcast = S_broadcast.repeat_interleave(s, dim=-2)
        S_broadcast = S_broadcast * sm_scale

        # Concatenate attention sink scores
        # QK: [b, nh, s, s], S_broadcast: [b, nh, s, 1] -> QK_with_sink: [b, nh, s, s+1]
        QK = torch.cat([QK, S_broadcast], dim=-1)

    # Apply softmax over extended dimension (including sink)
    W = torch.softmax(QK, dim=-1)

    if S is not None:
        # Slice off attention sink weights (they don't contribute to output)
        W = W[..., :-1]  # [b, nh, s, s]

    # Compute final output
    output = torch.matmul(W, V)  # [b, nh, s, d]

    return output


def reference_flash_attention_with_sinks(Q, K, V, S, is_causal=True, q_chunk_size=32, k_chunk_size=32):
    """
    Flash Attention implementation with attention sinks using chunked processing.

    Args:
        Q: Query tensor [b, nh, s, d]
        K: Key tensor [b, nh, s, d]
        V: Value tensor [b, nh, s, d]
        S: Attention sink tensor [b, nh] - one sink value per head (or None)
        is_causal: Whether to apply causal masking
        q_chunk_size: Size of Q chunks for tiling
        k_chunk_size: Size of K chunks for tiling

    Returns:
        Output tensor [b, nh, s, d]
    """
    b, nh, s, d = Q.shape
    assert K.shape == (b, nh, s, d)
    assert V.shape == (b, nh, s, d)
    assert S.shape == (1, nh, 1, 1), f"Expected S shape {(1, nh, 1, 1)}, got {S.shape}"

    # Compute scale
    sm_scale = 1.0 / math.sqrt(d)
    S = S.repeat_interleave(b, dim=0)

    # Initialize output accumulator
    output = torch.zeros_like(Q)

    # Process in Q chunks
    num_q_chunks = (s + q_chunk_size - 1) // q_chunk_size

    for q_chunk_idx in range(num_q_chunks):
        q_start = q_chunk_idx * q_chunk_size
        q_end = min(q_start + q_chunk_size, s)
        q_chunk_len = q_end - q_start

        Q_chunk = Q[:, :, q_start:q_end, :]  # [b, nh, q_chunk_len, d]

        # Initialize running statistics for this Q chunk
        running_max = torch.full((b, nh, q_chunk_len, 1), float("-inf"), device=Q.device, dtype=Q.dtype)
        running_sum = torch.zeros((b, nh, q_chunk_len, 1), device=Q.device, dtype=Q.dtype)
        running_output = torch.zeros((b, nh, q_chunk_len, d), device=Q.device, dtype=Q.dtype)

        # Process K chunks (all chunks up to and including current Q chunk for causal)
        # For causal attention, we attend to all K positions up to each Q position
        max_k_chunks = (
            (q_end + k_chunk_size - 1) // k_chunk_size if is_causal else (s + k_chunk_size - 1) // k_chunk_size
        )
        for k_chunk_idx in range(max_k_chunks):
            k_start = k_chunk_idx * k_chunk_size
            k_end = min(k_start + k_chunk_size, s)

            # Only process if this K chunk contains tokens that should be attended to (causal constraint)
            if is_causal and k_start >= q_end:
                break

            K_chunk = K[:, :, k_start:k_end, :]  # [b, nh, k_chunk_len, d]
            V_chunk = V[:, :, k_start:k_end, :]  # [b, nh, k_chunk_len, d]

            # Compute attention scores for this QK chunk pair
            QK = torch.matmul(Q_chunk, K_chunk.transpose(-2, -1)) * sm_scale  # [b, nh, q_chunk_len, k_chunk_len]

            # Apply causal mask if needed
            if is_causal:
                q_indices = torch.arange(q_start, q_end, device=Q.device)[:, None]
                k_indices = torch.arange(k_start, k_end, device=Q.device)[None, :]
                causal_mask = q_indices < k_indices
                QK = QK.masked_fill(causal_mask[None, None, :, :], float("-inf"))

            # Compute max for this chunk (handling -inf properly)
            chunk_max = QK.max(dim=-1, keepdim=True).values  # [b, nh, q_chunk_len, 1]

            # Update running max
            new_max = torch.maximum(running_max, chunk_max)

            # Rescale previous statistics if we're not on the first K chunk
            if k_chunk_idx > 0 or torch.any(running_max > float("-inf")):
                exp_diff_prev = torch.exp(running_max - new_max)
                # Handle -inf cases
                exp_diff_prev = torch.nan_to_num(exp_diff_prev, nan=0.0, posinf=0.0, neginf=0.0)
                running_sum = running_sum * exp_diff_prev
                running_output = running_output * exp_diff_prev

            # Compute softmax for current chunk
            QK_exp = torch.exp(QK - new_max)
            # Handle -inf cases (masked positions)
            QK_exp = torch.nan_to_num(QK_exp, nan=0.0, posinf=0.0, neginf=0.0)

            chunk_sum = QK_exp.sum(dim=-1, keepdim=True)

            # Update running sum
            running_sum = running_sum + chunk_sum

            # Accumulate weighted values
            running_output = running_output + torch.matmul(QK_exp, V_chunk)

            # Update running max
            running_max = new_max
        if S is not None:
            # Process attention sink as a virtual K chunk
            # S shape: [b, nh, 1, 1] -> broadcast to [b, nh, q_chunk_len, 1]
            S_scaled = S * sm_scale
            S_chunk = S_scaled.repeat_interleave(q_chunk_len, dim=-2)  # [b, nh, q_chunk_len, 1]

            # Update running max with sink values
            new_max = torch.maximum(running_max, S_chunk)

            # Rescale previous statistics
            exp_diff_prev = torch.exp(running_max - new_max)
            exp_diff_prev = torch.nan_to_num(exp_diff_prev, nan=0.0, posinf=0.0, neginf=0.0)
            running_sum = running_sum * exp_diff_prev
            running_output = running_output * exp_diff_prev

            # Add sink contribution to sum (sink doesn't contribute to output)
            sink_exp = torch.exp(S_chunk - new_max)
            running_sum = running_sum + sink_exp

        # Final normalization
        output[:, :, q_start:q_end, :] = running_output / running_sum

    # Output is already in [b, nh, s, d] format
    return output


def run_test_sdpa_tt(
    device,
    b,
    nh,
    nkv,
    s,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    use_high_precision_compute=False,
    rmse_threshold=None,
    memory_config=ttnn.DRAM_MEMORY_CONFIG,
):
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    if use_high_precision_compute:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
    else:
        compute_kernel_config = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2,
            math_approx_mode=True,
            fp32_dest_acc_en=False,
            packer_l1_acc=False,
        )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(
        Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device, pad_value=0.0
    )
    tt_K = ttnn.from_torch(
        K, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device, pad_value=0.0
    )
    tt_V = ttnn.from_torch(
        V, dtype=dtype, layout=ttnn.TILE_LAYOUT, memory_config=memory_config, device=device, pad_value=0.0
    )
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q, tt_K, tt_V, is_causal=True, program_config=program_config, compute_kernel_config=compute_kernel_config
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, d, S
    gt = torch.nn.functional.scaled_dot_product_attention(Q, K_repeated, V_repeated, is_causal=True)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    else:
        assert out_pass


def run_sdpa_noncausal(
    device,
    b,
    nh,
    nkv,
    sq,
    d,
    q_chunk_size,
    k_chunk_size,
    dtype,
    sk=None,
    use_mask=True,
    rmse_threshold=None,
    bcast_mask_batch_dim=False,
    bcast_mask_head_dim=True,
):
    torch.manual_seed(1234)
    if sk is None:
        sk = sq

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, sq, d)
    K = fa_rand(b, nkv, sk, d)
    V = fa_rand(b, nkv, sk, d)
    # Generate random non-causal attention mask
    tt_mask = None
    mask = None
    if use_mask:
        mask = torch.bernoulli(
            torch.full(
                (
                    1 if bcast_mask_batch_dim else b,
                    1 if bcast_mask_head_dim else nh,
                    sq,
                    sk,
                ),
                0.25,
            )
        )
        mask = mask * -1e9
        tt_mask = ttnn.from_torch(mask, dtype=ttnn.bfloat4_b, layout=ttnn.TILE_LAYOUT, device=device)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=False,
        attn_mask=tt_mask,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :sq, :]

    if nkv > 1 and nkv != nh:
        assert nh % nkv == 0
        K = K.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)
        V = V.reshape(b, nkv, 1, sk, d).repeat(1, 1, nh // nkv, 1, 1).reshape(b, nh, sk, d)

    gt = torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=False, attn_mask=mask)

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold

    assert out_pass


def run_test_sdpa_sliding_window(
    device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sliding_window, is_causal=True, rmse_threshold=None
):
    """Test sliding window attention in prefill mode."""
    torch.manual_seed(1234)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=is_causal,
        sliding_window_size=sliding_window,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    # Create reference with sliding window mask
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, s, s
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)  # b, nh, s, s

    # Create sliding window mask
    sliding_window_mask = create_sliding_window_mask_prefill(b, nh, s, sliding_window, is_causal)

    gt = torch.nn.functional.scaled_dot_product_attention(
        Q, K_repeated, V_repeated, attn_mask=sliding_window_mask, is_causal=False
    )

    out_pass, out_pcc = comp_pcc(gt, tt_back, 0.994)
    logger.debug(f"python vs pytorch: {out_pcc}")
    rmse = torch.sqrt(((gt - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold
    else:
        assert out_pass


def run_test_sdpa_with_attention_sink(
    device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sink_values=None, rmse_threshold=None
):
    """Test SDPA with attention sinks using per-head sink values."""
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=True,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=True,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    Q = fa_rand(b, nh, s, d)
    K = fa_rand(b, nkv, s, d)
    V = fa_rand(b, nkv, s, d)

    # Create per-head attention sink values
    # Shape: [1, nh, 1, 1] - one value per head,
    if sink_values is None:
        S_per_head = torch.rand(1, nh) * 4.0  # Random values scaled by to make closer to real distribution
    else:
        S_per_head = torch.tensor(sink_values).reshape(1, nh)

    # Prepare attention sink tensor for device: [1, nh, 1, 1]
    S_padded = S_per_head.reshape(1, nh, 1, 1)

    tt_Q = ttnn.from_torch(Q, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_K = ttnn.from_torch(K, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_V = ttnn.from_torch(V, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)
    tt_S = ttnn.from_torch(S_padded, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, pad_value=0.0)

    tt_back = ttnn.transformer.scaled_dot_product_attention(
        tt_Q,
        tt_K,
        tt_V,
        is_causal=True,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        attention_sink=tt_S,
    )
    tt_back = ttnn.to_torch(tt_back)
    # Slice out any tile-padding
    tt_back = tt_back[:, :, :s, :]

    # Compute reference with GQA expansion
    K_repeated = torch.cat([K[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)
    V_repeated = torch.cat([V[:, i : i + 1, :, :].repeat(1, nh // nkv, 1, 1) for i in range(nkv)], dim=1)

    # Compute reference output using per-head sink values
    gt = reference_sdpa_with_attention_sinks(
        Q,
        K_repeated,
        V_repeated,
        S_padded,
        is_causal=True,
    )
    gt_flash = reference_flash_attention_with_sinks(
        Q,
        K_repeated,
        V_repeated,
        S_padded,
        is_causal=True,
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
    )

    out_pass, out_pcc = comp_pcc(gt, gt_flash, 0.99)
    logger.debug(f"python vs python(flash): {out_pcc}")
    rmse = torch.sqrt(((gt - gt_flash) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")
    if rmse_threshold is not None:
        assert rmse < rmse_threshold, f"RMSE {rmse} exceeds threshold {rmse_threshold}"
    else:
        assert out_pass, f"PCC check failed: {out_pcc}"

    out_pass, out_pcc = comp_pcc(gt_flash, tt_back, 0.99)
    logger.debug(f"pytorch vs tt: {out_pcc}")
    rmse = torch.sqrt(((gt_flash - tt_back) ** 2).mean()).item()
    logger.debug(f"rmse: {rmse}")

    if rmse_threshold is not None:
        assert rmse < rmse_threshold, f"RMSE {rmse} exceeds threshold {rmse_threshold}"
    else:
        assert out_pass, f"PCC check failed: {out_pcc}"


# ---------------------------------------------------------------------------
# Test functions with reduced parametrizations for fast post-commit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b], ids=["bfp8"])
@pytest.mark.parametrize("memory_config", [ttnn.DRAM_MEMORY_CONFIG], ids=["dram_interleaved"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 2048, 128],),
)
def test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, memory_config):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")
    rmse_threshold = 0.0092 if (dtype == ttnn.bfloat8_b or dtype == ttnn.bfloat4_b) else 0.0093
    run_test_sdpa_tt(
        device,
        b,
        nh,
        nkv,
        s,
        d,
        q_chunk_size,
        k_chunk_size,
        dtype,
        rmse_threshold=rmse_threshold,
        memory_config=memory_config,
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 2048, 128],),
)
def test_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if s > 2048 and (q_chunk_size == 128 or k_chunk_size == 128):
        pytest.skip("Bad PCC for small chunks")
    rmse_threshold = 0.0069
    run_sdpa_noncausal(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold)


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b], ids=["bfp8"])
@pytest.mark.parametrize("q_chunk_size", [128], ids=["q128"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 2048, 128],),
)
def test_sdpa_tt_with_program_cache(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype):
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh == 8 and q_chunk_size == 128 and k_chunk_size == 128:
        pytest.skip("Can cause OOM if profiling is enabled.")

    for _ in range(2):
        run_test_sdpa_tt(device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype)

    assert device.num_program_cache_entries() == 1


@pytest.mark.parametrize("dtype", [ttnn.bfloat8_b], ids=["bfp8"])
@pytest.mark.parametrize("q_chunk_size", [256], ids=["q256"])
@pytest.mark.parametrize("k_chunk_size", [256], ids=["k256"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d, sliding_window",
    ([1, 8, 1, 2048, 128, 128],),
)
def test_sdpa_sliding_window(device, b, nh, nkv, s, d, dtype, q_chunk_size, k_chunk_size, sliding_window):
    """Test sliding window attention functionality in SDPA prefill."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if sliding_window >= s:
        pytest.skip("sliding_window must be smaller than sequence length")

    rmse_threshold = 0.01
    run_test_sdpa_sliding_window(
        device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, sliding_window, rmse_threshold=rmse_threshold
    )


@pytest.mark.parametrize("dtype", [ttnn.bfloat16], ids=["bf16"])
@pytest.mark.parametrize("q_chunk_size", [32], ids=["q32"])
@pytest.mark.parametrize("k_chunk_size", [128], ids=["k128"])
@pytest.mark.parametrize(
    "b, nh, nkv, s, d",
    ([1, 8, 1, 256, 32],),
)
def test_sdpa_with_attention_sink(device, b, nh, nkv, s, d, dtype, q_chunk_size, k_chunk_size, reset_seeds):
    """Test SDPA with per-head attention sinks on device."""
    if (s % q_chunk_size != 0) or (s % k_chunk_size != 0):
        pytest.skip("s must be divisible by q_chunk_size and k_chunk_size")
    if nh % nkv != 0:
        pytest.skip("nkv must divide nh")

    rmse_threshold = 0.02
    run_test_sdpa_with_attention_sink(
        device, b, nh, nkv, s, d, q_chunk_size, k_chunk_size, dtype, rmse_threshold=rmse_threshold
    )
