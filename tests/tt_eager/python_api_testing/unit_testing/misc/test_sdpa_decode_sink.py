# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from models.utility_functions import nearest_y

import ttnn
from loguru import logger
import pytest


def flash_decode_sdpa(Q, K_cache, V_cache, sink, sm_scale, sliding_window=0, block_size=128):
    """
    Flash Decode implementation for autoregressive generation.

    Args:
        Q: Query tensor [batch, 1, n_heads, q_mult, d_head] - only the new token
        K_cache: Key cache [batch, seq_len, n_heads, d_head] - all previous tokens
        sink: Attention sink tensor [batch, 1, n_heads, 1, 1] - used for scaling
        V_cache: Value cache [batch, seq_len, n_heads, d_head] - all previous tokens
        sm_scale: Scaling factor for attention scores
        sliding_window: Sliding window size (0 = no window)
        block_size: Block size for tiled computation

    Returns:
        attn: Attention output [batch, 1, n_heads * q_mult, d_head]
    """
    batch_size, q_len, n_heads, q_mult, d_head = Q.shape
    _, seq_len, _, _ = K_cache.shape

    assert q_len == 1, "Flash decode expects single query token"
    assert K_cache.shape == (batch_size, seq_len, n_heads, d_head)
    assert V_cache.shape == (batch_size, seq_len, n_heads, d_head)

    # Expand K and V cache to match Q's q_mult dimension
    K_cache = K_cache[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, seq_len, n_heads, q_mult, d_head]
    V_cache = V_cache[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, seq_len, n_heads, q_mult, d_head]

    # Initialize output and normalization tensors
    O = torch.zeros_like(Q).squeeze(1)  # [batch, n_heads, q_mult, d_head]
    l = torch.zeros(batch_size, n_heads, q_mult, device=Q.device, dtype=Q.dtype)  # row sums
    m = torch.full((batch_size, n_heads, q_mult), -float("inf"), device=Q.device, dtype=Q.dtype)  # row maxes

    Q = Q.squeeze(1)  # [batch, n_heads, q_mult, d_head]

    # Apply sliding window mask bounds
    start_idx = 0
    if sliding_window > 0:
        start_idx = max(0, seq_len - sliding_window)

    # Process in blocks for memory efficiency
    for block_start in range(start_idx, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)

        # Extract block
        K_block = K_cache[:, block_start:block_end, :, :, :]  # [batch, block_len, n_heads, q_mult, d_head]
        V_block = V_cache[:, block_start:block_end, :, :, :]  # [batch, block_len, n_heads, q_mult, d_head]

        # Compute attention scores for this block
        # Q: [batch, n_heads, q_mult, d_head]
        # K_block: [batch, block_len, n_heads, q_mult, d_head]
        S = torch.einsum("bhmd,bkhmd->bhmk", Q, K_block) * sm_scale  # [batch, n_heads, q_mult, block_len]

        # Apply causal mask - since we're at position seq_len, all previous tokens are visible
        # No additional masking needed for causality in decode phase

        # Online softmax update
        m_new = torch.maximum(m[:, :, :, None], S.max(dim=-1, keepdim=True)[0])  # [batch, n_heads, q_mult, 1]

        # Compute exponentials with numerically stable softmax
        alpha = torch.exp(m[:, :, :, None] - m_new)  # [batch, n_heads, q_mult, 1]
        exp_S = torch.exp(S - m_new)  # [batch, n_heads, q_mult, block_len]

        # Update row sum
        l_new = alpha.squeeze(-1) * l + exp_S.sum(dim=-1)  # [batch, n_heads, q_mult]

        # Update output
        O = alpha.squeeze(-1)[:, :, :, None] * O + torch.einsum("bhmk,bkhmd->bhmd", exp_S, V_block)

        # Update running statistics
        l = l_new
        m = m_new.squeeze(-1)

    if sink is not None:
        ##### Handle Attention Sink #####
        sink = sink.reshape(batch_size, n_heads, q_mult, 1)  # [batch, n_heads, q_mult, 1]

        # max(m, sink)
        m_new = torch.maximum(m[:, :, :, None], sink)  # [batch, n_heads, q_mult, 1]

        # exp(m - m_new)
        alpha = torch.exp(m[:, :, :, None] - m_new)  # [batch, n_heads, q_mult, 1]

        # sub_exp -> exp(sink - m_new)
        exp_sink = torch.exp(sink - m_new)  # [batch, n_heads, q_mult, 1]

        # l_new -> l * alpha + sub_exp, ie, re-scale l + exp_sink
        l_new = alpha.squeeze(-1) * l + exp_sink.sum(dim=-1)  # [batch, n_heads, q_mult]

        # O -> O * alpha
        O = alpha.squeeze(-1)[:, :, :, None] * O

        l = l_new
        ##### Done Handling Attention Sink #####

    # Final normalization
    O = O / l[:, :, :, None]

    # Reshape output to [batch, 1, n_heads * q_mult, d_head]
    O = O.reshape(batch_size, 1, n_heads * q_mult, d_head)

    return O


def scaled_dot_product_attention_reference(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    batch_size, n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (batch_size, n_tokens, n_heads, d_head)
    assert V.shape == (batch_size, n_tokens, n_heads, d_head)

    # Expand K and V to match Q's q_mult dimension
    K = K[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, n_tokens, n_heads, q_mult, d_head]
    V = V[:, :, :, None, :].expand(-1, -1, -1, q_mult, -1)  # [batch, n_tokens, n_heads, q_mult, d_head]

    # Expand S to match the required dimensions
    S = S.reshape(batch_size, n_heads, q_mult, 1, 1).expand(
        -1, -1, -1, n_tokens, -1
    )  # [batch, n_heads, q_mult, n_tokens, 1]

    # Create causal mask
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)

    # Add sliding window mask if specified
    if sliding_window > 0:
        mask += torch.tril(mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window)

    # Compute attention scores QK^T
    QK = torch.einsum("bqhmd,bkhmd->bhmqk", Q, K)  # [batch, n_heads, q_mult, n_tokens, n_tokens]
    QK *= sm_scale

    # Apply mask (broadcast across batch and head dimensions)
    QK += mask[None, None, None, :, :]  # broadcast mask to [batch, n_heads, q_mult, n_tokens, n_tokens]

    # Concatenate with S
    QK = torch.cat([QK, S], dim=-1)  # [batch, n_heads, q_mult, n_tokens, n_tokens+1]

    # Apply softmax
    W = torch.softmax(QK, dim=-1)

    # Remove the extra dimension added by S
    W = W[..., :-1]  # [batch, n_heads, q_mult, n_tokens, n_tokens]

    # Compute attention output
    attn = torch.einsum("bhmqk,bkhmd->bqhmd", W, V)  # [batch, n_tokens, n_heads, q_mult, d_head]

    # Reshape to [batch, n_tokens, n_heads * q_mult, d_head]
    return attn.reshape(batch_size, n_tokens, n_heads * q_mult, d_head)


def run_sdpa_decode_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    dim,
    q_dtype,
    dtype,
):
    use_ref_fa = True  # Use Torch impl for FlashDecode + attention sink

    num_iters = 1

    # Log the test parameters
    logger.debug(f"Running SDPA Decode with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"Dim: {dim}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, dim).float()  # (B, H, S, D)
    k = torch.randn(batch, nkv, seq_len, dim).float()  # (B, H, S, D)
    v = torch.randn(batch, nkv, seq_len, dim).float()  # (B, H, S, D)

    # TODO: Remove -inf after ttnn supports attention sink
    sink = torch.randn(batch, 1, nh, 1, 1) * -float("inf")  # (1, H, 1, 1)

    ref_q = q.permute(0, 2, 1, 3).view(batch, seq_len, nkv, nh // nkv, dim)
    ref_k = k.permute(0, 2, 1, 3)
    ref_v = v.permute(0, 2, 1, 3)

    tt_q_in = q[:, :, -1:, :].permute(2, 0, 1, 3)  # (D, B, H, S) -> (S, B, H, D)

    ######################
    ### TT Setup
    #######################
    q_chunk_size = 0  # Not used in decode
    k_chunk_size = 128

    scale = dim**-0.5
    start_indices = batch * [seq_len - 1]

    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=False,
    )

    # Set up input tensors
    q_mem_config = ttnn.DRAM_MEMORY_CONFIG
    out_mem_config = ttnn.DRAM_MEMORY_CONFIG

    tt_q = ttnn.from_torch(
        tt_q_in,
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=q_mem_config,
    )
    tt_k = ttnn.from_torch(
        k,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_v = ttnn.from_torch(
        v,
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    tt_start_indices = ttnn.from_torch(
        torch.tensor(start_indices),
        device=device,
        dtype=ttnn.int32,
    )

    ##########################
    ### SDPA Decode
    ##########################
    logger.info(f"Running SDPA Decode with TT Q shape: {tt_q.shape}, TT K shape: {tt_k.shape}, dtype: {dtype}")

    def run_op():
        if use_ref_fa:
            tt_out = flash_decode_sdpa(
                ref_q[:, -1:, ...],
                ref_k,
                ref_v,
                sink,
                scale,
                sliding_window=0,
            )
        else:
            tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
                tt_q,
                tt_k,
                tt_v,
                cur_pos_tensor=tt_start_indices,
                scale=scale,
                program_config=sdpa_program_config,
                compute_kernel_config=compute_kernel_config,
                memory_config=out_mem_config,
            )

        return tt_out

    tt_outs = []
    for i in range(num_iters):  # Check for program cache
        logger.debug(f"Running SDPA Decode operation iteration {i + 1}/{num_iters}")
        tt_out = run_op()
        tt_outs.append(tt_out)

        # Increment start indices for the next iteration
        ttnn.plus_one(tt_start_indices)

    ########################
    ### Validation
    ########################
    outs = []
    for _ in range(num_iters):
        out_t = scaled_dot_product_attention_reference(
            ref_q,
            ref_k,
            ref_v,
            sink,
            scale,
            sliding_window=0,
        )
        out_t = out_t[:, -1:, ...]
        outs.append(out_t)

        start_indices = [x + 1 for x in start_indices]

    pcc_threshold = 0.999
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.91
    if dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.98

    for i, (tt_out, out_t) in enumerate(zip(tt_outs, outs)):
        if not use_ref_fa:
            tt_out = ttnn.to_torch(tt_out)[..., :nh, :]  # (S, B, H, D)

        out_pass, out_pcc = comp_pcc(tt_out, out_t, pcc_threshold)
        logger.debug(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"

    # Check program cache entries
    num_program_cache_entries = device.num_program_cache_entries()

    # SDPA + PlusOne
    expected_num_program_cache_entries = 1 if use_ref_fa else 2
    assert (
        num_program_cache_entries == expected_num_program_cache_entries
    ), f"Expected {expected_num_program_cache_entries} program cache entries, got {num_program_cache_entries}."


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, dim",
    [
        (1, 256, 32, 4, 64),  # GPT-OSS 20B TP=2
        (32, 256, 32, 4, 64),
        (32, 256, 32, 1, 64),
        # (1, 256, 64, 8, 64),  # GPT-OSS 20B TP=1 (FIXME: Fails)
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
    ],
)
def test_sdpa_decode(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    dim,
    q_dtype,
    dtype,
    function_level_defaults,
    reset_seeds,
):
    run_sdpa_decode_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        dim,
        q_dtype,
        dtype,
    )
