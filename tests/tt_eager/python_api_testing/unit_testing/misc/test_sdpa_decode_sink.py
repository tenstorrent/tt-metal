# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_pcc,
)

import ttnn
from loguru import logger
import pytest
from models.utility_functions import skip_for_blackhole


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
    sink = torch.randn(1, 1, nh, 1, 1).repeat(batch, 1, 1, 1, 1)  # (batch, 1, H, 1, 1)
    sink *= 4.0  # Closer to real distribution

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

    # Setup sink for TTNN
    tt_sink_in = sink[:1, ...].reshape(nh, 1)
    tt_sink_in = torch.nn.functional.pad(tt_sink_in, (0, ttnn.TILE_SIZE - 1), "constant", 0)
    tt_sink_in /= scale  # Important!! GPT-OSS expects sink to not be scaled

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

    tt_sink = ttnn.from_torch(
        tt_sink_in,
        device=device,
        dtype=ttnn.bfloat16,
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
        torch_out = flash_decode_sdpa(
            ref_q[:, -1:, ...],
            ref_k,
            ref_v,
            sink,
            scale,
            sliding_window=0,
        )

        tt_out = ttnn.transformer.scaled_dot_product_attention_decode(
            tt_q,
            tt_k,
            tt_v,
            cur_pos_tensor=tt_start_indices,
            scale=scale,
            attention_sink=tt_sink,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=out_mem_config,
        )

        return torch_out, tt_out

    outs = []
    for i in range(num_iters):  # Check for program cache
        logger.debug(f"Running SDPA Decode operation iteration {i + 1}/{num_iters}")
        torch_out, tt_out = run_op()
        outs.append((torch_out, tt_out))

        # Increment start indices for the next iteration
        ttnn.plus_one(tt_start_indices)
        start_indices = [x + 1 for x in start_indices]

    ########################
    ### Validation
    ########################
    pcc_threshold = 0.999
    if dtype == ttnn.bfloat4_b:
        pcc_threshold = 0.98
    if dtype == ttnn.bfloat8_b:
        pcc_threshold = 0.999

    for i, (torch_out, tt_out) in enumerate(outs):
        tt_out = ttnn.to_torch(tt_out)[..., :nh, :]  # (S, B, H, D)

        out_pass, out_pcc = comp_pcc(torch_out, tt_out, pcc_threshold)
        logger.debug(f"Output PCC: {out_pcc}")

    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"

    # Check program cache entries
    num_program_cache_entries = device.num_program_cache_entries()

    # SDPA + PlusOne
    expected_num_program_cache_entries = 2
    assert (
        num_program_cache_entries == expected_num_program_cache_entries
    ), f"Expected {expected_num_program_cache_entries} program cache entries, got {num_program_cache_entries}."


@skip_for_blackhole("Failing on Blackhole, Issue #27193")
@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, dim",
    [
        (1, 256, 32, 4, 64),  # GPT-OSS 20B TP=2
        (64, 1024, 32, 8, 64),
        (16, 1024, 32, 1, 64),
        (32, 256, 32, 1, 128),
        (32, 128, 8, 1, 128),
        (32, 128, 8, 8, 128),
        # (1, 256, 64, 8, 64),  # GPT-OSS 20B TP=1 (FIXME: Fails)
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
        (ttnn.bfloat16, ttnn.bfloat4_b),
        (ttnn.bfloat8_b, ttnn.bfloat8_b),
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
    if nkv > 1 and q_dtype != ttnn.bfloat16:
        pytest.skip("Only bfloat16 is supported for multi-head queries")

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
