# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import os
import math
import torch
import time
import numpy as np
from tests.tt_eager.python_api_testing.sweep_tests.comparison_funcs import (
    comp_allclose,
    comp_pcc,
)
from tests.tt_eager.python_api_testing.unit_testing.misc.test_flash_multi_latent_attention_decode import (
    page_table_setup,
    to_paged_cache,
    from_paged_cache,
)
import ttnn
from loguru import logger
import pytest
from tracy import signpost

from models.tt_transformers.tt.common import (
    PagedAttentionConfig,
)


def nearest_n(x, n):
    return ((x + n - 1) // n) * n


def print_tensor_chunked(tensor, name="Tensor", chunk_size=32):
    """
    Print tensor in chunks of chunk_size along the last dimension.
    For a tensor with shape [B, H, S, D], prints chunks of D dimension.
    """
    if isinstance(tensor, ttnn.Tensor):
        tensor_np = ttnn.to_torch(tensor).float().numpy()
    else:
        tensor_np = tensor.float().numpy() if isinstance(tensor, torch.Tensor) else tensor

    shape = tensor_np.shape
    if len(shape) < 1:
        print(f"{name} (full): {tensor_np}")
        return

    # Last dimension index
    dim_idx = len(shape) - 1
    dim_size = shape[dim_idx]

    num_chunks = (dim_size + chunk_size - 1) // chunk_size

    print(f"{name} shape: {shape}")
    for chunk_idx in range(num_chunks):
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, dim_size)

        # Create slice for all dimensions
        slices = [slice(None)] * len(shape)
        slices[dim_idx] = slice(start, end)

        chunk = tensor_np[tuple(slices)]
        print(f"{name} [{start}:{end}] (dim {dim_idx}): {chunk}")


def print_tensor_tiled(tensor, name="Tensor", tile_size=32):
    """
    Print tensor as tiles of [tile_size, tile_size] along the last two dimensions.
    For a tensor with shape [B, H, S, D], prints tiles of [S, D] dimension.
    """
    if isinstance(tensor, ttnn.Tensor):
        tensor_np = ttnn.to_torch(tensor).float().numpy()
    else:
        tensor_np = tensor.float().numpy() if isinstance(tensor, torch.Tensor) else tensor

    shape = tensor_np.shape
    if len(shape) < 2:
        print(f"{name} (full): {tensor_np}")
        return

    # Last two dimension indices
    dim_2nd_last = len(shape) - 2
    dim_last = len(shape) - 1
    dim_2nd_last_size = shape[dim_2nd_last]
    dim_last_size = shape[dim_last]

    num_tiles_2nd_last = (dim_2nd_last_size + tile_size - 1) // tile_size
    num_tiles_last = (dim_last_size + tile_size - 1) // tile_size

    print(f"{name} shape: {shape}")
    tile_idx = 0
    for tile_2nd_last_idx in range(num_tiles_2nd_last):
        start_2nd_last = tile_2nd_last_idx * tile_size
        end_2nd_last = min(start_2nd_last + tile_size, dim_2nd_last_size)

        for tile_last_idx in range(num_tiles_last):
            start_last = tile_last_idx * tile_size
            end_last = min(start_last + tile_size, dim_last_size)

            # Create slice for all dimensions
            slices = [slice(None)] * len(shape)
            slices[dim_2nd_last] = slice(start_2nd_last, end_2nd_last)
            slices[dim_last] = slice(start_last, end_last)

            tile = tensor_np[tuple(slices)]
            print(f"{name} tile {tile_idx} = [{start_2nd_last}:{end_2nd_last}, {start_last}:{end_last}]: {tile}")
            tile_idx += 1


def nearest_pow_2(x):
    if x < 1:
        raise ValueError("x must be >= 1")
    import math

    power = math.ceil(math.log2(x))
    return 1 << power


def sdpa_online_cpu(
    Q,
    K,
    V,
    *,
    scale=None,
    is_causal=True,
    key_padding_mask=None,
    block_k=256,
):
    """
    Online softmax scaled dot-product attention (CPU reference).

    Shapes:
      Q: [B, H, Sq, D]
      K: [B, H, Sk, D]
      V: [B, H, Sk, Dv]

    key_padding_mask:
      - None
      - or [B, Sk] with True = keep, False = pad
    """

    B, H, Sq, D = Q.shape
    _, _, Sk, Dv = V.shape

    if scale is None:
        scale = 1.0 / (D**0.5)

    device = Q.device
    dtype = Q.dtype

    # Running statistics
    m = torch.full((B, H, Sq), -float("inf"), device=device, dtype=dtype)
    l = torch.zeros((B, H, Sq), device=device, dtype=dtype)
    out = torch.zeros((B, H, Sq, Dv), device=device, dtype=dtype)

    q_idx = torch.arange(Sq, device=device)

    for ks in range(0, Sk, block_k):
        ke = min(ks + block_k, Sk)

        K_blk = K[:, :, ks:ke, :]  # [B,H,Bk,D]
        V_blk = V[:, :, ks:ke, :]  # [B,H,Bk,Dv]

        # [B,H,Sq,Bk]
        scores = torch.einsum("bhqd,bhkd->bhqk", Q, K_blk) * scale

        # ---- causal mask (generated inside) ----
        if is_causal:
            k_idx = torch.arange(ks, ke, device=device)
            causal_mask = k_idx[None, None, None, :] > q_idx[None, None, :, None]
            scores = scores.masked_fill(causal_mask, -float("inf"))

        # ---- key padding mask (generated inside) ----
        if key_padding_mask is not None:
            pad_blk = key_padding_mask[:, ks:ke]  # [B,Bk]
            pad_blk = pad_blk[:, None, None, :]  # [B,1,1,Bk]
            scores = scores.masked_fill(~pad_blk, -float("inf"))

        # ---- online softmax update ----
        block_max = scores.max(dim=-1).values
        m_new = torch.maximum(m, block_max)

        exp_scores = torch.exp(scores - m_new.unsqueeze(-1))

        l_new = l * torch.exp(m - m_new) + exp_scores.sum(dim=-1)

        out = out * (l * torch.exp(m - m_new) / l_new).unsqueeze(-1) + torch.einsum(
            "bhqk,bhkd->bhqd", exp_scores, V_blk
        ) / l_new.unsqueeze(-1)

        m = m_new
        l = l_new

    return out


def scaled_dot_product_attention_reference(Q, K, V, scale, is_causal=True, use_online_softmax=True):
    """
    Full-sequence causal SDPA reference.
    Q: (B, nh, S, d_qk), K/V: (B, nkv, S, d)
    """

    b, nh, S, d_qk = Q.shape
    _, nkv, _, d_v = V.shape
    # Expand KV to match Q heads
    head_rep = nh // nkv
    K_exp = K.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_qk)
    V_exp = V.repeat_interleave(head_rep, dim=1)  # (B, nh, S, d_v)
    # print("K_exp shape: ", K_exp.shape)
    # print("V_exp shape: ", V_exp.shape)
    # Use PyTorch’s builtin causal attention
    if use_online_softmax:
        return sdpa_online_cpu(Q, K_exp, V_exp, scale=scale, is_causal=is_causal)
    else:
        return torch.nn.functional.scaled_dot_product_attention(
            Q, K_exp, V_exp, attn_mask=None, scale=scale, is_causal=is_causal
        )


def run_flash_mla_prefill_impl(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    v_head_dim,
):
    device.disable_and_clear_program_cache()
    # Log the test parameters
    logger.debug(f"Running FlashMLA Prefill with parameters: ")
    logger.debug(f"Batch: {batch}")
    logger.debug(f"Sequence Length: {seq_len}")
    logger.debug(f"Number of Heads (Q): {nh}")
    logger.debug(f"Number of Heads (KV): {nkv}")
    logger.debug(f"KV LoRA Rank: {kv_lora_rank}")
    logger.debug(f"Dimensionality of RoPE: {d_rope}")
    logger.debug(f"V Head Dim: {v_head_dim}")
    logger.debug(f"Query Data Type: {q_dtype}")
    logger.debug(f"Key-Value Data Type: {dtype}")

    ######################
    ### Torch Setup
    ######################
    q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S (1 for decode), D)
    k = torch.randn(batch, 1, seq_len, kv_lora_rank + d_rope).float()  # (B, H, S, D)
    v = k[..., :kv_lora_rank]  # (B, H, S, D)
    v_out = torch.randn(batch, nh, kv_lora_rank, v_head_dim).float()
    logger.debug(f"v_out shape: {v_out.shape}")
    ######################
    ### TT Setup
    #######################

    tt_k_torch = k
    q_chunk_size = 32
    k_chunk_size = 32

    scale = (kv_lora_rank + d_rope) ** -0.5
    sdpa_program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        q_chunk_size=q_chunk_size,
        k_chunk_size=k_chunk_size,
        exp_approx_mode=False,
    )

    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi2,
        math_approx_mode=False,
        fp32_dest_acc_en=False,
        packer_l1_acc=True,
    )

    tt_q = ttnn.from_torch(
        q,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v = ttnn.from_torch(
        v,  # (B, H, S, D)
        device=device,
        dtype=q_dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_k = ttnn.from_torch(
        tt_k_torch,  # (B, H, S, D)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    tt_v_out = ttnn.from_torch(
        v_out,  # (B, H, D, D_out)
        device=device,
        dtype=dtype,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    run_old_path = True
    # np.set_printoptions(
    #     precision=16, threshold=1000000, linewidth=200, edgeitems=20, suppress=True, floatmode="maxprec"
    # )

    ##########################
    ### FlashMLA Prefill
    ##########################
    if run_old_path:
        print("tt_q shape is: ", tt_q.shape)
        print("tt_k shape is: ", tt_k.shape)
        tt_flash_mla_prefill_out = ttnn.transformer.flash_mla_prefill(
            tt_q,
            tt_k,
            head_dim_v=kv_lora_rank,
            scale=scale,
            program_config=sdpa_program_config,
            compute_kernel_config=compute_kernel_config,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            attn_mask=None,
            is_causal=True,
        )
        signpost(header="Original v_out matmul")
        tt_out = ttnn.linear(tt_flash_mla_prefill_out, tt_v_out)

        print("q shape is: ", q.shape)
        print("k shape is: ", k.shape)
        print("v shape is: ", v.shape)
        ref_out_t = scaled_dot_product_attention_reference(
            q,
            k,
            v,
            scale,
            is_causal=True,
            use_online_softmax=False,
        )
        ref_out_t = ref_out_t @ v_out
        out_pass, out_pcc = comp_pcc(ref_out_t, ttnn.to_torch(tt_out), 0.99)
        print(f"Output PCC: {out_pcc}")
        if not out_pass:
            pytest.skip(f"Ref impl PCC {out_pcc} < 0.99")

    # Second path
    print("Running second SDPA path...")
    print("tt_v shape is: ", tt_v.shape)
    tt_v_post_repeat = ttnn.repeat(tt_v, [1, nh, 1, 1])
    signpost(header="New v_out matmul")
    print("tt_v_post_repeat shape is: ", tt_v_post_repeat.shape)
    print("tt_v_out shape is: ", tt_v_out.shape)
    tt_v_pre_sdpa = ttnn.linear(tt_v_post_repeat, tt_v_out)
    # Repeat K as a current limitation of the SDPA op
    tt_k_post_repeat = ttnn.repeat(tt_k, [1, nh, 1, 1])
    print("Calling SDPA with shapes: ", tt_q.shape, tt_k_post_repeat.shape, tt_v_pre_sdpa.shape)

    # print_tensor_chunked(tt_k_post_repeat, "TT_K")
    # print_tensor_tiled(tt_q, "TT_Q")
    # print_tensor_tiled(tt_k, "TT_K")
    # print_tensor_tiled(tt_v_pre_sdpa, "TT_V")
    print("TT_v_pre_sdpa tensor dtype is: ", tt_v_pre_sdpa.dtype)
    print("TT_Q_tensor dtype is: ", tt_q.dtype)
    print("tt_k_post_repeat tensor dtype is: ", tt_v_out.dtype)
    tt_new_sdpa_out = ttnn.transformer.scaled_dot_product_attention(
        tt_q,
        tt_k_post_repeat,
        tt_v_pre_sdpa,
        scale=scale,
        program_config=sdpa_program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        use_mla=True,
        head_dim_v=v_head_dim,
        is_causal=True,
        attn_mask=None,
    )
    print("tt_new_sdpa_out shape is: ", tt_new_sdpa_out.shape)
    # print_tensor_chunked(tt_new_sdpa_out, "TT_NEW_SDPA_OUT")
    if run_old_path:
        ref_tt_impl_out_torch = ttnn.to_torch(tt_out)
        new_tt_impl_out_torch = ttnn.to_torch(tt_new_sdpa_out)

        pcc_threshold = 0.99
        # if dtype == ttnn.bfloat4_b:
        #     pcc_threshold = 0.98

        out_pass, out_pcc = comp_pcc(ref_tt_impl_out_torch, new_tt_impl_out_torch, pcc_threshold)
        print(f"Output PCC: {out_pcc}")
    ttnn.synchronize_device(device)
    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


@pytest.mark.parametrize(
    "batch, seq_len, nh, nkv, kv_lora_rank, d_rope, v_head_dim",
    [
        # working cases
        # (1, 32, 1, 1, 512, 64, 128) ,
        (1, 4096, 16, 16, 512, 64, 128),
        # (1, 32, 1, 1, 256, 32, 32),
        #  (1, 32, 1, 1, 128, 64, 32) ,
        #  (1, 32, 1, 1, 128, 64, 32) ,
        #  (1, 32, 1, 1, 32, 64, 32) ,
        #  (1, 32, 1, 1, 96, 64, 32) ,
        #  (1, 32, 1, 1, 96, 96, 64) ,
    ],
)
@pytest.mark.parametrize(
    "q_dtype, dtype",
    [
        (ttnn.bfloat16, ttnn.bfloat8_b),
    ],
)
def test_flash_mla_prefill(
    device,
    batch,
    seq_len,
    nh,
    nkv,
    kv_lora_rank,
    d_rope,
    q_dtype,
    dtype,
    v_head_dim,
    function_level_defaults,
    reset_seeds,
):
    run_flash_mla_prefill_impl(
        device,
        batch,
        seq_len,
        nh,
        nkv,
        kv_lora_rank,
        d_rope,
        q_dtype,
        dtype,
        v_head_dim,
    )


@pytest.mark.parametrize(
    "num_heads, seq_len, kv_lora_rank, v_head_dim",
    [
        (16, 4096, 512, 128),
        (32, 4096, 512, 128),
    ],
)
def test_batched_mla_mm(
    device,
    num_heads,
    seq_len,
    kv_lora_rank,
    v_head_dim,
):
    in0_shape = [1, 1, seq_len, kv_lora_rank]
    in1_shape = [1, num_heads, kv_lora_rank, v_head_dim]

    in0 = torch.randn(in0_shape).float()
    in1 = torch.randn(in1_shape).float()

    in0_t = ttnn.from_torch(
        in0,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    in1_t = ttnn.from_torch(
        in1,
        device=device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
    )
    in0_t = ttnn.repeat(in0_t, [1, num_heads, 1, 1])  # get rid of this repeat
    out_t = ttnn.linear(in0_t, in1_t)
    out_t = ttnn.to_torch(out_t)
    out_ref = in0 @ in1
    out_pass, out_pcc = comp_pcc(out_ref, out_t, 0.99)
    print(f"Output PCC: {out_pcc}")
    assert out_pass, f"Output mismatch: PCC {out_pcc} < 0.99"


# @pytest.mark.parametrize(
#     "batch, seq_len, nh,    nkv, kv_lora_rank, d_rope, head_dim",
#     [
#         (1, 32, 1, 1, 32, 32, 32),
#     ],
# )
# def test_torch_ref_impl(
#     batch,
#     seq_len,
#     nh,
#     nkv,
#     kv_lora_rank,
#     d_rope,
#     head_dim,
#     function_level_defaults,
#     reset_seeds,
# ):
#     q = torch.randn(batch, nh, seq_len, kv_lora_rank + d_rope)
#     k = torch.randn(batch, nkv, seq_len, kv_lora_rank + d_rope)
#     v = k[..., :kv_lora_rank]
#     v_out = torch.randn(batch, nh, kv_lora_rank, head_dim)
#     print("")
#     print("q shape: ", q.shape)
#     print("k shape: ", k.shape)
#     print("v shape: ", v.shape)
#     print("v_out shape: ", v_out.shape)

#     scale = (kv_lora_rank + d_rope) ** -0.5
#     time_start = time.time()
#     out_t = scaled_dot_product_attention_reference(
#         q,
#         k,
#         v,
#         scale,
#         is_causal=True,
#     )
#     time_end = time.time()
#     sdpa1_time = time_end - time_start
#     print("out_t shape: ", out_t.shape)
#     out_v = out_t @ v_out
#     print("out_v shape: ", out_v.shape)

#     v_pre_sdpa = v @ v_out
#     print("v_pre_sdpa shape: ", v_pre_sdpa.shape)
#     time_start = time.time()
#     out_v_ref = scaled_dot_product_attention_reference(
#         q,
#         k,
#         v_pre_sdpa,
#         scale,
#         is_causal=True,
#     )
#     out_v_ref_2 = scaled_dot_product_attention_reference(
#         q,
#         k,
#         v_pre_sdpa,
#         scale,
#         is_causal=True,
#         use_online_softmax=False,
#     )
#     time_end = time.time()
#     sdpa2_time = time_end - time_start

#     print("out_v_ref shape: ", out_v_ref.shape)
#     print("SDPA1 time: ", sdpa1_time)
#     print("SDPA2 time: ", sdpa2_time)
#     pcc_threshold = 0.99999
#     out_pass, out_pcc = comp_pcc(out_v, out_v_ref, pcc_threshold)
#     out_pass_2, out_pcc_2 = comp_pcc(out_v_ref, out_v_ref_2, pcc_threshold)
#     print(f"Out PCC: {out_pcc}")
#     print(f"Out PCC 2: {out_pcc_2}")
