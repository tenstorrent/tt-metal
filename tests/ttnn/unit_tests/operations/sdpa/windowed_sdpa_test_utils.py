# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the windowed SDPA sanity and nightly tests."""

import torch

import ttnn


def windowed_mask(seq_len, cu_window_seqlens, is_causal=False):
    """Block-diagonal mask: token i attends only to tokens in the same window (and, if causal, only to
    tokens at or before i)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=torch.float32)
    for i in range(1, len(cu_window_seqlens)):
        start, end = cu_window_seqlens[i - 1], cu_window_seqlens[i]
        mask[start:end, start:end] = 0.0
    if is_causal:
        mask = mask + torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
    return mask


def to_cu_tensor(cu_window_seqlens, device, **kwargs):
    return ttnn.from_torch(
        torch.tensor(cu_window_seqlens, dtype=torch.int32),
        device=device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.uint32,
        **kwargs,
    )


def sdpa_configs(device, q_chunk, k_chunk, fp32_dest_acc_en=True):
    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=device.compute_with_storage_grid_size(),
        exp_approx_mode=False,
        q_chunk_size=q_chunk,
        k_chunk_size=k_chunk,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=fp32_dest_acc_en,
        packer_l1_acc=True,
    )
    return program_config, compute_kernel_config


def reference_sdpa(q, k, v, mask, scale):
    """torch SDPA in fp32, with GQA K/V heads repeated up to the Q head count."""
    rep = q.shape[1] // k.shape[1]
    k, v = (t.to(torch.float32).repeat_interleave(rep, dim=1) for t in (k, v))
    return torch.nn.functional.scaled_dot_product_attention(
        q.to(torch.float32), k, v, attn_mask=mask.unsqueeze(0).unsqueeze(0), scale=scale
    )
