# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.quasar.transformer.chunked_scaled_dot_product_attention``  (chunked/paged prefill SDPA).

Model call site (modules/attention/attention_1d.py:553-561, prefill_forward, chunked path):
    attn_output = ttnn.experimental.quasar.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=q_heads_sdpa,      # [B, n_heads, chunk_len, head_dim]
        input_tensor_k=keys,              # paged cache [num_blocks, n_kv_heads, block, head_dim]
        input_tensor_v=values,            # paged cache [num_blocks, n_kv_heads, block, head_dim]
        page_table_tensor=page_table,     # int32 [B, blocks_per_seq]
        chunk_start_idx=chunk_start_idx,  # int: absolute offset of this Q chunk
        compute_kernel_config=cfg.sdpa_prefill_compute_kernel_cfg,
        program_config=cfg.prefill_sdpa_prg_config(seq_len, chunk_start_idx),
    )

This test runs a single chunk covering the whole sequence (chunk_start_idx=0), so
the result equals a plain causal prefill SDPA over the (unshuffled) KV cache.
GQA: 32 query heads, 8 kv heads (head_dim 64).
Known-good paging + direct-call pattern mirrored from
tests/ttnn/nightly/unit_tests/operations/sdpa/test_sdpa_chunked.py:28-236.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.prototype_ops import op_utils as U

BATCH = 1  # chunked prefill is single-user (batched prefill uses the non-chunked path)
PAGE_BLOCK_SIZE = 128


def _page_cache(cache, b, nkv, blocks_per_seq, block_size, d, permutation):
    paged = (
        cache.reshape(b, nkv, blocks_per_seq, block_size, d)
        .transpose(1, 2)
        .reshape(b * blocks_per_seq, nkv, block_size, d)
    )
    return paged[permutation]


def _torch_sdpa_causal(q, k, v, scale):
    n_rep = q.shape[1] // k.shape[1]
    k_rep = k.repeat_interleave(n_rep, dim=1).float()
    v_rep = v.repeat_interleave(n_rep, dim=1).float()
    return torch.nn.functional.scaled_dot_product_attention(q.float(), k_rep, v_rep, is_causal=True, scale=scale)


@U.with_default_mesh()
@pytest.mark.parametrize("seq", U.PREFILL_SEQ_LENS, ids=[f"seq{s}" for s in U.PREFILL_SEQ_LENS])
def test_chunked_scaled_dot_product_attention(ttnn_mesh_device, reset_seeds, seq):
    mesh = ttnn_mesh_device
    scale = U.HEAD_DIM**-0.5

    blocks_per_seq = seq // PAGE_BLOCK_SIZE
    num_blocks = BATCH * blocks_per_seq

    q_torch = U.torch_rand((BATCH, U.N_HEADS, seq, U.HEAD_DIM))
    k_torch = U.torch_rand((BATCH, U.N_KV_HEADS, seq, U.HEAD_DIM))
    v_torch = U.torch_rand((BATCH, U.N_KV_HEADS, seq, U.HEAD_DIM))

    permutation = torch.randperm(num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(BATCH, blocks_per_seq)
    paged_k = _page_cache(k_torch, BATCH, U.N_KV_HEADS, blocks_per_seq, PAGE_BLOCK_SIZE, U.HEAD_DIM, permutation)
    paged_v = _page_cache(v_torch, BATCH, U.N_KV_HEADS, blocks_per_seq, PAGE_BLOCK_SIZE, U.HEAD_DIM, permutation)

    q = U.to_tt(q_torch, mesh)
    k = U.to_tt(paged_k, mesh)
    v = U.to_tt(paged_v, mesh)
    page_table_tt = U.to_tt(page_table.to(torch.int32), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
        q_chunk_size=128,
        k_chunk_size=128,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    # One chunk covering the whole sequence: chunk_start_idx=0.
    out = ttnn.experimental.quasar.transformer.chunked_scaled_dot_product_attention(
        input_tensor_q=q,
        input_tensor_k=k,
        input_tensor_v=v,
        page_table_tensor=page_table_tt,
        chunk_start_idx=0,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
    )

    ref = _torch_sdpa_causal(q_torch, k_torch, v_torch, scale)  # [B, n_heads, seq, head_dim]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
