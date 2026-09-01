# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.transformer.paged_scaled_dot_product_attention_decode``  (paged decode SDPA).

Model call site (modules/attention/attention_1d.py:836-847, _sdpa_decode_paged):
    return ttnn.transformer.paged_scaled_dot_product_attention_decode(
        q_heads,               # [1, batch, n_heads, head_dim]
        keys,                  # paged KV cache [num_blocks, n_kv_heads, block_size, head_dim]
        values,                # paged KV cache [num_blocks, n_kv_heads, block_size, head_dim]
        page_table_tensor=page_table,   # int32 [batch, blocks_per_seq]
        cur_pos_tensor=current_pos,     # int32 [batch] positions
        scale=cfg.scale,                # head_dim ** -0.5
        sliding_window_size=cfg.sliding_window,   # None for Llama-3.2-1B
        program_config=cfg.decode_sdpa_prg_config,
        compute_kernel_config=cfg.sdpa_decode_compute_kernel_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

Same math as non-paged decode, but the KV cache is stored in shuffled physical
blocks and a page_table maps each user's virtual blocks to physical ones.
GQA: 32 query heads, 8 kv heads (head_dim 64).
Known-good paging + direct-call pattern mirrored from
tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py:774-995.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

CACHE_SEQ = 256
CUR_POS = 128
BLOCK_SIZE = 64
K_CHUNK = 128


def _to_paged(cache, b, nkv, blocks_per_seq, block_size, d):
    return (
        cache.reshape(b, nkv, blocks_per_seq, block_size, d)
        .transpose(1, 2)
        .reshape(b * blocks_per_seq, nkv, block_size, d)
    )


def _torch_decode_ref(q, k, v, cur_pos, scale, padded_layer_len):
    b, nh = q.shape[1], q.shape[2]
    nkv = k.shape[1]
    q_slice = q.permute(1, 2, 0, 3).float()  # [b, nh, 1, d]
    k_slice = k[:, :, :padded_layer_len, :].repeat_interleave(nh // nkv, dim=1).float()
    v_slice = v[:, :, :padded_layer_len, :].repeat_interleave(nh // nkv, dim=1).float()
    mask = torch.zeros((b, nh, 1, padded_layer_len))
    for i in range(b):
        mask[i, :, :, cur_pos[i] + 1 :] = torch.finfo(torch.float32).min
    expect = torch.nn.functional.scaled_dot_product_attention(
        q_slice, k_slice, v_slice, mask, scale=scale, is_causal=False
    )
    return expect.squeeze(2).unsqueeze(0)  # [1, b, nh, d]


@U.with_default_mesh()
@pytest.mark.parametrize("batch", U.DECODE_BATCHES, ids=[f"batch{b}" for b in U.DECODE_BATCHES])
def test_paged_scaled_dot_product_attention_decode(ttnn_mesh_device, reset_seeds, batch):
    mesh = ttnn_mesh_device
    scale = U.HEAD_DIM**-0.5

    blocks_per_seq = CACHE_SEQ // BLOCK_SIZE
    num_blocks = batch * blocks_per_seq

    q_torch = U.torch_rand((1, batch, U.N_HEADS, U.HEAD_DIM))
    k_torch = U.torch_rand((batch, U.N_KV_HEADS, CACHE_SEQ, U.HEAD_DIM))
    v_torch = U.torch_rand((batch, U.N_KV_HEADS, CACHE_SEQ, U.HEAD_DIM))
    cur_pos = [CUR_POS] * batch

    # Page + shuffle the contiguous caches; page_table maps virtual -> physical block.
    paged_k = _to_paged(k_torch, batch, U.N_KV_HEADS, blocks_per_seq, BLOCK_SIZE, U.HEAD_DIM)
    paged_v = _to_paged(v_torch, batch, U.N_KV_HEADS, blocks_per_seq, BLOCK_SIZE, U.HEAD_DIM)
    permutation = torch.randperm(num_blocks)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(batch, blocks_per_seq)
    paged_k_shuffled = paged_k[permutation]
    paged_v_shuffled = paged_v[permutation]

    q = U.to_tt(q_torch, mesh)
    k = U.to_tt(paged_k_shuffled, mesh)
    v = U.to_tt(paged_v_shuffled, mesh)
    page_table_tt = U.to_tt(page_table.to(torch.int32), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    cur_pos_tt = U.to_tt(torch.tensor(cur_pos, dtype=torch.int32), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
        q_chunk_size=32,  # padded_num_heads for n_heads=32
        k_chunk_size=K_CHUNK,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    out = ttnn.experimental.quasar.transformer.paged_scaled_dot_product_attention_decode(
        q,
        k,
        v,
        page_table_tt,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    padded_layer_len = ((CUR_POS + 1 + K_CHUNK - 1) // K_CHUNK) * K_CHUNK
    ref = _torch_decode_ref(q_torch, k_torch, v_torch, cur_pos, scale, padded_layer_len)  # [1, b, nh, d]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
