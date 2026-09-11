# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Per-op test: ``ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode``  (non-paged decode SDPA).

Model call site (modules/attention/attention_1d.py:852-862, _sdpa_decode_non_paged):
    return ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        q_heads,               # [1, batch, n_heads, head_dim]
        keys,                  # KV cache [batch, n_kv_heads, max_seq, head_dim]
        values,                # KV cache [batch, n_kv_heads, max_seq, head_dim]
        cur_pos_tensor=current_pos,     # int32 [batch] positions
        scale=cfg.scale,                # head_dim ** -0.5
        sliding_window_size=cfg.sliding_window,   # None for Llama-3.2-1B
        program_config=cfg.decode_sdpa_prg_config,
        compute_kernel_config=cfg.sdpa_decode_compute_kernel_cfg,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

Decode does one token per user: Q is [1, batch, n_heads, head_dim], the KV cache
holds all past tokens, and cur_pos selects each user's current write index.
GQA: 32 query heads, 8 kv heads (head_dim 64).
Known-good direct-call pattern mirrored from
tests/ttnn/unit_tests/operations/sdpa/sdpa_test_utils.py:342-541.
"""

import pytest
import torch

import ttnn
from models.experimental.llama32_1b_quasar.tests.ops import op_utils as U

# (KV-cache capacity, current decode position, k_chunk) configs to sweep. The number of K-chunks a
# core processes is nearest_n(cur_pos + 1, k_chunk) / k_chunk. A 2-chunk-only test is a weak check: it
# hides online-softmax merge bugs because the final chunk contributes a single valid column, so its
# (possibly wrong) softmax correction is numerically negligible. The 8-chunk case exercises the real
# multi-chunk flash accumulation (batch32 = one core does all K-chunks) and the deeper multi-round tree
# reduction (batch1 = one K-chunk per core, combined across more cores) where merge bugs actually bite.
DECODE_SEQ_CONFIGS = [
    (256, 128, 128),  # 2 K-chunks: nearest_n(129, 128) == 256
    (1024, 896, 128),  # 8 K-chunks: nearest_n(897, 128) == 1024
]


def _torch_decode_ref(q, k, v, cur_pos, scale, padded_layer_len):
    # q: [1, b, nh, d]; k/v: [b, nkv, s, d]
    b, nh = q.shape[1], q.shape[2]
    nkv = k.shape[1]
    q_slice = q[:, :, :, :].permute(1, 2, 0, 3).float()  # [b, nh, 1, d]
    k_slice = k[:, :, :padded_layer_len, :].repeat_interleave(nh // nkv, dim=1).float()  # [b, nh, S, d]
    v_slice = v[:, :, :padded_layer_len, :].repeat_interleave(nh // nkv, dim=1).float()
    mask = torch.zeros((b, nh, 1, padded_layer_len))
    for i in range(b):
        mask[i, :, :, cur_pos[i] + 1 :] = torch.finfo(torch.float32).min
    expect = torch.nn.functional.scaled_dot_product_attention(
        q_slice, k_slice, v_slice, mask, scale=scale, is_causal=False
    )
    return expect.squeeze(2).unsqueeze(0)  # [1, b, nh, d]


@U.with_default_mesh()
@pytest.mark.parametrize(
    "cache_seq, cur_pos_val, k_chunk",
    DECODE_SEQ_CONFIGS,
    ids=[f"seq{s}_pos{p}_kc{kc}" for (s, p, kc) in DECODE_SEQ_CONFIGS],
)
@pytest.mark.parametrize("batch", U.DECODE_BATCHES, ids=[f"batch{b}" for b in U.DECODE_BATCHES])
def test_scaled_dot_product_attention_decode(ttnn_mesh_device, reset_seeds, batch, cache_seq, cur_pos_val, k_chunk):
    mesh = ttnn_mesh_device
    scale = U.HEAD_DIM**-0.5

    q_torch = U.torch_rand((1, batch, U.N_HEADS, U.HEAD_DIM))
    k_torch = U.torch_rand((batch, U.N_KV_HEADS, cache_seq, U.HEAD_DIM))
    v_torch = U.torch_rand((batch, U.N_KV_HEADS, cache_seq, U.HEAD_DIM))
    cur_pos = [cur_pos_val] * batch

    q = U.to_tt(q_torch, mesh)
    k = U.to_tt(k_torch, mesh)
    v = U.to_tt(v_torch, mesh)
    # cur_pos: int32 positions, one per user, ROW_MAJOR on device.
    cur_pos_tt = U.to_tt(torch.tensor(cur_pos, dtype=torch.int32), mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)

    program_config = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=mesh.compute_with_storage_grid_size(),
        q_chunk_size=32,  # padded_num_heads for n_heads=32
        k_chunk_size=k_chunk,
        exp_approx_mode=False,
    )
    compute_kernel_config = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
        packer_l1_acc=False,
    )

    out = ttnn.experimental.quasar.transformer.scaled_dot_product_attention_decode(
        q,
        k,
        v,
        cur_pos_tensor=cur_pos_tt,
        scale=scale,
        program_config=program_config,
        compute_kernel_config=compute_kernel_config,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )

    padded_layer_len = ((cur_pos_val + 1 + k_chunk - 1) // k_chunk) * k_chunk  # nearest_n(cur_pos+1, k_chunk)
    ref = _torch_decode_ref(q_torch, k_torch, v_torch, cur_pos, scale, padded_layer_len)  # [1, b, nh, d]
    U.assert_pcc(ref, out, pcc=0.99, mesh_device=mesh)
