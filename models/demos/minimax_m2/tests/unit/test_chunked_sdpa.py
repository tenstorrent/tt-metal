# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Foundational validation for the chunked/paged attention rewire (Tier 1).

Drives the GQA chunked-prefill op flow for MiniMax-M2's attention shape (48 q-heads /
8 kv-heads, head_dim 128) directly at the op level — independent of the model — so we
confirm the op signatures + paged-cache flow before rewiring attention/prefill.py:

  for each chunk c:
     paged_fill_cache(k_cache, K_chunk, chunk_page_table)        # write this chunk's K/V
     chunked_scaled_dot_product_attention(Q_chunk, k_cache, v_cache,
                                          page_table, chunk_start_idx=c*C)   # read [0..chunk_end)

Reference: full causal GQA SDPA in torch over the whole sequence. Running prefill in
chunks must equal the single-shot result (this is the paged-KV READ path that the
current prefill never exercises). Runs at mesh (1,1)/TP=1.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc

from ..test_factory import parametrize_mesh_with_fabric


def _torch_causal_gqa_sdpa(q, k, v, scale):
    """Full causal GQA reference: q [1,nq,S,d], k/v [1,nkv,S,d]."""
    nq, nkv = q.shape[1], k.shape[1]
    kr = k.repeat_interleave(nq // nkv, dim=1)
    vr = v.repeat_interleave(nq // nkv, dim=1)
    S = q.shape[2]
    attn = (q.float() @ kr.float().transpose(-1, -2)) * scale
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1)
    attn = (attn + mask).softmax(dim=-1)
    return attn @ vr.float()  # [1, nq, S, d]


@parametrize_mesh_with_fabric(mesh_shapes=[(1, 1)])
@pytest.mark.parametrize(
    "seq_len, chunk_size, block_size",
    [(256, 128, 128), (384, 128, 64)],
    ids=["s256_c128_b128", "s384_c128_b64"],
)
def test_chunked_paged_sdpa_gqa(mesh_device, device_params, seq_len, chunk_size, block_size, reset_seeds):
    nq, nkv, hd = 48, 8, 128
    scale = hd**-0.5
    num_blocks = seq_len // block_size

    q = torch.randn(1, nq, seq_len, hd)
    k = torch.randn(1, nkv, seq_len, hd)
    v = torch.randn(1, nkv, seq_len, hd)
    ref = _torch_causal_gqa_sdpa(q, k, v, scale)

    repl = ttnn.ReplicateTensorToMesh(mesh_device)

    # Paged KV cache [num_blocks, nkv, block_size, head_dim], bf8 (cache dtype).
    def cache():
        return ttnn.from_torch(
            torch.zeros(num_blocks, nkv, block_size, hd),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            mesh_mapper=repl,
        )

    k_cache, v_cache = cache(), cache()

    # Full page_table [1, num_blocks] padded to a multiple of 8 cols (chunked-SDPA stick
    # alignment); pad value 0 is read-safe (those positions are causally masked out).
    pt = torch.arange(num_blocks, dtype=torch.int32).reshape(1, -1)
    cols_padded = ((num_blocks + 7) // 8) * 8
    pt_padded = torch.zeros(1, cols_padded, dtype=torch.int32)
    pt_padded[:, :num_blocks] = pt
    page_table_tt = ttnn.from_torch(
        pt_padded, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT, device=mesh_device, mesh_mapper=repl
    )

    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=ttnn.CoreCoord(8, 8),
        q_chunk_size=chunk_size,
        k_chunk_size=chunk_size,
        exp_approx_mode=False,
    )
    kernel_cfg = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=False
    )

    outs = []
    for c in range(seq_len // chunk_size):
        cs = c * chunk_size
        blk0, blk1 = cs // block_size, (cs + chunk_size) // block_size

        def to_dev(t, dt=ttnn.bfloat8_b):
            return ttnn.from_torch(t, dtype=dt, layout=ttnn.TILE_LAYOUT, device=mesh_device, mesh_mapper=repl)

        q_tt = to_dev(q[:, :, cs : cs + chunk_size])
        k_tt = to_dev(k[:, :, cs : cs + chunk_size])
        v_tt = to_dev(v[:, :, cs : cs + chunk_size])

        # Write this chunk's K/V into its blocks.
        chunk_pt = ttnn.from_torch(
            pt[:, blk0:blk1].contiguous(),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=repl,
        )
        ttnn.experimental.paged_fill_cache(k_cache, k_tt, chunk_pt, batch_idx=0)
        ttnn.experimental.paged_fill_cache(v_cache, v_tt, chunk_pt, batch_idx=0)

        # Chunk's queries attend over cached K/V up to [0, cs+chunk_size).
        cs_tt = ttnn.from_torch(
            torch.tensor([cs], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh_device,
            mesh_mapper=repl,
        )
        out = ttnn.transformer.chunked_scaled_dot_product_attention(
            input_tensor_q=q_tt,
            input_tensor_k=k_cache,
            input_tensor_v=v_cache,
            page_table_tensor=page_table_tt,
            chunk_start_idx_tensor=cs_tt,  # flexible path (device [1] int32), like llama3_70b_galaxy
            program_config=prog,
            compute_kernel_config=kernel_cfg,
        )
        outs.append(ttnn.to_torch(ttnn.get_device_tensors(out)[0]).reshape(1, nq, chunk_size, hd))

    tt_out = torch.cat(outs, dim=2)  # [1, nq, seq, hd]
    passing, pcc = comp_pcc(ref, tt_out, 0.99)
    logger.info(f"chunked/paged GQA SDPA (seq={seq_len}, chunk={chunk_size}, block={block_size}): {pcc}")
    assert passing, f"chunked SDPA PCC fail: {pcc}"
