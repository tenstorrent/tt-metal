# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEVICE (>=8 chips): chunked sequence-parallel prefill through the full attention block.

Drives ``Attention`` twice over one sequence — chunk 0 then chunk 1 with ``cached_len = chunk_global``
— so the second chunk must attend the prefix the first one left in the cache. That is the path a
real 128K prefill takes (chunk by chunk into a growing block-cyclic KV cache) and it exercises three
things the single-chunk tests cannot:

  * the **indexed on-device RoPE**: the whole-cache cos/sin are built ONCE and
    ``rotary_embedding_indexed`` derives each chunk's rows from ``kv_actual_global`` plus the chip's
    SP coordinate, so a wrong chunk offset shows up as a phase error, not a crash;
  * the **cache-backed ring read** with a non-zero prefix;
  * the per-layer **KV write at a non-zero offset** (`kv_actual = chunk_global`).

The reference is a plain full-sequence causal GQA over both chunks, compared against chunk 1's output.

Run:  pytest models/demos/mistral_medium_d_p/tests/unit/test_attention_chunked_vs_ref.py -k 2x4
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_medium_d_p.reference.torch_reference import gqa_attention
from models.demos.mistral_medium_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig, allocate_kv_cache
from models.demos.mistral_medium_d_p.tt.rope import build_indexed_rope, build_transformation_mat
from models.demos.mistral_medium_d_p.tt.rope_tables import build_hf_cos_sin
from models.tt_transformers.tt.load_checkpoints import convert_hf_qkv_to_meta_format

from ..test_factory import mesh_setup, parametrize_mesh_with_fabric
from .shapes import EPS, HEAD_DIM, HIDDEN, N_KV, N_Q, YARN, per_chip


@parametrize_mesh_with_fabric(mesh_shapes=[(2, 4), (8, 4)])
@pytest.mark.parametrize("chunk_local", [128], ids=["c128"])
def test_chunked_sp_prefill_vs_ref(mesh_device, device_params, chunk_local, reset_seeds):
    """Two SP chunks through the block; chunk 1 must see chunk 0's KV."""
    rows, cols = tuple(mesh_device.shape)
    mesh_config, ccl = mesh_setup(mesh_device)
    sp, tp, sp_axis = mesh_config.sp, mesh_config.tp, mesh_config.sp_axis
    pc = per_chip(tp)

    chunk_global = sp * chunk_local
    total = 2 * chunk_global

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(3)
    w = {
        "q": torch.randn(N_Q * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "k": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "v": torch.randn(N_KV * HEAD_DIM, HIDDEN, generator=g) * 0.02,
        "o": torch.randn(HIDDEN, N_Q * HEAD_DIM, generator=g) * 0.02,
    }
    x_full = torch.randn(1, total, HIDDEN) * 0.1

    cos_hf, sin_hf = build_hf_cos_sin(total, HEAD_DIM, **YARN)
    ref_full = gqa_attention(x_full.float(), w, cos_hf, sin_hf, n_q=N_Q, n_kv=N_KV, head_dim=HEAD_DIM)
    ref_chunk1 = ref_full[:, chunk_global:, :]

    state = convert_hf_qkv_to_meta_format({f"{n}_proj.weight": w[n] for n in ("q", "k", "v", "o")}, HEAD_DIM)
    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=HIDDEN,
            num_heads=N_Q,
            num_kv_heads=N_KV,
            head_dim=HEAD_DIM,
            max_seq_len=total,
            rms_norm_eps=EPS,
            sequence_parallel=True,
        ),
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=total,
        sp_axis=sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
        n_kv_local=pc["n_kv"],
    )
    # Whole-cache block-cyclic SP cos/sin, built ONCE; the indexed op picks each chunk's rows.
    rope_mats = build_indexed_rope(
        mesh_device, head_dim=HEAD_DIM, max_seq_len=total, chunk_size=chunk_global, sp_axis=sp_axis, **YARN
    )

    outs = []
    for c in range(2):
        lo = c * chunk_global
        x_chunk = x_full[:, lo : lo + chunk_global, :]
        # SP-shard the chunk contiguously across the rows, as the runtime's make_chunk_input does.
        x_sp = x_chunk.reshape(sp, 1, chunk_local, HIDDEN)
        dims = [None, None]
        dims[sp_axis] = 0
        x_tt = ttnn.from_torch(
            x_sp,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=tuple(dims)),
        )
        out_tt = attn(x_tt, rope_mats=rope_mats, kv_cache=kv_cache, cached_len=lo, indexed_rope=True)
        assert out_tt.shape[-1] == HIDDEN // tp
        # Gather TP (hidden) then SP (sequence). Device index is row*cols + col; each row holds a
        # contiguous slice of this chunk's sequence, each col a slice of the hidden dim.
        dev = ttnn.get_device_tensors(out_tt)
        row_chunks = [torch.cat([ttnn.to_torch(dev[r * cols + c]) for c in range(cols)], dim=-1) for r in range(sp)]
        outs.append(torch.cat(row_chunks, dim=2).reshape(1, chunk_global, HIDDEN))

    got_chunk1 = outs[1]
    passing, pcc = comp_pcc(ref_chunk1, got_chunk1, 0.99)
    logger.info(f"chunked SP prefill (SP={sp} TP={tp}, chunk 1 of 2, prefix={chunk_global}): {pcc}")
    assert passing, f"chunked SP attention PCC fail: {pcc}"

    # Chunk 0 is causally independent of chunk 1, so it must equal a standalone prefill of chunk 0.
    ok0, pcc0 = comp_pcc(ref_full[:, :chunk_global, :], outs[0], 0.99)
    logger.info(f"chunked SP prefill (chunk 0, no prefix): {pcc0}")
    assert ok0, f"chunk 0 PCC fail: {pcc0}"
