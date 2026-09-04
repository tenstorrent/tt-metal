# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: a 2-chunk sequence pushed through the SAME ``Attention`` module, chunk 2 vs the golden.

Proves the cache-read path is actually WIRED into the production attention module, not merely
callable in isolation (that is ``test_ring_joint_cache_read_sp_vs_ref``). Chunk 0 is pushed with
``cached_len=0``, chunk 1 with ``cached_len=chunk_global``; chunk 1's output must match a one-shot
reference over the whole sequence, restricted to chunk 1's positions.

This is the D2/D3 statement of the invariant P2 later has to hold end-to-end.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig, allocate_kv_cache

from ..test_factory import (
    concat_sp,
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    shard_seq_on_sp,
)
from .test_kv_cache_write_vs_ref import gather_kv_cache

PCC = 0.99


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("chunk_global", [2048], ids=["c2048"])
def test_attention_chunked_vs_ref(mesh_device, device_params, chunk_global, reset_seeds):
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    n_kv_local = cfg.num_key_value_heads // mesh_config.tp
    n_chunks = 2
    seq_len = n_chunks * chunk_global

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_attn = ref.LlamaAttention(cfg, layer_idx=0)
    cos_hf, sin_hf = ref.build_cos_sin_hf(seq_len, cfg)
    ref_out, ref_k, ref_v = ref_attn(x, (cos_hf, sin_hf))
    ref_k = ref.hf_to_meta_head_perm(ref_k, hd)

    attn = Attention(
        mesh_device=mesh_device,
        config=AttentionConfig(
            hidden_size=cfg.hidden_size,
            num_heads=cfg.num_attention_heads,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=hd,
            max_seq_len=seq_len,
            sequence_parallel=True,
        ),
        state_dict=hf_to_meta_qk(
            {
                "q_proj.weight": ref_attn.q_proj.weight.data,
                "k_proj.weight": ref_attn.k_proj.weight.data,
                "v_proj.weight": ref_attn.v_proj.weight.data,
                "o_proj.weight": ref_attn.o_proj.weight.data,
            },
            hd,
        ),
        ccl_manager=make_ccl(mesh_device),
        mesh_config=mesh_config,
        program_config=ProgramConfig(),
        layer_idx=0,
        transformation_mats={"prefill": tt_rope.build_transformation_mat(mesh_device)},
        weight_dtype=ttnn.bfloat16,
    )

    # Whole-cache rope, sized to the FULL sequence so both chunks index into the same cos/sin.
    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=seq_len, chunk_size=chunk_global, sp_axis=mesh_config.sp_axis
    )
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=seq_len,
        sp_axis=mesh_config.sp_axis,
        num_users=1,
        head_dim=hd,
        num_kv_heads_local=n_kv_local,
    )

    outs = []
    for c in range(n_chunks):
        cached_len = c * chunk_global
        x_c = x[:, cached_len : cached_len + chunk_global, :].reshape(1, 1, chunk_global, cfg.hidden_size)
        out_tt = attn(
            shard_seq_on_sp(mesh_device, x_c, mesh_config),
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            user_id=0,
            cached_len=cached_len,
            indexed_rope=True,
        )
        outs.append(concat_sp(mesh_device, out_tt, mesh_config).reshape(1, chunk_global, cfg.hidden_size))
    ttnn.synchronize_device(mesh_device)

    # The cache must hold the whole sequence's K/V after both chunks.
    host_k = gather_kv_cache(mesh_device, kv_cache.k, n_kv_local, chunk_local=chunk_global // mesh_config.sp)
    host_v = gather_kv_cache(mesh_device, kv_cache.v, n_kv_local, chunk_local=chunk_global // mesh_config.sp)
    ok_k, pcc_k = comp_pcc(ref_k, host_k, PCC)
    ok_v, pcc_v = comp_pcc(ref_v, host_v, PCC)
    logger.info(f"chunked KV after {n_chunks} chunks: K={pcc_k} V={pcc_v}")
    assert ok_k, f"K cache mismatch after chunking: {pcc_k}"
    assert ok_v, f"V cache mismatch after chunking: {pcc_v}"

    # Chunk 1 read the prefix chunk 0 left in the cache: its output must match the one-shot golden.
    ok, pcc = comp_pcc(ref_out[:, chunk_global:, :], outs[1], PCC)
    logger.info(f"chunked attention chunk1 out: pcc={pcc}")
    assert ok, f"chunk-1 attention output mismatch: {pcc}"
