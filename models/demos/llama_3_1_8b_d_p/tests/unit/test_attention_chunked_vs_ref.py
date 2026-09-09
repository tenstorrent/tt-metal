# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""A 2-chunk sequence through the SAME `Attention` module, two ways — the second chunk must agree.

Structure follows `minimax_m3/tests/unit/test_attention_chunked_vs_ref.py`.

This is what proves the cache-read path is WIRED, not merely callable: chunk 1's queries attending
the prefix that chunk 0 left in the cache must produce what a single one-shot pass over the whole
sequence produces for those same rows. `test_ring_joint_cache_read_sp_vs_ref.py` checks the op in
isolation; this checks the module around it — the write seam, the layer-folded cache batch index,
the rope offset for the second chunk, and the `cached_len` / `logical_n` bookkeeping, all of which
are module-level and none of which the op-level test touches.
"""

import torch

import ttnn
from models.demos.llama_3_1_8b_d_p.reference.model import REF_DTYPE, RefAttention, RefRotaryEmbedding, causal_mask
from models.demos.llama_3_1_8b_d_p.tt.attention import Attention, allocate_kv_caches
from models.demos.llama_3_1_8b_d_p.tt.attention.config import LlamaAttentionProgramConfig
from models.demos.llama_3_1_8b_d_p.tt.layer import build_attention_config
from models.demos.llama_3_1_8b_d_p.tt.rope import create_rope_setup

from ..test_factory import ACT_DTYPE, KV_DTYPE, WEIGHT_DTYPE, assert_pcc, parametrize_target_mesh, sp_shard_rope
from .test_attention_vs_ref import meta_rope_tables

CHUNK = 1024
N_CHUNKS = 2
TOTAL = CHUNK * N_CHUNKS


@parametrize_target_mesh()
def test_attention_chunked_vs_ref(mesh_device, device_params, config, hf_config, mesh_config, ccl_manager, topology_name):
    """Chunk 1's output through the cache path vs the one-shot reference over the full sequence."""
    torch.manual_seed(0)
    x = torch.randn(1, TOTAL, config.hidden_size, dtype=REF_DTYPE)
    attn_ref = RefAttention(config)
    rope = RefRotaryEmbedding(config)
    cos, sin = rope(torch.arange(TOTAL)[None, :], dtype=REF_DTYPE)
    with torch.no_grad():
        golden = attn_ref(x, (cos, sin), causal_mask(TOTAL))
    state_dict = {
        f"{n}.weight": getattr(attn_ref, n).weight.detach().clone() for n in ("q_proj", "k_proj", "v_proj", "o_proj")
    }

    rope_setup = create_rope_setup(mesh_device, hf_config, max_seq_len=TOTAL)
    attention = Attention(
        mesh_device=mesh_device,
        config=build_attention_config(hf_config, max_seq_len=TOTAL, chunk_size=CHUNK),
        state_dict=state_dict,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=LlamaAttentionProgramConfig(),
        global_layer_idx=0,
        transformation_mats=rope_setup.transformation_mat_prefill,
        weight_dtype=WEIGHT_DTYPE,
    )
    kv = allocate_kv_caches(
        mesh_device,
        num_layers=1,
        max_seq_len=TOTAL,
        chunk_size=CHUNK,
        sp_axis=mesh_config.sp_axis,
        tp_axis=mesh_config.tp_axis,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        cache_dtype=KV_DTYPE,
    )

    def to_dev(t, dims):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            dtype=ACT_DTYPE,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=dims),
        )

    out = None
    for c in range(N_CHUNKS):
        lo, hi = c * CHUNK, (c + 1) * CHUNK
        tt_x = to_dev(x[:, lo:hi, :].reshape(1, 1, CHUNK, config.hidden_size), [2, None])
        # Meta-format tables for THIS chunk's absolute positions [lo, hi) — the offset is what
        # makes chunk 1 rotate at positions 1024..2047 rather than starting over at 0.
        cos_meta, sin_meta = meta_rope_tables(mesh_device, rope_setup, config.head_dim, lo, hi)
        # SP-sharded, not replicated: each SP row needs the cos/sin rows for ITS positions.
        rope_mats = (sp_shard_rope(mesh_device, mesh_config, cos_meta), sp_shard_rope(mesh_device, mesh_config, sin_meta))
        out = attention(tt_x, rope_mats=rope_mats, kv_cache=kv, slot_idx=0, cached_len=lo, logical_n=hi)
    ttnn.synchronize_device(mesh_device)

    got = ttnn.to_torch(
        out, mesh_composer=ttnn.ConcatMesh2dToTensor(mesh_device, dims=(2, 3), mesh_shape=tuple(mesh_device.shape))
    ).to(REF_DTYPE)
    golden_last = golden[:, (N_CHUNKS - 1) * CHUNK :, :].reshape(1, 1, CHUNK, config.hidden_size)
    assert_pcc("attention_chunked[chunk1]", golden_last, got, topology_name)
