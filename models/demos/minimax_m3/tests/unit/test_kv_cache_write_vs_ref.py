# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Per-layer KV cache WRITE through the prefill seam, SP=8 x TP=4, dense + sparse — vs torch ref.

Drives a single ``Attention`` (one dense layer and one MSA/sparse layer) through ``prefill_forward``
with an externally-allocated ``MiniMaxKVCache``, then reads the cache back and PCC-checks its contents
against a self-authored HF-convention torch reference:

  * K cache   — post-RoPE K (per-head gemma QK-norm -> partial RoPE), 4 KV heads, TP-sharded on cols.
  * V cache   — raw V (no norm, no RoPE), 4 KV heads, TP-sharded on cols.
  * index_k   — (sparse layer only) post-norm/post-RoPE MSA index key, the single shared head,
                REPLICATED across the TP cols (we read it back from col 0).

This validates step 2 (the ``update_padded_kv_cache`` write wired at the seam): right tensor (post-RoPE
K / raw V / index_k), right per-chip layout, right bf8 round-trip. The SP-sharded sequence ordering of
the op itself is separately covered by ``test_kv_cache_gqa_sp_vs_ref``; here ``kv_actual=0`` + a cache
sized to the prompt makes the block-cyclic layout the identity, so readback is in natural order.

NOTE: the cache is write-only for now (step 4 wires the read), so this asserts ONLY on the cache, not
on the attention output. The index-branch RoPE convention is shared with the main Q/K path (same
``convert_hf_qkv_to_meta_format_partial`` swizzle + partial RoPE), so the ref is self-consistent with
the TT path; end-to-end index correctness vs the real model is the golden/step-3 job.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.minimax_m3.config import MeshConfig
from models.demos.minimax_m3.tt.attention import Attention, AttentionConfig, allocate_kv_caches
from models.demos.minimax_m3.tt.attention_configs import MiniMaxM3AttentionProgramConfig
from models.demos.minimax_m3.tt.ccl import CCLManager
from models.demos.minimax_m3.tt.model import create_rope_setup
from models.demos.minimax_m3.utils.general_utils import get_default_num_links
from models.demos.minimax_m3.utils.weight_conversion import convert_hf_qkv_to_meta_format_partial

from ..test_factory import parametrize_mesh_with_fabric

# M3 attention dims (text_config).
HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
NIDX, INDEX_DIM = 4, 128  # MSA indexer: 4 index q-heads, 1 shared index-k head, dim 128


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _partial_rope(t, cos, sin):
    """Rotate the first ROTARY_DIM dims (HF rotate_half convention), pass the rest through."""
    t_rot, t_pass = t[..., :ROTARY_DIM], t[..., ROTARY_DIM:]
    t_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat([t_rot, t_pass], dim=-1)


def _gemma_head_norm(x, weight):
    """RMSNorm over the last (head) dim with gemma (1 + w)."""
    var = x.pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(var + EPS)) * (1.0 + weight.float())


def _ref_kv(x, w, cos, sin):
    """Reference post-RoPE K [1,NKV,S,HD] and raw V [1,NKV,S,HD] (what the K/V cache should hold)."""
    B, S, _ = x.shape
    k = (x @ w["k"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    v = (x @ w["v"].t()).view(B, S, NKV, HEAD_DIM).transpose(1, 2)
    k = _partial_rope(_gemma_head_norm(k, w["k_norm"]), cos, sin)
    return k, v


def _ref_index_k(x, w, cos, sin):
    """Reference post-norm/post-RoPE MSA index_k [1,1,S,INDEX_DIM] (the single shared index head)."""
    B, S, _ = x.shape
    ik = (x @ w["index_k"].t()).view(B, S, 1, INDEX_DIM).transpose(1, 2)
    return _partial_rope(_gemma_head_norm(ik, w["index_k_norm"]), cos, sin)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
# 640/row at SP=8. MSA needs the gathered context to satisfy: blocks (S/128=40) >= topk_blocks (16)
# AND S % indexer k_chunk_size (1024) == 0 -> S=5120 is the smallest validated multiple that clears
# both (2560 fails Tt % KC == 0). Dense is block-count-agnostic but runs at the same S here.
@pytest.mark.parametrize("seq_len", [5120], ids=["s5120"])
@pytest.mark.parametrize("layer_kind", ["dense", "sparse"])
def test_kv_cache_write_vs_ref(mesh_device, device_params, layer_kind, seq_len, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4), "TP=4 x SP=8 layout expected"
    sp, tp, sp_axis = rows, cols, 0
    s_local = seq_len // sp
    is_sparse = layer_kind == "sparse"

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    w = {
        "q": torch.rand(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.rand(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }
    if is_sparse:
        w.update(
            {
                "index_q": torch.rand(NIDX * INDEX_DIM, HIDDEN) * 0.02,
                "index_k": torch.rand(INDEX_DIM, HIDDEN) * 0.02,
                "index_q_norm": torch.randn(INDEX_DIM) * 0.1,
                "index_k_norm": torch.randn(INDEX_DIM) * 0.1,
            }
        )

    # HF-convention partial-RoPE cos/sin (theta 5e6, rotary_dim 64), shape [1,1,S,rotary_dim].
    inv_freq = 1.0 / (THETA ** (torch.arange(0, ROTARY_DIM, 2).float() / ROTARY_DIM))
    emb = torch.cat([torch.outer(torch.arange(seq_len).float(), inv_freq)] * 2, dim=-1)
    cos_ref, sin_ref = emb.cos()[None, None], emb.sin()[None, None]

    ref_k, ref_v = _ref_kv(x.float(), w, cos_ref, sin_ref)  # [1,NKV,S,HD]
    ref_index_k = _ref_index_k(x.float(), w, cos_ref, sin_ref) if is_sparse else None

    # The cache stores K / index_k in Meta-RoPE-swizzled head layout: the q/k/index projection weights
    # were permuted by convert_hf_qkv_to_meta_format_partial and apply_rope uses the Meta transformation
    # matrix, so the post-RoPE K in the cache equals permute(HF K) over the rotary slice. Swizzle the HF
    # ref the same way (Meta idx m <- HF idx half*(m%2) + m//2 over [0:rotary_dim], identity tail). V is
    # raw (v_proj unswizzled, no RoPE) so it is NOT permuted.
    half = ROTARY_DIM // 2
    src = list(range(HEAD_DIM))
    for m in range(ROTARY_DIM):
        src[m] = half * (m % 2) + (m // 2)
    src = torch.tensor(src, dtype=torch.long)
    ref_k = ref_k[..., src]
    if ref_index_k is not None:
        ref_index_k = ref_index_k[..., src]

    # --- TT Attention from the same weights, Meta-RoPE-swizzled (q/k + index_q/k proj + their norms) ---
    hf_state = {
        "q_proj.weight": w["q"],
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
        "q_norm.weight": w["q_norm"],
        "k_norm.weight": w["k_norm"],
    }
    if is_sparse:
        hf_state.update(
            {
                "index_q_proj.weight": w["index_q"],
                "index_k_proj.weight": w["index_k"],
                "index_q_norm.weight": w["index_q_norm"],
                "index_k_norm.weight": w["index_k_norm"],
            }
        )
    # convert matches by substring, so index_{q,k}_proj / index_{q,k}_norm are swizzled too (head count
    # inferred from shape: 4 for index_q, 1 for index_k) — same partial-RoPE convention as main Q/K.
    state = convert_hf_qkv_to_meta_format_partial(hf_state, HEAD_DIM, ROTARY_DIM)

    mesh_config = MeshConfig((rows, cols), tp=tp)  # prefill auto: SP=rows, TP=cols
    ccl_manager = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)

    from types import SimpleNamespace

    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rope_scaling=None,
        rms_norm_eps=EPS,
        max_position_embeddings=max(seq_len, 128),
        use_qk_norm=True,
        use_gemma_norm=True,
    )
    rope_setup = create_rope_setup(mesh_device=mesh_device, hf_config=hf_config, datatype=ttnn.bfloat16)

    attn_config = AttentionConfig(
        hidden_size=HIDDEN,
        num_heads=NQ,
        num_kv_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rms_norm_eps=EPS,
        use_qk_norm=True,
        use_gemma_norm=True,
        max_seq_len=max(seq_len, 128),
        max_local_batch_size=1,
        is_sparse=is_sparse,
        sequence_parallel=True,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=ccl_manager,
        mesh_config=mesh_config,
        program_config=MiniMaxM3AttentionProgramConfig(),
        layer_idx=0,
        transformation_mats=rope_setup.get_both_trans_mats(),
    )

    # --- SP input shard: one sequence, S/sp rows per device, hidden replicated across TP cols ---
    in_dims = [None, None]
    in_dims[sp_axis] = 2  # seq -> SP rows
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
    )

    # Per-row RoPE: re-shard the model's own prefill cos/sin so row r rotates positions [r*s_local:...].
    def reshard_rope(dev_tensor):
        full = ttnn.to_torch(ttnn.get_device_tensors(dev_tensor)[0])[:, :, :seq_len, :]
        return ttnn.from_torch(
            full,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    rope_sp = [reshard_rope(rope_setup.cos_matrix_prefill), reshard_rope(rope_setup.sin_matrix_prefill)]

    # Externally-owned packed cache (one layer, one user) sized to the prompt -> block-cyclic == identity.
    kv_cache = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=seq_len, sp_axis=sp_axis, num_users=1)

    attn(x_tt, rope_mats=rope_sp, position_idx=None, kv_cache=kv_cache, user_id=0)
    ttnn.synchronize_device(mesh_device)

    # --- readback. Per chip the cache is [1, 1, s_local, HD]; heads on cols, seq on rows, batch slot 0.
    def gather_heads(cache_tensor, n_heads):
        """K/V: head c lives on col c; concat the SP rows' seq shards, then the head cols. -> [1,n,S,HD]."""
        dts = ttnn.get_device_tensors(cache_tensor)
        heads = [
            torch.cat([ttnn.to_torch(dts[r * cols + c]).float() for r in range(rows)], dim=2) for c in range(n_heads)
        ]
        return torch.cat(heads, dim=1)

    host_k = gather_heads(kv_cache.k, NKV)  # [1, NKV, S, HD]
    host_v = gather_heads(kv_cache.v, NKV)

    ok_k, pcc_k = comp_pcc(ref_k, host_k, 0.99)
    ok_v, pcc_v = comp_pcc(ref_v, host_v, 0.99)
    logger.info(f"[{layer_kind}] KV write: K pcc={pcc_k} V pcc={pcc_v}")
    assert ok_k, f"K cache mismatch: {pcc_k}"
    assert ok_v, f"V cache mismatch: {pcc_v}"

    if is_sparse:
        # index_k is the single shared head, replicated across TP cols -> read col 0, concat SP rows.
        dts = ttnn.get_device_tensors(kv_cache.index_k)
        host_ik = torch.cat([ttnn.to_torch(dts[r * cols + 0]).float() for r in range(rows)], dim=2)  # [1,1,S,HD]
        ok_ik, pcc_ik = comp_pcc(ref_index_k, host_ik, 0.99)
        logger.info(f"[sparse] index_k write: pcc={pcc_ik}")
        assert ok_ik, f"index_k cache mismatch: {pcc_ik}"
