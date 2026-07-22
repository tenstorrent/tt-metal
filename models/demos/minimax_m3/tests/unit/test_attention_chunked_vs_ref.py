# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Chunked prefill through the ATTENTION LAYER (Attention.__call__), SP=8 × TP=4, dense + sparse.

End-to-end check of the wired cache-read path: process a 2-chunk sequence two ways through the SAME
Attention module and assert the second chunk's output matches.

  * single-shot: one forward over the full 2*chunk sequence (cached_len=0, no-cache path) — the golden.
  * chunked:     chunk 0 (cached_len=0, writes cache) then chunk 1 (cached_len=chunk, CACHE-READ path),
                 where chunk 1's queries attend the accumulated [chunk0 ; chunk1] read back from the cache.

The fundamental correctness property of chunked prefill is "chunked == single-shot over the full
sequence", so the second chunk's output must match the single-shot's [chunk:2*chunk] slice. Both paths
use the model's own ops (device-to-device compare; no torch ref / no Meta-swizzle reconciliation). The
RoPE for chunk 1 uses absolute positions [chunk:2*chunk] — the chunk's true offset in the sequence.
"""

from types import SimpleNamespace

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

HIDDEN, NQ, NKV, HEAD_DIM, ROTARY_DIM, THETA, EPS = 6144, 64, 4, 128, 64, 5_000_000.0, 1e-6
NIDX, INDEX_DIM = 4, 128


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize("chunk_local", [640], ids=["chunk640"])  # chunk=5120; 2 chunks -> T=10240 (40 blocks)
# TODO(block-aware): re-add "sparse" once the indexer/sparse_sdpa_msa kernels read the block-cyclic
# NdShard cache directly (slab-aware in-kernel cache-read, like ring_joint). Until then the sparse
# chunked path goes through msa_sp_attention's host-side to_memory_config+slice gather, which (a) crashes
# on the single-layer whole-tensor-slice alias and (b) with random weights would flip the MSA top-k block
# selection chunked-vs-single-shot. MSA compute is covered by test_msa_layer_vs_ref (real weights); real
# chunked MSA end-to-end by galaxy_prefill_kv_pcc. It's a small kernel change to enable this.
@pytest.mark.parametrize("layer_kind", ["dense"])
def test_attention_chunked(mesh_device, device_params, layer_kind, chunk_local, reset_seeds):
    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    chunk = sp * chunk_local  # 5120
    total = 2 * chunk  # 10240
    is_sparse = layer_kind == "sparse"

    torch.manual_seed(0)
    x = torch.randn(1, total, HIDDEN) * 0.1

    w = {
        "q": torch.rand(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.rand(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }
    hf_state = {
        "q_proj.weight": w["q"],
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
        "q_norm.weight": w["q_norm"],
        "k_norm.weight": w["k_norm"],
    }
    if is_sparse:
        w |= {
            "index_q": torch.rand(NIDX * INDEX_DIM, HIDDEN) * 0.02,
            "index_k": torch.rand(INDEX_DIM, HIDDEN) * 0.02,
            "index_q_norm": torch.randn(INDEX_DIM) * 0.1,
            "index_k_norm": torch.randn(INDEX_DIM) * 0.1,
        }
        hf_state |= {
            "index_q_proj.weight": w["index_q"],
            "index_k_proj.weight": w["index_k"],
            "index_q_norm.weight": w["index_q_norm"],
            "index_k_norm.weight": w["index_k_norm"],
        }
    state = convert_hf_qkv_to_meta_format_partial(hf_state, HEAD_DIM, ROTARY_DIM)

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rope_scaling=None,
        rms_norm_eps=EPS,
        max_position_embeddings=total,
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
        max_seq_len=total,
        max_local_batch_size=1,
        is_sparse=is_sparse,
        sequence_parallel=True,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=MiniMaxM3AttentionProgramConfig(),
        layer_idx=0,
        transformation_mats=rope_setup.get_both_trans_mats(),
    )

    in_dims = [None, None]
    in_dims[sp_axis] = 2  # seq -> SP rows (contiguous)

    def shard_seq(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    def rope_range(a, b):
        """Per-row RoPE for absolute positions [a:b], SP-resharded across rows (chip r -> its slice)."""

        def rs(dev):
            full = ttnn.to_torch(ttnn.get_device_tensors(dev)[0])[:, :, a:b, :]
            return shard_seq(full)

        return [rs(rope_setup.cos_matrix_prefill), rs(rope_setup.sin_matrix_prefill)]

    def run(x_chunk, a, b, cached_len, kv_cache):
        L = b - a
        x_tt = shard_seq(x_chunk.reshape(1, 1, L, HIDDEN))
        return attn(x_tt, rope_mats=rope_range(a, b), kv_cache=kv_cache, user_id=0, cached_len=cached_len)

    def gather_seq(out):
        # out: SP-sharded on rows (seq), full hidden on each TP col after o_proj reduce. Take col 0.
        dts = ttnn.get_device_tensors(out)
        return torch.cat([ttnn.to_torch(dts[r * cols + 0]).float() for r in range(rows)], dim=2)  # [1,1,S,H]

    # --- single-shot golden over the full sequence (fresh cache) ---
    kvc_ss = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=total, sp_axis=sp_axis, num_users=1)
    out_ss = run(x, 0, total, 0, kvc_ss)
    ttnn.synchronize_device(mesh_device)
    golden_c1 = gather_seq(out_ss)[:, :, chunk:total, :]  # [1,1,chunk,H]

    # --- chunked: chunk 0 (writes cache) then chunk 1 (cache-read over [chunk0 ; chunk1]) ---
    kvc = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=total, sp_axis=sp_axis, num_users=1)
    run(x[:, :chunk], 0, chunk, 0, kvc)
    out_c1 = run(x[:, chunk:], chunk, total, chunk, kvc)
    ttnn.synchronize_device(mesh_device)
    chunked_c1 = gather_seq(out_c1)  # [1,1,chunk,H]

    passing, pcc = comp_pcc(golden_c1, chunked_c1, 0.99)
    logger.info(f"[{layer_kind}] chunked-prefill chunk1 vs single-shot[{chunk}:{total}]: pcc={pcc}")
    assert passing, f"{layer_kind} chunked attention PCC fail: {pcc}"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)], linear_fabric=True)
@pytest.mark.parametrize(
    "chunk_local", [256], ids=["chunk256"]
)  # chunk=2048, T=4096 (32 blocks) — keeps the fp32 CPU ref fast
# TODO(block-aware): re-add "sparse" once the indexer/sparse_sdpa_msa kernels read the block-cyclic
# NdShard cache directly (slab-aware in-kernel cache-read, like ring_joint). Until then the sparse
# chunked path goes through msa_sp_attention's host-side to_memory_config+slice gather, which (a) crashes
# on the single-layer whole-tensor-slice alias and (b) with random weights would flip the MSA top-k block
# selection chunked-vs-single-shot. MSA compute is covered by test_msa_layer_vs_ref (real weights); real
# chunked MSA end-to-end by galaxy_prefill_kv_pcc. It's a small kernel change to enable this.
@pytest.mark.parametrize("layer_kind", ["dense"])
def test_attention_chunked_vs_cpu_ref(mesh_device, device_params, layer_kind, chunk_local, reset_seeds):
    """Chunked-prefill chunk-1 output vs the self-contained torch CPU reference (absolute correctness).

    Unlike test_attention_chunked (chunked == device single-shot — a self-consistency check that a shared
    bug could pass), this pins the device chunked path to the CPU reference's attention output. The device
    runs chunk0-then-chunk1 (chunk1 = cache-read); the reference runs the full sequence and we compare the
    [chunk:2*chunk] slice. The attention OUTPUT (post o_proj) is convention-agnostic, so no Meta-swizzle
    reconciliation is needed — the reference gets RAW HF weights, the device the swizzled ones.
    """
    from models.demos.minimax_m3.reference.model import DictWeights, MiniMaxM3Config, MiniMaxM3TextModel, build_rope

    rows, cols = tuple(mesh_device.shape)
    assert (rows, cols) == (8, 4)
    sp, tp, sp_axis = rows, cols, 0
    chunk = sp * chunk_local
    total = 2 * chunk
    is_sparse = layer_kind == "sparse"

    torch.manual_seed(0)
    x = torch.randn(1, total, HIDDEN) * 0.1
    w = {
        "q": torch.rand(NQ * HEAD_DIM, HIDDEN) * 0.02,
        "k": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "v": torch.rand(NKV * HEAD_DIM, HIDDEN) * 0.02,
        "o": torch.rand(HIDDEN, NQ * HEAD_DIM) * 0.02,
        "q_norm": torch.randn(HEAD_DIM) * 0.1,
        "k_norm": torch.randn(HEAD_DIM) * 0.1,
    }
    if is_sparse:
        w |= {
            "index_q": torch.rand(NIDX * INDEX_DIM, HIDDEN) * 0.02,
            "index_k": torch.rand(INDEX_DIM, HIDDEN) * 0.02,
            "index_q_norm": torch.randn(INDEX_DIM) * 0.1,
            "index_k_norm": torch.randn(INDEX_DIM) * 0.1,
        }

    # --- CPU reference: the self-contained torch model's _attention over the FULL sequence (RAW HF weights) ---
    p = "model.layers.0."
    ref_state = {
        p + "self_attn.q_proj.weight": w["q"],
        p + "self_attn.k_proj.weight": w["k"],
        p + "self_attn.v_proj.weight": w["v"],
        p + "self_attn.o_proj.weight": w["o"],
        p + "self_attn.q_norm.weight": w["q_norm"],
        p + "self_attn.k_norm.weight": w["k_norm"],
    }
    if is_sparse:
        ref_state |= {
            p + "self_attn.index_q_proj.weight": w["index_q"],
            p + "self_attn.index_k_proj.weight": w["index_k"],
            p + "self_attn.index_q_norm.weight": w["index_q_norm"],
            p + "self_attn.index_k_norm.weight": w["index_k_norm"],
        }
    ref_model = MiniMaxM3TextModel(MiniMaxM3Config(), DictWeights(ref_state))
    cos_ref, sin_ref = build_rope(total, ROTARY_DIM, THETA)
    ref_attn = ref_model._attention(x.float(), p, cos_ref, sin_ref, is_sparse)[0]  # [1, total, H]
    ref_c1 = ref_attn[:, chunk:total, :].reshape(1, chunk, HIDDEN)

    # --- device chunked: swizzled weights -> Attention; chunk0 then chunk1 (cache-read) ---
    hf_state = {
        "q_proj.weight": w["q"],
        "k_proj.weight": w["k"],
        "v_proj.weight": w["v"],
        "o_proj.weight": w["o"],
        "q_norm.weight": w["q_norm"],
        "k_norm.weight": w["k_norm"],
    }
    if is_sparse:
        hf_state |= {
            "index_q_proj.weight": w["index_q"],
            "index_k_proj.weight": w["index_k"],
            "index_q_norm.weight": w["index_q_norm"],
            "index_k_norm.weight": w["index_k_norm"],
        }
    state = convert_hf_qkv_to_meta_format_partial(hf_state, HEAD_DIM, ROTARY_DIM)

    mesh_config = MeshConfig((rows, cols), tp=tp)
    ccl = CCLManager(mesh_device, num_links=get_default_num_links(mesh_device), topology=ttnn.Topology.Linear)
    hf_config = SimpleNamespace(
        hidden_size=HIDDEN,
        num_attention_heads=NQ,
        num_key_value_heads=NKV,
        head_dim=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        rope_theta=THETA,
        rope_scaling=None,
        rms_norm_eps=EPS,
        max_position_embeddings=total,
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
        max_seq_len=total,
        max_local_batch_size=1,
        is_sparse=is_sparse,
        sequence_parallel=True,
    )
    attn = Attention(
        mesh_device=mesh_device,
        config=attn_config,
        state_dict=state,
        ccl_manager=ccl,
        mesh_config=mesh_config,
        program_config=MiniMaxM3AttentionProgramConfig(),
        layer_idx=0,
        transformation_mats=rope_setup.get_both_trans_mats(),
    )

    in_dims = [None, None]
    in_dims[sp_axis] = 2

    def shard_seq(t):
        return ttnn.from_torch(
            t,
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=(rows, cols), dims=in_dims),
        )

    def rope_range(a, b):
        def rs(dev):
            return shard_seq(ttnn.to_torch(ttnn.get_device_tensors(dev)[0])[:, :, a:b, :])

        return [rs(rope_setup.cos_matrix_prefill), rs(rope_setup.sin_matrix_prefill)]

    def run(x_chunk, a, b, cached_len, kv_cache):
        x_tt = shard_seq(x_chunk.reshape(1, 1, b - a, HIDDEN))
        return attn(x_tt, rope_mats=rope_range(a, b), kv_cache=kv_cache, user_id=0, cached_len=cached_len)

    def gather_seq(out):
        dts = ttnn.get_device_tensors(out)
        return torch.cat([ttnn.to_torch(dts[r * cols + 0]).float() for r in range(rows)], dim=2)

    kvc = allocate_kv_caches(mesh_device, num_layers=1, max_seq_len=total, sp_axis=sp_axis, num_users=1)
    run(x[:, :chunk], 0, chunk, 0, kvc)
    out_c1 = run(x[:, chunk:], chunk, total, chunk, kvc)
    ttnn.synchronize_device(mesh_device)
    dev_c1 = gather_seq(out_c1).reshape(1, chunk, HIDDEN)

    passing, pcc = comp_pcc(ref_c1, dev_c1, 0.99)
    logger.info(f"[{layer_kind}] chunked-prefill chunk1 vs CPU reference[{chunk}:{total}]: pcc={pcc}")
    assert passing, f"{layer_kind} chunked vs CPU ref PCC fail: {pcc}"
