# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The trace-safe METADATA-TENSOR path must be indistinguishable from the host-scalar path.

A ttnn trace is recorded once, so a per-chunk value read from a host Python int is frozen at capture
time. The fix is for the ops to read it from a persistent device tensor instead
(``tt/trace.py``). Two of the three ops on this model's chunk path have that form —
``rotary_embedding_indexed`` and ``update_padded_kv_cache`` — and this test pins that switching them
over changes nothing about the result.

That matters for two reasons: it is the prerequisite for tracing at all, and it is the half of the
work that can be validated *today*, before the ring SDPA gains its tensor form (see
``tt/trace.py`` for why chunked trace capture is refused until it does).

Host-only companions check that the guard refuses exactly the configurations a capture would break.
"""

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.llama3_1_8b_d_p.reference import model as ref
from models.demos.llama3_1_8b_d_p.tt import rope as tt_rope
from models.demos.llama3_1_8b_d_p.tt import trace as trace_util
from models.demos.llama3_1_8b_d_p.tt.attention import Attention, AttentionConfig, ProgramConfig, allocate_kv_cache

from ..test_factory import (
    hf_to_meta_qk,
    llama_config,
    make_ccl,
    make_mesh_config,
    parametrize_mesh_with_fabric,
    shard_seq_on_sp,
)
from .test_kv_cache_write_vs_ref import gather_kv_cache


def test_guard_refuses_the_chunked_ring_path():
    """Host-only. The chunked path must be refused: its offsets are host scalars on the ring op, so a
    capture would freeze chunk 0's values and every later chunk would read the cache wrongly."""
    with pytest.raises(trace_util.TraceUnsupported) as e:
        trace_util.assert_traceable(uses_cache_backed_ring=True, num_users=1)
    assert "ring_joint_scaled_dot_product_attention" in str(e.value)


def test_guard_refuses_multi_user():
    """Host-only. kv_cache_batch_idx is a host scalar on the ring op, so one capture cannot serve a
    second slot even where the cache-read path is not taken."""
    with pytest.raises(trace_util.TraceUnsupported):
        trace_util.assert_traceable(uses_cache_backed_ring=False, num_users=2)


def test_guard_allows_the_one_shot_single_user_case():
    """Host-only. The configuration that IS capturable is allowed through."""
    trace_util.assert_traceable(uses_cache_backed_ring=False, num_users=1)


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("seq_len", [2048], ids=["s2048"])
def test_metadata_path_matches_scalar_path(mesh_device, device_params, seq_len, reset_seeds):
    """Same weights, same input, same offsets — once with host ints, once with metadata tensors.

    The two runs must write the SAME KV. Anything else means the metadata overloads of
    ``rotary_embedding_indexed`` / ``update_padded_kv_cache`` are being fed differently from the
    scalars, which is precisely the bug a trace would then bake in permanently.
    """
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    n_kv_local = cfg.num_key_value_heads // mesh_config.tp

    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.1
    ref_attn = ref.LlamaAttention(cfg, layer_idx=0)
    state_dict = hf_to_meta_qk(
        {
            "q_proj.weight": ref_attn.q_proj.weight.data,
            "k_proj.weight": ref_attn.k_proj.weight.data,
            "v_proj.weight": ref_attn.v_proj.weight.data,
            "o_proj.weight": ref_attn.o_proj.weight.data,
        },
        hd,
    )

    attn_cfg = AttentionConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd,
        max_seq_len=seq_len,
        sequence_parallel=True,
    )
    ccl = make_ccl(mesh_device)
    trans = {"prefill": tt_rope.build_transformation_mat(mesh_device)}
    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=seq_len, chunk_size=seq_len, sp_axis=mesh_config.sp_axis
    )

    def run(metadata):
        attn = Attention(
            mesh_device=mesh_device,
            config=attn_cfg,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            program_config=ProgramConfig(),
            layer_idx=0,
            transformation_mats=trans,
            weight_dtype=ttnn.bfloat16,
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
        attn(
            shard_seq_on_sp(mesh_device, x.reshape(1, 1, seq_len, cfg.hidden_size), mesh_config),
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            user_id=0,
            cached_len=0,
            indexed_rope=True,
            metadata=metadata,
        )
        ttnn.synchronize_device(mesh_device)
        return (
            gather_kv_cache(mesh_device, kv_cache.k, n_kv_local),
            gather_kv_cache(mesh_device, kv_cache.v, n_kv_local),
        )

    scalar_k, scalar_v = run(None)
    meta = trace_util.make_metadata(mesh_device, slot_idx=0, kv_actual=0)
    meta_k, meta_v = run(meta)

    # Same program, same inputs, only the metadata SOURCE differs — expect equality, not a PCC gate.
    ok_k, pcc_k = comp_pcc(scalar_k, meta_k, 0.9999)
    ok_v, pcc_v = comp_pcc(scalar_v, meta_v, 0.9999)
    logger.info(f"metadata-tensor vs host-scalar KV: K={pcc_k} V={pcc_v}")
    assert ok_k, f"K differs between the metadata and scalar paths: {pcc_k}"
    assert ok_v, f"V differs between the metadata and scalar paths: {pcc_v}"
    assert torch.equal(scalar_v, meta_v), "V should be bit-identical: it is neither rotated nor offset"


@parametrize_mesh_with_fabric(mesh_shapes=[(8, 4)])
@pytest.mark.parametrize("cached_len", [512, 1024], ids=["c512", "c1024"])
def test_metadata_path_matches_scalar_path_at_offset(mesh_device, device_params, cached_len, reset_seeds):
    """The same equivalence at a NON-ZERO cache offset — where a frozen capture would diverge.

    At ``kv_actual = 0`` a stale metadata value happens to be right, so an offset case is what
    actually exercises the tensor being read.
    """
    cfg = llama_config()
    hd = cfg.head_dim
    mesh_config = make_mesh_config(mesh_device)
    n_kv_local = cfg.num_key_value_heads // mesh_config.tp
    chunk = 512
    max_seq_len = 2048

    x = torch.randn(1, chunk, cfg.hidden_size) * 0.1
    ref_attn = ref.LlamaAttention(cfg, layer_idx=0)
    state_dict = hf_to_meta_qk(
        {
            "q_proj.weight": ref_attn.q_proj.weight.data,
            "k_proj.weight": ref_attn.k_proj.weight.data,
            "v_proj.weight": ref_attn.v_proj.weight.data,
            "o_proj.weight": ref_attn.o_proj.weight.data,
        },
        hd,
    )
    attn_cfg = AttentionConfig(
        hidden_size=cfg.hidden_size,
        num_heads=cfg.num_attention_heads,
        num_kv_heads=cfg.num_key_value_heads,
        head_dim=hd,
        max_seq_len=max_seq_len,
        sequence_parallel=True,
    )
    ccl = make_ccl(mesh_device)
    trans = {"prefill": tt_rope.build_transformation_mat(mesh_device)}
    rope_mats = tt_rope.build_indexed_rope(
        mesh_device, head_dim=hd, max_seq_len=max_seq_len, chunk_size=chunk, sp_axis=mesh_config.sp_axis
    )

    def run(metadata):
        attn = Attention(
            mesh_device=mesh_device,
            config=attn_cfg,
            state_dict=state_dict,
            ccl_manager=ccl,
            mesh_config=mesh_config,
            program_config=ProgramConfig(),
            layer_idx=0,
            transformation_mats=trans,
            weight_dtype=ttnn.bfloat16,
        )
        kv_cache = allocate_kv_cache(
            mesh_device,
            num_layers=1,
            max_seq_len=max_seq_len,
            sp_axis=mesh_config.sp_axis,
            num_users=1,
            head_dim=hd,
            num_kv_heads_local=n_kv_local,
        )
        if metadata is not None:
            metadata.update(slot_idx=0, kv_actual=cached_len)
        attn(
            shard_seq_on_sp(mesh_device, x.reshape(1, 1, chunk, cfg.hidden_size), mesh_config),
            rope_mats=rope_mats,
            kv_cache=kv_cache,
            user_id=0,
            cached_len=cached_len,
            indexed_rope=True,
            metadata=metadata,
        )
        ttnn.synchronize_device(mesh_device)
        return (
            gather_kv_cache(mesh_device, kv_cache.k, n_kv_local, chunk_local=chunk // mesh_config.sp),
            gather_kv_cache(mesh_device, kv_cache.v, n_kv_local, chunk_local=chunk // mesh_config.sp),
        )

    scalar_k, scalar_v = run(None)
    meta_k, meta_v = run(trace_util.make_metadata(mesh_device))

    ok_k, pcc_k = comp_pcc(scalar_k, meta_k, 0.9999)
    ok_v, pcc_v = comp_pcc(scalar_v, meta_v, 0.9999)
    logger.info(f"metadata vs scalar at cached_len={cached_len}: K={pcc_k} V={pcc_v}")
    assert ok_k, f"K differs at cached_len={cached_len}: {pcc_k}"
    assert ok_v, f"V differs at cached_len={cached_len}: {pcc_v}"
