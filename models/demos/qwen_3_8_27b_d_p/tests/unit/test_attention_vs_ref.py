# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Full-attention block vs the torch reference, at the model's real dims and target SP x TP.

Layered so a failure localises: the fused projection + head split first, then the whole block
one-shot, then the KV cache write and read-back, then a 2-chunk run through the *same* module —
which is the mechanism chunked prefill depends on, and the only test that proves the cache-read
path is wired rather than merely callable.

Same random weights both sides; the cos/sin are shared, so this measures attention and not the
RoPE constants.
"""

from __future__ import annotations

import torch

import ttnn
from models.demos.qwen_3_8_27b_d_p.reference.modeling import Qwen35Attention, Qwen35RotaryEmbedding, init_random_weights
from models.demos.qwen_3_8_27b_d_p.tt.attention.prefill import Attention
from models.demos.qwen_3_8_27b_d_p.tt.caches import allocate_prefill_caches
from models.demos.qwen_3_8_27b_d_p.tt.rope import RotarySetup

from ..test_factory import mesh_setup, parametrize_mesh, unit_test_config
from .helpers import CACHE_DTYPE, WEIGHT_DTYPE, assert_tp_replicated, check_pcc, from_sp_sharded, randn, to_sp_sharded

S_LOCAL = 128  # per SP row; the chunk is S_LOCAL * sp
LAYER_IDX = 3  # the first full-attention layer


def _reference(cfg, seed: int = 31) -> Qwen35Attention:
    ref = Qwen35Attention(cfg, LAYER_IDX).eval()
    init_random_weights(ref, seed=seed)
    return ref


def _tt_attention(mesh, cfg, mesh_config, ccl, ref) -> Attention:
    return Attention(
        mesh,
        cfg,
        ref.state_dict(),
        mesh_config=mesh_config,
        ccl_manager=ccl,
        layer_idx=LAYER_IDX,
        weight_dtype=WEIGHT_DTYPE,
        cache_dtype=CACHE_DTYPE,
    )


@parametrize_mesh()
def test_fused_qkv_gate_projection_vs_ref(mesh, submesh_shape, device_params):
    """The rank-block-major fusion: each TP column must get ITS heads' q/k/v and gate.

    A wrong permutation here is the classic invisible bug — every tensor has the right shape and
    the block still produces a smooth output, just with the heads shuffled between columns.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=32, scale=0.5)
    with torch.no_grad():
        # project_raw, not project: QK-norm is close to a per-row rescale, so comparing against
        # the normed reference would stay above 0.99 even with the norm missing from the device.
        ref_q, ref_k, ref_v, ref_gate = ref.project_raw(x)

    attn = _tt_attention(mesh, cfg, mesh_config, ccl, ref)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    q, k, v, gate = attn.project(tt_x)

    q_local, kv_local = attn.n_q_local, attn.n_kv_local
    for c in range(mesh_config.tp):
        # Compose each TP column's own heads, gathering its SP rows back into the full sequence.
        def _col(t, n_heads, head_dim):
            dev = ttnn.get_device_tensors(t)
            rows = [ttnn.to_torch(dev[r * mesh_config.tp + c]) for r in range(mesh_config.sp)]
            return torch.cat(rows, dim=2).reshape(1, n_heads, total, head_dim)

        check_pcc(
            f"attn_q[col{c}]",
            ref_q[:, c * q_local : (c + 1) * q_local],
            _col(q, q_local, cfg.head_dim),
        )
        check_pcc(
            f"attn_k[col{c}]",
            ref_k[:, c * kv_local : (c + 1) * kv_local],
            _col(k, kv_local, cfg.head_dim),
        )
        check_pcc(
            f"attn_v[col{c}]",
            ref_v[:, c * kv_local : (c + 1) * kv_local],
            _col(v, kv_local, cfg.head_dim),
        )
        gate_dev = ttnn.get_device_tensors(gate)
        gate_col = torch.cat(
            [ttnn.to_torch(gate_dev[r * mesh_config.tp + c]) for r in range(mesh_config.sp)], dim=2
        ).reshape(1, total, q_local * cfg.head_dim)
        check_pcc(
            f"attn_gate[col{c}]",
            ref_gate[:, :, c * q_local * cfg.head_dim : (c + 1) * q_local * cfg.head_dim],
            gate_col,
        )


@parametrize_mesh()
def test_attention_vs_ref(mesh, submesh_shape, device_params):
    """The whole block one-shot: projection -> QK-norm -> partial RoPE -> causal ring SDPA ->
    output gate -> o_proj -> TP all-reduce."""
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=33, scale=0.5)

    cos, sin = Qwen35RotaryEmbedding(cfg)(torch.arange(total)[None, :])
    with torch.no_grad():
        expected, _ = ref(x, (cos, sin))

    attn = _tt_attention(mesh, cfg, mesh_config, ccl, ref)
    caches = allocate_prefill_caches(
        mesh,
        cfg,
        mesh_config=mesh_config,
        max_seq_len=total,
        layer_indices=[LAYER_IDX],
        cache_dtype=CACHE_DTYPE,
    )
    tt_cos, tt_sin = RotarySetup(mesh, cfg, mesh_config).chunk_mats(0, total)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    out = attn(tt_x, cos=tt_cos, sin=tt_sin, caches=caches, cached_len=0)
    assert_tp_replicated(out, mesh_config, "attention output")
    got = from_sp_sharded(out, mesh_config)
    check_pcc("attention", expected.reshape(1, 1, total, cfg.hidden_size), got)


@parametrize_mesh()
def test_kv_cache_write_and_read_back(mesh, submesh_shape, device_params):
    """Cache contents after a write through the production seam, read back and PCC'd against the
    reference's post-RoPE K and raw V.

    Two things are under test at once and both are silent when wrong: that the write lands at the
    right ``slot = user * num_kv_layers + kv_slot`` (not at ``layer_idx``, which for layer 3 of 64
    would be off by three), and that the block-cyclic SP layout puts row ``r``'s tokens where the
    reader expects them.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=34, scale=0.5)
    cos, sin = Qwen35RotaryEmbedding(cfg)(torch.arange(total)[None, :])
    with torch.no_grad():
        _out, capture = ref(x, (cos, sin))

    attn = _tt_attention(mesh, cfg, mesh_config, ccl, ref)
    caches = allocate_prefill_caches(
        mesh,
        cfg,
        mesh_config=mesh_config,
        max_seq_len=total,
        layer_indices=[LAYER_IDX],
        cache_dtype=CACHE_DTYPE,
    )
    tt_cos, tt_sin = RotarySetup(mesh, cfg, mesh_config).chunk_mats(0, total)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    attn(tt_x, cos=tt_cos, sin=tt_sin, caches=caches, cached_len=0)

    kv = caches.kv
    slot = kv.slot(0, attn.kv_slot)
    seq_local = total // mesh_config.sp
    kv_local = attn.n_kv_local
    for name, cache, expected in (("k", kv.k, capture.key), ("v", kv.v, capture.value)):
        interleaved = ttnn.to_memory_config(cache, ttnn.DRAM_MEMORY_CONFIG)
        dev = ttnn.get_device_tensors(interleaved)
        for c in range(mesh_config.tp):
            rows = []
            for r in range(mesh_config.sp):
                shard = ttnn.to_torch(dev[r * mesh_config.tp + c])
                rows.append(shard[slot : slot + 1, :, :seq_local, :])
            got = torch.cat(rows, dim=2).reshape(1, kv_local, total, cfg.head_dim)
            check_pcc(
                f"kv_cache_{name}[col{c}]",
                expected[:, c * kv_local : (c + 1) * kv_local],
                got,
            )


@parametrize_mesh()
def test_attention_chunked_vs_ref(mesh, submesh_shape, device_params):
    """A 2-chunk sequence through the SAME module, the second chunk attending the prefix left in
    the cache. Asserts the second chunk's output matches a one-shot run's second half.

    This is the test the whole chunked-prefill path rests on: the cache-read ring SDPA is only
    exercised when ``cached_len > 0``, so a module that never takes that branch passes every other
    attention test here.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    chunk = S_LOCAL * mesh_config.sp
    total = 2 * chunk
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=35, scale=0.5)
    cos, sin = Qwen35RotaryEmbedding(cfg)(torch.arange(total)[None, :])
    with torch.no_grad():
        expected, _ = ref(x, (cos, sin))
    expected_second = expected[:, chunk:]

    attn = _tt_attention(mesh, cfg, mesh_config, ccl, ref)
    caches = allocate_prefill_caches(
        mesh,
        cfg,
        mesh_config=mesh_config,
        max_seq_len=total,
        layer_indices=[LAYER_IDX],
        cache_dtype=CACHE_DTYPE,
    )
    setup = RotarySetup(mesh, cfg, mesh_config)
    for c in range(2):
        tt_cos, tt_sin = setup.chunk_mats(c * chunk, chunk)
        tt_x = to_sp_sharded(x[:, c * chunk : (c + 1) * chunk].reshape(1, 1, chunk, cfg.hidden_size), mesh, mesh_config)
        out = attn(tt_x, cos=tt_cos, sin=tt_sin, caches=caches, cached_len=c * chunk)
        if c == 1:
            got = from_sp_sharded(out, mesh_config)
    check_pcc("attention_chunk1_cache_read", expected_second.reshape(1, 1, chunk, cfg.hidden_size), got)


@parametrize_mesh()
def test_qk_norm_is_applied_in_the_block(mesh, submesh_shape, device_params):
    """``Attention.qk_norm`` against the reference's normed q/k.

    Separate from the projection test above precisely because the normalisation is a near-rescale:
    the two have to be measured apart or neither is really measured.
    """
    cfg = unit_test_config()
    mesh_config, ccl = mesh_setup(mesh)
    total = S_LOCAL * mesh_config.sp
    ref = _reference(cfg)
    x = randn(1, total, cfg.hidden_size, seed=36, scale=0.5)
    with torch.no_grad():
        ref_q, ref_k, _v, _gate = ref.project(x)

    attn = _tt_attention(mesh, cfg, mesh_config, ccl, ref)
    tt_x = to_sp_sharded(x.reshape(1, 1, total, cfg.hidden_size), mesh, mesh_config)
    q, k, _v_tt, _gate_tt = attn.project(tt_x)
    q, k = attn.qk_norm(q, k)

    q_local, kv_local = attn.n_q_local, attn.n_kv_local

    def _col(t, c, n_heads):
        dev = ttnn.get_device_tensors(t)
        rows = [ttnn.to_torch(dev[r * mesh_config.tp + c]) for r in range(mesh_config.sp)]
        return torch.cat(rows, dim=2).reshape(1, n_heads, total, cfg.head_dim)

    for c in range(mesh_config.tp):
        check_pcc(f"qk_norm_q[col{c}]", ref_q[:, c * q_local : (c + 1) * q_local], _col(q, c, q_local))
        check_pcc(f"qk_norm_k[col{c}]", ref_k[:, c * kv_local : (c + 1) * kv_local], _col(k, c, kv_local))
