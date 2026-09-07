# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Cache contents after a write through the PRODUCTION prefill seam, read back and PCC'd against the
torch reference's K/V. Pattern: ``minimax_m3/tests/unit/test_kv_cache_write_vs_ref.py``.

This is the sensitive test of the whole rope + swizzle + cache-layout chain, because it compares
post-RoPE K ELEMENT-WISE rather than through an attention output (where a wrong channel pairing
largely cancels inside q.k^T — see the note in ``test_attention_vs_ref.py``). Three things have to
be simultaneously right for it to pass:

  1. the q/k Meta swizzle (``convert_hf_qkv_to_meta_format``) and the Meta-interleaved rope table,
     reconciled against the HF-convention golden by the ``hf_to_meta_head_permutation`` index;
  2. the KV write landing at the right ``(slot, layer)`` and the right sequence offset —
     ``slot = user_id * num_layers + layer_idx``, user-major;
  3. the SP block-cyclic sequence layout, inverted on read-back by ``blockcyclic_positions``.

The negative control (unswizzled q/k) belongs here for the same reason: at this level of comparison
the convention error is unmissable, so the row actually discriminates.

Target mesh (SP=4 x TP=8): KV head ``c`` lives on TP column ``c`` (8 heads, 8 columns), and the
sequence is SP-sharded block-cyclic over the 4 rows.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.deepseek_v3_d_p.tt.mla.utils import block_cyclic_reorder, blockcyclic_positions
from models.demos.mistral_3_5_d_p.reference import model as reference
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.reference.mistral_config import reduced_text_config
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.attention import allocate_kv_cache, write_kv_chunk
from models.demos.mistral_3_5_d_p.tt.rope import build_indexed_rope, hf_to_meta_head_permutation

from ..test_factory import build_mesh_and_ccl, parametrize_mesh, sp_tp_shard_mapper
from .test_attention_vs_ref import HEAD_DIM, HIDDEN, build_attention, build_cos_sin, random_attention_weights


def read_cache_natural(cache_tensor, mesh_device, *, slot, chunk_size, capacity, n_kv, n_tokens):
    """Read one cache slot back in NATURAL token order: ``[n_kv, n_tokens, head_dim]``.

    Per chip the cache is ``[num_users*num_layers, 1, seq_local, head_dim]`` with KV head ``c`` on TP
    column ``c`` and the sequence block-cyclic over the SP rows, so: concatenate the rows' shards,
    then scatter through ``blockcyclic_positions`` (the inverse of what the writer kernel did).
    """
    rows, cols = tuple(mesh_device.shape)
    positions = blockcyclic_positions(rows, chunk_size, capacity)
    shards = ttnn.get_device_tensors(cache_tensor)

    def one_head(col):
        device_order = torch.cat([ttnn.to_torch(shards[r * cols + col])[slot, 0].float() for r in range(rows)], dim=0)
        natural = torch.empty_like(device_order)
        natural[positions] = device_order
        return natural[:n_tokens]

    return torch.stack([one_head(c) for c in range(n_kv)], dim=0)


@parametrize_mesh()
@pytest.mark.parametrize("num_users, num_layers", [(2, 3)], ids=["u2xl3"])
@pytest.mark.parametrize("seq_len", [256], ids=["s256"])
def test_kv_cache_write_read_roundtrip(mesh_device, device_params, num_users, num_layers, seq_len, reset_seeds):
    """Write known K/V into every ``(user, layer)`` slot, read back, PCC vs what was written.

    Validates the layout in isolation from the model: the two DRAM NdShard caches, the user-major
    slot packing, the head-sharded + SP-sharded sequence, and the bf8 round-trip. A slot-formula
    error shows up as one slot holding another's data, which this catches per slot.
    """
    rows, cols = tuple(mesh_device.shape)
    sp, n_kv = rows, cols
    assert n_kv == C.NUM_KEY_VALUE_HEADS, f"this layout maps KV head c -> TP column c; needs {n_kv} == 8 columns"
    assert seq_len % (ttnn.TILE_SIZE * sp) == 0, f"seq_len {seq_len} must be a multiple of {ttnn.TILE_SIZE * sp}"
    capacity = seq_len  # one chunk == the whole cache, so block-cyclic is exercised but kv_actual is 0

    sent_k = torch.randn(num_users, num_layers, n_kv, seq_len, HEAD_DIM)
    sent_v = torch.randn(num_users, num_layers, n_kv, seq_len, HEAD_DIM)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=num_users,
        head_dim=HEAD_DIM,
    )
    assert kv_cache.k.dtype == SPEC.kv_cache_dtype, "the cache must be allocated at the spec's kv dataformat"
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)

    def to_chunk(natural):
        """[n_kv, seq, hd] natural -> device [1, n_kv, seq, hd], block-cyclic then SP/TP sharded."""
        bc = block_cyclic_reorder(natural.reshape(1, n_kv, seq_len, HEAD_DIM), seq_len // sp, sp, seq_dim=2)
        return ttnn.from_torch(
            bc,
            device=mesh_device,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper,
        )

    for u in range(num_users):
        for layer in range(num_layers):
            tt_k, tt_v = to_chunk(sent_k[u, layer]), to_chunk(sent_v[u, layer])
            write_kv_chunk(kv_cache, tt_k, tt_v, slot_idx=u, layer_idx=layer, kv_actual=0, sp_axis=SPEC.sp_axis)
            tt_k.deallocate(True)
            tt_v.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    for u in range(num_users):
        for layer in range(num_layers):
            slot = u * num_layers + layer
            got_k = read_cache_natural(
                kv_cache.k,
                mesh_device,
                slot=slot,
                chunk_size=seq_len,
                capacity=capacity,
                n_kv=n_kv,
                n_tokens=seq_len,
            )
            got_v = read_cache_natural(
                kv_cache.v,
                mesh_device,
                slot=slot,
                chunk_size=seq_len,
                capacity=capacity,
                n_kv=n_kv,
                n_tokens=seq_len,
            )
            ok_k, pcc_k = comp_pcc(sent_k[u, layer], got_k, SPEC.pcc)
            ok_v, pcc_v = comp_pcc(sent_v[u, layer], got_v, SPEC.pcc)
            logger.info(f"(user={u}, layer={layer}) slot={slot}: K pcc={pcc_k} V pcc={pcc_v}")
            assert ok_k, f"K cache mismatch (user={u}, layer={layer}): {pcc_k}"
            assert ok_v, f"V cache mismatch (user={u}, layer={layer}): {pcc_v}"


@parametrize_mesh()
@pytest.mark.parametrize("n_chunks", [3], ids=["c3"])
def test_kv_cache_multichunk_write_offsets(mesh_device, device_params, n_chunks, reset_seeds):
    """Write N chunks at increasing tile-aligned ``kv_actual`` offsets and verify every position.

    The round-trip above only ever writes ``kv_actual=0``, so this is the only coverage of
    ``update_padded_kv_cache``'s nonzero-offset placement — the thing chunked prefill depends on, and
    the thing a misaligned chunk_size would corrupt silently.
    """
    rows, cols = tuple(mesh_device.shape)
    sp, n_kv = rows, cols
    chunk = ttnn.TILE_SIZE * sp * 2  # 256 at sp=4: tile-aligned per chip
    capacity = chunk * n_chunks

    sent_k = torch.randn(n_kv, capacity, HEAD_DIM)
    sent_v = torch.randn(n_kv, capacity, HEAD_DIM)

    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=1,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    mapper = sp_tp_shard_mapper(mesh_device, seq_dim=2, head_dim=1)
    positions = blockcyclic_positions(sp, chunk, capacity)

    for i in range(n_chunks):
        # Gather this chunk's rows in the order the per-chip block-cyclic write expects: the cache
        # rows belonging to global positions [i*chunk, (i+1)*chunk).
        rows_for_chunk = [r for r in range(capacity) if i * chunk <= int(positions[r]) < (i + 1) * chunk]
        idx = torch.tensor([int(positions[r]) for r in rows_for_chunk], dtype=torch.long)
        for cache, src in ((kv_cache.k, sent_k), (kv_cache.v, sent_v)):
            tt = ttnn.from_torch(
                src[:, idx, :].reshape(1, n_kv, chunk, HEAD_DIM),
                device=mesh_device,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=mapper,
            )
            ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                cache,
                ttnn.typecast(tt, cache.dtype),
                slot_idx=0,
                layer_idx=0,
                num_layers=1,
                kv_actual_global=i * chunk,
                cluster_axis=SPEC.sp_axis,
            )
            tt.deallocate(True)
    ttnn.synchronize_device(mesh_device)

    got_k = read_cache_natural(
        kv_cache.k, mesh_device, slot=0, chunk_size=chunk, capacity=capacity, n_kv=n_kv, n_tokens=capacity
    )
    got_v = read_cache_natural(
        kv_cache.v, mesh_device, slot=0, chunk_size=chunk, capacity=capacity, n_kv=n_kv, n_tokens=capacity
    )
    ok_k, pcc_k = comp_pcc(sent_k, got_k, SPEC.pcc)
    ok_v, pcc_v = comp_pcc(sent_v, got_v, SPEC.pcc)
    logger.info(f"multi-chunk write ({n_chunks} x {chunk} tokens): K pcc={pcc_k} V pcc={pcc_v}")
    assert ok_k, f"K multi-chunk mismatch: {pcc_k}"
    assert ok_v, f"V multi-chunk mismatch: {pcc_v}"


@parametrize_mesh()
@pytest.mark.parametrize("swizzle", [True, False], ids=["meta_swizzled", "unswizzled_control"])
def test_attention_writes_post_rope_kv(mesh_device, device_params, swizzle, reset_seeds):
    """Post-RoPE K and raw V, written by the PRODUCTION attention seam, vs the torch reference.

    Runs the real sequence-parallel configuration: the chunk is SP-sharded across the 4 rows (so each
    chip writes ``seq/sp`` cache rows, which is what ``update_padded_kv_cache`` requires) and the
    rope is the on-device INDEXED one built from the whole-cache block-cyclic table. That makes this
    the first test where the rope table, the SP shard layout and the cache write all have to agree.

    ``swizzle=True`` is the real configuration and must clear the PCC bar. ``swizzle=False`` is the
    negative control: with HF-convention q/k the device rotates the wrong channel pairs, and because
    this comparison is element-wise on K (not on an attention output) the error is unmissable —
    which is precisely why the control lives in this file. V carries no rope, so the control must
    leave V correct; that is what pins the failure to the rope convention and not to the write.
    """
    rows, cols = tuple(mesh_device.shape)
    sp, n_kv = rows, cols
    seq_len = ttnn.TILE_SIZE * sp * 2  # 256 at sp=4 -> 64 cache rows per chip
    capacity = seq_len  # one chunk == the whole cache
    num_layers = 2

    hf_state = random_attention_weights()
    x = torch.randn(1, seq_len, HIDDEN) * 0.1

    # Golden post-RoPE K / raw V over the whole sequence, HF convention.
    (cos_hf, sin_hf), _ = build_cos_sin(seq_len)
    ref = reference.attention_reference(
        x, hf_state, reduced_text_config(), cos_sin=(cos_hf.unsqueeze(0), sin_hf.unsqueeze(0))
    )
    # The device K is Meta-swizzled over the full head_dim, so permute the golden HF -> Meta.
    perm = hf_to_meta_head_permutation(HEAD_DIM)
    golden_k = ref.k_post_rope[0][..., perm]  # [n_kv, seq, head_dim]
    golden_v = ref.v[0]

    mesh_config, ccl = build_mesh_and_ccl(mesh_device)
    kv_cache = allocate_kv_cache(
        mesh_device,
        num_layers=num_layers,
        max_seq_len=capacity,
        sp_axis=SPEC.sp_axis,
        num_users=1,
        head_dim=HEAD_DIM,
    )
    attn = build_attention(
        mesh_device,
        mesh_config,
        ccl,
        hf_state,
        max_seq_len=capacity,
        sequence_parallel=True,
        swizzle=swizzle,
    )
    # Layer 1 of 2 on purpose: a slot formula that ignored layer_idx would write layer 0's slot and
    # this read of layer 1 would come back zeroed.
    layer_idx = 1
    attn.layer_idx = layer_idx

    rope_mats = build_indexed_rope(
        mesh_device,
        head_dim=HEAD_DIM,
        max_seq_len=capacity,
        chunk_size=seq_len,
        sp_axis=SPEC.sp_axis,
        rope_theta=C.ROPE_THETA,
        yarn_factor=C.YARN_FACTOR,
        yarn_orig_max_pos=C.YARN_ORIG_MAX_POS,
        yarn_beta_fast=C.YARN_BETA_FAST,
        yarn_beta_slow=C.YARN_BETA_SLOW,
        truncate=C.YARN_TRUNCATE,
    )
    # The chunk is SP-sharded on the sequence dim and replicated across TP — the production layout.
    x_tt = ttnn.from_torch(
        x.reshape(1, 1, seq_len, HIDDEN),
        device=mesh_device,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=sp_tp_shard_mapper(mesh_device, seq_dim=2),
    )
    attn(x_tt, rope_mats=rope_mats, kv_cache=kv_cache, user_id=0, cached_len=0, indexed_rope=True)
    ttnn.synchronize_device(mesh_device)

    slot = 0 * num_layers + layer_idx
    got_k = read_cache_natural(
        kv_cache.k, mesh_device, slot=slot, chunk_size=seq_len, capacity=capacity, n_kv=n_kv, n_tokens=seq_len
    )
    got_v = read_cache_natural(
        kv_cache.v, mesh_device, slot=slot, chunk_size=seq_len, capacity=capacity, n_kv=n_kv, n_tokens=seq_len
    )
    ok_k, pcc_k = comp_pcc(golden_k, got_k, SPEC.pcc)
    ok_v, pcc_v = comp_pcc(golden_v, got_v, SPEC.pcc)
    logger.info(f"attention-seam KV ({'swizzled' if swizzle else 'CONTROL'}): K pcc={pcc_k} V pcc={pcc_v}")
    if swizzle:
        assert ok_k, f"post-RoPE K written by the attention seam does not match the golden: {pcc_k}"
        assert ok_v, f"raw V written by the attention seam does not match the golden: {pcc_v}"
    else:
        assert not ok_k, f"unswizzled q/k still produced matching K (pcc={pcc_k}); the swizzle is not applied"
        assert ok_v, f"the control broke V (pcc={pcc_v}), so it is not isolating the rope convention"
