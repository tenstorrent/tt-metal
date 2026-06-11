# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Dead-simple BATCH-32 decode gate: 32 IDENTICAL users -> 32 IDENTICAL outputs.

The key insight (and the whole point of this test): if all N decode users get
the SAME input AND the SAME recurrent/KV state, the model MUST produce IDENTICAL
logits for all N rows. So the oracle is TRIVIAL and CPU-free — we just compare
every output row i (1..N-1) to row 0 with PCC. No CPU reference, no per-user
distinct seeding, no vocab math. The oracle is correct by construction (identical
in -> identical out), so ANY failure is a real model bug, not a reference mismatch.

We go straight to N=32 (not N=2): it fills the whole ``tile_padded_batch_rows=32``
tile and exercises the real batch-32 grid / memory paths now.

EXPECTED RESULT
---------------
  - PRE-PHASE-1 (dim-2 sliced to row 0): the decode backbone collapses dim-2 to
    a single logical row (full-attn ``_forward_decode_qwen36`` slices x_3d dim-2
    back to 1; the decoder branches slice attn_out dim-2 back to 1). Only user 0
    survives; rows 1..31 are garbage. So ``row0 != row_i`` -> this test FAILS.
  - POST-PHASE-1 (de-slice): all 32 rows match row0 -> this test goes GREEN.

Two sub-tests (each at N parametrized, default [32]):

  1. ``test_gdn_decode_batch32_identical`` — GatedDeltaNet (layer 0). Seed all N
     users' DeltaNet recurrent + conv state IDENTICALLY (build ONE user's state,
     broadcast to all N slices). Identical decode input in all N rows.
  2. ``test_full_attn_decode_batch32_identical`` — full_attention (layer 3, paged
     KV). Prefill the SAME random stream for every user so all users' KV pages
     hold identical data. Identical decode input + identical current_pos.

Reuses the PROVEN building blocks from ``test_decode_batch2_isolated_pcc.py``
(fixture, weight loader, ``_build_tt_model`` with ``max_batch_size=N``, GDN
state-seed helpers, full-attn paged-prefill helper, ``_send_col_sharded_decode_rows``,
the logits gather).

Run (N=32):

    export TT_METAL_HOME=$(pwd) PYTHONPATH=$(pwd) \\
        && source python_env/bin/activate \\
        && QWEN36_TT_LANG_BETA_G=0 python -m pytest --noconftest \\
            models/demos/qwen3_6_galaxy_v2/tests/test_decode_batch32_smoke.py \\
            -v -s
"""
from __future__ import annotations

import math
import os

import pytest
import torch

import ttnn

# Reuse the PROVEN building blocks verbatim from the batch-2 isolated PCC test.
from models.demos.qwen3_6_galaxy_v2.tests.test_decode_batch2_isolated_pcc import (
    bh_glx_mesh,  # module-scoped mesh fixture
)
from models.demos.qwen3_6_galaxy_v2.tests.test_decode_batch2_isolated_pcc import (  # noqa: E501
    _FA_LAYER_IDX,
    _GDN_LAYER_IDX,
    _H,
    _PAGED_BLOCK_SIZE,
    _PCC_THRESH,
    _SNAPSHOT,
    _T_PREFILL,
    _build_partial_rope_cos_sin_tt,
    _build_ref_gdn,
    _build_tt_model,
    _build_zero_rope_cos_sin_tt,
    _gather_decode_logits,
    _load_state_dict_for_layer,
    _pcc,
    _seed_conv_state,
    _seed_dn_state,
    _send_col_sharded_decode_rows,
    _send_col_sharded_hidden,
    _user_page_table_tt,
)

# ---------------------------------------------------------------------------
# Trivial CPU-free oracle: every row i must match row 0.
# ---------------------------------------------------------------------------


def _assert_all_rows_match_row0(tt_logits: torch.Tensor, N: int, tag: str):
    """``tt_logits`` is the gathered ``[N_pad, vocab]``; the N users live in dim-0.

    Oracle = identical-in -> identical-out: every row i in 1..N-1 must match row 0
    (PCC > thresh). No CPU reference. Prints the MIN PCC across rows and row 0's
    argmax (a non-garbage sanity check on row 0 itself).
    """
    vocab = tt_logits.shape[-1]
    flat = tt_logits.reshape(-1, vocab)  # [N_pad, vocab]
    row0 = flat[0].reshape(vocab)
    row0_argmax = int(row0.argmax().item())
    failures = []
    min_pcc = float("inf")
    min_pcc_row = -1
    for i in range(1, N):
        pcc_i = _pcc(flat[i].reshape(vocab), row0)
        if pcc_i < min_pcc:
            min_pcc, min_pcc_row = pcc_i, i
        if not (pcc_i > _PCC_THRESH):
            failures.append((i, pcc_i))
    print(
        f"[{tag}] N={N}: min row-vs-row0 PCC = {min_pcc:.6f} (row {min_pcc_row}, "
        f"thresh={_PCC_THRESH})  row0 argmax={row0_argmax}  "
        f"({len(failures)} of {N - 1} compared rows FAIL)"
    )
    assert not failures, (
        f"[{tag}] batch-{N} identical-users decode: rows differ from row 0 "
        f"(PCC < {_PCC_THRESH}) for rows "
        + ", ".join(f"{i}(pcc={p:.4f})" for i, p in failures[:8])
        + (f", ... (+{len(failures) - 8} more)" if len(failures) > 8 else "")
        + "  (EXPECTED pre-Phase-1: decode backbone slices dim-2 to row 0, so "
        "rows 1..N-1 are garbage; Phase 1 de-slice turns this green)"
    )
    print(f"[{tag}] PASSED (all {N} rows identical to row 0, PCC > {_PCC_THRESH})")


# ===========================================================================
# Sub-test 1: GatedDeltaNet (linear_attention) decode at batch-32, identical users
# ===========================================================================


@pytest.mark.hardware
@pytest.mark.parametrize("N", [32], ids=lambda n: f"N{n}")
def test_gdn_decode_batch32_identical(bh_glx_mesh, N):
    """All N GDN-decode users get the SAME input + SAME state -> SAME logits.

    Oracle is CPU-free: every row must equal row 0. Pre-Phase-1 the decoder
    collapses dim-2 to row 0 so rows 1..N-1 are garbage (FAIL); Phase 1 de-slice
    makes all rows equal row 0 (PASS).
    """
    os.environ.setdefault("QWEN36_TT_LANG_BETA_G", "0")  # B>1 uses the 6-op chain

    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _GDN_LAYER_IDX)
    print(f"[gdn-N{N}] loaded {len(state_dict)} weights")

    # ---- Build ONE user's GDN state from a single random prefill stream, then
    #      broadcast it to ALL N users (every state slice IDENTICAL). The actual
    #      values don't matter — only that all N users share the SAME state.
    gdn_ref = _build_ref_gdn(state_dict)
    torch.manual_seed(44)
    x_full = torch.randn(1, _T_PREFILL + 1, _H, dtype=torch.bfloat16).float()  # ONE shared stream
    with torch.no_grad():
        _, conv_s, recur_s = gdn_ref(x_full[:, :_T_PREFILL, :], conv_state=None, recurrent_state=None)
    conv_states = [conv_s for _ in range(N)]  # same state for every user
    recur_states = [recur_s for _ in range(N)]
    print(f"[gdn-N{N}] built ONE shared state, broadcast to {N} users")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, "linear_attention", N)
    assert getattr(model.layers[0], "is_linear_attention_layer", False) is True
    attn = model.layers[0].attention
    print(f"[gdn-N{N}] TT model built; max_batch_size={attn.max_batch_size}")

    attn.clear_state()
    _seed_dn_state(attn, recur_states)
    _seed_conv_state(attn, conv_states)
    print(f"[gdn-N{N}] seeded IDENTICAL recurrent + conv state for {N} users")

    # ---- ONE batched decode step. ALL N dim-2 rows carry the SAME input row.
    one_row = x_full[:, _T_PREFILL, :].bfloat16()  # [1, H]
    decode_rows = one_row.repeat(N, 1)  # [N, H], every row identical
    x_decode_tt = _send_col_sharded_decode_rows(decode_rows, bh_glx_mesh, args)
    cos_tt, sin_tt = _build_zero_rope_cos_sin_tt(bh_glx_mesh, 1)
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([_T_PREFILL] * N, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ReplicateTensorToMesh(bh_glx_mesh),
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=cur_pos_tt,
        rot_mats=(cos_tt, sin_tt),
        user_id=0,
        mode="decode",
        page_table=None,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=_T_PREFILL,
        get_last_token=-1,
        kv_cache=None,
        batch_size=N,
    )
    tt_logits = _gather_decode_logits(tt_out, bh_glx_mesh, args)  # [N_pad, vocab]
    _assert_all_rows_match_row0(tt_logits, N, tag=f"gdn-N{N}")


# ===========================================================================
# Sub-test 2: full_attention decode at batch-32, identical users (paged KV)
# ===========================================================================


@pytest.mark.hardware
@pytest.mark.parametrize("N", [32], ids=lambda n: f"N{n}")
def test_full_attn_decode_batch32_identical(bh_glx_mesh, N):
    """All N full-attn users get the SAME prefill stream + SAME decode input +
    SAME current_pos -> SAME logits. CPU-free oracle: every row must equal row 0.

    Pre-Phase-1 ``_forward_decode_qwen36`` slices x dim-2 back to 1 so only user 0
    survives (rows 1..N-1 garbage -> FAIL); Phase 1 de-slice turns it green.
    """
    from models.demos.llama3_70b_galaxy.tt.llama_common import PagedAttentionConfig

    # Size the page table for N users: per-user blocks for T=128 with block_size=32
    # is ceil(128/32)=4 blocks/user; +1 spare block/user for the decode step.
    blocks_per_user = (_T_PREFILL + _PAGED_BLOCK_SIZE - 1) // _PAGED_BLOCK_SIZE + 1  # 5
    max_num_blocks = blocks_per_user * N
    # Round up to a multiple of N so it reshapes cleanly to [N, max_blocks/N].
    max_num_blocks = int(math.ceil(max_num_blocks / N) * N)
    paged_attention_config = PagedAttentionConfig(block_size=_PAGED_BLOCK_SIZE, max_num_blocks=max_num_blocks)

    state_dict = _load_state_dict_for_layer(_SNAPSHOT, _FA_LAYER_IDX)
    print(f"[fa-N{N}] loaded {len(state_dict)} weights (max_num_blocks={max_num_blocks})")

    model, args = _build_tt_model(bh_glx_mesh, state_dict, "full_attention", N, paged_attention_config)
    assert getattr(model.layers[0], "is_linear_attention_layer", True) is False
    print(f"[fa-N{N}] TT model built (paged KV); max_batch_size={args.max_batch_size}")

    # Build the page table INLINE with STRICTLY-DISJOINT, contiguous physical blocks
    # per user (mirrors the PROVEN probe's distinct-per-user mapping). The argsort/
    # reshape permutation in ``_build_paged_page_table`` reshapes to [N, max_blocks//N]
    # columns, but the from_torch KV fill below only writes the first
    # ``n_blocks_per_user_fill`` (=4) columns — so any user whose *unfilled* trailing
    # column happens to alias another user's *filled* block is corrupted. Here we give
    # user u the contiguous run [u*blocks_per_user : (u+1)*blocks_per_user], which is
    # provably disjoint across users (each physical block index appears in exactly one
    # row), guaranteeing every user's KV is independent. Both the from_torch fill AND
    # the decode ``page_table_tt`` consume this SAME ``page_table_torch`` handle.
    assert max_num_blocks >= N * blocks_per_user, (
        f"page table needs >= {N * blocks_per_user} blocks for {N} disjoint users, " f"have {max_num_blocks}"
    )
    page_table_torch = torch.arange(N * blocks_per_user, dtype=torch.int32).reshape(N, blocks_per_user)
    # Sanity: provably no physical block is shared across users.
    assert (
        page_table_torch.unique().numel() == page_table_torch.numel()
    ), "page-table physical blocks must be distinct across all users (no overlap)"
    # Batch-DP (seam A/B) reads a COLUMN-sharded page_table/current_pos (8 users/col,
    # dims=(None,-2)/(None,0)) — match the model's QWEN36_FA_BATCH_DP gate. The KV
    # cache is col-replicated so col c's read of users [8c:8c+8] hits filled blocks.
    # FA_BATCH_DP=0 → replicated (legacy path).
    _fa_bdp = os.environ.get("QWEN36_FA_BATCH_DP", "1") != "0"
    page_table_tt = ttnn.from_torch(
        page_table_torch,
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=ttnn.ShardTensor2dMesh(
            bh_glx_mesh, dims=((None, -2) if _fa_bdp else (None, None)), mesh_shape=args.cluster_shape
        ),
    )

    # ---- Seed the paged KV cache directly via ``from_torch`` (NOT TT prefill).
    #      This MIRRORS the PROVEN seeding in ``test_sdpa_decode_batch32_probe.py``
    #      (which passes 32/32): build ONE user's keys/values for a 128-token
    #      context, then write that SAME content into EVERY user's DISTINCT
    #      physical blocks (distinct blocks per user, IDENTICAL content). We reuse
    #      the smoke's already-built ``page_table_torch`` (same handle decode uses)
    #      so the decode step reads exactly the blocks we filled. This isolates the
    #      DECODE path from the per-user TT-prefill seeding path.
    #
    #      The model's KV cache lives in ``attn.layer_past`` (kv_cache=None at decode
    #      ⇒ the model reads ``self.layer_past``). Its shape/dtype/sharding are set
    #      in ``llama_attention.init_kv_cache`` for qwen3.6:
    #        cache shape per-mesh = [max_num_blocks, n_kv_full=8, block_size, head_dim]
    #        row-sharded dims=(1, None) (n_kv across cluster rows), bfloat16, TILE, DRAM.
    #      The probe builds this exact structure; we replicate it here and overwrite
    #      ``attn.layer_past`` in place.
    from models.demos.qwen3_6_galaxy_v2.tt.llama_attention import _qwen36_kv_cache_dtype

    attn = model.layers[0].attention
    n_kv_full = attn.n_kv_heads  # 8 (padded); cache dim=1 sharded across cluster rows
    head_dim = attn.head_dim  # 256
    block_size = _PAGED_BLOCK_SIZE  # 32
    n_blocks_per_user_fill = (_T_PREFILL + block_size - 1) // block_size  # ceil(128/32)=4

    # ONE shared random keys/values stream for a fixed _T_PREFILL-token context.
    torch.manual_seed(44)
    ctx_pad = n_blocks_per_user_fill * block_size  # 128
    k_real = torch.randn(ctx_pad, head_dim, dtype=torch.float32) * 0.5
    v_real = torch.randn(ctx_pad, head_dim, dtype=torch.float32) * 0.5

    # Fill every user's physical blocks with the SAME content (replicated across all
    # n_kv_full head slots so each row-device holds it) — exactly the probe's fill.
    cache_k = torch.zeros(paged_attention_config.max_num_blocks, n_kv_full, block_size, head_dim)
    cache_v = torch.zeros(paged_attention_config.max_num_blocks, n_kv_full, block_size, head_dim)
    for u in range(N):
        for vblk in range(n_blocks_per_user_fill):
            phys = int(page_table_torch[u, vblk])
            t0 = vblk * block_size
            cache_k[phys, :, :, :] = k_real[t0 : t0 + block_size].unsqueeze(0)
            cache_v[phys, :, :, :] = v_real[t0 : t0 + block_size].unsqueeze(0)

    row_shard_kv = ttnn.ShardTensor2dMesh(bh_glx_mesh, dims=(1, None), mesh_shape=args.cluster_shape)
    _kv_dtype = _qwen36_kv_cache_dtype()
    for old in attn.layer_past:
        old.deallocate(True)
    attn.layer_past = [
        ttnn.from_torch(
            kv,
            device=bh_glx_mesh,
            dtype=_kv_dtype,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=row_shard_kv,
        )
        for kv in [cache_k, cache_v]
    ]
    print(
        f"[fa-N{N}] from_torch-seeded paged KV (distinct blocks/user, IDENTICAL content) "
        f"for {N} users; ctx={ctx_pad}, blocks/user={n_blocks_per_user_fill}"
    )

    # ---- ONE batched decode step. ALL N rows carry the SAME input; SAME pos.
    #      The decode input is a single shared hidden-state row, replicated to all N
    #      users (identical-in). Its exact value is irrelevant to the oracle — only
    #      that every row is identical (and the seeded KV is identical per user).
    one_row = (torch.randn(1, _H, dtype=torch.float32) * 0.5).bfloat16()  # [1, H]
    decode_rows = one_row.repeat(N, 1)  # [N, H], every row identical
    x_decode_tt = _send_col_sharded_decode_rows(decode_rows, bh_glx_mesh, args)
    cos_dec, sin_dec = _build_partial_rope_cos_sin_tt(bh_glx_mesh, torch.tensor([_T_PREFILL], dtype=torch.long))
    cur_pos_tt = ttnn.from_torch(
        torch.tensor([_T_PREFILL] * N, dtype=torch.int32),
        device=bh_glx_mesh,
        dtype=ttnn.int32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_mapper=(
            ttnn.ShardTensor2dMesh(bh_glx_mesh, dims=(None, 0), mesh_shape=args.cluster_shape)
            if _fa_bdp
            else ttnn.ReplicateTensorToMesh(bh_glx_mesh)
        ),
    )
    tt_out = model.forward(
        x_decode_tt,
        current_pos=cur_pos_tt,
        rot_mats=(cos_dec, sin_dec),
        user_id=0,
        mode="decode",
        page_table=page_table_tt,
        chunk_page_table=None,
        chunk_start_idx=None,
        start_pos=_T_PREFILL,
        get_last_token=-1,
        kv_cache=None,
        batch_size=N,
    )
    tt_logits = _gather_decode_logits(tt_out, bh_glx_mesh, args)  # [N_pad, vocab]
    _assert_all_rows_match_row0(tt_logits, N, tag=f"fa-N{N}")
