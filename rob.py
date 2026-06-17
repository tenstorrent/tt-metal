# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Quick standalone harness for TPAttention prefill + decode (Qwen3.5 gated full-attention).

Builds one HF Qwen3_5Attention layer with random weights, loads those SAME weights into
the TP module, then PCC-checks the TT module against the HF golden. Two selectable modes
(set ROB_MODE; default "batched"):

  * "single" — the original flow: a length-SEQ_LEN causal prompt prefill (filling both the
    HF DynamicCache and the TT module's internal KV cache), then one decode token at
    position SEQ_LEN read straight off those prefill-filled caches. Exercises the
    prefill→decode hand-off a pure output PCC misses.
  * "batched" — pushes a real batch dimension through forward_prefill in ONE call:
    B_BATCH users, each a length-S_BATCH causal prompt, as a single [B,1,S,dim] tensor,
    then steps all B users through N_DECODE decode tokens together. The framework prefill
    convention is one user per call ([1,1,S,dim], looped over user_id), so this deliberately
    stresses whether the attention math (QKV head split, q/k-norm, partial RoPE, causal SDPA,
    gate, o_proj) AND the per-user batched KV-cache fill → multi-step decode hand-off hold for
    B>1. It builds its own module sized for B users (the shared one is max_batch_size=1).

  * "batched_paged" — the same B_BATCH×S_BATCH prefill + N_DECODE decode as "batched", but
    through a PAGED KV cache: positions are scattered across physical blocks via a
    [B, blocks_per_user] page table (a scrambled reverse-permutation, so a correct read can't
    rely on blocks being contiguous). The whole batch goes through forward_prefill in ONE call
    with the page table (the B>1 paged-fill assert was lifted — forward_prefill loops the
    paged_fill_cache over users internally); decode then steps the whole batch with the page
    table, as production (vLLM) does. This is the mode that validates the paged path end-to-end.

  * "both" — run single then batched.

It's the script form of tests/unit/test_attention.py (no pytest fixtures), so either op
can be poked at a breakpoint (set ROB_INTERACTIVE=1 to enable the breakpoints).
"""
import os

import torch
from loguru import logger

import ttnn
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention

from models.common.utility_functions import comp_pcc
from models.tt_transformers.tt.ccl import TT_CCL
from models.tt_transformers.tt.common import PagedAttentionConfig
from models.demos.blackhole.qwen3_5_9b.tt.attention import Qwen35Attention
from models.demos.blackhole.qwen3_5_9b.tt.attention.kv_cache import init_kv_cache
from models.demos.blackhole.qwen3_5_9b.tt.attention.rope_tp import rot_mats_decode, rot_mats_prefill
from models.demos.blackhole.qwen3_5_9b.tt.model_config import Qwen35ModelArgs
from models.demos.blackhole.qwen3_5_9b.tests.unit.reference import causal_mask, hf_rope

# HF_MODEL is the single source of truth for dims (Qwen35ModelArgs parses it). Default to the
# local FP8 snapshot that's actually on this host — a bare hub id ("Qwen/Qwen3.5-9B") would
# trigger a multi-GB snapshot_download. Override with `export HF_MODEL=...` to target another.
os.environ.setdefault("HF_MODEL", "Qwen/Qwen3.5-9B")

SEQ_LEN = 2026  # single-user prefill length
B_BATCH = 32  # batched-prefill: number of users pushed through one forward_prefill call
S_BATCH = 512  # batched-prefill: per-user prompt length
N_DECODE = 5  # batched: number of decode tokens stepped after the prefill
PAGE_BLOCK_SIZE = 64  # batched_paged: KV-cache page (block) size in token positions

MODE = os.environ.get("ROB_MODE", "batched")  # "single" | "batched" | "batched_paged" | "both"
INTERACTIVE = os.environ.get("ROB_INTERACTIVE", "0") == "1"


def maybe_break():
    """breakpoint() only when ROB_INTERACTIVE=1, so the script runs clean under `python rob.py`."""
    if INTERACTIVE:
        breakpoint()


def run_single(mesh_device, args, attn, hf_attn, cfg, rope, composer):
    """Original single-user prefill (length SEQ_LEN) → decode-one-token flow.

    Prefill fills both caches (HF DynamicCache + the TT internal cache); decode continues
    from exactly those filled caches, so this validates the prefill→decode hand-off too.
    """
    torch.manual_seed(0)
    x = torch.randn(1, SEQ_LEN, args.dim, dtype=torch.float32)

    # HF golden (causal). For text-only position_ids the interleaved M-RoPE reduces to the
    # standard partial RoPE that rope_tp builds, so the two are directly comparable. The
    # DynamicCache captures the post-RoPE K/V (positions 0..S-1) that decode reads back.
    cache = DynamicCache()
    with torch.no_grad():
        cos, sin = rope(x, torch.arange(SEQ_LEN).unsqueeze(0))
        ref, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=causal_mask(SEQ_LEN), past_key_values=cache)

    # TT prefill: x replicated as [1,1,S,dim]; cos/sin tables for positions 0..S-1.
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(1, 1, SEQ_LEN, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, SEQ_LEN, args.rope_theta)

    out = attn.forward_prefill(x_tt, cos_tt, sin_tt)
    out_torch = ttnn.to_torch(out, mesh_composer=composer)[0, 0].float()  # [S, dim]

    passing, pcc = comp_pcc(ref[0], out_torch, 0.99)
    logger.info(f"forward_prefill PCC (S={SEQ_LEN}) = {pcc}  passing={passing}")
    maybe_break()

    # ── Decode: one token past the prompt, continuing from the prefill-filled caches ──
    B = args.max_batch_size
    pos = SEQ_LEN
    x_dec = torch.randn(B, 1, args.dim, dtype=torch.float32)

    with torch.no_grad():
        cos_d, sin_d = rope(x_dec, torch.full((B, 1), pos, dtype=torch.long))
        ref_dec, _ = hf_attn(x_dec, position_embeddings=(cos_d, sin_d), attention_mask=None, past_key_values=cache)
    ref_dec = ref_dec[:, 0]  # [B, dim]

    x_dec_tt = ttnn.from_torch(
        x_dec.to(torch.bfloat16).reshape(1, 1, B, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    positions = torch.full((B,), pos, dtype=torch.int32)
    pos_tt = ttnn.from_torch(
        positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
    )
    cos_dec_tt, sin_dec_tt = rot_mats_decode(
        mesh_device, args.rope_head_dim, args.max_seq_len, args.rope_theta, positions
    )

    out_dec = attn.forward_decode(x_dec_tt, pos_tt, cos_dec_tt, sin_dec_tt)
    out_dec_torch = ttnn.to_torch(out_dec, mesh_composer=composer)[0, 0].float()  # [B, dim]

    passing_d, pcc_d = comp_pcc(ref_dec, out_dec_torch, 0.99)
    logger.info(f"forward_decode PCC (pos={pos}, B={B}) = {pcc_d}  passing={passing_d}")
    maybe_break()


def run_batched(mesh_device, args, state_dict, hf_attn, tt_ccl, rope, composer):
    """Batched prefill (B_BATCH users × S_BATCH) in ONE forward call, then N_DECODE decode steps.

    Feeds a real [B,1,S,dim] tensor (vs the framework's per-user [1,1,S,dim]) to stress whether
    forward_prefill's attention math holds for B>1, then steps all B users through several decode
    tokens together — exercising the per-user batched cache fill → multi-step decode hand-off that
    a pure prefill PCC can't see.

    The shared top-level module is sized max_batch_size=1 (its KV cache holds a single user), so
    this builds its own module sized for B users. That cache is kept snug (max_seq_len ≈ S+N_DECODE)
    because a full-length cache × B=32 users OOMs a single device (see the GDN batch>1 OOM note).
    """
    B, S = B_BATCH, S_BATCH
    logger.info(f"=== Batched prefill + decode: B={B}, S={S}, N_DECODE={N_DECODE}, dim={args.dim} ===")

    # Dedicated B-user module + cache. max_seq_len is rounded up to a tile (32) so the cache spans
    # exactly the S prompt positions plus the N_DECODE decode positions and nothing more.
    max_seq_b = ((S + N_DECODE + 31) // 32) * 32
    args_b = Qwen35ModelArgs(mesh_device, max_seq_len=max_seq_b, max_batch_size=B)
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=args_b,
        tt_ccl=tt_ccl,
        create_kv_cache=True,
    )

    torch.manual_seed(1)
    x = torch.randn(B, S, args.dim, dtype=torch.float32)

    # HF golden, batched. position_ids 0..S-1 per user; causal_mask is [1,1,S,S] (broadcasts over
    # batch). past_key_values=cache captures the post-RoPE K/V for positions 0..S-1 of all B users,
    # so the decode steps below continue from exactly the same history the TT cache holds.
    cache = DynamicCache()
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)  # [B, S]
    with torch.no_grad():
        cos, sin = rope(x, position_ids)
        ref, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=causal_mask(S), past_key_values=cache)
    # ref: [B, S, dim]

    # TT prefill: x as [B,1,S,dim] replicated to the mesh; the prefill RoPE tables are shared across
    # users (every user sees positions 0..S-1). Cache ENABLED this time so forward_prefill fills
    # slots 0..B-1 (one fill_cache per user) — the per-user batched fill the decode steps read back.
    # The fill is a side write that doesn't change the prefill output, so the PCC check is unaffected.
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(B, 1, S, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)

    out = attn.forward_prefill(x_tt, cos_tt, sin_tt)

    # forward_prefill returns [B,1,S,dim] (reduce-scattered along dim=3 for TP>1). On a 1-dev
    # mesh the composer's dim=0 fallback is a no-op; for TP>1 it concats the fractured dim.
    out_torch = ttnn.to_torch(out, mesh_composer=composer).float()
    logger.info(f"batched forward_prefill returned shape {tuple(out_torch.shape)}")
    out_torch = out_torch.reshape(B, S, args.dim)

    passing, pcc = comp_pcc(ref, out_torch, 0.99)
    logger.info(f"BATCHED forward_prefill PCC (B={B}, S={S}) = {pcc}  passing={passing}")

    # ── Decode: N_DECODE tokens past the prompt, all B users stepping together, continuing from
    #    the prefill-filled caches (HF DynamicCache + the TT internal cache). ──
    for step in range(N_DECODE):
        pos = S + step  # 0-based: prefill filled 0..S-1, so the first decode token sits at S
        x_dec = torch.randn(B, 1, args.dim, dtype=torch.float32)

        with torch.no_grad():
            cos_d, sin_d = rope(x_dec, torch.full((B, 1), pos, dtype=torch.long))
            ref_dec, _ = hf_attn(x_dec, position_embeddings=(cos_d, sin_d), attention_mask=None, past_key_values=cache)
        ref_dec = ref_dec[:, 0]  # [B, dim]

        # TT decode framework layout: x as [1,1,B,dim]; per-user positions [B] drive both the
        # cache-update index and the decode RoPE tables (sized to the snug args_b.max_seq_len).
        x_dec_tt = ttnn.from_torch(
            x_dec.to(torch.bfloat16).reshape(1, 1, B, args.dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        positions = torch.full((B,), pos, dtype=torch.int32)
        pos_tt = ttnn.from_torch(
            positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )
        cos_dec_tt, sin_dec_tt = rot_mats_decode(
            mesh_device, args.rope_head_dim, args_b.max_seq_len, args.rope_theta, positions
        )

        out_dec = attn.forward_decode(x_dec_tt, pos_tt, cos_dec_tt, sin_dec_tt)
        out_dec_torch = ttnn.to_torch(out_dec, mesh_composer=composer)[0, 0].float()  # [B, dim]

        passing_d, pcc_d = comp_pcc(ref_dec, out_dec_torch, 0.99)
        logger.info(f"  decode step {step} PCC (pos={pos}, B={B}) = {pcc_d}  passing={passing_d}")

    maybe_break()


def run_batched_paged(mesh_device, args, state_dict, hf_attn, tt_ccl, rope, composer):
    """Batched prefill + multi-step decode through a PAGED KV cache, to validate the page_table path.

    Same B_BATCH×S_BATCH prefill + N_DECODE decode as run_batched, but the KV cache is paged: every
    user's positions are scattered across physical blocks by a [B, blocks_per_user] page table, so a
    correct cache read MUST follow the indirection. The page table is a scrambled reverse-permutation
    (argsort of a randperm) — a bijection over physical blocks, so no two users share a block and a
    bug that ignored the mapping (e.g. assumed contiguous per-user storage) would tank PCC.

    forward_prefill now accepts B>1 with a page_table: it runs ONE fused [B,1,S,dim] prefill (batched
    QKV/norm/RoPE/causal-SDPA) and fills the paged cache with a per-user loop of paged_fill_cache
    (batch_idx=user_id+b selects user b's page-table row). So this pushes the real batched paged
    prefill in a single call, then steps the whole batch through the page table — exactly the
    production (vLLM) prefill→decode path.
    """
    B, S = B_BATCH, S_BATCH
    block_size = PAGE_BLOCK_SIZE
    # blocks_per_user must cover the S prompt positions plus the N_DECODE decode positions.
    blocks_per_user = (S + N_DECODE + block_size - 1) // block_size
    max_num_blocks = B * blocks_per_user
    paged_cfg = PagedAttentionConfig(block_size=block_size, max_num_blocks=max_num_blocks)
    logger.info(
        f"=== PAGED batched prefill + decode: B={B}, S={S}, N_DECODE={N_DECODE}, "
        f"block_size={block_size}, blocks/user={blocks_per_user}, max_num_blocks={max_num_blocks} ==="
    )

    # Scrambled [B, blocks_per_user] page table: each user's logical block i maps to a shuffled
    # physical block. reverse_permutation = argsort(randperm) is a bijection over all physical
    # blocks, so users never collide and the read can't accidentally be contiguous.
    generator = torch.Generator().manual_seed(0)
    permutation = torch.randperm(max_num_blocks, generator=generator)
    reverse_permutation = torch.argsort(permutation)
    page_table = reverse_permutation.reshape(B, blocks_per_user).to(torch.int32)
    page_table_tt = ttnn.from_torch(
        page_table,
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.int32,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    # Dedicated B-user module bound to a PAGED cache (shape [max_num_blocks, NKV, block_size, HD]).
    # create_kv_cache=False then bind the paged cache by hand — the contiguous and paged caches are
    # mutually exclusive, and init_kv_cache with a paged_attention_config builds the block-shaped one.
    max_seq_b = blocks_per_user * block_size
    args_b = Qwen35ModelArgs(mesh_device, max_seq_len=max_seq_b, max_batch_size=B)
    attn = Qwen35Attention(
        mesh_device=mesh_device,
        state_dict=state_dict,
        args=args_b,
        tt_ccl=tt_ccl,
        create_kv_cache=False,
    )
    attn.kv_cache = init_kv_cache(
        mesh_device=mesh_device,
        args=args_b,
        max_batch_size=B,
        max_seq_len=max_seq_b,
        paged_attention_config=paged_cfg,
        cache_dtype=ttnn.bfloat16,
    )

    torch.manual_seed(1)
    x = torch.randn(B, S, args.dim, dtype=torch.float32)

    # HF golden, batched (identical to run_batched): fills the DynamicCache for positions 0..S-1 of
    # all B users so the decode steps continue from the same history the paged cache holds.
    cache = DynamicCache()
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)  # [B, S]
    with torch.no_grad():
        cos, sin = rope(x, position_ids)
        ref, _ = hf_attn(x, position_embeddings=(cos, sin), attention_mask=causal_mask(S), past_key_values=cache)
    # ref: [B, S, dim]

    cos_tt, sin_tt = rot_mats_prefill(mesh_device, args.rope_head_dim, S, args.rope_theta)

    # Batched paged prefill in ONE call: the full [B,1,S,dim] tensor through forward_prefill with the
    # page table. forward_prefill now loops internally over the B users, filling user b's blocks via
    # paged_fill_cache(..., batch_idx=user_id+b), so user_id=0 seeds page-table rows 0..B-1 — exactly
    # the call production batched prefill will issue. (This used to be a per-user B=1 loop because the
    # paged branch asserted B==1; that assert is now lifted, so we exercise the real batched path.)
    x_tt = ttnn.from_torch(
        x.to(torch.bfloat16).reshape(B, 1, S, args.dim),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        dtype=ttnn.bfloat16,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )
    out = attn.forward_prefill(x_tt, cos_tt, sin_tt, page_table=page_table_tt, user_id=0)
    out_torch = ttnn.to_torch(out, mesh_composer=composer).float().reshape(B, S, args.dim)

    passing, pcc = comp_pcc(ref, out_torch, 0.99)
    logger.info(f"PAGED batched forward_prefill PCC (B={B}, S={S}) = {pcc}  passing={passing}")

    # ── Decode: N_DECODE tokens past the prompt, all B users stepping together, reading the PAGED
    #    cache through the page table (paged_update_cache + paged_scaled_dot_product_attention_decode). ──
    for step in range(N_DECODE):
        pos = S + step  # prefill filled 0..S-1, so the first decode token sits at S
        x_dec = torch.randn(B, 1, args.dim, dtype=torch.float32)

        with torch.no_grad():
            cos_d, sin_d = rope(x_dec, torch.full((B, 1), pos, dtype=torch.long))
            ref_dec, _ = hf_attn(x_dec, position_embeddings=(cos_d, sin_d), attention_mask=None, past_key_values=cache)
        ref_dec = ref_dec[:, 0]  # [B, dim]

        x_dec_tt = ttnn.from_torch(
            x_dec.to(torch.bfloat16).reshape(1, 1, B, args.dim),
            device=mesh_device,
            layout=ttnn.TILE_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            dtype=ttnn.bfloat16,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )
        positions = torch.full((B,), pos, dtype=torch.int32)
        pos_tt = ttnn.from_torch(
            positions, dtype=ttnn.int32, device=mesh_device, mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)
        )
        cos_dec_tt, sin_dec_tt = rot_mats_decode(
            mesh_device, args.rope_head_dim, args_b.max_seq_len, args.rope_theta, positions
        )

        out_dec = attn.forward_decode(x_dec_tt, pos_tt, cos_dec_tt, sin_dec_tt, page_table=page_table_tt)
        out_dec_torch = ttnn.to_torch(out_dec, mesh_composer=composer)[0, 0].float()  # [B, dim]

        passing_d, pcc_d = comp_pcc(ref_dec, out_dec_torch, 0.99)
        logger.info(f"  PAGED decode step {step} PCC (pos={pos}, B={B}) = {pcc_d}  passing={passing_d}")

    maybe_break()


# ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
mesh_device = ttnn.open_mesh_device(mesh_shape=ttnn.MeshShape(1, 1))

try:
    max_seq = max(SEQ_LEN, S_BATCH)
    args = Qwen35ModelArgs(mesh_device, max_seq_len=max_seq)

    # Build the HF reference from the SAME parsed config so its q/k/v/o dims match what args
    # (and the sharded TT weights) expect. eager attn so it honours the explicit causal mask;
    # state_dict keeps RAW q_norm/k_norm (HF adds the +1 internally, the TT loader bakes it in).
    cfg = args.hf_config.get_text_config()
    cfg._attn_implementation = "eager"
    hf_attn = Qwen3_5Attention(config=cfg, layer_idx=0).to(torch.float32).eval()
    state_dict = hf_attn.state_dict()

    # One CCL handle shared by both modes (run_batched builds its own attn module but reuses this).
    tt_ccl = TT_CCL(mesh_device)

    rope = hf_rope(cfg)
    composer = ttnn.ConcatMeshToTensor(mesh_device, dim=3 if mesh_device.get_num_devices() > 1 else 0)

    if MODE in ("single", "both"):
        # Single-user flow uses the default args (max_batch_size=1): its KV cache holds one user,
        # matching the single-user prefill→decode it runs.
        attn = Qwen35Attention(
            mesh_device=mesh_device,
            state_dict=state_dict,
            args=args,
            tt_ccl=tt_ccl,
            create_kv_cache=True,
        )
        run_single(mesh_device, args, attn, hf_attn, cfg, rope, composer)
    if MODE in ("batched", "both"):
        # Batched flow builds its own B-user module internally (the default one is too small), so
        # it only needs the shared state_dict + tt_ccl handed down.
        run_batched(mesh_device, args, state_dict, hf_attn, tt_ccl, rope, composer)
    if MODE == "batched_paged":
        # Paged variant of the batched flow — same per-user prefill + batched decode, but through a
        # paged KV cache + page table. Builds its own paged B-user module, so reuses state_dict + tt_ccl.
        run_batched_paged(mesh_device, args, state_dict, hf_attn, tt_ccl, rope, composer)
finally:
    ttnn.close_mesh_device(mesh_device)
    # ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)  # mirror the fixture's teardown
