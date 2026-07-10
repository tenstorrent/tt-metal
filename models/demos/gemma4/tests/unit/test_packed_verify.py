# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Packed-query verify (spec decode) vs the sequential and batch-dim verifies.

The packed verify folds the K+1 candidates into the query-heads dim of ONE
batch=1 forward (head-major packed SDPA with an additive mask) and writes the
new KV loop-free through the persistent staging + paged_fill_cache path. Greedy
argmax must match the sequential single-token verify chain position-for-position
(up to the same near-tie caveat as the batch-dim verify); the loop-free KV write
must also leave the cache state correct across iterations (a chain of packed
verifies, with rollovers across page boundaries, keeps matching sequential).

Requires HF_MODEL (target). The drafter is irrelevant here.
"""

import math
import os

import pytest
import torch
from loguru import logger

import ttnn

from ...tests.test_factory import parametrize_mesh_with_fabric


def _is_moe_model(model_path):
    """True for Mixture-of-Experts checkpoints (e.g. gemma-4-26B-A4B).

    Packed verify folds the K+1 candidates into ONE multi-token forward, so it
    drives the model with seq_len = P = draft_len+1 (e.g. 5) — not a multiple of
    32. On an MoE checkpoint the experts block routes any seq_len != 1 through
    its prefill path, which asserts ``seq_len % 32 == 0`` (see
    gemma4/tt/experts/__init__.py). Dense models (12B, 31B) have no experts block
    and are unaffected. Padding P up to 32 would run the experts on 32 tokens per
    verify and erase the packed-verify win, so packed verify is unsupported on
    MoE. Detect it cheaply from the config (no weights loaded) so we can skip
    before ``from_pretrained`` — which also avoids writing the weight cache on the
    read-only NAS mount used by CI e2e.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        return bool(getattr(tc, "num_experts", 0))
    except Exception:
        return False


_MOE_UNSUPPORTED_REASON = (
    "packed verify drives the model with seq_len=P=draft_len+1 (not a multiple of 32); "
    "MoE experts prefill requires seq_len%32==0, so packed verify is unsupported on MoE "
    "checkpoints (e.g. gemma-4-26B-A4B). Dense targets (12B, 31B) run this test."
)


def _is_pli_model(model_path):
    """True for per-layer-input (MatFormer) checkpoints (e.g. gemma-4-E2B/E4B).

    These models carry a per-layer input embedding (``hidden_size_per_layer_input``)
    that must be fed to every layer. Packed verify drives ``self(...)`` through
    ``ttnn_packed_verify_forward`` without threading PLI, so a PLI model falls into
    ``Gemma4Model._compute_per_layer_inputs(None, None)`` and raises. PLI is not
    wired through the speculative-verify path yet, so packed verify is unsupported
    on these checkpoints (tracked in #49022). Dense non-PLI targets (12B, 31B) are
    unaffected. Detect it cheaply from the config (no weights loaded) so we can skip
    before ``from_pretrained`` — which also avoids writing the weight cache on the
    read-only NAS mount used by CI e2e.
    """
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        tc = getattr(cfg, "text_config", cfg)
        return bool(getattr(tc, "hidden_size_per_layer_input", 0))
    except Exception:
        return False


_PLI_UNSUPPORTED_REASON = (
    "packed verify does not thread per-layer inputs (PLI) through the verify forward, so "
    "PLI/MatFormer checkpoints (e.g. gemma-4-E2B/E4B) hit _compute_per_layer_inputs(None, None) "
    "and raise. Unsupported until PLI is wired through the speculative-verify path (see #49022). "
    "Dense non-PLI targets (12B, 31B) run this test."
)

_L1_OVERFLOW_REASON = (
    "packed-verify global-layer SDPA (head_dim=512, fp32-accum, P candidates folded into "
    "heads) exceeds this core's L1 at the current TP. The per-core footprint shrinks ~TP×, "
    "so larger targets need a larger TP mesh: e.g. gemma-4-31B overflows L1 at TP=4 "
    "(4-chip box) but fits at TP>=8. 12B fits at TP=4. Skipping on this SKU/mesh."
)


def _is_l1_cb_overflow(exc):
    """True if `exc` is the ttnn L1 circular-buffer capacity throw.

    This is a deterministic, host-side program-validation failure (raised before
    any on-device launch, so the device stays healthy), matching e.g.:
        "Statically allocated circular buffers on core range [...] grow to
         2005824 B which is beyond max L1 size of 1572864 B".
    Keyed off the message text so it stays model/SKU/mesh-agnostic.
    """
    s = str(exc).lower()
    return "circular buffer" in s and "l1 size" in s


@parametrize_mesh_with_fabric()
def test_packed_verify_matches_sequential(mesh_device, reset_seeds):
    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")

    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    if _is_pli_model(model_path):
        pytest.skip(_PLI_UNSUPPORTED_REASON)

    # Single-device (TP=1) is unsupported for packed verify on the 12B model: the
    # global layers (global_head_dim=512) run the packed SDPA with fp32_dest_acc_en
    # (the odd-PNHt MUL_BCAST power-of-2 fix) AND cannot head-split (nkv_local>1 ⇒ no
    # single shared KV head), so the per-core flash cross-reduction CBs overflow the
    # 1.5 MB L1 (~2.8 MB). Capping the reduction fan-in fits L1 but deadlocks the SDPA
    # kernel on-device. Supported on multi-device (TP) meshes, where TP shrinks the
    # per-core packed-head footprint ~TP×.
    if mesh_device.get_num_devices() == 1:
        pytest.skip(
            "single-device L1 limit: 12B global-layer packed SDPA exceeds single-device L1 (use a TP mesh, e.g. 1x4)"
        )

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    max_seq_len = 1024
    block_size = 64
    paged_attention_config = PagedAttentionConfig(
        block_size=block_size, max_num_blocks=math.ceil(max_seq_len / block_size)
    )
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)
    prompt = "The capital of France is"
    in_pt, encoded, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, 24, max_prefill_len=max_seq_len
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    anchor_token = int(encoded[0][prefill_lens[0] - 1])
    anchor_pos = prefill_lens[0] - 1

    # assistant_model unused — only the verify paths run.
    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=4,
    )
    P = spec.draft_len + 1

    def _prefill():
        # warmup_prefill=False skips the prefill-trace warmup (and its on-device
        # sampling/penalty sweep). This correctness test reads logits to host and
        # does argmax itself, so device sampling is irrelevant here; skipping the
        # warmup also avoids an unrelated TP>1 penalty-path shape mismatch that
        # otherwise aborts before the verify code runs.
        generator.prefill_forward_text(
            in_pt,
            page_table=page_table,
            kv_cache=tt_kv_cache,
            prompt_lens=decoding_pos,
            warmup_prefill=False,
        )

    # ── Reference: sequential single-token verify chain (batch=1 takes the
    # plain verify path — packing only engages for multi-token calls).
    _prefill()
    seq = []
    tok, pos = anchor_token, anchor_pos
    for _ in range(3 * P):  # several iterations ⇒ exercises staging rollovers
        logits, h = spec._verify([tok], [pos])
        h.deallocate(True)
        tok = int(torch.argmax(logits[0]))
        seq.append(tok)
        pos += 1

    # ── Packed: chain of packed verifies committing the matched chain tokens.
    _prefill()
    spec._pv_a_prev = -1
    packed = []
    pos = anchor_pos
    chain = [anchor_token] + seq
    it = 0
    while len(packed) < 3 * P:
        tokens = chain[it * P : it * P + P]  # the already-verified greedy chain
        positions = [pos + j for j in range(P)]
        try:
            lh, h = spec._verify(tokens, positions)  # first call compiles the packed SDPA
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        h.deallocate(True)
        packed.extend(int(torch.argmax(lh[j])) for j in range(P))
        pos += P
        it += 1
    packed = packed[: len(seq)]

    logger.info(f"sequential: {seq}")
    logger.info(f"packed:     {packed}")
    n_match = sum(1 for a, b in zip(seq, packed) if a == b)
    logger.info(f"match {n_match}/{len(seq)}")
    assert packed == seq, f"packed verify diverges from sequential: packed={packed} seq={seq}"


# ───────────────────────── batched (B>1) packed-verify bench ─────────────────
#
# The PR's KV-bandwidth win lives entirely in the verify step: a packed verify
# reads each user's KV ONCE (batch dim = B, the P=K+1 candidates folded into the
# query-heads), whereas the batch-alias verify spends one full KV read per
# pseudo-user (batch dim = B*(K+1)). At batch=1 the decode is weight-bound, so
# this never shows; only once weights are amortized across a batch does per-user
# cost become KV-bound and the B-vs-B*(K+1) read ratio matter.
#
# This bench drives ``ttnn_packed_verify_forward`` directly at B>1. The packed
# attention path (``packed_decode_forward`` → ``_packed_verify_sdpa``) is already
# B-aware (``B = rows // P``, page_table ``[B, blocks]``, mask ``[B, 1, H*P,
# S_k]``); the only batch=1 lock-ins are the spec-decode host builders and the
# loop-free *staging* KV write. We sidestep the staging (a write-side optimization
# that is irrelevant to the read amortization and dwarfed by the full-context read
# at long ctx) by using the per-position ``paged_update_cache`` *fallback* write
# (``kv_write_idxs``, no staging), so this isolates exactly the SDPA read claim.
#
# Asymmetry that *is* the batch>1 story: the batch-alias verify rides the ordinary
# decode path, whose per-user position vector is capped at 32 → ``B*(K+1) <= 32``
# ⇒ B<=8 at P=4. The packed verify keeps batch=B and folds P into heads, so it
# scales to B=32 where a single-pass batch-alias verify cannot even run; there the
# honest baseline is P sequential batch-B decodes (the cost of verifying P
# positions without packing).
#
# Env knobs: GEMMA4_BENCH_B="1,8" (comma list), GEMMA4_BENCH_CTX=2048 (prefill
# length per user), GEMMA4_SPEC_DRAFT_LEN=3 (K; P=K+1), GEMMA4_BENCH_ITERS=20.
# SKU-adaptive mesh (CI picks the largest that fits), matching matches_sequential.
# Pinning a fixed TP (e.g. 1x4) breaks on boxes whose NAS weight cache was built at
# a different TP: the demo populates the cache at the largest mesh's TP, so a TP4
# pin on an 8-chip box misses the cache and tries to *write* it into the read-only
# mount. trace_region_size is needed for this test's trace capture.
@parametrize_mesh_with_fabric(device_params_extra={"trace_region_size": 256_000_000})
def test_packed_verify_batch_perf(mesh_device, reset_seeds):
    import time

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    if _is_moe_model(model_path):
        pytest.skip(_MOE_UNSUPPORTED_REASON)

    if _is_pli_model(model_path):
        pytest.skip(_PLI_UNSUPPORTED_REASON)
    if mesh_device.get_num_devices() == 1:
        pytest.skip("single-device L1 limit: use a TP mesh (e.g. 1x4)")

    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig

    # CI is a shared, time-boxed runner: perf numbers there are meaningless and
    # the full sweep (B up to 32, ctx=2048, 20 reps) on a 31B target blows the
    # job timeout (65k-token prefill + 4 B-values × 20 reps × 2 paths). Under
    # CI=true default to a light sweep so this still runs as a B>1 packed-verify
    # smoke/correctness check; local runs keep full fidelity. Any explicit
    # GEMMA4_BENCH_* override wins in both cases. NOTE: ctx must keep
    # max_seq_len >= the sliding window (1024) — the sliding-window decode slices
    # `window` keys from the cache, so a shorter cache overruns it. ctx=1024 gives
    # max_seq_len=1088 (>= 1024); do not lower it below the window.
    _ci = os.getenv("CI") == "true"
    Bs = [int(b) for b in os.environ.get("GEMMA4_BENCH_B", "1,8" if _ci else "1,8,16,32").split(",") if b.strip()]
    ctx = int(os.environ.get("GEMMA4_BENCH_CTX", 1024 if _ci else 2048))
    K = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
    P = K + 1
    reps = int(os.environ.get("GEMMA4_BENCH_ITERS", 3 if _ci else 20))
    # A packed-vs-alias argmax flip is a real bug only if the reference (alias)
    # preferred its token by more than this logit margin; smaller gaps are
    # near-ties that the ~1e-1 batched-SDPA noise is expected to flip.
    CONFIDENT_GAP = float(os.environ.get("GEMMA4_BENCH_CONFIDENT_GAP", 5.0))
    Bmax = max(Bs)

    block_size = 64
    # Key length S_k = max_seq_len: must cover positions 0..ctx+K and be a
    # multiple of the SDPA k_chunk (64). The additive mask zeroes out anything
    # past each row's causal bound, so over-allocating keys is harmless.
    max_seq_len = math.ceil((ctx + P) / block_size) * block_size
    blocks_per_user = max_seq_len // block_size
    paged_attention_config = PagedAttentionConfig(block_size=block_size, max_num_blocks=Bmax * blocks_per_user)

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=Bmax,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    target = generator.model[0]
    page_table_torch = create_tt_page_table(Bmax, paged_attention_config)  # [Bmax, blocks_per_user], distinct per user

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=None,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table_torch,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=K,
    )
    mapper = target._replicate_to_mesh_mapper()
    tp = target.mesh_config.tp if target.mesh_config else 1
    H = target.layers[0].self_attn.config.num_attention_heads // tp
    window = target.hf_config.sliding_window
    vocab = target.vocab_size
    NEG = -1e9

    # Prefill Bmax identical users to length `ctx` (distinct KV blocks per user
    # via the page table). Content is irrelevant to the bench — packed and
    # batch-alias read the SAME prefilled cache. warmup_prefill=False avoids the
    # TP>1 prefill-warmup sampling path (see the correctness test above).
    in_pt = torch.randint(low=10, high=2000, size=(Bmax, ctx), dtype=torch.int32)
    generator.prefill_forward_text(
        in_pt, page_table=page_table_torch, kv_cache=tt_kv_cache, prompt_lens=[ctx] * Bmax, warmup_prefill=False
    )

    c = ctx  # committed/anchor position; verify P fresh positions c..c+K
    S_k = max_seq_len
    tokens_per_user = [int(in_pt[0, 0])] + [int(in_pt[0, min(1 + j, ctx - 1)]) for j in range(K)]

    def _from(t, dtype, layout=ttnn.ROW_MAJOR_LAYOUT):
        return ttnn.from_torch(t, device=mesh_device, layout=layout, dtype=dtype, mesh_mapper=mapper)

    def _masks(B):
        # head-major rows h*P+p; per-user identical (same c) → tile across B.
        j = torch.arange(S_k)
        rf = torch.empty(P, S_k)
        rs = torch.empty(P, S_k)
        for p in range(P):
            upper = c + p
            rf[p] = torch.where(j <= upper, 0.0, NEG)
            rs[p] = torch.where((j <= upper) & (j > upper - window), 0.0, NEG)
        mf = rf.repeat(H, 1).reshape(1, 1, H * P, S_k).repeat(B, 1, 1, 1).to(torch.bfloat16)
        ms = rs.repeat(H, 1).reshape(1, 1, H * P, S_k).repeat(B, 1, 1, 1).to(torch.bfloat16)
        return _from(mf, ttnn.bfloat16, ttnn.TILE_LAYOUT), _from(ms, ttnn.bfloat16, ttnn.TILE_LAYOUT)

    def _run(call_fn, n_rows):
        """Compile (+read logits for correctness), capture, time `reps` replays."""
        logits, hidden = call_fn()
        ttnn.synchronize_device(mesh_device)
        lh = spec._logits_to_host(logits).reshape(n_rows, vocab).float().clone()
        logits.deallocate(True)
        hidden.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh_device, cq_id=0)
        logits, hidden = call_fn()
        ttnn.end_trace_capture(mesh_device, tid, cq_id=0)
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(reps):
            ttnn.execute_trace(mesh_device, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        ms = (time.perf_counter() - t0) / reps * 1e3
        ttnn.release_trace(mesh_device, tid)
        logits.deallocate(True)
        hidden.deallocate(True)
        return ms, lh

    logger.info(
        f"=== packed-verify batch bench | ctx={ctx} P={P} (K={K}) reps={reps} tp={tp} H_local={H} S_k={S_k} ==="
    )
    rows = []
    for B in Bs:
        pt_b = page_table_torch[:B].to(torch.int32)
        # ── packed: batch dim B, P folded into heads, fallback per-p KV write ──
        # rows user-major / position-minor: row u*P+p is user u's p-th candidate.
        x_p = _from(torch.tensor([tokens_per_user] * B, dtype=torch.int64).reshape(1, B * P), ttnn.uint32)
        pos_p = _from(torch.tensor([[c + p for u in range(B) for p in range(P)]], dtype=torch.int64), ttnn.uint32)
        mask_full, mask_slide = _masks(B)
        write_idxs = [_from(torch.full((B,), c + p, dtype=torch.int32), ttnn.int32) for p in range(P)]
        pt_packed = _from(pt_b, ttnn.int32)

        def _packed_call():
            return target.ttnn_packed_verify_forward(
                x=x_p,
                position_idx=pos_p,
                attn_mask_full=mask_full,
                attn_mask_sliding=mask_slide,
                packed_p=P,
                page_table=pt_packed,
                kv_cache=spec.tt_kv_cache,
                kv_write_idxs=write_idxs,
                embed_idx_full=None,
                embed_idx_sliding=None,
                hot_pt=None,
            )

        try:
            packed_ms, packed_lh = _run(_packed_call, B * P)  # first call compiles the packed SDPA
        except RuntimeError as e:
            if _is_l1_cb_overflow(e):
                pytest.skip(_L1_OVERFLOW_REASON)
            raise
        for t in (x_p, pos_p, mask_full, mask_slide, pt_packed, *write_idxs):
            t.deallocate(True)

        # ── batch-alias baseline ──────────────────────────────────────────────
        if B * P <= 32:
            # single-pass: B*P pseudo-users, each user's row replicated P times.
            x_a = spec._tokens_tensor([tokens_per_user[p] for u in range(B) for p in range(P)])
            pu_a, pi_a = spec._pos_tensors([c + p for u in range(B) for p in range(P)])
            pt_alias = _from(pt_b.repeat_interleave(P, dim=0), ttnn.int32)

            def _alias_call():
                return target.ttnn_verify_forward(
                    x=x_a, current_pos=pu_a, current_pos_cache=pi_a, page_table=pt_alias, kv_cache=spec.tt_kv_cache
                )

            alias_ms, alias_lh = _run(_alias_call, B * P)
            for t in (x_a, pu_a, pi_a, pt_alias):
                t.deallocate(True)
            # correctness: packed argmax vs batch-alias argmax per row. Both are
            # batched-SDPA paths whose per-user RoPE + cross-core reductions differ
            # by ~1e-1, so a divergence is only a bug at a CONFIDENT token — i.e.
            # when the reference (alias) logit gap between its argmax and packed's
            # pick is large. Near-ties (small gap) are expected to flip.
            pa = packed_lh.argmax(dim=-1)
            aa = alias_lh.argmax(dim=-1)
            n_match = int((pa == aa).sum())
            max_diff = float((packed_lh - alias_lh).abs().max())
            confident_flips = 0
            for r in range(B * P):
                if int(pa[r]) != int(aa[r]):
                    gap = float(alias_lh[r, int(aa[r])] - alias_lh[r, int(pa[r])])
                    if gap > CONFIDENT_GAP:
                        confident_flips += 1
            baseline_kind = f"alias(B*P={B*P})"
        else:
            # batch-alias can't fit B*(K+1)>32 users in one decode pass; the
            # no-packing cost is P sequential batch-B decodes.
            x_a = spec._tokens_tensor([tokens_per_user[0]] * B)
            pu_a, pi_a = spec._pos_tensors([c] * B)
            pt_alias = _from(pt_b, ttnn.int32)

            def _alias_call():
                return target.ttnn_verify_forward(
                    x=x_a, current_pos=pu_a, current_pos_cache=pi_a, page_table=pt_alias, kv_cache=spec.tt_kv_cache
                )

            one_ms, _ = _run(_alias_call, B)
            for t in (x_a, pu_a, pi_a, pt_alias):
                t.deallocate(True)
            alias_ms = one_ms * P
            n_match, max_diff, confident_flips = None, None, None
            baseline_kind = f"{P}x batch-{B} decode (B*(K+1)>32: no single-pass alias)"

        speedup = alias_ms / packed_ms if packed_ms > 0 else float("nan")
        rows.append((B, packed_ms, alias_ms, baseline_kind, speedup, n_match, max_diff, confident_flips))
        if n_match is not None:
            corr = f"corr {n_match}/{B*P} match, confident-flips={confident_flips} (max|Δlogit|={max_diff:.2f})"
        else:
            corr = "corr n/a (no single-pass alias reference at this B)"
        logger.info(
            f"[bench] B={B:>2}: packed={packed_ms:6.2f} ms | baseline({baseline_kind})={alias_ms:6.2f} ms | "
            f"speedup={speedup:4.2f}x | {corr}"
        )

    logger.info("=== summary (verify ms/iter; speedup = baseline/packed) ===")
    for B, p_ms, a_ms, kind, sp, nm, md, cf in rows:
        logger.info(f"  B={B:>2}  packed {p_ms:6.2f} ms  baseline {a_ms:6.2f} ms [{kind}]  →  {sp:4.2f}x")

    # Bug gate: a CONFIDENT-token flip (reference gap > CONFIDENT_GAP) is a real
    # divergence; near-tie flips under random-context noise are expected.
    for B, p_ms, a_ms, kind, sp, nm, md, cf in rows:
        if cf is not None:
            assert cf == 0, f"B={B}: {cf} confident-token flips packed vs batch-alias (max|Δlogit|={md:.2f})"
