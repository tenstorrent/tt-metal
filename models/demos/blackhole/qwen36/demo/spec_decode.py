# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Speculative decoding for Qwen3.6-27B using the built-in MTP drafter.

Architecture:
  Drafter  = mtp.* module (fc-fusion + 1 full-attention transformer layer)
  Target   = backbone (64-layer Qwen3.6 with interleaved GDN + full-attn layers)

Batched verify (one pass for K+1 tokens):
  1. Draft K tokens via MTP chaining — zero backbone GDN advance.
  2. ONE backbone prefill on K+1 tokens → all-position logits + h_last.
  3. Accept j tokens (greedy or speculative sampling).
  4. GDN correction if j < K: restore S_prev + short j+1-token prefill.
  5. Advance position by j+1.

GDN state management:
  - Draft: GDN stays at S_prev (MTP doesn't touch backbone GDN).
  - Verify: GDN → S_{K+1}.
  - Correction (j<K): GDN → S_{j+1} via j+1-token prefill from S_prev.
  - j==K: no correction needed; GDN at S_{K+1} is correct.

Expected speedup at K=3, α=0.6: ~2.1× over non-speculative decode.
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import torch

import ttnn

# Prefill SDPA's q_chunk_size is computed as T / num_cores and must be >= TILE_SIZE=32.
# With the backbone's default 8x8 grid, we need T >= 32*16 = 512... but chunk_size=128
# limits SDPA to at most one 128-token chunk, so T=128 is both sufficient and correct.
_MIN_PREFILL_T = 128


def _pad_tokens(token_ids: torch.Tensor, valid_len: int):
    """Pad token_ids to max(T, _MIN_PREFILL_T) if needed. valid_len is unchanged."""
    T = token_ids.shape[1]
    if T < _MIN_PREFILL_T:
        token_ids = torch.cat([token_ids, token_ids.new_zeros(1, _MIN_PREFILL_T - T)], dim=1)
    return token_ids


# ---------------------------------------------------------------------------
# Acceptance helpers
# ---------------------------------------------------------------------------


def _to_probs(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """[vocab] logits → probability distribution."""
    if temperature <= 0.0:
        p = torch.zeros_like(logits)
        p[logits.argmax()] = 1.0
        return p
    return torch.softmax(logits / temperature, dim=-1)


def _sample(logits: torch.Tensor, temperature: float) -> int:
    if temperature <= 0.0:
        return int(logits.argmax().item())
    return int(torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1).item())


def accept_tokens(
    draft_tokens: List[int],
    verify_logits: torch.Tensor,
    temperature: float = 0.0,
) -> Tuple[List[int], int]:
    """Accept a prefix of draft_tokens based on target verify_logits.

    verify_logits[k] = target distribution at verify step k (predicts token at pos+k+1).
    draft_tokens[k] is the draft token for position pos+k+1.

    Returns (accepted_list, j) where j tokens were accepted and accepted_list
    has j+1 entries (j accepted drafts + 1 bonus/fallback token).
    """
    K = len(draft_tokens)
    for k in range(K):
        target = verify_logits[k]
        if temperature <= 0.0:
            ok = target.argmax().item() == draft_tokens[k]
        else:
            p = _to_probs(target, temperature)
            q = _to_probs(torch.zeros_like(target), temperature)  # uniform fallback
            ratio = (p[draft_tokens[k]] / (p[draft_tokens[k]] + 1e-9)).clamp(max=1.0)
            ok = torch.rand(1).item() < ratio.item()
        if not ok:
            fallback = _sample(target, temperature)
            return draft_tokens[:k] + [fallback], k
    # All K accepted: bonus token from verify_logits[K]
    bonus = _sample(verify_logits[K], temperature)
    return draft_tokens + [bonus], K


# ---------------------------------------------------------------------------
# GDN snapshot / restore
# ---------------------------------------------------------------------------


class GDNStateManager:
    """Snapshot/restore for Qwen3.6 GDN recurrent state across TP devices."""

    def __init__(self, model, mesh):
        self._mesh = mesh
        self._gdn = [layer.attention for layer in model.layers if not layer.is_full_attention]

    def snapshot(self) -> list:
        comp = ttnn.ConcatMeshToTensor(self._mesh, dim=0)
        return [
            (
                ttnn.to_torch(dn.rec_state, mesh_composer=comp),
                [ttnn.to_torch(c, mesh_composer=comp) for c in dn.conv_states],
            )
            for dn in self._gdn
        ]

    def restore(self, snap: list):
        mapper = ttnn.ShardTensorToMesh(self._mesh, dim=0)

        def _back(t, dtype):
            return ttnn.from_torch(t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=self._mesh, mesh_mapper=mapper)

        for dn, (rec, convs) in zip(self._gdn, snap):
            r = _back(rec, dn.rec_state.dtype)
            ttnn.copy(r, dn.rec_state)
            ttnn.deallocate(r)
            for j, c in enumerate(convs):
                cc = _back(c, dn.conv_states[j].dtype)
                ttnn.copy(cc, dn.conv_states[j])
                ttnn.deallocate(cc)


# ---------------------------------------------------------------------------
# Verify + GDN correction
# ---------------------------------------------------------------------------


def batched_verify(
    model,
    current_tok: int,
    draft_tokens: List[int],
    gdn_mgr: GDNStateManager,
    s_prev: list,
    page_table=None,
    pos: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ONE backbone prefill on [current_tok, d_0..d_{K-1}], returning all-position logits.

    Uses model.prefill_tp(return_all_logits=True) — one forward pass through all 64
    layers, then norm+lm_head on all K+1 positions. This is the "batched" in batched verify.

    Returns:
        (verify_logits, h_last)
          verify_logits: [K+1, vocab] — verify_logits[k] predicts token at pos+k+1.
          h_last: device tensor [1,1,1,dim_frac] — sharded backbone hidden at pos+K.
    """
    gdn_mgr.restore(s_prev)
    K = len(draft_tokens)
    tokens = torch.tensor([[current_tok] + list(draft_tokens)], dtype=torch.long)  # [1, K+1]
    tokens = _pad_tokens(tokens, K + 1)
    all_logits, h_last = model.prefill_tp(
        tokens,
        valid_len=K + 1,
        return_all_logits=True,
        page_table=page_table,
        chunk_start_idx=pos,
    )
    return all_logits, h_last  # [K+1, vocab], [1,1,dim]


def gdn_correction_pass(
    model,
    current_tok: int,
    accepted_tokens: List[int],
    j: int,
    gdn_mgr: GDNStateManager,
    s_prev: list,
    page_table=None,
    pos: int = 0,
):
    """Advance GDN from S_prev by j+1 tokens to reach S_{j+1}.

    Called when j < K. Runs a short prefill on [current_tok, accepted[0..j-1]]
    (j+1 tokens) so GDN lands at S_{j+1} = the correct state after accepting j drafts.
    The KV writes are identical to verify pass 1's writes for those positions.
    """
    gdn_mgr.restore(s_prev)
    if j == 0:
        return  # S_prev is already the correct state
    correction = [current_tok] + list(accepted_tokens[:j])
    valid_len = len(correction)
    tokens = _pad_tokens(torch.tensor([correction], dtype=torch.long), valid_len)
    model.prefill_tp(tokens, valid_len=valid_len, page_table=page_table, chunk_start_idx=pos)


# ---------------------------------------------------------------------------
# Main speculative decode loop
# ---------------------------------------------------------------------------


def speculative_decode_loop(
    model,
    mtp_drafter,
    mesh,
    nxt: int,
    pos: int,
    max_tokens: int,
    K: int = 3,
    temperature: float = 0.0,
    page_table=None,
    warmup_tokens: Optional[List[int]] = None,
) -> Tuple[List[int], dict]:
    """Speculative decode loop for Qwen3.6-27B using the built-in MTP drafter.

    Per-iteration cost breakdown:
      Draft:       0 backbone steps + K MTP steps (negligible vs backbone)
      Verify:      1 backbone prefill on K+1 tokens
      Correction:  0 (j==K) or 1 short prefill on j+1 tokens (j<K, prob 1-α)
      Total GPU:   1 + (1-α) × j/(K+1) backbone-prefill-equivalents per iteration
      Tokens out:  1 + K×α  per iteration
      Speedup:     (1 + K×α) / (1 + (1-α)×j/(K+1)) ≈ 2× at K=3, α=0.6

    Priming (first iteration): runs one 1-token prefill to get h_backbone.

    Args:
        model: Qwen36Model with model.mtp != None (QWEN36_SPEC_DECODE=1).
        mtp_drafter: model.mtp (Qwen36MTPTTModule — on-device TTNN drafter).
        mesh: ttnn mesh device.
        nxt: first decode token (last token from prefill).
        pos: current sequence position (= prompt length T).
        max_tokens: max new tokens to generate.
        K: draft tokens per iteration (3-4 recommended).
        temperature: 0=greedy, >0=sampling.
        page_table: torch int32 [1, num_blocks] — paged KV page table (same one used for
          prefill and traced decode). Passed to prefill_tp so verify/correction use the
          correct paged KV path instead of the concat-KV path.
        warmup_tokens: prompt token IDs (list[int]) used to warm up the MTP KV cache.
          When provided, mtp_drafter.prefill() is called before the main loop so the
          MTP attention has context for positions 0..T-1.

    Returns:
        (generated_tokens, perf_dict)
    """
    gdn_mgr = GDNStateManager(model, mesh)
    generated: List[int] = []
    mtp_drafter.reset_kv_cache()

    # Warm up MTP KV cache over the prompt (positions 0..pos-1)
    if warmup_tokens is not None and hasattr(mtp_drafter, "prefill"):
        mtp_drafter.prefill(warmup_tokens, pos_start=0)

    total_drafts = 0
    total_accepted = 0
    decode_times: List[float] = []
    h_backbone = None  # device tensor [1,1,1,dim_frac], primed first iteration

    while len(generated) < max_tokens:
        t0 = time.time()

        # ------------------------------------------------------------------
        # Prime: get backbone hidden for MTP chaining (one-time, first iter only)
        # ------------------------------------------------------------------
        if h_backbone is None:
            tok_t = _pad_tokens(torch.tensor([[nxt]], dtype=torch.long), 1)
            logits_prime, h_backbone = model.prefill_tp(
                tok_t,
                valid_len=1,
                return_hidden=True,
                page_table=page_table,
                chunk_start_idx=pos,
            )
            nxt = _sample(logits_prime, temperature)
            generated.append(nxt)
            pos += 1
            decode_times.append(time.time() - t0)
            continue

        # ------------------------------------------------------------------
        # 1. Snapshot GDN — draft phase leaves it unchanged
        # ------------------------------------------------------------------
        s_prev = gdn_mgr.snapshot()

        # ------------------------------------------------------------------
        # 2. Draft K tokens via MTP chaining (zero backbone GDN advance)
        # ------------------------------------------------------------------
        draft_tokens, _ = mtp_drafter.chain(h_backbone, nxt, K, pos_start=pos)
        total_drafts += K

        # ------------------------------------------------------------------
        # 3. ONE batched verify: K+1-token prefill → all-position logits
        # ------------------------------------------------------------------
        verify_logits, h_last = batched_verify(
            model,
            nxt,
            draft_tokens,
            gdn_mgr,
            s_prev,
            page_table=page_table,
            pos=pos,
        )
        # verify_logits[k] predicts the token at position pos+k+1.
        # h_last = backbone hidden at pos+K (last verify position).

        # ------------------------------------------------------------------
        # 4. Accept/reject
        # ------------------------------------------------------------------
        accepted, j = accept_tokens(draft_tokens, verify_logits, temperature)
        # accepted has j+1 entries: j accepted drafts + 1 bonus/fallback.
        total_accepted += j

        # ------------------------------------------------------------------
        # 5. GDN correction when j < K (verify advanced GDN by K+1, need j+1)
        # ------------------------------------------------------------------
        if j < K:
            gdn_correction_pass(
                model,
                nxt,
                accepted,
                j,
                gdn_mgr,
                s_prev,
                page_table=page_table,
                pos=pos,
            )
            # h_backbone for next iter: re-prime on the last accepted token.
            # GDN is now at S_{j+1}; this 1-token prefill is causally correct.
            last_accepted_tok = accepted[j - 1] if j > 0 else nxt
            tok_t = _pad_tokens(torch.tensor([[last_accepted_tok]], dtype=torch.long), 1)
            _, h_backbone = model.prefill_tp(
                tok_t,
                valid_len=1,
                return_hidden=True,
                page_table=page_table,
                chunk_start_idx=pos + j,
            )
        else:
            # All K accepted: GDN from verify is at S_{K+1} (correct).
            # h_last is the backbone hidden at the last accepted position (pos+K).
            h_backbone = h_last

        # ------------------------------------------------------------------
        # 6. Record tokens and advance position
        # ------------------------------------------------------------------
        for tok in accepted:
            generated.append(tok)
            pos += 1
            if len(generated) >= max_tokens:
                break
        nxt = generated[-1]
        decode_times.append(time.time() - t0)

    acceptance_rate = total_accepted / max(total_drafts, 1)
    # Drop the priming step from timing (index 0 is the 1-token prime)
    steady = decode_times[2:] if len(decode_times) > 2 else decode_times[1:]
    avg_iter_s = (sum(steady) / len(steady)) if steady else float("inf")
    spec_iters = max(len(decode_times) - 1, 1)
    avg_tok_per_iter = (total_accepted + spec_iters) / spec_iters

    return generated, {
        "spec_acceptance_rate": acceptance_rate,
        "spec_K": K,
        "spec_avg_accepted_per_iter": avg_tok_per_iter,
        "decode_iter_s": avg_iter_s,
        "decode_tok_s": avg_tok_per_iter / avg_iter_s if avg_iter_s > 0 else 0.0,
    }
