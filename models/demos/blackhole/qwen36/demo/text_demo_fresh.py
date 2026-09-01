# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fresh, correctness-focused Qwen3.6-27B generation demo, with and without MTP.

Deliberately does NOT use allocate_kv_caches (paged KV) or the traced chunk-outer
prefill (capture_prefill_trace_chunked / _run_tp_generation in text_demo.py) --
this session found two distinct GDN bugs in that machinery on Wormhole T3K:
  1. the fused/phased GDN chunk kernel's returned state for continuing into decode
     (fixed here via QWEN_GDN_FUSED_CHUNK=0, see tt/gdn/fused_chunk.py);
  2. a second, still-unresolved bug specifically in GDN's _stable_state=True path
     (what allocate_kv_caches turns on for every GDN layer, for decode-trace address
     stability) -- proven independent of bug 1 by direct tensor-value comparison.
Qwen36Model.prefill_tp/decode_tp/generate_tp never touch _stable_state (no paged
KV), so with fix 1 applied they are a fully correct generation path -- confirmed via
generate_tp() matching the real HF model 10/10 tokens. Everything here builds on
those primitives directly instead of the production trace/paged path, so it needs
QWEN_GDN_FUSED_CHUNK=0 but is NOT exposed to bug 2 at all.

The MTP drafter is tt/mtp_fresh.py::Qwen36MTPDrafter, the from-scratch ttnn port
(independent of tt/mtp.py) validated against reference/mtp_torch.py earlier this
session. Verify here is intentionally simple and sequential (one real decode_tp-
equivalent step per draft candidate, comparing greedy picks) -- a correctness
reference for the draft/accept/reject mechanism, not the production one-pass-verify
performance design (that lives in tt/spec_decode.py, built on the currently-broken
paged path).

Run without MTP:
    QWEN_GDN_FUSED_CHUNK=0 MESH_DEVICE=T3K HF_HUB_OFFLINE=1 \
      pytest models/demos/blackhole/qwen36/demo/text_demo_fresh.py -v -s -k "no_mtp"

Run with MTP:
    QWEN_GDN_FUSED_CHUNK=0 MESH_DEVICE=T3K HF_HUB_OFFLINE=1 \
      pytest models/demos/blackhole/qwen36/demo/text_demo_fresh.py -v -s -k "with_mtp"
"""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import (
    _MESH_SHAPE,
    BLOCK_SIZE,
    DEVICE_PARAMS,
    _blocks_for,
    _get_prompt,
)
from models.demos.blackhole.qwen36.tt.model import Qwen36Model
from models.demos.blackhole.qwen36.tt.mtp_fresh import Qwen36MTPDrafter
from models.demos.blackhole.qwen36.tt.weight_mapping import (
    checkpoint_has_mtp,
    load_qwen36_mtp_state_dict,
    load_qwen36_shared_head_weights,
)


class _RealEmbeddingLookup:
    """Host-lookup embedding for the drafter, using the checkpoint's REAL embed_tokens
    table (replicated) -- tok_tt arrives already replicated across the mesh, so reading
    it back needs a mesh_composer on TP."""

    def __init__(self, table, mesh_device):
        self.table = table
        self.mesh_device = mesh_device
        nd = mesh_device.get_num_devices()
        self._mesh_kwargs = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if nd > 1 else {}
        self._readback_composer = ttnn.ConcatMeshToTensor(mesh_device, dim=0) if nd > 1 else None

    def __call__(self, tok_tt):
        if self._readback_composer is not None:
            tid = int(ttnn.to_torch(tok_tt, mesh_composer=self._readback_composer).reshape(-1)[0])
        else:
            tid = int(ttnn.to_torch(tok_tt).reshape(-1)[0])
        return ttnn.from_torch(
            self.table[tid].reshape(1, 1, -1).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            **self._mesh_kwargs,
        )


def _prefill_with_hidden(model, token_ids, valid_len):
    """prefill_tp's own computation, but also returning the per-position post-final-norm
    hidden for every real prompt position (needed to seed the drafter's KV; prefill_tp
    itself only returns the last position's logits)."""
    from models.demos.blackhole.qwen36.tt import tp_common as tpc
    from models.tt_transformers.tt.common import Mode

    T = token_ids.shape[1]
    mesh = model.mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)
    model._build_request_rope(token_ids[:, :valid_len], None)

    tok = ttnn.from_torch(token_ids.to(torch.int32), dtype=ttnn.uint32, device=mesh, mesh_mapper=rep)
    x = model.embd(tok)
    ttnn.deallocate(tok)
    x = ttnn.reshape(x, (1, 1, T, x.shape[-1]))

    if tpc.wh_9b_n300(model.args):
        cos, sin = model._rope_tp_cos_sin_dev(0, T)
    else:
        cos_t, sin_t = model._rope_tp_cos_sin_torch(0, T)
        cos = ttnn.from_torch(cos_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)
        sin = ttnn.from_torch(sin_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, mesh_mapper=rep)

    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="prefill", chunk_size=128, valid_len=valid_len)
    x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
    x_normed = model.norm(x, mode=Mode.PREFILL)  # per-row RMSNorm -- independent of sequence position
    ttnn.deallocate(x)

    # DistributedNorm all-gathers internally and returns a REPLICATED full-width tensor
    # (matches prefill_tp's own usage: it feeds x_normed straight to _lm_head with no
    # further mesh composition) -- read back ONE device's copy, not a dim=-1 concat
    # (which would wrongly concatenate 8 copies of the full width instead of 8 shards).
    if model.num_devices > 1:
        hidden_all = ttnn.to_torch(ttnn.get_device_tensors(x_normed)[0]).reshape(T, -1).float()[:valid_len]
    else:
        hidden_all = ttnn.to_torch(x_normed).reshape(T, -1).float()[:valid_len]

    sel_tt = model._row_selector(T, valid_len - 1, (1, 1, 1, T), x_normed.dtype)
    x_last = ttnn.matmul(sel_tt, x_normed)
    ttnn.deallocate(sel_tt)
    ttnn.deallocate(x_normed)
    logits = model._lm_head(x_last)
    ttnn.deallocate(x_last)
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    ttnn.deallocate(logits)
    return lt[0].reshape(-1)[: model.vocab_size].float(), hidden_all


def _decode_with_hidden(model, token_id, position):
    """decode_tp's own computation, but also returning the pre-lm_head hidden -- needed
    to re-seed the drafter after each accept/reject cycle."""
    from models.demos.blackhole.qwen36.tt.attention.rope_tp import rot_mats_decode

    mesh = model.mesh_device
    rep = ttnn.ReplicateTensorToMesh(mesh)

    tok = ttnn.from_torch(
        torch.tensor([[token_id]], dtype=torch.int32), dtype=ttnn.uint32, device=mesh, mesh_mapper=rep
    )
    x = model._embed(tok)
    ttnn.deallocate(tok)
    x = ttnn.reshape(x, (1, 1, 1, x.shape[-1]))
    cos, sin = rot_mats_decode(
        mesh,
        model.args.rope_head_dim,
        model.args.max_seq_len,
        model.args.rope_theta,
        torch.tensor([position + model.rope.rope_delta], dtype=torch.int32),
        full_head_dim=model.rope.full_head_dim,
    )
    pos_tt = ttnn.from_torch(
        torch.tensor([position], dtype=torch.int32), dtype=ttnn.int32, device=mesh, mesh_mapper=rep
    )

    for layer in model.layers:
        x = layer.forward(x, cos=cos, sin=sin, mode="decode", position_tensor=pos_tt)
    ttnn.deallocate(pos_tt)
    hidden_tt = model._final_norm_decode(x)
    ttnn.deallocate(x)
    logits = model._lm_head(hidden_tt)
    lt = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
    # _final_norm_decode is the same DistributedNorm object as prefill's -- all-gathers
    # internally and returns a REPLICATED full-width tensor (see _prefill_with_hidden's
    # comment); read back one device's copy, not a dim=-1 concat.
    if model.num_devices > 1:
        hidden = ttnn.to_torch(ttnn.get_device_tensors(hidden_tt)[0]).reshape(-1).float()
    else:
        hidden = ttnn.to_torch(hidden_tt).reshape(-1).float()
    ttnn.deallocate(hidden_tt)
    ttnn.deallocate(logits)
    return lt[0].reshape(-1)[: model.vocab_size].float(), hidden


def run_fresh_generation(model, tokenizer, token_ids, max_generated_tokens, drafter=None, draft_len=4):
    """Generate via prefill_tp/decode_tp (never allocate_kv_caches), optionally drafting
    `draft_len` tokens at a time with `drafter` and verifying each sequentially against a
    real decode step. Returns (generated_ids, stats).

    Position bookkeeping (two conventions in play, kept explicit rather than implicit):
      - "decode position" of a token = the position at which THAT token's own embedding
        gets written into KV when fed to _decode_with_hidden -- e.g. cur_token's decode
        position is `decode_pos`, and feeding it there produces logits for decode_pos+1
        AND the hidden state AT decode_pos (needed to seed the NEXT drafter step).
      - "drafter slot" i holds the pair (hidden[i], token[i+1]) and predicts token[i+2]
        (tt/mtp_fresh.py's convention) -- i.e. drafter slot i == decode position i+1's
        cur_token, one less than the decode-position numbering above.
    """
    T = token_ids.shape[1]
    model.reset_tp()
    t_prefill_start = time.time()
    first_logits, hidden_all = _prefill_with_hidden(model, token_ids, valid_len=T)
    ttft_s = time.time() - t_prefill_start
    cur_token = int(torch.argmax(first_logits).item())
    generated = [cur_token]
    decode_pos = T  # cur_token's decode position

    def _timing_stats(t_decode_start, extra=None):
        # Every step above ends in a blocking ttnn.to_torch() readback, so wall-clock time
        # here already reflects real device completion, not just async dispatch.
        decode_s = time.time() - t_decode_start
        n_decoded = len(generated) - 1  # excludes the prefill-produced first token
        stats = {
            "ttft_s": ttft_s,
            "decode_s": decode_s,
            "decode_tok_s": n_decoded / decode_s if decode_s > 0 else float("inf"),
            "total_s": ttft_s + decode_s,
            "overall_tok_s": len(generated) / (ttft_s + decode_s) if (ttft_s + decode_s) > 0 else float("inf"),
        }
        stats.update(extra or {})
        return stats

    if drafter is None:
        t_decode_start = time.time()
        while len(generated) < max_generated_tokens:
            logits, _ = _decode_with_hidden(model, cur_token, decode_pos)
            cur_token = int(torch.argmax(logits).item())
            decode_pos += 1
            generated.append(cur_token)
        return generated, _timing_stats(t_decode_start, {"accept_stats": None})

    # Seed the drafter's KV over the real prompt: slot i = (hidden[i], token[i+1]), for
    # every real prompt position except the last (that one becomes target_hidden below,
    # fed live into the first draft step rather than pre-stepped through).
    for i in range(T - 1):
        drafter.step(int(token_ids[0, i + 1]), hidden_all[i], i)
    target_hidden = hidden_all[T - 1]

    accept_lens = []
    t_decode_start = time.time()
    while len(generated) < max_generated_tokens:
        base_pos = decode_pos  # cur_token's decode position at the start of this cycle
        drafter_pos = base_pos - 1  # the slot target_hidden/cur_token occupy

        draft_tokens = []
        step_token, step_hidden = cur_token, target_hidden
        for j in range(draft_len):
            logits, step_hidden = drafter.step(step_token, step_hidden, drafter_pos + j)
            step_token = int(torch.argmax(logits).item())
            draft_tokens.append(step_token)

        # Sequential real verify: one real decode step per candidate, greedy-compared.
        # (The production design verifies all candidates in ONE pass; this simpler,
        # unambiguously-correct sequential form is the point of a from-scratch
        # correctness reference, not a speed demonstration.)
        committed_tokens, committed_hiddens = [], []
        verify_token, verify_pos = cur_token, base_pos
        accept = 0
        for j in range(draft_len):
            logits, real_hidden = _decode_with_hidden(model, verify_token, verify_pos)
            real_token = int(torch.argmax(logits).item())
            committed_tokens.append(real_token)
            committed_hiddens.append(real_hidden)
            matched = draft_tokens[j] == real_token
            verify_token, verify_pos = real_token, verify_pos + 1
            if matched:
                accept += 1
            else:
                break
        accept_lens.append(accept)

        n_new = len(committed_tokens)  # accept, if every candidate matched; accept+1 otherwise
        generated.extend(committed_tokens[: max_generated_tokens - len(generated)])

        # Re-seed the drafter's KV for slots [base_pos, base_pos+n_new-2] with the REAL
        # (hidden, token) pairs the verify loop just produced -- the draft chain wrote its
        # OWN (possibly wrong) speculative guesses there. Slot base_pos-1 needs no fix: the
        # very first draft step above already re-wrote it with the real (target_hidden,
        # cur_token) pair. The final new slot needs no separate step() either -- it becomes
        # the live target_hidden/cur_token for the next cycle below.
        for k in range(n_new - 1):
            drafter.step(committed_tokens[k + 1], committed_hiddens[k], base_pos + k)

        cur_token = committed_tokens[-1]
        target_hidden = committed_hiddens[-1]
        decode_pos = base_pos + n_new

    return generated, _timing_stats(
        t_decode_start,
        {
            "accept_stats": accept_lens,
            "avg_accept": sum(accept_lens) / len(accept_lens) if accept_lens else 0.0,
            "iterations": len(accept_lens),
        },
    )


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "use_mtp, max_generated_tokens, draft_len",
    [
        pytest.param(False, 30, None, id="no_mtp"),
        pytest.param(True, 30, 4, id="with_mtp"),
        # Matches demo/text_demo.py's spec_128 id (seqlen=128, 200 generated tokens) for a
        # fair tok/s comparison against the production TT_SPEC_DECODE=1 batched-verify path.
        pytest.param(False, 200, None, id="no_mtp_200"),
    ],
)
def test_fresh_generation(mesh_device, use_mtp, max_generated_tokens, draft_len):
    from transformers import AutoTokenizer

    device = mesh_device
    device.enable_program_cache()
    seqlen = 128
    num_blocks = _blocks_for(seqlen, max_generated_tokens)
    max_seq_len = num_blocks * BLOCK_SIZE

    model = Qwen36Model.from_pretrained(device, max_batch_size=1, max_seq_len=max_seq_len)
    tokenizer = AutoTokenizer.from_pretrained(model.args.CKPT_DIR, trust_remote_code=True)
    max_prompt_len = ((max_seq_len - max_generated_tokens) // 128) * 128
    token_ids = _get_prompt(seqlen, tokenizer, max_prompt_len=max_prompt_len)

    drafter = None
    if use_mtp:
        ckpt = model.args.CKPT_DIR
        if not checkpoint_has_mtp(ckpt):
            pytest.skip(f"checkpoint {ckpt} has no mtp.* weights")
        mtp_sd = load_qwen36_mtp_state_dict(ckpt)
        heads = load_qwen36_shared_head_weights(ckpt)
        mesh_kwargs = dict(mesh_mapper=ttnn.ReplicateTensorToMesh(device)) if model.num_devices > 1 else {}
        lm_head_tt = ttnn.as_tensor(
            heads["lm_head_weight"].T.contiguous().to(torch.bfloat16),
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **mesh_kwargs,
        )
        drafter = Qwen36MTPDrafter(
            device,
            model.args,
            mtp_sd,
            embedding=_RealEmbeddingLookup(heads["embed_weight"], device),
            lm_head_weight=lm_head_tt,
            rope=model.rope,
        )
        drafter.allocate_kv_cache(num_blocks_per_user=8)

    generated, stats = run_fresh_generation(
        model, tokenizer, token_ids, max_generated_tokens, drafter=drafter, draft_len=draft_len or 4
    )
    if drafter is not None:
        drafter.free_kv_cache()

    text = tokenizer.decode(generated, skip_special_tokens=True)
    logger.info(f"[fresh, mtp={use_mtp}] GENERATED: {text!r}")
    logger.info(
        f"[fresh, mtp={use_mtp}] {len(generated)} tokens: ttft={stats['ttft_s']:.2f}s "
        f"decode={stats['decode_s']:.2f}s ({stats['decode_tok_s']:.2f} tok/s decode-only) "
        f"total={stats['total_s']:.2f}s ({stats['overall_tok_s']:.2f} tok/s overall)"
    )
    if stats["accept_stats"] is not None:
        logger.info(
            f"[fresh, mtp={use_mtp}] {stats['iterations']} iterations, accept lengths: {stats['accept_stats']} "
            f"(avg {stats['avg_accept']:.2f}/{draft_len})"
        )
    assert len(generated) >= 1 and not any(t < 0 for t in generated)
