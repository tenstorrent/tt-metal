# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DEBUG-ONLY: first-draft-token comparison between the TARGET and the MTP drafter.

Why this exists
---------------
Speculative-decode acceptance is the fraction of drafter tokens that survive the
target verify. When acceptance is low, the loss is in exactly one of three
places, and this test isolates the FIRST draft token so they can be told apart:

  1. The SEED — the target hidden at the anchor that is handed to the drafter.
     Produced by ``target.ttnn_verify_forward()`` at ``[anchor_token]`` /
     ``[anchor_pos]``, which is what ``SpeculativeDecoder.seed()`` wraps. If the
     hidden the drafter receives differs from the one the target produced (the
     clone/slice/layout round-trip), the drafter is guessing from bad state.
  2. The DRAFTER — ``assistant.step()``: one MTP/EAGLE step consuming
     (anchor_token, anchor_hidden) and returning ``(logits, next_hidden)``. Its
     argmax is draft token d1.
  3. The VERIFY — ``target.ttnn_verify_forward()`` over
     ``[anchor_token, d1]`` at ``[anchor_pos, anchor_pos+1]``. Greedy acceptance
     compares d1 against ``argmax(verify_logits[0])``: row 0 is the target's own
     next token AFTER the anchor, so a mismatch there rejects d1
     (see ``SpeculativeDecoder._accept_greedy``).

The test asserts only numerical sanity (finite logits, matching vocab). A
MISMATCH is a finding to read, not a failure — that is the thing being
investigated. Production speculative-decode logic is not modified or monkey-
patched: everything here calls the real methods.

All device tensors are ttnn.Tensor; ``_to_host`` converts them (reading the
device-0 replica under TP) before any torch op. Never call torch APIs on a
ttnn.Tensor directly.

Run (same fused-traced greedy path as ``test_isl_sweep.py --speculative`` /
``_run_spec_decode`` — shift seed, ``generate_fused`` trace replay by default):

    MESH_DEVICE=T3K pytest \\
      models/demos/gemma4/tests/unit/test_spec_decode_first_token_debug.py -sv --timeout 3600

    # 10 decode iterations, batch-1 JSON prompt, draft_len=3 table each iter:
    GEMMA4_DEBUG_RUNS=10 GEMMA4_SPEC_DRAFT_LEN=3 MESH_DEVICE=T3K \\
      HF_MODEL=google/gemma-4-31B-it GEMMA4_ASSISTANT_MODEL=google/gemma-4-31B-it-assistant \\
      pytest models/demos/gemma4/tests/unit/test_spec_decode_first_token_debug.py -sv --timeout 3600

    # A/B untraced fused (still on-device ``_fused_iter``, not the legacy reseed loop):
    GEMMA4_SPEC_TRACE=0 MESH_DEVICE=T3K pytest ... -sv --timeout 3600

    # Legacy reseed ``_draft`` + ``_verify`` loop (pre-demo path, for comparison only):
    GEMMA4_DEBUG_LEGACY_PATH=1 MESH_DEVICE=T3K pytest ... -sv --timeout 3600
"""

_BATCH1_PROMPT_FILE = "models/tt_transformers/demo/sample_prompts/input_data_questions_prefill_128.json"

import math
import os

import pytest
import torch
from loguru import logger

import ttnn

# Reuse the demo's mesh/device parametrization and page-table helper so this test
# opens the mesh exactly like test_demo_spec_decode / _run_spec_decode do.
from models.demos.gemma4.demo.text_demo_v2 import (
    _device_params,
    _mesh_device_param,
    _model_path,
    _prepare_demo_prefill_warmup,
    create_tt_page_table,
)


def _to_host(tensor, tp):
    """ttnn.Tensor -> torch.Tensor. Under TP the tensor is replicated, so read
    the device-0 replica (same convention as SpeculativeDecoder._logits_to_host)."""
    if tp > 1:
        return ttnn.to_torch(ttnn.get_device_tensors(tensor)[0])
    return ttnn.to_torch(tensor)


def _stats(label, t):
    """Log shape + summary statistics for a host torch tensor."""
    f = t.float()
    logger.info(
        f"    {label}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"min={float(f.min()):+.6f} max={float(f.max()):+.6f} "
        f"mean={float(f.mean()):+.6f} std={float(f.std()):.6f} "
        f"absmax={float(f.abs().max()):.6f} "
        f"n_nan={int(torch.isnan(f).sum())} n_inf={int(torch.isinf(f).sum())}"
    )


def _decode_tok(tokenizer, tok_id):
    try:
        return repr(tokenizer.decode([int(tok_id)]))
    except Exception:
        return "<undecodable>"


def _word(tokenizer, tok_id):
    try:
        return tokenizer.decode([int(tok_id)])
    except Exception:
        return "<?>"


def _logit_at(logits_row, tok_id):
    return float(logits_row.float()[int(tok_id)])


def _demo_enable_trace():
    """Match batch-1 ``test_isl_sweep`` / ``_run_spec_decode`` decode tracing default."""
    _decode_trace = os.environ.get("GEMMA4_DECODE_TRACE")
    if _decode_trace is not None:
        return _decode_trace.lower() in ("1", "true", "yes")
    return True


def _configure_spec_trace(spec, enable_trace):
    """Same ``spec._use_trace`` policy as ``text_demo_v2._run_spec_decode``."""
    _trace_env = os.environ.get("GEMMA4_SPEC_TRACE")
    spec._use_trace = enable_trace if _trace_env is None else (_trace_env == "1")


def _fused_accept(drafts, target_ids):
    """Greedy accept count + committed tokens (shift-mode fused ids)."""
    k = len(drafts)
    m = next((i for i in range(k) if drafts[i] != target_ids[i]), k)
    committed = drafts[:m] + [target_ids[m]]
    return m, committed


def _read_verify_x_drafts(spec, verify_x_host, draft_len):
    """Draft ids from persistent ``verify_x`` (reseed vs shift layout)."""
    vx = verify_x_host.reshape(-1)
    return [int(vx[j if spec._fused_reseed else 1 + j]) for j in range(draft_len)]


def _resolve_debug_prompt():
    """Default: first prompt from the batch-1 ISL sweep JSON."""
    if "GEMMA4_DEBUG_PROMPT" in os.environ:
        return os.environ["GEMMA4_DEBUG_PROMPT"], "GEMMA4_DEBUG_PROMPT"
    from models.demos.gemma4.demo.text_demo_v2 import load_inputs

    prompt = load_inputs(_BATCH1_PROMPT_FILE, batch=1, instruct=True)[0]
    return prompt, _BATCH1_PROMPT_FILE


def _print_clean_summary(
    tokenizer,
    draft_len,
    drafts,
    draft_logits,
    target_greedy,
    verify_logits,
    prefix_match,
    committed,
    *,
    run_idx=None,
    num_runs=None,
    anchor_pos=None,
    anchor_token=None,
):
    """Print a compact side-by-side table: MTP vs TARGET for every draft step."""
    w = 78
    print("\n" + "=" * w)
    hdr = f" SPEC-DECODE SUMMARY  (batch=1, draft_len={draft_len})"
    if run_idx is not None and num_runs is not None:
        hdr = f" SPEC-DECODE SUMMARY  run {run_idx + 1}/{num_runs}  (batch=1, draft_len={draft_len})"
    print(hdr)
    if anchor_pos is not None and anchor_token is not None:
        print(f" anchor_pos={anchor_pos}  anchor_token={anchor_token}  {_decode_tok(tokenizer, anchor_token)}")
    print("=" * w)
    print(
        f" {'step':^4} │ {'MTP id':^8} │ {'MTP word':^14} │ {'MTP logit':^10} │"
        f" {'TARGET id':^9} │ {'TARGET word':^14} │ {'TARGET logit':^12} │ {'MATCH':^5}"
    )
    print("-" * w)
    have_logits = draft_logits is not None and verify_logits is not None
    for i in range(draft_len):
        mtp_id = int(drafts[i])
        tgt_id = int(target_greedy[i])
        mtp_word = _word(tokenizer, mtp_id).replace("\n", "\\n")[:14]
        tgt_word = _word(tokenizer, tgt_id).replace("\n", "\\n")[:14]
        if have_logits:
            mtp_logit = _logit_at(draft_logits[i], mtp_id)
            tgt_logit = _logit_at(verify_logits[i], tgt_id)
            mtp_logit_s = f"{mtp_logit:+10.4f}"
            tgt_logit_s = f"{tgt_logit:+12.4f}"
        else:
            mtp_logit_s = "       n/a"
            tgt_logit_s = "         n/a"
        matched = mtp_id == tgt_id
        print(
            f" {i:^4} │ {mtp_id:^8} │ {mtp_word:^14} │ {mtp_logit_s} │"
            f" {tgt_id:^9} │ {tgt_word:^14} │ {tgt_logit_s} │ {'YES' if matched else 'NO':^5}"
        )
        if have_logits:
            mtp_scores_tgt = _logit_at(draft_logits[i], tgt_id)
            tgt_scores_mtp = _logit_at(verify_logits[i], mtp_id)
            print(
                f"      │          │ (tgt tok in MTP: {mtp_scores_tgt:+.4f})"
                f" │            │ (mtp tok in TARGET: {tgt_scores_mtp:+.4f}) │"
            )
    print("-" * w)
    print(f" prefix-match (greedy accept): {prefix_match}/{draft_len}")
    committed_words = " ".join(repr(_word(tokenizer, t)) for t in committed)
    print(f" committed after accept   : ids={committed}  words=[{committed_words}]")
    print("=" * w + "\n")


def _log_topk(label, logits_row, tokenizer, k=5):
    """Log the top-k tokens of a [vocab] logits row plus the top1-top2 gap."""
    f = logits_row.float()
    vals, idx = torch.topk(f, k)
    logger.info(f"    {label} top-{k}:")
    for r in range(k):
        logger.info(
            f"      #{r + 1} id={int(idx[r]):<8d} logit={float(vals[r]):+.4f} " f"tok={_decode_tok(tokenizer, idx[r])}"
        )
    logger.info(f"    {label} top1-top2 gap: {float(vals[0] - vals[1]):.4f} (small gap => near-tie, unstable argmax)")


def _update_fused_trace_inputs(spec, cur_token, cur_pos):
    """Host-side input refresh between fused trace replays (``_generate_fused_traced``)."""
    tr = spec._fused_trace
    k = spec.draft_len
    h_tok = spec._host_tokens([cur_token])
    ttnn.copy_host_to_device_tensor(h_tok, tr["anchor_tok"])
    h_tok.deallocate(True)
    d_hpu, d_hpi = spec._host_pos([cur_pos])
    ttnn.copy_host_to_device_tensor(d_hpu, tr["d_pu"])
    ttnn.copy_host_to_device_tensor(d_hpi, tr["d_pi"])
    d_hpu.deallocate(True)
    d_hpi.deallocate(True)
    v_pos = [cur_pos + 1 + j for j in range(k)] if spec._fused_reseed else [cur_pos + j for j in range(k + 1)]
    v_hpu, v_hpi = spec._host_pos(v_pos)
    ttnn.copy_host_to_device_tensor(v_hpu, tr["v_pu"])
    ttnn.copy_host_to_device_tensor(v_hpi, tr["v_pi"])
    v_hpu.deallocate(True)
    v_hpi.deallocate(True)


def _read_fused_iter_ids(spec, tp):
    """Read draft + target ids after one fused (traced) iteration."""
    tr = spec._fused_trace
    k = spec.draft_len
    vx = ttnn.to_torch(ttnn.get_device_tensors(tr["verify_x"])[0]) if tp > 1 else ttnn.to_torch(tr["verify_x"])
    drafts = _read_verify_x_drafts(spec, vx, k)
    target_ids = spec._ids_to_host(tr["vidx"], k + 1)
    return drafts, target_ids


def _debug_fused_traced_loop(spec, mesh_device, tokenizer, tp, anchor_token, anchor_pos, num_runs, stop_tokens):
    """Step through ``_generate_fused_traced`` with per-iter tables (demo path)."""
    k = spec.draft_len
    spec._use_trace = False
    anchor_hidden = spec.seed(anchor_token, anchor_pos)
    spec._use_trace = True
    spec._capture_fused_trace(anchor_token, anchor_hidden, anchor_pos, capture_logits=True)
    anchor_hidden.deallocate(True)

    generated, accepts = [], []
    cur_token, cur_pos = anchor_token, anchor_pos
    first = True
    runs_completed = 0
    for run_idx in range(num_runs):
        runs_completed = run_idx + 1
        if not first:
            _update_fused_trace_inputs(spec, cur_token, cur_pos)
        first = False

        ttnn.execute_trace(mesh_device, spec._fused_trace["id"], cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh_device)
        drafts, target_ids = _read_fused_iter_ids(spec, tp)
        prefix_match, committed = _fused_accept(drafts, target_ids)
        accepts.append(prefix_match)
        draft_logits, verify_logits = spec._read_fused_capture_logits(spec._fused_trace)

        _print_clean_summary(
            tokenizer,
            k,
            drafts,
            draft_logits,
            target_ids[:k],
            verify_logits,
            prefix_match,
            committed,
            run_idx=run_idx,
            num_runs=num_runs,
            anchor_pos=cur_pos,
            anchor_token=cur_token,
        )

        generated.extend(committed)
        if any(tok in stop_tokens for tok in committed):
            break

        if not spec._fused_reseed:
            spec._hidden_row_to_device(spec._fused_shift_seed_row(prefix_match, k))
        cur_pos = cur_pos + prefix_match + 1
        cur_token = committed[-1]

    return generated, accepts, runs_completed


def _debug_fused_eager_loop(spec, tokenizer, tp, anchor_token, anchor_pos, num_runs, stop_tokens):
    """Step through untraced ``generate_fused`` (``_fused_iter``) with per-iter tables."""
    k = spec.draft_len
    spec._pv_a_prev = -1
    generated, accepts = [], []
    anchor_hidden = spec.seed(anchor_token, anchor_pos)
    anchor_tok_tt = spec._tokens_tensor([anchor_token])
    cur_pos = anchor_pos
    cur_token = anchor_token
    runs_completed = 0

    for run_idx in range(num_runs):
        runs_completed = run_idx + 1
        if spec._fused_reseed:
            new_anchor_hidden = spec.seed(cur_token, cur_pos)
            anchor_hidden.deallocate(True)
            anchor_hidden = new_anchor_hidden

        drafts, target_ids, vhidden, draft_logits, verify_logits = spec._fused_iter(
            anchor_tok_tt, anchor_hidden, cur_pos, capture_logits=True
        )
        prefix_match, committed = _fused_accept(drafts, target_ids)
        accepts.append(prefix_match)

        _print_clean_summary(
            tokenizer,
            k,
            drafts,
            draft_logits,
            target_ids[:k],
            verify_logits,
            prefix_match,
            committed,
            run_idx=run_idx,
            num_runs=num_runs,
            anchor_pos=cur_pos,
            anchor_token=cur_token,
        )

        generated.extend(committed)
        if any(tok in stop_tokens for tok in committed):
            vhidden.deallocate(True)
            anchor_tok_tt.deallocate(True)
            anchor_hidden.deallocate(True)
            break

        new_pos = cur_pos + prefix_match + 1
        new_token = committed[-1]
        if spec._fused_reseed:
            new_anchor_hidden = anchor_hidden
        else:
            row = spec._fused_shift_seed_row(prefix_match, k)
            new_anchor_hidden = ttnn.clone(vhidden[:, :, row : row + 1, :])
            anchor_hidden.deallocate(True)
        vhidden.deallocate(True)
        anchor_hidden = new_anchor_hidden
        anchor_tok_tt.deallocate(True)
        anchor_tok_tt = spec._tokens_tensor([new_token])
        cur_pos = new_pos
        cur_token = new_token

    else:
        anchor_hidden.deallocate(True)
        anchor_tok_tt.deallocate(True)

    return generated, accepts, runs_completed


def _debug_legacy_reseed_loop(spec, tokenizer, tp, anchor_token, anchor_pos, num_runs, stop_tokens, draft_len, verbose):
    """Legacy ``_draft`` + ``_verify`` + exact reseed (pre-demo comparison path)."""
    generated = []
    accepts = []
    anchor_hidden = None
    owns_anchor_hidden = True
    runs_completed = 0

    for run_idx in range(num_runs):
        runs_completed = run_idx + 1
        if anchor_hidden is None:
            anchor_hidden = spec.seed(anchor_token, anchor_pos)
            owns_anchor_hidden = True

        if run_idx == 0 or verbose:
            raw_hidden_host = _to_host(anchor_hidden, tp)
            logger.info("")
            logger.info(f"── RUN {run_idx + 1}/{num_runs} ── TARGET anchor ───────────────")
            logger.info(f"  anchor token id   : {anchor_token}  tok={_decode_tok(tokenizer, anchor_token)}")
            logger.info(f"  anchor position   : {anchor_pos}")
            logger.info(f"  hidden state shape: {tuple(raw_hidden_host.shape)}")
            _stats("anchor_hidden (target output)", raw_hidden_host)

        drafts, draft_logits = spec._draft(anchor_token, anchor_hidden, anchor_pos)

        if run_idx == 0 or verbose:
            logger.info("")
            logger.info(f"── RUN {run_idx + 1}/{num_runs} ── MTP / ASSISTANT (_draft, K={draft_len}) ──")
            for i, (d_tok, d_logits) in enumerate(zip(drafts, draft_logits)):
                logger.info(f"  draft step {i}:")
                _stats(f"    logits step{i}", d_logits)
                _log_topk(f"MTP step{i}", d_logits, tokenizer)
                logger.info(f"    predicted token id : {d_tok}  tok={_decode_tok(tokenizer, d_tok)}")

        verify_tokens = [anchor_token] + drafts
        verify_positions = [anchor_pos + j for j in range(len(verify_tokens))]
        verify_logits, verify_hidden = spec._verify(verify_tokens, verify_positions)

        target_greedy = [int(torch.argmax(verify_logits[j])) for j in range(len(drafts) + 1)]
        prefix_match = 0
        for i, d in enumerate(drafts):
            if d == target_greedy[i]:
                prefix_match += 1
            else:
                break
        m_accept, committed = spec._accept_greedy(drafts, verify_logits)
        accepts.append(m_accept)

        if run_idx == 0 or verbose:
            logger.info("")
            logger.info(f"── RUN {run_idx + 1}/{num_runs} ── TARGET VERIFICATION ─────────")
            logger.info(f"  verify tokens    : {verify_tokens} @ positions {verify_positions}")
            logger.info(f"  prefix-match: {prefix_match}/{draft_len}  committed={committed}")

        _print_clean_summary(
            tokenizer,
            draft_len,
            drafts,
            draft_logits,
            target_greedy,
            verify_logits,
            prefix_match,
            committed,
            run_idx=run_idx,
            num_runs=num_runs,
            anchor_pos=anchor_pos,
            anchor_token=anchor_token,
        )

        assert draft_logits[0].numel() == verify_logits.shape[-1]
        for i, d_logits in enumerate(draft_logits):
            assert torch.isfinite(d_logits.float()).all(), f"run {run_idx} MTP logits step {i} contain NaN/Inf"
        assert torch.isfinite(verify_logits.float()).all(), f"run {run_idx} target verify logits contain NaN/Inf"

        generated.extend(committed)
        if any(tok in stop_tokens for tok in committed):
            verify_hidden.deallocate(True)
            break

        new_pos = anchor_pos + m_accept + 1
        new_token = committed[-1]
        verify_hidden.deallocate(True)
        new_anchor_hidden = spec.seed(new_token, new_pos)
        if owns_anchor_hidden:
            anchor_hidden.deallocate(True)
        anchor_hidden = new_anchor_hidden
        owns_anchor_hidden = True
        anchor_pos = new_pos
        anchor_token = new_token

    if owns_anchor_hidden and anchor_hidden is not None:
        anchor_hidden.deallocate(True)

    return generated, accepts, runs_completed


@pytest.mark.parametrize("device_params", [_device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [_mesh_device_param()], indirect=True)
def test_spec_first_token_debug(mesh_device, reset_seeds):
    """Compare TARGET vs MTP/assistant over multiple greedy spec-decode iterations."""
    from models.demos.gemma4.tt.common import create_assistant_model
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.demos.gemma4.tt.spec_decode import SpeculativeDecoder
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    # ── config: identical to _run_spec_decode() ───────────────────────────────
    model_path = _model_path()
    assistant_path = os.getenv("GEMMA4_ASSISTANT_MODEL") or f"{model_path}-assistant"
    prompt, prompt_source = _resolve_debug_prompt()
    instruct = True
    batch_size = 1
    num_runs = int(os.environ.get("GEMMA4_DEBUG_RUNS", 10))
    verbose = os.environ.get("GEMMA4_DEBUG_VERBOSE", "0") == "1"
    max_seq_len = int(os.environ.get("GEMMA4_MAX_SEQ_LEN", 4096))
    max_generated_tokens = 32  # only bounds the prefill length here
    block_size = 64  # page_params["page_block_size"] in test_demo_spec_decode

    paged_attention_config = PagedAttentionConfig(
        block_size=block_size,
        max_num_blocks=batch_size * math.ceil(max_seq_len / block_size),
    )

    draft_len = int(os.environ.get("GEMMA4_SPEC_DRAFT_LEN", 3))
    enable_trace = _demo_enable_trace()
    legacy_path = os.environ.get("GEMMA4_DEBUG_LEGACY_PATH", "0") == "1"
    logger.info("=" * 78)
    logger.info("SPEC-DECODE DEBUG (multi-iteration)")
    logger.info(f"  target    : {model_path}")
    logger.info(f"  drafter   : {assistant_path}")
    logger.info(f"  prompt_src: {prompt_source}")
    logger.info(f"  prompt    : {prompt[:120]!r}{'...' if len(prompt) > 120 else ''}")
    logger.info(f"  runs      : {num_runs} (set GEMMA4_DEBUG_RUNS to change)")
    logger.info(f"  draft_len : {draft_len} (set GEMMA4_SPEC_DRAFT_LEN to change)")
    logger.info(f"  path      : {'legacy reseed (_draft/_verify)' if legacy_path else 'fused greedy (demo)'}")
    logger.info(f"  decode_trace: {enable_trace} (GEMMA4_DECODE_TRACE / batch-1 demo default)")
    logger.info("=" * 78)

    # ── 1. target model, exactly as _run_spec_decode() builds it ──────────────
    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=batch_size,
        max_seq_len=max_seq_len,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,  # spec-decode needs unbounded sliding KV
    )
    target = generator.model[0]
    model_args = generator.model_args
    tp = target.mesh_config.tp if target.mesh_config else 1

    page_table = create_tt_page_table(batch_size, paged_attention_config)

    sampling_params = {"temperature": 0, "top_p": 0.08}
    model_args_list = model_args if isinstance(model_args, (list, tuple)) else [model_args]
    prefill_enable_trace, _device_sampling_params = _prepare_demo_prefill_warmup(
        generator=generator,
        tt_kv_cache=tt_kv_cache,
        sampling_params=sampling_params,
        enable_trace=enable_trace,
        max_seq_len=max_seq_len,
        model_args_list=model_args_list,
        batch_size=batch_size,
        input_prompts=_BATCH1_PROMPT_FILE,
    )

    # ── 2-3. tokenize + target prefill ────────────────────────────────────────
    input_tokens_prefill_pt, encoded_prompts, decoding_pos, prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, model_args, instruct, max_generated_tokens, max_prefill_len=max_seq_len
    )
    input_tokens_prefill_pt = torch.stack(input_tokens_prefill_pt).view(batch_size, -1)

    logger.info("Running target prefill...")
    prefill_logits = generator.prefill_forward_text(
        input_tokens_prefill_pt,
        page_table=page_table,
        kv_cache=tt_kv_cache,
        prompt_lens=decoding_pos,
        warmup_prefill=False,
        enable_trace=prefill_enable_trace,
    )
    ttnn.synchronize_device(mesh_device)
    if hasattr(prefill_logits, "deallocate"):
        prefill_logits.deallocate(True)

    # ── 4. anchor (same derivation as _run_spec_decode) ───────────────────────
    prompt_len = int(decoding_pos[0])
    anchor_pos = prompt_len - 1
    anchor_token = int(encoded_prompts[0][anchor_pos])

    # ── 5. assistant, created AFTER prefill (production ordering) ─────────────
    _, assistant = create_assistant_model(
        mesh_device=mesh_device,
        target_model=target,
        mesh_config=target.mesh_config,
        ccl_manager=target.ccl_manager,
        assistant_path=assistant_path,
    )

    spec = SpeculativeDecoder(
        target_model=target,
        assistant_model=assistant,
        mesh_device=mesh_device,
        tt_kv_cache=tt_kv_cache,
        page_table_torch=page_table,
        stop_tokens=tokenizer.stop_tokens,
        draft_len=draft_len,
    )
    _configure_spec_trace(spec, enable_trace)
    logger.info(
        f"Spec-decode debug loop: path={'legacy' if legacy_path else 'fused'}, "
        f"trace={spec._use_trace}, seed={'reseed' if spec._fused_reseed else 'shift'}, "
        f"shift_seed={getattr(spec, '_fused_shift_seed', 'n/a')}"
    )

    if legacy_path:
        spec._use_trace = False
        generated, accepts, runs_completed = _debug_legacy_reseed_loop(
            spec,
            tokenizer,
            tp,
            anchor_token,
            anchor_pos,
            num_runs,
            tokenizer.stop_tokens,
            draft_len,
            verbose,
        )
    elif spec._use_trace:
        generated, accepts, runs_completed = _debug_fused_traced_loop(
            spec,
            mesh_device,
            tokenizer,
            tp,
            anchor_token,
            anchor_pos,
            num_runs,
            tokenizer.stop_tokens,
        )
    else:
        generated, accepts, runs_completed = _debug_fused_eager_loop(
            spec,
            tokenizer,
            tp,
            anchor_token,
            anchor_pos,
            num_runs,
            tokenizer.stop_tokens,
        )

    gen_text = tokenizer.decode(generated)
    mean_accept = (sum(accepts) / len(accepts)) if accepts else 0.0
    logger.info("")
    logger.info("=" * 78)
    logger.info(f"GENERATED {len(generated)} tokens over {runs_completed} runs")
    logger.info(f"  token ids : {generated}")
    logger.info(f"  text      : {gen_text!r}")
    logger.info(
        f"  mean accepted {mean_accept:.2f}/{draft_len} drafts/iter "
        f"(tokens/iter: {mean_accept + 1:.2f}) — same metric as text_demo_v2"
    )
    logger.info("=" * 78)
