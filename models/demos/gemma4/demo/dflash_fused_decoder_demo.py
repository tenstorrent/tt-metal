# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Demo: DFlash speculative-decoding generation via ``DFlashFusedDecoder``
(tt/dflash_drafter.py) -- the same B=1 fused-trace loop that backs
``Gemma4DFlashForCausalLM`` in vLLM serving (tt/generator_vllm.py).

Replicates ``Gemma4DFlashForCausalLM``'s exact prefill_forward /
_spec_get_drafter / _spec_bootstrap / decode_forward sequencing directly
against a real ``Gemma4Generator`` target model, so it can be run and watched
without standing up vLLM. Prints the generated text plus a few headline
numbers (tok/s, mean accepted-drafts/iteration).

Requires HF_MODEL (target, e.g. google/gemma-4-31B-it) and the z-lab dFlash
drafter snapshot in the HF cache (auto-discovered) or GEMMA4_DFLASH_DRAFTER.

GEMMA4_DFLASH_SHARD_ARGMAX=1 is required: the default sampling path reuses
the target model's shared TTSampling module, whose route_full_row=False
branch does a broadcast-add (models/common/sampling/tt_sampling.py) that
TT_THROWs ("Invalid subtile broadcast type") on the drafter's multi-row block
logits -- a pre-existing, documented TTSampling limitation (see
spec_decode.py's own negative-results notes on scalar-wise broadcast), not
specific to this demo. This env var routes dFlash through its own
purpose-built ``_shard_argmax`` on-device argmax instead.

Run:
    HF_MODEL=google/gemma-4-31B-it GEMMA4_DFLASH_SHARD_ARGMAX=1 pytest \
        models/demos/gemma4/demo/dflash_fused_decoder_demo.py -k 1x8 -s

    # Different prompt / generation length / ISL:
    GEMMA4_DFLASH_PROMPT="..." GEMMA4_DFLASH_MAX_NEW=128 GEMMA4_DFLASH_MAX_SEQ_LEN=8192 pytest \
        models/demos/gemma4/demo/dflash_fused_decoder_demo.py -k 1x8 -s

    # A prompt too long for a single env var/argv string (Linux MAX_ARG_STRLEN, 128 KiB):
    GEMMA4_DFLASH_PROMPT_FILE=/path/to/prompt.txt GEMMA4_DFLASH_MAX_SEQ_LEN=34816 pytest \
        models/demos/gemma4/demo/dflash_fused_decoder_demo.py -k 1x8 -s
"""

import collections
import math
import os
import time

import pytest
import torch
from loguru import logger

from ..tests.test_factory import parametrize_mesh_with_fabric

DEFAULT_PROMPT = (
    "Write a Python function that checks whether a given string is a "
    "palindrome, ignoring spaces, punctuation and case. Include a short "
    "docstring and two example calls."
)
# 2048 was a hard ceiling before two fixes landed: (1) the drafter's growing-
# context sliding-window mask used to be wrong once ctx_len approached/exceeded
# the drafter's own sliding_window=2048 (see generate.py's module docstring,
# "FIXED (was a KNOWN LATENT LIMITATION...)"); (2) the target model's own
# multi-chunk prefill (needed for any ISL > max_prefill_chunk_size=2048) used a
# traced-chunk-replay path with a TT_FATAL at large chunk counts -- worked
# around by disabling that path's auto-enable (generator_trace.py's
# maybe_auto_enable_chunked_prefill_trace), which this demo already benefits
# from since it never opts into GEMMA4_CHUNKED_PREFILL_TRACE itself, so its
# prefill always took the (correct, if slower) eager per-chunk path anyway.
# Both are fixed/worked-around now; MAX_SEQ_LEN is configurable via
# GEMMA4_DFLASH_MAX_SEQ_LEN. The model's own HF-declared max_position_embeddings
# is 262144 (both google/gemma-4-31b-it and the z-lab DFlash drafter checkpoint
# agree) -- that is the architectural ceiling, not a value verified to fit in
# practice: a real per-layer, unbounded (bounded_sliding_kv_cache=False, a
# DFlash requirement) KV cache at that width, times 60 target layers plus the
# drafter's own 5, is a large DRAM footprint that has NOT been checked against
# T3K's actual budget at that scale. Verified working end-to-end on real
# hardware (coherent output, correct mean-accepted-drafts, no crash) at real
# prefill lengths of 8200 and 16716 tokens; not verified beyond that.
#
# MUST be a multiple of the target model's own max_prefill_chunk_size (2048 in
# every config seen so far on WH T3K) -- the eager per-chunk prefill loop
# ttnn_prefill_forward falls into above that chunk size rounds its LAST chunk
# up to a full chunk width regardless of how much real content remains, and
# that rounded-up position can exceed a non-chunk-aligned MAX_SEQ_LEN: hit a
# real TT_FATAL this way (RoPE table sliced to position 10240 against a table
# built only 9216 (a non-multiple) rows wide -- ttnn/cpp/.../slice_device_
# operation.cpp's "Ends 10240 must be less than or equal to the shape of the
# tensor 9216"). Not asserted here (max_prefill_chunk_size is resolved
# dynamically and this demo doesn't import that resolver), but every verified
# value above is a clean multiple of 2048 -- keep GEMMA4_DFLASH_MAX_SEQ_LEN
# that way.
MAX_SEQ_LEN = int(os.environ.get("GEMMA4_DFLASH_MAX_SEQ_LEN", 2048))
PAGE_BLOCK_SIZE = 64
# Target model's own max_prefill_chunk_size (2048 on WH T3K) -- see the block
# above re: why GEMMA4_DFLASH_MAX_SEQ_LEN must be a multiple of this.
PREFILL_CHUNK_SIZE = 2048
# Traced prefill (single-chunk only, MAX_SEQ_LEN <= one prefill chunk). ON BY
# DEFAULT as of 2026-09-11 -- validated on real hardware: acceptance rate
# (mean accepted-drafts/iter) is IDENTICAL to eager prefill, confirming the
# trace-safe tap buffers capture the same data as the eager clone-append path.
# Profiling showed DFlash's eager prefill spends ~96% of its measured window
# in host-dispatch gaps between ops, not compute (859ms gap vs 38ms device
# compute on an 8-layer/4k-token profile); tracing amortizes that dispatch
# cost across repeated calls at the same shape, same mechanism the plain
# (non-DFlash) baseline already uses. Measured end-to-end effect (44-token
# prompt, 260 generated tokens): the FIRST call is ~10% SLOWER than eager
# (35.9 -> 32.1 tok/s end-to-end -- one-time trace-capture tax), but every
# call after that is ~67% FASTER (35.9 -> 60.1 tok/s) since prefill collapses
# from ~3.3s to ~0.1s on replay -- the realistic case for repeated requests
# at the same padded shape (e.g. vLLM serving). Set GEMMA4_DFLASH_PREFILL_TRACE=0
# to force the old eager/untraced path (e.g. for a single one-off run where
# the capture tax isn't worth paying, or to bisect a regression against this
# path).
# Multi-chunk prefill (MAX_SEQ_LEN > one chunk) is NOT supported yet: the tap
# hook's copy-mode buffers are indexed per FORWARD CALL, not per chunk
# position, so a captured trace replayed once per chunk would overwrite the
# same buffer slot on every chunk instead of accumulating across the full
# prompt -- see the dflash_capture_taps docstring in tt/model.py. The
# MAX_SEQ_LEN <= PREFILL_CHUNK_SIZE gate below auto-falls-back to eager for that case.
GEMMA4_DFLASH_PREFILL_TRACE = os.environ.get("GEMMA4_DFLASH_PREFILL_TRACE", "1") == "1"

# Adaptive fallback: some workloads (open-ended/conversational prompts) have
# genuinely low draft/target agreement regardless of block_size -- DFlash's
# per-iteration overhead (drafter forward + verify pass) then costs more than
# it recovers, measured net *below* plain autoregressive decode (e.g. 0.94-
# 0.97x on an MT-Bench-style prompt). Detect this early and drop to plain
# decode for the rest of the session instead of paying that tax for the whole
# generation. One-way (no re-enabling DFlash mid-session) -- simplest policy,
# no oscillation risk; a workload whose predictability genuinely improves
# later in a very long generation is the tradeoff accepted for that
# simplicity. Off entirely if GEMMA4_DFLASH_FALLBACK=0.
GEMMA4_DFLASH_FALLBACK = os.environ.get("GEMMA4_DFLASH_FALLBACK", "1") == "1"
GEMMA4_DFLASH_FALLBACK_WINDOW = int(os.environ.get("GEMMA4_DFLASH_FALLBACK_WINDOW", 10))
# Mean accepted-drafts/iter (bonus token excluded) below this over the trailing
# window means DFlash is committing barely more than the bonus token alone --
# i.e. paying for a block draft + verify pass to get what plain decode would
# have gotten anyway. Calibrated against the measured MT-Bench regression
# (mean accepted ~1.13-1.26 there) vs. healthy workloads (>=2.46 at the
# worst-measured code/math/extraction bucket) -- see README's block-size and
# ISL sweep tables.
GEMMA4_DFLASH_FALLBACK_THRESHOLD = float(os.environ.get("GEMMA4_DFLASH_FALLBACK_THRESHOLD", 1.0))
# Switching costs a one-time trace-compile tax (~1.5s, measured) for the plain-
# decode path's first traced call. That's only worth paying if enough budget
# remains afterward to earn it back: DFlash's own *sustained* full-session
# regression on MT-Bench-style prompts is mild in steady state (~46.4ms/tok vs.
# baseline's ~43.4ms/tok, a ~3ms/tok gap) even though the rolling-window
# acceptance that triggers detection looks much worse locally -- the trigger
# condition is a local dip, not necessarily the whole remaining session's rate.
# Break-even: ~1500ms / 3ms-per-token =~ 500 tokens. Below that remaining
# budget, switching is a net loss (measured: forcing it anyway on a 256-token
# MT-Bench session, with only ~186 tokens left at detection, dropped 0.94-0.97x
# to 0.85x -- worse than just letting DFlash finish). Conservative margin (500
# -> 512) rather than cutting it exactly at the measured break-even.
GEMMA4_DFLASH_FALLBACK_MIN_REMAINING = int(os.environ.get("GEMMA4_DFLASH_FALLBACK_MIN_REMAINING", 512))


def _dflash_default_snapshot():
    """Locate the z-lab drafter snapshot in the HF cache (mirrors
    generator_vllm.py's own helper -- reimplemented here rather than imported,
    since importing generator_vllm.py pulls in tt_transformers'
    generator_vllm.py -> vllm, which some dev environments don't have."""
    import glob

    hits = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--z-lab--gemma-4-31B-it-DFlash/snapshots/*/"))
    return hits[0] if hits else None


def _unwrap_kv_layers(kv_cache):
    """Same defensive unwrap as Gemma4DFlashForCausalLM._spec_bootstrap."""
    kv_layers = kv_cache
    if (
        isinstance(kv_layers, (list, tuple))
        and kv_layers
        and isinstance(kv_layers[0], (list, tuple))
        and kv_layers[0]
        and isinstance(kv_layers[0][0], (list, tuple))
    ):
        kv_layers = kv_layers[0]
    return kv_layers


@parametrize_mesh_with_fabric(
    [(1, 8)],
    device_params_extra={
        "trace_region_size": 256_000_000,
        # CCL all_gather allocates semaphores in L1_SMALL when this is > 0 --
        # without it, they fragment the main L1 pool. At the 1024-token
        # prefill bucket (anything >128 real tokens rounds up to it) this was
        # observed to TT_THROW "Statically allocated circular buffers...
        # clash with L1 buffers" during prefill warmup -- a general
        # Gemma4-31B TP=8 issue, not specific to dFlash's own tap-capture
        # (confirmed: the plain, non-dFlash text_demo.py::test_demo hit the
        # identical TT_THROW at the identical L1 addresses at this bucket
        # before the same fix was applied there too).
        #
        # 8192, not text_demo_v2.py's 24576: a later commit
        # (7c183561a7d, "Optimize DFlash ctx K/V commit and hidden_norm for
        # trace-safe fused decode") grew DFlashDrafter's own L1 footprint
        # enough that 24576 newly clashed with the TARGET model's prefill
        # SDPA at the SMALLER 128-token bucket (a different clash than the
        # one above -- same TT_THROW signature, different root cause: not
        # enough main-pool L1 left once that reservation is carved out,
        # rather than main-pool fragmentation from too little of one). 8192
        # is the smallest value that resolved both the 1024-bucket clash and
        # the 128-bucket one in the same run -- verified at both buckets on
        # real hardware; 0 (no reservation) fixes 128 but reintroduces the
        # 1024 clash, 24576 (and higher, tried up to 131072) fixes 1024 but
        # breaks 128.
        "l1_small_size": int(os.environ.get("GEMMA4_L1_SMALL_SIZE", 8192)),
    },
)
def test_demo_dflash_fused_decoder(mesh_device, device_params, reset_seeds):
    # DFlashDrafter's persistent L1 footprint + fp32 per-head norm buffers clash
    # at the 128-token prefill bucket; keep the bf16 norm path for this demo only.
    os.environ.setdefault("GEMMA4_PREFILL_HEAD_NORM_FP32", "0")
    import ttnn
    from models.demos.gemma4.demo.text_demo_v2 import create_tt_page_table
    from models.demos.gemma4.tt.dflash_drafter import DFlashDrafter, DFlashFusedDecoder
    from models.demos.gemma4.tt.generator import Gemma4Generator
    from models.tt_transformers.tt.common import PagedAttentionConfig, preprocess_inputs_prefill

    model_path = os.getenv("HF_MODEL")
    if not model_path:
        pytest.skip("set HF_MODEL (target) to run")
    weights_dir = os.environ.get("MODEL_WEIGHTS_DIR")
    if not weights_dir:
        # _spec_get_drafter needs this to load the target's own embed_tokens
        # weight for the drafter's tied lm_head -- point it at the target
        # snapshot dir when the caller hasn't set it explicitly.
        import glob

        hits = glob.glob(
            os.path.expanduser(f"~/.cache/huggingface/hub/models--*--{model_path.split('/')[-1]}/snapshots/*/")
        )
        weights_dir = hits[0].rstrip("/") if hits else None
    if not weights_dir or not os.path.isdir(weights_dir):
        pytest.skip(f"MODEL_WEIGHTS_DIR not found/set (tried {weights_dir!r}); set it explicitly")

    snap = os.environ.get("GEMMA4_DFLASH_DRAFTER") or _dflash_default_snapshot()
    if not snap:
        pytest.skip("dFlash drafter snapshot not found; set GEMMA4_DFLASH_DRAFTER")

    if os.environ.get("GEMMA4_DFLASH_SHARD_ARGMAX") != "1":
        pytest.skip(
            "set GEMMA4_DFLASH_SHARD_ARGMAX=1 -- required to route around a known TTSampling "
            "multi-row broadcast limitation (see module docstring)"
        )

    # GEMMA4_DFLASH_PROMPT_FILE takes priority: Linux caps a single env var/argv
    # string at MAX_ARG_STRLEN (128 KiB) -- a real prompt long enough to exercise
    # a large ISL (tens of thousands of tokens, well over that many bytes) TT_FATALs
    # the shell itself ("Argument list too long") before ever reaching Python if
    # passed via GEMMA4_DFLASH_PROMPT directly.
    prompt_file = os.environ.get("GEMMA4_DFLASH_PROMPT_FILE")
    if prompt_file:
        with open(prompt_file) as f:
            prompt = f.read()
    else:
        prompt = os.environ.get("GEMMA4_DFLASH_PROMPT", DEFAULT_PROMPT)
    max_new = int(os.environ.get("GEMMA4_DFLASH_MAX_NEW", 256))

    paged_attention_config = PagedAttentionConfig(
        block_size=PAGE_BLOCK_SIZE, max_num_blocks=math.ceil(MAX_SEQ_LEN / PAGE_BLOCK_SIZE)
    )

    generator, tt_kv_cache, tokenizer = Gemma4Generator.from_pretrained(
        mesh_device=mesh_device,
        model_path=model_path,
        max_batch_size=1,
        max_seq_len=MAX_SEQ_LEN,
        num_layers=None,
        paged_attention_config=paged_attention_config,
        bounded_sliding_kv_cache=False,
    )
    model0 = generator.model[0]
    page_table = create_tt_page_table(1, paged_attention_config)

    def _embed_loader():
        import json

        from safetensors import safe_open

        idx = json.load(open(f"{weights_dir}/model.safetensors.index.json"))
        key = next(
            k
            for k in idx["weight_map"]
            if k.endswith("language_model.embed_tokens.weight") or k.endswith("model.embed_tokens.weight")
        )
        with safe_open(f"{weights_dir}/{idx['weight_map'][key]}", framework="pt") as f:
            return f.get_tensor(key)

    drafter = DFlashDrafter(
        mesh_device=mesh_device,
        drafter_path=snap,
        target_embed_weight_loader=_embed_loader,
        mesh_config=model0.mesh_config,
        ccl_manager=model0.ccl_manager,
        tensor_cache_path=None,
        # This demo knows its target ISL bucket upfront (MAX_SEQ_LEN); let the
        # drafter shrink its default block_size once that estimate exceeds
        # GEMMA4_DFLASH_LONG_CTX_THRESHOLD -- see recommended_dflash_block_size.
        # GEMMA4_DFLASH_BLOCK, if set, still overrides this unconditionally.
        ctx_len_hint=MAX_SEQ_LEN,
    )

    # Chat/instruct-formatted prompt: what the dFlash drafter was validated
    # against (dflash_drafter.py module docstring: "mean greedy acceptance
    # ~3-4.3 of a 16 block") and what real vLLM serving always feeds it. A raw
    # continuation prompt is off-distribution for this it-tuned pair and
    # degenerates to near-zero acceptance.
    in_pt, encoded, decoding_pos, _prefill_lens = preprocess_inputs_prefill(
        [prompt], tokenizer, generator.model_args, True, max_new + 32, max_prefill_len=MAX_SEQ_LEN
    )
    in_pt = torch.stack(in_pt).view(1, -1)
    n = int(decoding_pos[0])
    anchor_token = int(encoded[0][n - 1])

    logger.info("=" * 70)
    logger.info(f"DFlash fused-decoder demo -- prompt: {prompt!r}")
    logger.info(f"prompt tokens: {n}  |  max_new: {max_new}  |  model: {model_path}")
    logger.info("=" * 70)

    # Real prefill with dFlash tap capture (Gemma4DFlashForCausalLM.prefill_forward).
    #
    # Default path (GEMMA4_DFLASH_PREFILL_TRACE=1): arms the tap hook's
    # copy-into-persistent-buffers mode -- the same trace-safe mechanism the
    # steady-state verify step already uses inside DFlashFusedDecoder's fused
    # trace -- and calls prefill_forward_text with enable_trace=True. ONE call:
    # this session's first (and, in this one-shot demo, only) prefill pays the
    # one-time compile+capture cost, same as eager would; the win is for the
    # NEXT call at the same shape (a second demo run, or -- the realistic
    # case -- vLLM serving repeated requests at the same padded shape), which
    # would replay the captured trace instead of dispatching each op fresh.
    # Set GEMMA4_DFLASH_PREFILL_TRACE_BENCH=1 to run a SECOND back-to-back call
    # and log the capture-vs-replay timing split (adds a throwaway prefill
    # purely for that measurement -- not representative of a real session's
    # cost, only useful to reproduce/verify the speedup number itself).
    #
    # Fallback path (GEMMA4_DFLASH_PREFILL_TRACE=0, or MAX_SEQ_LEN > one
    # chunk): enable_trace=False, taps captured by the hook's OTHER mode
    # (python-side clone-append) -- a traced replay skips that hook entirely,
    # so this is the only correct mode outside the single-chunk case above.
    n_taps = len(drafter.target_layer_ids)
    use_prefill_trace = GEMMA4_DFLASH_PREFILL_TRACE and MAX_SEQ_LEN <= PREFILL_CHUNK_SIZE
    if GEMMA4_DFLASH_PREFILL_TRACE and not use_prefill_trace:
        logger.warning(
            f"GEMMA4_DFLASH_PREFILL_TRACE=1 but MAX_SEQ_LEN={MAX_SEQ_LEN} exceeds one prefill chunk "
            f"({PREFILL_CHUNK_SIZE}) -- multi-chunk traced prefill isn't supported yet (see the "
            "tap-buffer comment above). Falling back to eager prefill."
        )

    if use_prefill_trace:
        bench = os.environ.get("GEMMA4_DFLASH_PREFILL_TRACE_BENCH", "0") == "1"
        tap_buffers = [None] * n_taps
        model0.dflash_capture_taps(drafter.target_layer_ids, buffers=tap_buffers)
        try:
            t_pf0 = time.perf_counter()
            generator.prefill_forward_text(
                in_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                enable_trace=True,
                warmup_prefill=False,
            )
            t_pf1 = time.perf_counter()
            if bench:
                generator.prefill_forward_text(
                    in_pt,
                    page_table=page_table,
                    kv_cache=tt_kv_cache,
                    prompt_lens=decoding_pos,
                    enable_trace=True,
                    warmup_prefill=False,
                )
                t_pf2 = time.perf_counter()
                logger.info(
                    f"[prefill-trace] capture+first call: {t_pf1 - t_pf0:.3f}s  |  "
                    f"replay: {t_pf2 - t_pf1:.3f}s  |  speedup: {(t_pf1 - t_pf0) / max(t_pf2 - t_pf1, 1e-9):.1f}x"
                )
            else:
                logger.info(f"[prefill-trace] capture+first call: {t_pf1 - t_pf0:.3f}s")
        finally:
            model0.dflash_capture_taps(None)
        # prefill_ingest() deallocates every tensor it consumes -- correct for
        # the default clone-append mode (each request's taps are single-use),
        # but tap_buffers are persistent/boot-owned and must survive for the
        # NEXT traced replay (this session's or a future one reusing the same
        # generator). Hand it disposable clones instead of the originals.
        taps = [ttnn.clone(b) for b in tap_buffers]
    else:
        model0.dflash_capture_taps(drafter.target_layer_ids, keep_last=12)
        try:
            t_pf0 = time.perf_counter()
            generator.prefill_forward_text(
                in_pt,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                prompt_lens=decoding_pos,
                enable_trace=False,
                warmup_prefill=False,
            )
            logger.info(f"[prefill-eager] call: {time.perf_counter() - t_pf0:.3f}s")
        finally:
            taps = model0.pop_dflash_taps()
            model0.dflash_capture_taps(None)

    # NOT pre-warming the plain-decode fallback trace here (tried it): capturing
    # it before DFlash's own fused trace exists reserves L1 circular-buffer space
    # that DFlash's own capture then can't fit into --
    # "Statically allocated circular buffers in program 772 clash with L1
    # buffers... L1 buffer allocated at 1121536 and static circular buffer
    # region ends at 1266912" -- a hard TT_THROW, not a soft perf hit. Fixing
    # that would mean re-tuning L1 reservations for two simultaneously-live
    # traces (same class of work as this file's own l1_small_size history) --
    # out of scope here. The fallback (below) still captures its trace lazily
    # on first use instead, paying a one-time compile tax only in the sessions
    # that actually need it.

    # Fused decoder bootstrap (Gemma4DFlashForCausalLM._spec_bootstrap): one-time
    # drafter ctx ingest + fused-trace compile/capture.
    kv_layers = _unwrap_kv_layers(tt_kv_cache)
    dec = DFlashFusedDecoder(model0, drafter, kv_layers, page_table[:1])
    dec.prefill_ingest(taps, n)
    t_cap0 = time.perf_counter()
    dec.capture(anchor_token, n, max_new=max_new)
    logger.info(f"trace capture (one-time compile): {time.perf_counter() - t_cap0:.1f}s")

    # Steady-state decode loop (Gemma4DFlashForCausalLM.decode_forward's
    # internal block loop): call .step() back to back until max_new tokens
    # have been committed.
    eos = getattr(model0.hf_config, "eos_token_id", 1)
    eos_set = set(eos) if isinstance(eos, (list, tuple)) else {int(eos)}
    vocab = drafter.vocab

    committed = []
    iters = 0
    first = True
    hit_eos = False
    fell_back = False
    fallback_iter = None
    fallback_eval_enabled = True
    accept_history = collections.deque(maxlen=GEMMA4_DFLASH_FALLBACK_WINDOW)
    t_dec0 = time.perf_counter()
    while len(committed) < max_new:
        accepted, bonus, _produced = dec.step(first=first)
        first = False
        iters += 1
        toks = list(accepted) + [bonus]
        toks = [t if 0 <= t < vocab else int(bonus if 0 <= bonus < vocab else 1) for t in toks]
        committed.extend(toks)
        if eos_set & set(toks):
            hit_eos = True
            break
        if GEMMA4_DFLASH_FALLBACK and fallback_eval_enabled:
            accept_history.append(len(accepted))
            if len(accept_history) == accept_history.maxlen:
                mean_accept = sum(accept_history) / len(accept_history)
                if mean_accept < GEMMA4_DFLASH_FALLBACK_THRESHOLD:
                    remaining = max_new - len(committed)
                    if remaining >= GEMMA4_DFLASH_FALLBACK_MIN_REMAINING:
                        fell_back = True
                        fallback_iter = iters
                        logger.warning(
                            f"Adaptive fallback: mean accepted-drafts/iter over last "
                            f"{accept_history.maxlen} iterations = {mean_accept:.2f} < "
                            f"{GEMMA4_DFLASH_FALLBACK_THRESHOLD} -- DFlash isn't paying for its own "
                            f"overhead on this prompt. Switching to plain autoregressive decode for "
                            f"the remaining {remaining} tokens."
                        )
                        break
                    else:
                        # Acceptance has collapsed, but too little budget remains to
                        # earn back the one-time switch-trace-compile cost (see
                        # GEMMA4_DFLASH_FALLBACK_MIN_REMAINING's derivation above) --
                        # switching would make this session slower, not faster.
                        # Stay on DFlash and stop re-evaluating (remaining budget only
                        # shrinks from here, so this verdict can't flip later in the
                        # same session).
                        fallback_eval_enabled = False
                        logger.info(
                            f"Adaptive fallback: mean accepted-drafts/iter over last "
                            f"{accept_history.maxlen} iterations = {mean_accept:.2f} < "
                            f"{GEMMA4_DFLASH_FALLBACK_THRESHOLD}, but only {remaining} tokens remain "
                            f"(< {GEMMA4_DFLASH_FALLBACK_MIN_REMAINING} needed to amortize the switch "
                            f"cost) -- staying on DFlash for the rest of this session."
                        )
    dflash_iters = iters
    dflash_committed = len(committed)
    dflash_wall = time.perf_counter() - t_dec0

    fallback_wall = 0.0
    if fell_back and not hit_eos and len(committed) < max_new:
        from models.demos.gemma4.demo.sampling_utils import build_device_sampling_params, model_can_sample_on_device

        # DFlash's fused trace leaves its tap-capture hook armed on model0 (still
        # copying hidden_states into the block-sized tap buffers every forward
        # call, for its own steady-state use) -- plain decode's single-token
        # hidden_states shape doesn't match those buffers and TT_FATALs in
        # Gemma4Model.__call__'s ttnn.copy otherwise. Same call the demo already
        # uses to release the hook after prefill (see above); needed again here
        # since DFlashFusedDecoder re-arms it for the steady-state loop.
        model0.dflash_capture_taps(None)

        can_sample = model_can_sample_on_device(model0)
        device_sampling_params = build_device_sampling_params({"temperature": 0}, can_sample=can_sample)
        # dec.anchor / dec.start are exactly the last-committed-token / next-write-
        # position bookkeeping DFlashFusedDecoder itself uses (see .step()) --
        # continuing plain decode from here needs no other handoff state: the
        # target's own KV cache (kv_layers, the same tensors as tt_kv_cache) already
        # holds real, verified K/V up to dec.start from DFlash's own verify passes.
        out_tok = torch.tensor([[dec.anchor]], dtype=torch.long)
        current_pos = torch.tensor([dec.start], dtype=torch.long)
        t_fb0 = time.perf_counter()
        # One-step-of-slack async pipelining (mirrors text_demo_v2.run_demo_text's
        # decode loop) -- a fully synchronous submit-then-read loop here measured
        # ~7 tok/s (host-dispatch-bound, no overlap between device execution and
        # host readback) vs. baseline's ~23 tok/s, which would make the fallback
        # itself the bottleneck. out_tok/current_pos are NOT updated from the
        # readback below: once the decode trace is captured, Gemma4 tracks its own
        # sampled-token/position feedback in trace-persistent device buffers and
        # ignores the host values on replay (see device_tracks_decode_on_device in
        # sampling_utils.py) -- same as text_demo_v2's own pipelined branch.
        pending = []
        while len(committed) < max_new:
            decode_out = generator.decode_forward(
                out_tok,
                current_pos,
                page_table=page_table,
                kv_cache=tt_kv_cache,
                sampling_params=device_sampling_params,
                enable_trace=True,
                read_from_device=False,
            )
            pending.append(generator.read_decode_output(decode_out, async_read=True))
            current_pos = current_pos + 1
            if len(pending) > 1:
                host_out, read_events = pending.pop(0)
                for event in read_events:
                    ttnn.event_synchronize(event)
                toks, _ = generator.process_decode_output_host(host_out, is_tokens=True)
                next_tok = int(toks.reshape(-1)[0])
                committed.append(next_tok)
                if next_tok in eos_set:
                    hit_eos = True
                    break
        # Drain the one in-flight read left over from the one-step-of-slack loop.
        if not hit_eos and pending and len(committed) < max_new:
            host_out, read_events = pending.pop(0)
            for event in read_events:
                ttnn.event_synchronize(event)
            toks, _ = generator.process_decode_output_host(host_out, is_tokens=True)
            next_tok = int(toks.reshape(-1)[0])
            committed.append(next_tok)
            if next_tok in eos_set:
                hit_eos = True
        fallback_wall = time.perf_counter() - t_fb0

    wall = dflash_wall + fallback_wall
    tps = len(committed) / wall if wall > 0 else float("nan")
    # bonus token excluded per iter -- only over the DFlash portion; fallback
    # tokens are plain autoregressive, "accepted/iter" doesn't apply to them.
    avg_accept = (dflash_committed - dflash_iters) / dflash_iters if dflash_iters else float("nan")
    fallback_committed = len(committed) - dflash_committed
    text = tokenizer.decode(committed)

    logger.info("=" * 70)
    logger.info("Generated text:")
    logger.info(text)
    logger.info("=" * 70)
    logger.info(
        f"{len(committed)} tokens total ({dflash_committed} via {dflash_iters} dFlash iterations"
        f"{f' + {fallback_committed} via plain decode' if fell_back else ''}), {wall:.2f}s "
        f"-> {tps:.1f} tok/s overall  |  mean accepted-drafts/iter (DFlash phase): {avg_accept:.2f}  |  "
        f"hit_eos={hit_eos}"
    )
    if fell_back:
        logger.info(
            f"Adaptive fallback triggered at iteration {fallback_iter} "
            f"({dflash_wall:.2f}s DFlash + {fallback_wall:.2f}s plain decode)"
        )
    logger.info("=" * 70)
