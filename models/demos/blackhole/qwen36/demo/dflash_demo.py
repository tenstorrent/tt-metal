# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6-27B end-to-end text generation with DFlash speculative decoding on a T3K mesh.

The sibling of text_demo.py with the decode loop replaced: each step drafts a 16-slot block with the
DFlash drafter and verifies all 16 slots in one target forward. Greedy verification accepts a drafted
slot only when it matches the target's own argmax, so the emitted tokens are exactly the tokens the
27B would produce on its own; the drafter changes only how fast they arrive. The demo reports
acceptance length alongside tok/s and compares against production traced decode (text_demo.py).

The verify forward runs as a captured trace (Qwen36Model.capture_verify_trace); the drafter runs
eagerly.

Run::

    export DFLASH_RUN_TARGET=1
    export MESH_DEVICE=T3K
    export HF_MODEL=Qwen/Qwen3.6-27B
    export DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash
    export TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B
    pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s

    # one case
    ... pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s -k "spec_128"

Optional env vars: DFLASH_PROMPT overrides the prompt with a literal string. The anchor bucket is
sized to the request by TtTarget.anchor_for; DFLASH_ANCHOR=<rows> or DFLASH_AUTO_ANCHOR=0 (fixed 128)
override it, e.g. to exercise the anchor-crossing path.
"""

import json
import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import (
    SAMPLE_PROMPTS_DIR,
    _assert_output_quality,
    _get_prompt,
    _load_and_cache_context,
)
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.tt.dflash.config import (
    PAGED_BLOCK_SIZE,
    PRODUCTION_DECODE_TOK_S_T3K,
    TRACE_REGION_SIZE,
    load_drafter_state_dict,
    paged_blocks_for,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.speculative_drafter import TtDrafter
from models.demos.blackhole.qwen36.tt.dflash.target import TtTarget
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

#: DFlash is a T3K (8-chip) path: the 27B is TP=8 and the drafter is replicated across the same mesh.
_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))

DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": TRACE_REGION_SIZE,
    }
]


def _build(mesh_device, num_blocks, ctx_capacity=None, anchor=None):
    """The 27B target, the DFlash drafter, and the speculative loop's view of both."""
    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    max_seq_len = num_blocks * PAGED_BLOCK_SIZE
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=max_seq_len)
    kv_shape = [num_blocks, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)

    # device_taps=True keeps the target's residual taps on the mesh for the ttnn drafter instead of
    # reading them back to host every step.
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True, anchor=anchor)
    # ctx_capacity makes the drafter's KV history a persistent fixed-capacity buffer, which limits its
    # per-step shape space to the block width alone -- the precondition for warm_block_widths.
    # The drafter shares the model's TT_CCL for its tap all-gather.
    drafter = TtDrafter(
        TtDFlashDrafter(
            mesh_device,
            cfg,
            load_drafter_state_dict(drafter_path),
            tt_ccl=model.tt_ccl,
            max_seq_len=max_seq_len,
            ctx_capacity=ctx_capacity,
        ),
        target,
    )
    return model, target, drafter, cfg


def _long_prompt(isl, tokenizer):
    """At most ``isl`` tokens: an excerpt of Frankenstein plus a request to summarise it.

    The same source and instruction text_demo.py uses for its long ISLs. Acceptance depends on how
    predictable the output is, so the prompt must ask for new text: a document clipped mid-way and
    left to continue is copied near-verbatim, which the drafter predicts almost perfectly.
    """
    with open(f"{SAMPLE_PROMPTS_DIR}/eval_frankenstein_long.json") as f:
        entries = json.load(f)
    # Entries grow from 70k characters of Frankenstein (~17k tokens) to the whole of War and Peace;
    # take the first long enough for this ISL (~4 characters per token, with margin).
    entry = next((e for e in entries if int(e.get("max_length", 0)) >= 5 * isl), entries[-1])
    text = _load_and_cache_context(entry["context"], entry.get("max_length"))
    # Skip the Project Gutenberg licence header so the context is the novel itself.
    marker = "*** START OF THE PROJECT GUTENBERG EBOOK"
    if marker in text:
        text = text[text.index(marker) :].split("\n", 1)[1]

    def _ids(context):
        messages = [{"role": "user", "content": f"{context}\n\n{entry['prompt']}"}]
        chat = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return tokenizer(chat, add_special_tokens=False, return_tensors="pt").input_ids

    doc_ids = tokenizer(text, add_special_tokens=False).input_ids
    n = isl - _ids("").shape[1]
    assert len(doc_ids) >= n, f"context is too short for ISL {isl}"
    ids = _ids(tokenizer.decode(doc_ids[:n]))
    # Re-tokenising the clipped context can merge or split a token at the cut; trim from the context.
    while ids.shape[1] > isl:
        n -= ids.shape[1] - isl
        ids = _ids(tokenizer.decode(doc_ids[:n]))
    return ids


def _log_results(perf, prompt_len, stats, text):
    logger.info("=" * 70)
    logger.info(f"  Compile (warmup):      {perf['compile_s']:.3f}s")
    logger.info(f"  Prompt:                {prompt_len} tokens")
    logger.info(f"  TTFT:                  {perf['ttft_s'] * 1000:.1f} ms")
    logger.info(
        f"  Decode TPS:            {perf['decode_tok_s']:.2f} tok/s  ({1000 / perf['decode_tok_s']:.1f} ms/token)"
    )
    logger.info(f"  End-to-end:            {perf['tok_s']:.2f} tok/s  (prefill included)")
    logger.info(f"  Acceptance:            {stats.mean_acceptance_length:.3f} tok per target forward")
    logger.info(f"  Steps:                 {len(stats.acceptance_lengths)} for {stats.num_output_tokens} tokens")
    logger.info(
        f"  vs production decode:  {perf['decode_tok_s'] / PRODUCTION_DECODE_TOK_S_T3K:.2f}x "
        f"({PRODUCTION_DECODE_TOK_S_T3K} tok/s)"
    )
    logger.info(f"  Text: {text[:6000]}")
    logger.info("=" * 70)


@run_for_wormhole_b0_or_blackhole()
@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [_MESH_SHAPE], indirect=True)
@pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)
@pytest.mark.parametrize(
    "seqlen, max_generated_tokens",
    [
        pytest.param(128, 100, id="spec_128"),
        pytest.param(128, 256, id="spec_128_long"),
        pytest.param(512, 100, id="isl_512"),
        pytest.param(1024, 100, id="isl_1k"),
        pytest.param(2048, 100, id="isl_2k"),
        pytest.param(3072, 100, id="isl_3k"),
        pytest.param(3968, 100, id="isl_3968"),
        pytest.param(8192, 100, id="isl_8k"),
        pytest.param(16384, 100, id="isl_16k"),
        pytest.param(24576, 100, id="isl_24k"),
        pytest.param(32768, 100, id="isl_32k"),
    ],
)
def test_demo_dflash(mesh_device, device_params, seqlen, max_generated_tokens, reset_seeds, ensure_gc):
    """Speculative generation with a traced verify: throughput, acceptance, and the text."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to load the full 27B and its drafter")

    from transformers import AutoTokenizer

    # Tokenize before building: the drafter's capacity and the anchor are sized from the actual
    # prompt length, which with DFLASH_PROMPT is not `seqlen`.
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    # DFLASH_PROMPT overrides the shared prompt file with a literal, un-padded string.
    override = os.environ.get("DFLASH_PROMPT")
    if override:
        token_ids = tokenizer(override, return_tensors="pt").input_ids
        logger.info(f"DFLASH_PROMPT override: {override!r} -> {token_ids.shape[1]} tokens")
    else:
        # Up to 256 tokens: the shared question prompt. Longer: a Frankenstein excerpt to summarise.
        token_ids = (
            _get_prompt(seqlen, tokenizer, max_prompt_len=seqlen) if seqlen <= 256 else _long_prompt(seqlen, tokenizer)
        )
    prompt_len = token_ids.shape[1]

    try:
        # Fixed-capacity drafter history by default (DFLASH_CTX_CAPACITY=0 opts out), sized to this
        # request. It keeps the drafter's KV history at stable addresses across generations -- a
        # buffer reallocated under a parked trace corrupts it -- and bounds the drafter's shape space
        # so every block width can be compiled before the capture.
        cap = None
        if os.environ.get("DFLASH_CTX_CAPACITY", "1") != "0":
            cap = -(-(prompt_len + max_generated_tokens + 32) // 32) * 32
        # The anchor is sized to this request: a crossing costs an eager whole-bucket forward and a
        # trace re-capture, while a wider bucket costs padded rows on every step. TtTarget.anchor_for
        # weighs both, with the per-step term paid over the generation budget. DFLASH_ANCHOR
        # overrides it, and DFLASH_AUTO_ANCHOR=0 restores the fixed 128.
        anchor = None
        if os.environ.get("DFLASH_AUTO_ANCHOR", "1") != "0" and not os.environ.get("DFLASH_ANCHOR"):
            total = prompt_len + max_generated_tokens
            anchor = TtTarget.anchor_for(total, new_tokens=max_generated_tokens)
            logger.info(
                f"auto anchor {anchor} for {prompt_len} + {max_generated_tokens} tokens "
                f"({TtTarget.crossings_for(total, anchor)} crossings)"
            )
        num_blocks = paged_blocks_for(prompt_len + max_generated_tokens)
        model, target, drafter, cfg = _build(mesh_device, num_blocks, ctx_capacity=cap, anchor=anchor)
    except Exception as e:  # noqa: BLE001 -- a missing drafter checkpoint is a skip, not a failure
        if "drafter" in str(e).lower() or "DFLASH_HF_MODEL" in str(e):
            pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")
        raise

    # DFLASH_NUM_SPEC=<n> drafts n tokens per step (a block of n + 1 slots) instead of the checkpoint's
    # own num_speculative_tokens. warm_block_widths compiles every width up to the checkpoint's block,
    # so any n below it is safe under the parked trace.
    num_spec = int(os.environ.get("DFLASH_NUM_SPEC", cfg.num_speculative_tokens))
    assert (
        1 <= num_spec <= cfg.num_speculative_tokens
    ), f"DFLASH_NUM_SPEC={num_spec} must be in [1, {cfg.num_speculative_tokens}]"
    block_size = num_spec + 1
    logger.info(
        f"prompt {prompt_len} tokens, generating {max_generated_tokens}, block_size {block_size} ({num_spec} drafted)"
    )

    # Run an eager generation first, then capture. capture_verify_trace warms only its own forward;
    # the loop also needs the drafter projections, tap gather, LM head at the drafted width and the
    # whole-bucket eager fallback. Any program first compiled while a trace is parked hangs the
    # process and wedges the device (TtTarget.enable_traced_verify asserts against it).
    #
    # The eager warm-up must cover every shape the measured run will use, so it runs at the real
    # token budget: each step takes `verify_size = min(block_size, max_length - start,
    # target.max_block(start))`, so a long generation produces narrow tail blocks (and, past the
    # anchor, `max_block`-capped ones) that a short warm-up never reaches. This is why `compile_s`
    # is large and reported separately.
    t0 = time.perf_counter()
    if cap is not None:
        logger.info(f"fixed-capacity drafter ({cap} rows); warming every block width before capture")
        drafter.drafter.warm_block_widths()
        # Traced verify past the anchor is on by default (DFLASH_TRACE_PAST_ANCHOR=0 opts out): after
        # a crossing the verify trace is re-captured (TtTarget._recapture_after_anchor) instead of
        # falling back to eager verifies for the rest of the generation. It changes step time only;
        # greedy verification accepts exact argmax matches, so the tokens are the same either way.
        target.allow_trace_past_anchor = os.environ.get("DFLASH_TRACE_PAST_ANCHOR", "1") != "0"
    gen_kwargs = {"max_new_tokens": max_generated_tokens, "block_size": block_size}

    eager_ids = dflash_generate(drafter, target, token_ids, **gen_kwargs)
    # Narrow head on by default (DFLASH_NARROW_HEAD=0 opts out): the verify LM head runs over a
    # 32/64-row tile-aligned window instead of the whole 128-row bucket. It cannot change the tokens.
    target.enable_traced_verify(narrow_head=os.environ.get("DFLASH_NARROW_HEAD", "1") != "0")
    # One traced generation before the measured one.
    dflash_generate(drafter, target, token_ids, **gen_kwargs)
    compile_s = time.perf_counter() - t0

    # TTFT comes from dflash_generate itself (DFlashStats.ttft_s); a separate prefill probe after
    # the capture would call target.reset(), which reallocates the anchor's GDN snapshot while the
    # trace is active and corrupts the traced run.
    t0 = time.perf_counter()
    stats = dflash_generate(drafter, target, token_ids, return_stats=True, **gen_kwargs)
    total_s = time.perf_counter() - t0

    n = stats.num_output_tokens
    text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
    perf = {
        "compile_s": compile_s,
        "ttft_s": stats.ttft_s,
        # Decode rate after the first token, the TPS convention text_demo.py reports.
        "decode_tok_s": (n - 1) / (total_s - stats.ttft_s),
        "tok_s": n / total_s,
    }
    _log_results(perf, prompt_len, stats, text)

    target.model.release_verify_trace()

    # Acceptance >= 1 holds trivially (the target's bonus token is committed even when every draft is
    # rejected), so require more than one committed token per target forward on average.
    assert n >= 1, "generated nothing"
    # Greedy verification accepts only the target's own argmax, so the traced run must emit exactly
    # the tokens of the eager warm-up -- across every anchor crossing and trace re-capture.
    assert torch.equal(stats.output_ids, eager_ids), "traced generation diverged from the eager warm-up"
    assert stats.mean_acceptance_length > 1.0, (
        f"acceptance {stats.mean_acceptance_length:.3f} tok/step means every draft is being rejected "
        "and the loop has degenerated to plain autoregressive decoding through a slower path"
    )
    # Shared with text_demo: non-empty, no token >60% of output, no 8-gram repeated >10x. Greedy
    # verification makes the text the target's own, so a failure points at the target or its traced
    # verify rather than the drafter.
    _assert_output_quality(text, n)
