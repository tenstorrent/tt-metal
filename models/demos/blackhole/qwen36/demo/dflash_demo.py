# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6-27B end-to-end text generation with DFLASH SPECULATIVE DECODING, on a T3K mesh.

The sibling of text_demo.py, with the decode loop replaced. text_demo generates one token per
target forward; this drafts a 16-slot block with the 1.73B DFlash drafter and verifies all 16 in
ONE target forward, committing every slot the target would have produced anyway.

WHAT THIS IS FOR. Speculation is only worth having if it beats simply running the model, and that
comparison is easy to get wrong -- a drafter that emits plausible text at high speed can be
silently wrong. Two things make the claim checkable here:

* **The output is exact, not approximate.** Greedy speculation accepts a drafted slot only when it
  matches the target's own argmax, so the emitted tokens are the tokens the 27B would have emitted
  on its own. The drafter can only change HOW FAST they arrive, never WHAT they are. That is why
  this demo reports acceptance length alongside tok/s: acceptance is the speedup, and the text is
  the correctness.
* **The baseline is named.** Production traced decode on this model measures 17.87 tok/s / 56.0
  ms/tok at ISL 128 (text_demo.py, use_trace=True). A speculative loop slower than that number is
  not an optimization however impressive its own before/after looks.

WHAT RUNS TRACED. The verify forward is a captured trace (Qwen36Model.capture_verify_trace): a
masked, mid-sequence, all-row forward that no existing capture covers, and the single largest win
in this path (0.21x -> 1.19x of production traced decode). The DRAFTER runs eagerly, deliberately:
tracing it was built, measured at 1.00x / 0.91x / 0.71x across three configurations and abandoned,
because staging and the KV commit -- neither of which a capture can hold -- cost what the traced
dispatch saves (tests/reference/test_dflash_drafter_trace.py).

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s

    # one case
    ... pytest models/demos/blackhole/qwen36/demo/dflash_demo.py -v -s -k "spec_128"
"""

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_wormhole_b0_or_blackhole
from models.demos.blackhole.qwen36.demo.text_demo import _assert_output_quality, _get_prompt
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

#: DFlash is a T3K (8-chip) path: the 27B is TP=8 and the drafter is replicated across the same mesh.
_MESH_SHAPE = {"N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(os.environ.get("MESH_DEVICE"), (1, 8))

#: Must hold the verify trace. One capture of a 128-row masked forward over 64 layers plus the LM
#: head; 250 MB measured sufficient, and the capture fails loudly ("Cannot load new binaries") rather
#: than silently if it is not.
_TRACE_REGION_SIZE = 250_000_000
DEVICE_PARAMS = [
    {
        "l1_small_size": 24576,
        "fabric_config": ttnn.FabricConfig.FABRIC_1D,
        "trace_region_size": _TRACE_REGION_SIZE,
    }
]

PAGED_BLOCK_SIZE = 64
#: 64 x 64 = 4096 tokens of paged KV, which is also TtTarget's capacity ceiling (page_table width x
#: block size). Prompt + generation must fit.
NUM_BLOCKS = 64

#: Production traced decode on this model, for the only comparison that decides whether speculation
#: was worth doing (text_demo.py traced_128, ISL 128).
PRODUCTION_TOK_S = 17.87


def _build(mesh_device, ctx_capacity=None):
    """The 27B target, the DFlash drafter, and the speculative loop's view of both."""
    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    # device_taps=True keeps the target's residual taps on the mesh for the ttnn drafter -- five
    # fewer PCIe round trips per step than reading them back to host.
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    # ctx_capacity makes the drafter's KV history a persistent fixed buffer, which collapses its
    # per-step shape space to the block width alone -- the precondition for warm_block_widths, and
    # so for retiring the compile-under-a-parked-trace hang. Acceptance is unaffected: priced at
    # 5.182 tok/step both ways (tests/reference/test_dflash_acceptance_capacity.py).
    # DFLASH_SHARE_CCL=0 gives the drafter its OWN TT_CCL instead of the model's.
    #
    # TT_CCL hands out global semaphores round-robin -- get_and_cycle_ag_semaphore_handles advances
    # (idx + 1) % 2 over TWO per axis -- and a captured trace bakes whichever handle it held. Sharing
    # the instance means the drafter's tap all-gather cycles the same pool the target's TRACED
    # collectives use, which is a period-2 mechanism, and acceptance here alternates with period 2
    # (DFLASH_HANDOFF.md: traced generations 7.000 / 1.500 / 7.000; the demo measures the bad one and
    # runs 2.99 tok/s instead of 17.54). The known-good test_dflash_traced_throughput.py does NOT
    # share, which is the structural difference between them.
    #
    # An earlier check "refuted" shared tt_ccl, but it compared sharing against not-sharing on the
    # PROSE test's control -- it never tested it against the parity effect, which is what this knob
    # is for.
    drafter = TtDrafter(
        TtDFlashDrafter(
            mesh_device,
            cfg,
            load_drafter_state_dict(drafter_path),
            **({"tt_ccl": model.tt_ccl} if os.environ.get("DFLASH_SHARE_CCL", "1") == "1" else {}),
            ctx_capacity=ctx_capacity,
        ),
        target,
    )
    return model, target, drafter, cfg


def _log_results(perf, prompt_len, stats, text):
    tok_s, ms_tok = perf["tok_s"], perf["ms_tok"]
    logger.info("=" * 70)
    logger.info(f"  Compile (warmup):      {perf['compile_s']:.3f}s")
    logger.info(f"  Prompt:                {prompt_len} tokens (prefill included in the rate below)")
    logger.info(f"  Generate:              {ms_tok:.1f} ms/token  ({tok_s:.2f} tok/s)")
    logger.info(f"  Acceptance:            {stats.mean_acceptance_length:.3f} tok per target forward")
    logger.info(f"  Steps:                 {len(stats.acceptance_lengths)} for {stats.num_output_tokens} tokens")
    logger.info(f"  vs production decode:  {tok_s / PRODUCTION_TOK_S:.2f}x ({PRODUCTION_TOK_S} tok/s)")
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
        pytest.param(512, 100, id="spec_512"),
    ],
)
def test_demo_dflash(mesh_device, device_params, seqlen, max_generated_tokens, reset_seeds, ensure_gc):
    """Speculative generation with a traced verify: throughput, acceptance, and the text."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to load the full 27B and its drafter")

    from transformers import AutoTokenizer

    assert seqlen + max_generated_tokens <= NUM_BLOCKS * PAGED_BLOCK_SIZE, (
        f"prompt {seqlen} + generation {max_generated_tokens} exceeds the "
        f"{NUM_BLOCKS * PAGED_BLOCK_SIZE}-token paged KV; raise NUM_BLOCKS"
    )

    try:
        # DFLASH_CTX_CAPACITY=1 sizes a fixed-capacity drafter history to this case and warms
        # every block width before the capture. See warm_block_widths for why the growing-history
        # default cannot be warmed at all.
        # Fixed-capacity drafter by DEFAULT (DFLASH_CTX_CAPACITY=0 opts out). It is what lets the
        # drafter keep its KV history at stable addresses across generations, which is the fix for
        # BOTH the generation-parity collapse and the anchor-crossing collapse -- they were one bug,
        # a buffer reallocated under a parked trace. It also bounds the drafter's shape space so
        # every block width can be compiled before the capture, retiring the hang.
        cap = None
        if os.environ.get("DFLASH_CTX_CAPACITY", "1") != "0":
            cap = -(-(seqlen + max_generated_tokens + 32) // 32) * 32
        model, target, drafter, cfg = _build(mesh_device, ctx_capacity=cap)
    except Exception as e:  # noqa: BLE001 -- a missing drafter checkpoint is a skip, not a failure
        if "drafter" in str(e).lower() or "DFLASH_HF_MODEL" in str(e):
            pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")
        raise

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    # DFLASH_PROMPT overrides the shared prompt file with a literal, un-padded string. The point is
    # diagnostic: the shipped ISL-128 prompt is the condiment question repeated and clipped
    # mid-sentence, while every acceptance figure in this work was measured on "The capital of
    # France is". Setting this runs the demo's own loop on the reference prompt, which separates
    # "prose is harder to draft" from "acceptance is decaying" -- they predict different numbers.
    override = os.environ.get("DFLASH_PROMPT")
    if override:
        token_ids = tokenizer(override, return_tensors="pt").input_ids
        logger.info(f"DFLASH_PROMPT override: {override!r} -> {token_ids.shape[1]} tokens")
    else:
        token_ids = _get_prompt(seqlen, tokenizer, max_prompt_len=seqlen)
    prompt_len = token_ids.shape[1]
    logger.info(f"prompt {prompt_len} tokens, generating {max_generated_tokens}, block_size {cfg.block_size}")

    # Capture the verify trace, then WARM. The first generation compiles programs and would other-
    # wise be charged to the measured run -- the same bias that made an A/B in this work read 0.85x
    # when the truth was 1.00x. The warm-up is short; it only has to touch each program once.
    # EAGER generation FIRST, then capture. capture_verify_trace warms only its own forward; the
    # loop needs more (drafter projections, tap gather, LM head at the drafted width, the whole-
    # bucket eager fallback). Capturing first leaves those uncompiled and the first traced
    # generation compiles them with a trace parked -- which hangs the process and wedges the device.
    # TtTarget.enable_traced_verify now asserts this rather than letting it happen.
    #
    # AND ONE EAGER GENERATION IS NOT ENOUGH -- IT MUST COVER THE SHAPES THE MEASURED RUN WILL USE.
    # A warm-up compiles only the programs ITS OWN shapes touch. Any shape first met after the
    # capture compiles with a trace parked, which is the same hang the rule above exists to prevent.
    # `max_new_tokens=8` was the bug: it never reaches the block widths a long generation ends on.
    # Every step takes `verify_size = min(block_size, max_length - start, target.max_block(start))`,
    # so a generation of `max_generated_tokens` produces narrow tail blocks (and, past the 128-row
    # anchor, `max_block`-capped ones) that an 8-token warm-up never produces. Measured 2026-09-15:
    # test_dflash_prose_throughput.py hung deterministically on exactly this, at a 2-wide final
    # block, and warming at the real budget fixed it. See DFLASH_HANDOFF.md §0.
    #
    # So warm at the REAL token budget. This costs one eager generation of the full length, which is
    # why `compile_s` below is large -- that is the point of reporting it separately.
    t0 = time.perf_counter()
    if cap is not None:
        logger.info(f"fixed-capacity drafter ({cap} rows); warming every block width before capture")
        drafter.drafter.warm_block_widths()
        # Safe now that the drafter's history sits at stable addresses: the trace can serve every
        # bucket rather than only the first, which is worth 6.19 -> 14.88 tok/s on this prompt.
        target.allow_trace_past_anchor = True
    dflash_generate(drafter, target, token_ids, max_new_tokens=max_generated_tokens)
    # DFLASH_NARROW_HEAD=1 runs the verify LM head over a 32/64-row tile-aligned window instead of
    # the whole 128-row bucket (+20.8 % on the reference prompt, tokens bit-identical -- see
    # tests/reference/test_dflash_narrow_head.py). Off by default: it is validated on one prompt at
    # one length so far. It scales STEP TIME only, so it cannot rescue a run whose acceptance has
    # collapsed -- throughput is acceptance / step_time.
    # Narrow head ON by default (DFLASH_NARROW_HEAD=0 opts out): the verify LM head runs over a
    # 32/64-row tile-aligned window instead of the whole 128-row bucket. Tokens are bit-identical
    # -- greedy verification cannot change them -- and it is worth 11.31 -> 14.88 tok/s here, on
    # top of the +20.8 % measured in isolation (tests/reference/test_dflash_narrow_head.py).
    target.enable_traced_verify(narrow_head=os.environ.get("DFLASH_NARROW_HEAD", "1") != "0")
    # DFLASH_TRACED_WARM: how many traced generations to run before the measured one. Acceptance
    # ALTERNATES with period 2 on the traced path -- odd traced generations give 7.000, even ones
    # 1.500 (tests/reference/test_dflash_warmup_position.py). One traced warm-up, the historical
    # value, lands the measurement on generation 2, i.e. the bad parity, every single run. This is
    # a DIAGNOSTIC KNOB, not a fix: the right fix is whatever makes the parity stop mattering.
    for _ in range(int(os.environ.get("DFLASH_TRACED_WARM", "1"))):
        dflash_generate(drafter, target, token_ids, max_new_tokens=max_generated_tokens)
    compile_s = time.perf_counter() - t0

    # NO separate TTFT probe here, deliberately. An earlier version timed the prefill by calling
    # target.reset() + target.forward() between the capture and the measured run. reset()
    # REALLOCATES the anchor's GDN snapshot, which Metal flags as "Allocating device buffers is
    # potentially unsafe due to the existence of an active trace" -- and the run produced fluent-
    # looking garbage ("...enhance different dishes. Do you don" then multilingual token soup) that
    # _assert_output_quality passed, because that check finds repetition and this was not repetitive.
    # The sequence below is the one test_dflash_traced_throughput.py exercises and trusts.
    ttft_s = float("nan")

    t0 = time.perf_counter()
    stats = dflash_generate(drafter, target, token_ids, max_new_tokens=max_generated_tokens, return_stats=True)
    total_s = time.perf_counter() - t0

    n = stats.num_output_tokens
    # Whole call, prefill included. Separating them needs an extra forward, and doing that after the
    # capture is what broke the run above.
    gen_s = total_s
    text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
    perf = {
        "compile_s": compile_s,
        "ttft_s": ttft_s,
        "tok_s": n / gen_s,
        "ms_tok": gen_s * 1000 / n,
    }
    _log_results(perf, prompt_len, stats, text)
    print(
        f"\n>>> DFlash {n} tokens @ {perf['tok_s']:.2f} tok/s ({perf['ms_tok']:.1f} ms/tok), "
        f"acceptance {stats.mean_acceptance_length:.2f}, {perf['tok_s'] / PRODUCTION_TOK_S:.2f}x production\n"
        f">>> {text[:400]!r}\n"
    )

    target.model.release_verify_trace()

    # Correctness. Acceptance >= 1 always holds trivially (the target's own bonus token is committed
    # even when every draft is rejected), so assert something stronger: speculation must actually be
    # speculating, i.e. committing more than one token per target forward on average.
    assert n >= 1, "generated nothing"
    assert stats.mean_acceptance_length > 1.0, (
        f"acceptance {stats.mean_acceptance_length:.3f} tok/step means every draft is being rejected "
        "and the loop has degenerated to plain autoregressive decoding through a slower path"
    )
    # Shared with text_demo: non-empty, no token >60% of output, no 8-gram repeated >10x. A drafter
    # fault cannot reach here (greedy verify makes the text the target's own), so a failure points at
    # the target or its traced verify.
    _assert_output_quality(text, n)
