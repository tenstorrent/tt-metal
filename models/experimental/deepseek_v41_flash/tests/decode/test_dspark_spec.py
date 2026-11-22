# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Speculative decode on the full ``DeepSeekV41Model``.

``test_dspark_spec_decode`` runs the model's ``persistent_moe`` layout on the 8x4 mesh: one TP4 stage (row 4) holds
every layer's Engram, attention, hyper-connections and norms, every layer's MoE runs on the persistent TP12
``fused_experts`` launch on rows 5-7, and DSpark runs alone on row 3. It needs the TP12 expert cache
(``tools/fill_expert_cache.py --all-layers --tp-size 12 --mesh-rows 3``). ``test_parallel_verify_rows`` keeps the
seven-stage TP4 pipeline (rows 0-6, DSpark on row 7).

By default (``DEEPSEEK_V4_SPEC_VERIFY=block``) the model is built with ``verify_rows=4`` and every DSpark round is
ONE traced step of 4 rows (``[next, d_1, d_2, d_3]``: 3 drafts + 1) through every layer's ``(4, 32)``-tiled block
ops (``DeepSeekV41Model.decode_verify_block``), the persistent MoE serving 4-row requests; greedy only (temperature
0), fixed at 3 drafts. See :func:`_block_spec_decode`: it logs tok/s/u of plain greedy decode and of DSpark on the
same model, and checks exactness. ``DEEPSEEK_V4_SPEC_VERIFY=pipelined`` runs the older pipelined verify described
below (one-row steps per verified row, a sweep over draft counts).

One model serves three runs of the same conversation, each from position 0: the demo prompt, then
whenever the model ends its answer (end of sentence) the next of ``_FOLLOW_UPS`` as a new user turn
after it, until ``DEEPSEEK_V4_SPEC_TOKENS`` (default 2000) tokens are generated or the follow-ups run out:

1. plain greedy decode, the baseline;
2. speculative decode fed the baseline's own tokens with a third of them corrupted, so every
   round rejects somewhere -- the output must equal the baseline token for token, which holds only
   if the rejected rows leave no trace in the rings and compressor windows;
3. speculative decode with the DSpark drafter, sampled at ``DEEPSEEK_V4_SPEC_TEMPERATURE`` (default 1) and
   ``DEEPSEEK_V4_SPEC_TOP_P`` (default 1) with DeepInfer's accept rule (``DSparkSpeculator``) -- its
   acceptance, throughput and text are logged. At temperature 0 it is greedy and must equal the baseline.
   With the pipelined verify it runs once per draft count in ``DEEPSEEK_V4_SPEC_DRAFTS`` (default ``all``: 1
   to the most a round takes, see :func:`_draft_counts`; or a comma list such as ``2,4``), and a table of
   tokens/round, ms/round and tok/s per draft count, greedy as 0 drafts, is logged at the end. Temperature 0
   compares the counts on the same text.

Runs 2 and 3 verify a round's ``DRAFTS + 1`` rows as pipelined single-token steps. The persistent MoE takes one
row per request, so the parallel verify (``DEEPSEEK_V4_SPEC_VERIFY=parallel`` / ``both``) skips this test.
``DEEPSEEK_V4_SPEC_EXACT=0`` logs a divergence from the baseline instead of failing.
With the pipelined verify alone, every run is on the Rust host runtime (``model.start_host_runtime``), as the
server runs, and an unbuilt extension (``host_rs/build.sh``) fails the test; ``DEEPSEEK_V41_RUST_HOST=0`` keeps the
Python dispatch and reader instead.

Logging: by default each run logs one summary line (tokens, acceptance, tok/s), and the DSpark sweep ends with
its table. ``DEEPSEEK_V4_SPEC_VERBOSE=1`` adds the progress lines: every spec round's start position, accepted
drafts and phase times (``[spec] round ...``), every verified row (``[spec] round p=...: row ...``), each prefill,
extend and end of sentence, the greedy loop every 200 tokens, and every run's decoded text. A phase that runs
longer than ``DEEPSEEK_V4_SPEC_STALL_SECONDS`` (default 30) dumps the model's dispatch state and every thread's
stack (``[spec stall]``) either way.

The DSpark experts (bfloat8_b) come from ``tools/fill_expert_cache.py --dspark --dtype bfloat8_b``.

    DEEPSEEK_V4_SPEC_TOKENS=2000 pytest -s models/experimental/deepseek_v41_flash/tests/decode/test_dspark_spec.py
"""

from __future__ import annotations

import contextlib
import os
import random
import time

import pytest
import torch
from loguru import logger
from transformers import AutoTokenizer

import ttnn
from models.experimental.deepseek_v41_flash.tests.decode.test_full_model_decode_demo import (
    _DEFAULT_TEXT,
    _available_bytes,
    _build_rope,
    _checkpoint_available,
    _construct_model,
    _tokenize_chat,
    _traced_max_seq,
)
from models.experimental.deepseek_v41_flash.tt import host_rs
from models.experimental.deepseek_v41_flash.tt import spec as spec_module
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    ENGRAM_RAM_MARGIN,
    engram_ram_needed,
    stage_engram_tables,
)
from models.experimental.deepseek_v41_flash.tt.spec import DSparkSpeculator
from models.experimental.deepseek_v4_flash.encoding_dsv4 import encode_messages, eos_token
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tt_py_test_utils_common.tensor_prefetcher import tensor_prefetcher_session

DRAFTS = 4
# The pipelined DSpark runs' drafts per round: "all" or a comma list (see _draft_counts).
DRAFT_COUNTS = os.environ.get("DEEPSEEK_V4_SPEC_DRAFTS", "all")
CORRUPT = 1 / 3
VERIFY = os.environ.get("DEEPSEEK_V4_SPEC_VERIFY", "block")
# The block verify's rows: 3 drafts + 1.
BLOCK_ROWS = 4
# Block-verify exactness: a token that is not the block path's argmax must be within this many logits of it (a
# near-tie). The block path is deterministic, and the persistent MoE's rows reduce exactly (a row's output is the same
# bits whatever the other rows of its block route to), but a token's logits still depend on where its block starts:
# the same stream teacher-forced at the four block alignments disagrees on ~2-3% of tokens by up to ~1.4 logits
# (measured with DEEPSEEK_V4_SPEC_ALIGN_CONTROL=1; 300 tokens, after the MoE fix: spec runs' worst 1.25, the
# alignments' 1.375), so a spec run, whose blocks start wherever the last round ended, does too. Over a whole run the
# noise has a tail: the per-layer replay (DEEPSEEK_V4_SPEC_REPLAY=1) shows the kept rows bit-identical whatever the
# rejected rows hold, while the two alignments' logits differ by 1.3 median / 5.4 max, and a default-prompt run's worst
# deficit reached 2.125. The bit-exact checks (the replay, the probing oracle) are the correctness gates; this bound
# only flags gross errors.
NEAR_TIE = float(os.environ.get("DEEPSEEK_V4_SPEC_NEAR_TIE", "2.5"))
# Also teacher-force each checked stream with the blocks shifted by 1..3 positions: the alignment noise floor.
ALIGN_CONTROL = os.environ.get("DEEPSEEK_V4_SPEC_ALIGN_CONTROL", "0") == "1"
# Block verify: the bitwise rollback check (rejected rows leave no trace), before the spec runs.
ROLLBACK_CHECK = os.environ.get("DEEPSEEK_V4_SPEC_ROLLBACK_CHECK", "0") == "1"
# DEEPSEEK_V4_SPEC_PROFILE=n: profile n plain and n verify steps after the greedy baseline, then stop (see _profile_steps).
PROFILE_STEPS = int(os.environ.get("DEEPSEEK_V4_SPEC_PROFILE", "0"))
# Progress lines: the spec rounds' and rows', each prefill / extend / end of sentence, the greedy loop every 200
# steps and every run's text. Off: one summary line per run (see the module docstring).
_VERBOSE = os.environ.get("DEEPSEEK_V4_SPEC_VERBOSE", "0") not in ("0", "", "false", "False")
# 0: log a divergence from the baseline instead of failing (for throughput runs).
EXACT = os.environ.get("DEEPSEEK_V4_SPEC_EXACT", "1") != "0"
# The DSpark run's sampling (DSparkSpeculator.temperature / top_p); temperature 0 is greedy.
TEMPERATURE = float(os.environ.get("DEEPSEEK_V4_SPEC_TEMPERATURE", "1"))
TOP_P = float(os.environ.get("DEEPSEEK_V4_SPEC_TOP_P", "1"))
# DEEPSEEK_V4_SPEC_DUMP=<file.pt> (block verify): after the greedy baseline, teacher-force the prompt + baseline through
# eager verify blocks and save, per position, the tap rows the drafter receives, the target-layer stream means, the
# block path's argmax, and the device drafter's greedy proposals at every frontier (see _dump_drafter_inputs).
DUMP_PATH = os.environ.get("DEEPSEEK_V4_SPEC_DUMP", "")
# DEEPSEEK_V4_SPEC_REPLAY=1 (block verify): replay the DSpark run's rounds eagerly, as run and with the rejected rows
# replaced by noise, and against the teacher-forced blocks; log where the kept rows' streams part (_replay_rounds).
REPLAY = os.environ.get("DEEPSEEK_V4_SPEC_REPLAY", "0") == "1"
# DEEPSEEK_V4_SPEC_PROMPT (block verify): the user turn, a name in _PROMPTS or the text itself (default the demo's).
PROMPT = os.environ.get("DEEPSEEK_V4_SPEC_PROMPT", "demo")
_PROMPTS = {
    "demo": _DEFAULT_TEXT,
    # GSM8K test problems (the domain of the CPU reference's published acceptance).
    "gsm8k": (
        "Solve these three problems step by step, and give each final answer as a number.\n"
        "1. Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
        "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
        "How much in dollars does she make every day at the farmers' market?\n"
        "2. A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?\n"
        "3. Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. "
        "This increased the value of the house by 150%. How much profit did he make?"
    ),
}
# test_parallel_verify_rows' decode rows; the eighth row is the drafter's.
DECODE_STAGES = 7
# Asked in order, one after each end of sentence, with the conversation so far as context.
_FOLLOW_UPS = (
    "Great list! Now do the same for the top 10 best and worst TV series of all time, with the same format.",
    "Pick three movies from your best list and write a short, funny review of each, about a paragraph long.",
    "Plan the movie night for me: a schedule from 6 pm to midnight, snacks matched to each film, and games "
    "to play in between.",
    "Write a funny invitation message for my friends that mentions some of the movies we will watch.",
    "Suggest ten more underrated movies that would fit the night, each with a one-line pitch and a joke.",
    "Write a short trivia quiz of ten questions about the movies on your lists, with the answers at the end.",
    "Now give me a top 10 list of the best movie soundtracks, with a sentence about each and a joke.",
    "Finally, write a funny closing speech I can give at the end of the night to thank everyone.",
)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(14400)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            # As in test_full_model_decode_demo: without L1_SMALL the CCL semaphores and all-reduce scratch land in
            # L1 and cap fused_experts' circular buffers. The parallel verify's S-row all-reduces add scratch the
            # demo's 8 KB cannot hold.
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "16384")),
        }
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_1stage_persistent_moe12"])
def test_dspark_spec_decode(mesh_device, monkeypatch):
    # spec.py reads DEEPSEEK_V4_SPEC_VERBOSE at import, defaulting to on; this test defaults it off.
    monkeypatch.setattr(spec_module, "_VERBOSE", _VERBOSE)
    if VERIFY == "block":
        _block_spec_decode(mesh_device)
        return
    if VERIFY != "pipelined":
        pytest.skip(f"DEEPSEEK_V4_SPEC_VERIFY={VERIFY!r}: block or pipelined")
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    nbytes = engram_ram_needed(loader, config.engram_layer_ids)
    if _available_bytes() < ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")

    n_tokens = int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "2000"))
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)
    turns = [_tokenize_turn(tokenizer, text) for text in _FOLLOW_UPS]
    # Every follow-up asked, plus a verify block of the most drafts past the last position.
    most_drafts = max(DRAFTS, config.dspark_block_size)
    needed = len(prompt) + n_tokens + sum(len(turn) for turn in turns) + 2 * (most_drafts + 1)
    max_seq = _traced_max_seq(config, needed, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)
    modes = (False,)  # pipelined verify only (see the skip above)

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device,
            stack,
            loader=loader,
            config=config,
            num_stages=1,
            persistent_moe=True,
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        model.prepare_spec_decode(rope["main"])
        if model.use_prefetcher:
            stack.enter_context(tensor_prefetcher_session(model.dspark.device))
        model.activate_session(model.open_session())
        model.capture_spec_traces(prompt[0], 0)
        if host_rs.enabled() and modes == (False,):
            model.start_host_runtime()
        logger.info(f"[test] host runtime: {'Rust' if model.host_runtime_running else 'Python'}")

        # 1. Plain greedy decode. Each end of sentence asks the next follow-up, until n_tokens are generated
        # or the follow-ups run out.
        out = model.decode_prompt_traced(prompt, 0)
        stream = list(prompt)  # stream[q] is the token at position q
        baseline = []  # the generated tokens, without the follow-ups
        # Greedy's top1 - top2 logit gap per token: a divergence at a near-tie is bf16 / bfp8 reduction
        # order, not a wrong row (DSPARK_PARALLEL_VERIFY.md, Numerics).
        gaps = []
        plain_seconds, plain_steps = 0.0, 0
        while True:
            baseline.append(int(out.reshape(-1).float().argmax()))
            gaps.append(_top2_gap(out))
            asked = baseline.count(eos_id)
            if len(baseline) == n_tokens or (baseline[-1] == eos_id and asked > len(turns)):
                stream.append(baseline[-1])
                break
            if baseline[-1] == eos_id:
                feed = [eos_id, *turns[asked - 1]]
                if _VERBOSE:
                    logger.info(
                        f"[test] greedy: end of sentence {asked}/{len(turns)} at {len(baseline)} tokens, "
                        f"feeding a {len(feed)}-token user turn from {len(stream)}"
                    )
                out = model.decode_prompt_traced(feed, len(stream))
                stream += feed
            else:
                t0 = time.perf_counter()
                out = model.decode_traced(baseline[-1], len(stream))
                plain_seconds += time.perf_counter() - t0
                plain_steps += 1
                stream.append(baseline[-1])
                if _VERBOSE and plain_steps % 200 == 0:
                    logger.info(
                        f"[test] greedy: {len(baseline)}/{n_tokens} tokens, pos {len(stream)}, "
                        f"{plain_steps / plain_seconds:.2f} tok/s"
                    )
        greedy_tps = plain_steps / plain_seconds
        logger.info(
            f"greedy: {len(baseline)} tokens, {baseline.count(eos_id)} end(s) of sentence, {greedy_tps:.2f} tok/s"
        )
        if _VERBOSE:
            logger.info(f"greedy output: {tokenizer.decode(stream[len(prompt):])!r}")

        def check(run: str, got: list[int], exact: bool) -> None:
            if got == baseline[: len(got)]:
                return
            diff = _first_diff(got, baseline)
            message = (
                f"{run} diverged at token {diff}: {got[diff]} vs greedy {baseline[diff]}, "
                f"greedy top1 - top2 gap {gaps[diff]:.4f}"
            )
            assert not exact, message
            logger.warning(message)

        def converse(spec: DSparkSpeculator) -> list[int]:
            """The baseline's conversation through ``spec``: its generated tokens, without the follow-ups."""
            got = [spec.prefill(prompt)]
            if _VERBOSE:
                logger.info(
                    f"[spec] run start: {'parallel' if spec.parallel_verify else 'pipelined'} verify, "
                    f"prefill gave {got[0]}, pos {spec.position}"
                )
            while len(got) < n_tokens:
                asked = got.count(eos_id)
                if got[-1] == eos_id:
                    if asked > len(turns):
                        break
                    if _VERBOSE:
                        logger.info(f"[spec] end of sentence {asked}/{len(turns)}: extending with the next user turn")
                    got.append(spec.extend(turns[asked - 1]))
                else:
                    got += spec.generate(n_tokens - len(got), stop_ids=(eos_id,))
            if _VERBOSE:
                logger.info(f"[spec] run done: {len(got)} tokens, pos {spec.position}")
            return got

        draft_counts = _draft_counts(model)
        sweep = {}  # drafts -> the pipelined DSpark run's SpecStats
        for parallel_verify in modes:
            verify = "parallel verify" if parallel_verify else "pipelined verify"

            # 2. Corrupted oracle drafts: exactness under rejection. The same corruptions in both modes.
            rng = random.Random(0)

            def propose(token, frontier):
                return [
                    stream[q] if q < len(stream) and rng.random() > CORRUPT else rng.randrange(config.vocab_size)
                    for q in range(frontier + 2, frontier + 2 + DRAFTS)
                ]

            oracle = DSparkSpeculator(model, DRAFTS, propose=propose, parallel_verify=parallel_verify)
            got = converse(oracle)
            logger.info(f"oracle ({DRAFTS} drafts, 1/3 corrupted), {verify}: {oracle.stats.summary(DRAFTS)}")
            check(f"oracle spec ({verify})", got, EXACT)

            # 3. The DSpark drafter, at every draft count (the parallel verify's traces take DRAFTS + 1 rows only).
            # Sampling gives up equality with the baseline.
            for drafts in [DRAFTS] if parallel_verify else draft_counts:
                spec = DSparkSpeculator(
                    model, drafts, parallel_verify=parallel_verify, temperature=TEMPERATURE, top_p=TOP_P
                )
                got = converse(spec)
                logger.info(
                    f"DSpark, {verify}, {drafts} drafts, temperature {TEMPERATURE}, top_p {TOP_P}: "
                    f"{spec.stats.summary(drafts)} (greedy {greedy_tps:.2f} tok/s)"
                )
                if _VERBOSE:
                    logger.info(f"DSpark output ({verify}, {drafts} drafts): {tokenizer.decode(got)!r}")
                if not parallel_verify:
                    sweep[drafts] = spec.stats
                if TEMPERATURE == 0:
                    check(f"DSpark spec ({verify}, {drafts} drafts)", got, EXACT)

        if sweep:
            lines = [
                f"DSpark tok/s by drafts per round (pipelined verify, temperature {TEMPERATURE}, top_p {TOP_P}):",
                "  drafts  tokens/round  ms/round   tok/s  vs greedy",
                f"  {0:>6}  {1:>12.2f}  {1000 / greedy_tps:>8.1f}  {greedy_tps:>6.2f}  {1:>8.2f}x",
            ]
            for drafts, stats in sweep.items():
                rounds = len(stats.accepted)
                if not rounds:
                    lines.append(f"  {drafts:>6}  no rounds")
                    continue
                tps = stats.tokens / stats.seconds
                lines.append(
                    f"  {drafts:>6}  {stats.tokens / rounds:>12.2f}  {1000 * stats.seconds / rounds:>8.1f}  "
                    f"{tps:>6.2f}  {tps / greedy_tps:>8.2f}x"
                )
            logger.info("\n".join(lines))


def _profile_steps(model, baseline: list[int], n_prompt: int, rows: int) -> None:
    """``DEEPSEEK_V4_SPEC_PROFILE=n``: ``n`` plain traced steps, then ``n`` block verify steps (after a warmup one),
    each between device profiler reads of the stage and the drafter (run under tracy with ``-r``), and the host-side
    timing of each. The verify steps feed the greedy stream, so every row is accepted."""
    from tracy import signpost

    devices = [model.submeshes_io[0]["device"]] + ([model.dspark.device] if model.dspark is not None else [])

    def read_profiler():
        for device in devices:
            ttnn.ReadDeviceProfiler(device)

    read_profiler()
    plain_ms = []
    for i in range(PROFILE_STEPS):
        pos = n_prompt + len(baseline) - 1
        signpost(f"PLAIN_{i}")
        t0 = time.perf_counter()
        out = model.decode_traced(baseline[-1], pos)
        plain_ms.append(1000 * (time.perf_counter() - t0))
        baseline.append(int(out.reshape(-1)[0]))
        read_profiler()
    pos = n_prompt + len(baseline) - 1
    stream = [baseline[-1]] * rows
    model.decode_verify_block(stream, pos)  # the server switch to block requests, outside the profile
    model.spec_fixup(list(range(pos, pos + rows)))
    model.spec_fixup([])
    read_profiler()
    model._verify_timing = []
    verify_ms = []
    for i in range(PROFILE_STEPS):
        signpost(f"VERIFY_{i}")
        t0 = time.perf_counter()
        model.decode_verify_block(stream, pos)
        verify_ms.append(1000 * (time.perf_counter() - t0))
        read_profiler()
        model.spec_fixup(list(range(pos, pos + rows)))
        model.spec_fixup([])
        read_profiler()
    logger.info(f"[profile] plain step ms {[round(x, 2) for x in plain_ms]}")
    logger.info(f"[profile] verify step ms {[round(x, 2) for x in verify_ms]}")
    logger.info(
        f"[profile] verify host timing {[{k: round(1000 * v, 2) for k, v in d.items()} for d in model._verify_timing]}"
    )
    model._verify_timing = None


def _block_spec_decode(mesh_device) -> None:
    """The block-verify mode of :func:`test_dspark_spec_decode`.

    One model (``verify_rows=4``, device sampling) runs, from position 0 on the demo prompt:

    1. plain greedy decode on the one-row traces, ``DEEPSEEK_V4_SPEC_TOKENS`` (default 2000) tokens, with the next of
       ``_FOLLOW_UPS`` as a new user turn after each end of sentence: the baseline and its tok/s/u;
    2. DSpark rounds fed the baseline's own tokens with a third corrupted (every round rejects somewhere);
    3. DSpark with its drafter, greedy: tok/s/u, tokens/round, ms/round and its split.

    Exactness at temperature 0. The 4-row block ops compute in a different bf16 / bfp8 order than the one-row ops,
    so at a near-tie the block path may pick the other token (the gaps are a few bf16 ulps). So each spec run is
    checked against the block path itself, teacher-forced on the run's own output (eager block steps after a
    one-row prefill of the prompt): every emitted token must be that step's argmax, or within ``NEAR_TIE`` logits
    of it. A rollback that left a rejected row's trace (ring, compressor window, Engram or drafter state) breaks
    this by far more. The first divergence from the greedy baseline is logged with the block path's top1 - top2 gap
    there. The block verify needs the dense select (prompt + tokens below 508 positions), so the spec runs cover the
    first ``block_tokens`` generated tokens (to the first end of sentence); the DSpark run then carries on with plain
    one-row greedy decode (and the follow-up turns) to ``DEEPSEEK_V4_SPEC_TOKENS``, and its whole conversation is
    printed.
    """
    rows, drafts = BLOCK_ROWS, BLOCK_ROWS - 1
    if PROFILE_STEPS:
        # The modules' own mid-forward profiler reads sync the stage, which can deadlock a compile step waiting on its
        # packet: only _profile_steps reads.
        import sys

        from models.experimental.deepseek_v4_flash.tt import common as v4_common

        for module in list(sys.modules.values()):
            if getattr(module, "_profile", None) is v4_common._profile:
                module._profile = lambda device: None
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    nbytes = engram_ram_needed(loader, config.engram_layer_ids)
    if _available_bytes() < ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    prompt = _tokenize_chat(tokenizer, _PROMPTS.get(PROMPT, PROMPT))
    dense_limit = config.index_topk - 2 * rows  # every compressed group in the dense select
    n_tokens = int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "2000"))
    # The spec runs' share: the block verify stops at the dense select's end.
    block_tokens = min(n_tokens, dense_limit - len(prompt))
    # Past the block verify's range, plain decode carries on, with follow-up turns after each end of sentence. A run
    # within the range stops at the end of sentence and sizes the traces to it (no one-row select variants).
    turns = [_tokenize_turn(tokenizer, text) for text in _FOLLOW_UPS] if n_tokens > block_tokens else []
    max_seq = _traced_max_seq(
        config, len(prompt) + n_tokens + sum(len(t) + 1 for t in turns) + 2 * rows, config.num_hidden_layers
    )
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=1, persistent_moe=True, verify_rows=rows
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head, device_sampling=True)
        model.device_sampler.set_sampling(0.0, 0)
        model.prepare_spec_decode(rope["main"])
        if model.use_prefetcher:
            stack.enter_context(tensor_prefetcher_session(model.dspark.device))
        model.activate_session(model.open_session())
        t0 = time.perf_counter()
        model.capture_spec_traces(prompt[0], 0)
        logger.info(f"[test] traces compiled and captured in {time.perf_counter() - t0:.0f}s")

        def token_of(out) -> int:
            return int(out.reshape(-1)[0])

        def decode_plain(conv: list[int], gen: list[int], timing: list[float]) -> None:
            """Plain one-row greedy decode until ``gen`` (the generated tokens, follow-up turns excluded) holds
            ``n_tokens``. ``conv`` is the conversation in the model (positions ``0 .. len(conv) - 1``); ``gen[-1]``
            goes in next, at ``len(conv)``. After an end of sentence the next follow-up turn goes in as a prompt
            (until they run out). ``timing`` accumulates [seconds, steps] of the one-row steps."""
            while len(gen) < n_tokens:
                asked = gen.count(eos_id)
                if gen[-1] == eos_id:
                    if asked > len(turns):
                        if turns:
                            logger.warning(f"the follow-up turns ran out after {len(gen)} tokens")
                        break
                    feed = [eos_id, *turns[asked - 1]]
                    out = model.decode_prompt_traced(feed, len(conv))
                    conv += feed
                else:
                    t0 = time.perf_counter()
                    out = model.decode_traced(gen[-1], len(conv))
                    timing[0] += time.perf_counter() - t0
                    timing[1] += 1
                    conv.append(gen[-1])
                gen.append(token_of(out))
                if _VERBOSE and len(gen) % 200 == 0:
                    logger.info(f"[test] plain decode: {len(gen)}/{n_tokens} tokens, pos {len(conv)}")

        def text_of(conv: list[int], gen: list[int]) -> str:
            """The conversation after the prompt: generated tokens and follow-up turns."""
            return tokenizer.decode(conv[len(prompt) :] + gen[-1:])

        # 1. Plain greedy decode on the one-row traces.
        out = model.decode_prompt_traced(prompt, 0)
        greedy_full, greedy_conv, timing = [token_of(out)], list(prompt), [0.0, 0]
        decode_plain(greedy_conv, greedy_full, timing)
        greedy_tps = timing[1] / timing[0]
        logger.info(
            f"greedy: {len(greedy_full)} tokens, {greedy_full.count(eos_id)} end(s) of sentence, "
            f"{greedy_tps:.2f} tok/s/u ({1000 / greedy_tps:.2f} ms/token)"
        )
        print(
            f"\n===== greedy output ({len(greedy_full)} tokens) =====\n{text_of(greedy_conv, greedy_full)}\n",
            flush=True,
        )
        # The spec runs' reference: the block-verify share, to the first end of sentence.
        first_eos = greedy_full.index(eos_id) + 1 if eos_id in greedy_full else len(greedy_full)
        baseline = greedy_full[: min(block_tokens, first_eos)]
        stream = prompt + baseline

        def converse(spec: DSparkSpeculator) -> list[int]:
            got = [spec.prefill(prompt)]
            if got[-1] != eos_id:
                got += spec.generate(len(baseline) - 1, stop_ids=(eos_id,))
            spec.finish()
            return got[: len(baseline)]

        def teacher_forced(got: list[int], shift: int = 0) -> list[tuple[int, float, float]]:
            """The block path on ``prompt + got``: per generated token, (argmax, top1 logit, the token's logit). The
            blocks start ``shift`` tokens past the prompt (the first ``shift`` tokens' entries are the stream's own,
            unchecked: one-row prefill)."""
            seq = prompt + got
            # Prefilled as the spec runs do: verify blocks (padding rows restored).
            _, padded = model.prefill_blocks(seq[: len(prompt) + shift], 0)
            model.spec_fixup(padded)
            res = [(got[i + 1], 0.0, 0.0) for i in range(shift)]
            for p in range(len(prompt) + shift, len(seq), rows):
                block = seq[p : p + rows]
                block += [block[-1]] * (rows - len(block))
                logits = model.debug_block_logits(block, p)
                for j in range(rows):
                    q = p + j + 1  # the position row j predicts
                    if q - len(prompt) >= len(got):
                        break
                    top = float(logits[j].max())
                    res.append((int(logits[j].argmax()), top, float(logits[j, seq[q]])))
            # The stream's first token is the prefill's (the prompt's last row): one-row, not checked here.
            return [(got[0], 0.0, 0.0)] + res[: len(got) - 1]

        block_ref: list[int] = []

        def check(run: str, got: list[int]) -> None:
            forced = teacher_forced(got)
            off = [(i, top - mine) for i, (arg, top, mine) in enumerate(forced) if arg != got[i]]
            worst = max((d for _, d in off), default=0.0)
            if ALIGN_CONTROL:
                # The noise floor: the same stream teacher-forced with the blocks shifted by 1..3 positions (no
                # rejections at all), each token's deficit to that alignment's argmax.
                for shift in range(1, rows):
                    other = teacher_forced(got, shift)
                    o = [(i, top - mine) for i, (arg, top, mine) in enumerate(other) if arg != got[i]]
                    logger.info(
                        f"{run}: blocks shifted by {shift}: {len(o)} tokens not the argmax, deficits up to "
                        f"{max((d for _, d in o), default=0.0):.3f} at {o[:10]}"
                    )
            logger.info(
                f"{run}: {len(got)} tokens; {len(off)} not the block path's argmax (all near-ties up to "
                f"{worst:.3f} logits, at {[i for i, _ in off][:8]})"
            )
            message = f"{run}: token(s) {off} are not the block path's greedy pick (> {NEAR_TIE})"
            assert worst <= NEAR_TIE or not EXACT, message
            if worst > NEAR_TIE:
                logger.warning(message)
            for name, ref in (("one-row greedy baseline", baseline), ("block greedy stream", block_ref)):
                if not ref:
                    continue
                if got == ref[: len(got)]:
                    logger.info(f"{run}: equal to the {name} over {len(got)} tokens")
                    continue
                diff = _first_diff(got, ref)
                if diff == 0:
                    logger.warning(f"{run} diverged from the {name} at the prefill's token")
                    continue
                # The block path at the divergence, on the common prefix: its pick, and its logit gap to the
                # reference's token there.
                arg, top, _ = teacher_forced(got[: diff + 1])[diff]
                _, _, theirs = teacher_forced(got[:diff] + [ref[diff]])[diff]
                same_after = sum(a == b for a, b in zip(got[diff + 1 :], ref[diff + 1 :]))
                logger.warning(
                    f"{run} diverged from the {name} at token {diff}: {got[diff]} vs {ref[diff]}; there the block "
                    f"path picks {arg}, top logit {top:.3f}, {top - theirs:.3f} above the reference's token; "
                    f"{same_after}/{len(got) - diff - 1} later tokens equal position by position"
                )

        def rollback_check() -> None:
            """A block whose rows past ``m`` are rejected: from the greedy prefix at ``p``, verify ``[s_p .. s_{p+m},
            tail]``, roll back rows ``m+1 ..``, verify the next block at ``p + m + 1``, for a true tail (twice: the
            path is deterministic, bit for bit) and two junk tails. Logged, not asserted: the kept rows' own logits
            already differ with the tail at the rounding level (see NEAR_TIE), and the next block by as much."""
            junk_rng = random.Random(1)
            for p in (len(prompt) + 8, len(prompt) + 9, len(prompt) + 130):
                if p + rows + 2 * rows > len(stream):
                    continue
                for m in range(rows - 1):
                    outs, kept = [], []
                    for variant in ("true", "true", "junk_a", "junk_b"):
                        _, padded = model.prefill_blocks(stream[:p], 0)
                        model.spec_fixup(padded)
                        block = stream[p : p + m + 1]
                        tail = stream[p + m + 1 : p + rows]
                        if variant != "true":
                            tail = [junk_rng.randrange(config.vocab_size) for _ in tail]
                        first = model.debug_block_logits(block + tail, p)
                        kept.append(first[: m + 1])
                        model.spec_fixup(list(range(p + m + 1, p + rows)))
                        q = p + m + 1
                        outs.append(model.debug_block_logits(stream[q : q + rows], q))
                    same = [torch.equal(outs[0], o) for o in outs[1:]]
                    diff = [round(float((outs[0] - o).abs().max()), 4) for o in outs[1:]]
                    kdiff = [round(float((kept[0] - k).abs().max()), 4) for k in kept[1:]]
                    logger.info(
                        f"rollback p={p} m={m}: next block equal to the true-rows run {same} (max diff {diff}); "
                        f"the kept rows' own logits differ by {kdiff}"
                    )
                    assert same[0], f"rollback at p={p}, m={m}: the block path is not deterministic"

        if PROFILE_STEPS:
            _profile_steps(model, baseline, len(prompt), rows)
            return

        if ROLLBACK_CHECK:
            rollback_check()

        if os.environ.get("DEEPSEEK_V4_SPEC_CAUSAL_PROBE", "0") == "1":
            # Same prefix, same rows 0..k, other tokens in rows k+1..: rows 0..k must not move. Per layer, the
            # first layer where a kept row's streams differ between two tails.
            junk_rng = random.Random(2)
            p = len(prompt) + 8
            for k in range(rows - 1):
                taps, logits, subs = [], [], []
                layer0 = model.layers[model.submeshes_io[0]["layers"][0]]
                for variant in range(3):
                    layer0._debug_taps = {}
                    _, padded = model.prefill_blocks(stream[:p], 0)
                    model.spec_fixup(padded)
                    tail = stream[p + k + 1 : p + rows]
                    if variant:
                        tail = [junk_rng.randrange(config.vocab_size) for _ in tail]
                    model._block_debug_streams = []
                    logits.append(model.debug_block_logits(stream[p : p + k + 1] + tail, p))
                    model._block_debug_streams = None
                    taps.append(model.block_debug_taps)
                    subs.append(layer0._debug_taps)
                    layer0._debug_taps = None
                for v in (1, 2):
                    sub = {}
                    for name, a in subs[0].items():
                        b = subs[v][name]
                        sub[name] = round(float((a - b).abs().max()), 5) if a.shape == b.shape else f"shape {a.shape}"
                    logger.info(f"causal probe kept rows 0..{k}, tail {v}: layer 0 sublayer max diffs (all rows) {sub}")
                    for name in ("tokens", "attn_out", "row"):
                        a, b = subs[0][name], subs[v][name]
                        logger.info(
                            f"  {name}: per-row max diff {[round(float((a[j] - b[j]).abs().max()), 5) for j in range(a.shape[0])]}"
                        )
                    for name in ("mixed", "out"):
                        a, b = subs[0][name], subs[v][name]
                        logger.info(
                            f"  {name}: per-row max diff {[round(float((a[:, j] - b[:, j]).abs().max()), 5) for j in range(a.shape[1])]}"
                        )
                hc = taps[0].shape[1] // rows
                for v in (1, 2):
                    a, b = taps[0].reshape(len(taps[0]), hc, rows, -1), taps[v].reshape(len(taps[v]), hc, rows, -1)
                    first_bad = {}
                    for j in range(k + 1):
                        diffs = (a[:, :, j] - b[:, :, j]).abs().amax(dim=(1, 2))
                        bad = (diffs > 0).nonzero()
                        first_bad[j] = (int(bad[0]) if len(bad) else None, [round(float(x), 4) for x in diffs[:6]])
                    lg = [round(float((logits[0][j] - logits[v][j]).abs().max()), 3) for j in range(rows)]
                    logger.info(
                        f"causal probe kept rows 0..{k}, tail {v}: logits diff by row {lg}; first layer "
                        f"where kept row j moves (and its first diffs): {first_bad}"
                    )

        if DUMP_PATH:
            _dump_drafter_inputs(model, prompt, baseline, rows, DUMP_PATH)

        results = {}
        # 2. Block greedy: block verify rounds whose drafts are all the noise id (never the target's pick), one token
        # per round. The block path's own greedy stream, the exactness reference of the spec runs below: the block ops
        # round differently from the one-row ops, so it parts from the one-row baseline at its first near-tie.
        noise = config.dspark_noise_token_id
        block_greedy = DSparkSpeculator(
            model, drafts, propose=lambda token, frontier: [noise] * drafts, parallel_verify=True, temperature=0.0
        )
        block_stream = converse(block_greedy)
        logger.info(f"block greedy (1 token per round): {block_greedy.stats.summary(drafts)}")
        results["blockgr"] = (block_greedy.stats, block_stream)

        # 3. Corrupted oracle drafts: exactness under rejection, and the verify / accept / rollback plumbing. No fixed
        # stream is an oracle for the block path: a block's logits depend on where it starts at the bf16-ulp level, so
        # a stream parts from any reference at its first near-tie. So the oracle probes the block path itself: before
        # each round at p it runs verify blocks [next, t_1 .. t_j, noise ..] at p (rolled back each time) to get the
        # block path's own t_1 .. t_k from these rows, then corrupts each with probability CORRUPT. Rows are causal and
        # bit-exact whatever the later rows hold, so the round must accept exactly the drafts before the first
        # corrupted one; any other count is a verify, accept or rollback bug.
        rng = random.Random(0)
        expected: list[int] = []

        def propose(token, frontier):
            p, seq = frontier + 1, [token]
            for j in range(drafts):
                out = model.decode_verify_block(seq + [noise] * (rows - len(seq)), p)
                model.spec_fixup(list(range(p, p + rows)))
                seq.append(out[j])
            corrupt = [rng.random() < CORRUPT for _ in range(drafts)]
            expected.append(next((j for j, c in enumerate(corrupt) if c), drafts))
            return [rng.randrange(config.vocab_size) if c else t for t, c in zip(seq[1:], corrupt)]

        oracle = DSparkSpeculator(model, drafts, propose=propose, parallel_verify=True, temperature=0.0)
        got = converse(oracle)
        accepted = oracle.stats.accepted
        # The last round may be cut at the token budget.
        wrong = [(i, a, e) for i, (a, e) in enumerate(zip(accepted[:-1], expected)) if a != e]
        logger.info(
            f"oracle ({drafts} drafts, 1/3 corrupted, probed; its draft time is the probes'), block verify: "
            f"{oracle.stats.summary(drafts)}; rounds accepting other than the uncorrupted prefix: {len(wrong)} of "
            f"{len(accepted)} {wrong[:10]}"
        )
        assert not wrong or not EXACT, f"oracle rounds (round, accepted, expected) {wrong[:10]}"
        results["oracle"] = (oracle.stats, got)

        # 4. The DSpark drafter, greedy.
        spec = DSparkSpeculator(model, drafts, parallel_verify=True, temperature=0.0)
        history = []
        if REPLAY:
            verify_block = model.decode_verify_block

            def recorded(token_ids, pos):
                tokens = verify_block(token_ids, pos)
                history.append((list(token_ids), pos, list(tokens)))
                return tokens

            model.decode_verify_block = recorded
        try:
            got = converse(spec)
        finally:
            if REPLAY:
                del model.decode_verify_block
        logger.info(f"DSpark, block verify, {drafts} drafts, greedy: {spec.stats.summary(drafts)}")
        results["dspark"] = (spec.stats, got)
        # Past the dense select: plain one-row decode from where DSpark stopped, to n_tokens.
        assert spec.position == len(prompt) + len(got) - 1, (spec.position, len(prompt), len(got))
        dspark_full, dspark_conv, tail_timing = list(got), prompt + got[:-1], [0.0, 0]
        decode_plain(dspark_conv, dspark_full, tail_timing)
        tail = len(dspark_full) - len(got)
        whole_s = spec.stats.seconds + tail_timing[0]
        logger.info(
            f"DSpark run: {len(got)} tokens with DSpark (block verify, {spec.stats.tokens / spec.stats.seconds:.2f} "
            f"tok/s/u), then {tail} with plain decode past the dense select "
            f"({tail_timing[1] / tail_timing[0] if tail_timing[1] else 0:.2f} tok/s/u); "
            f"{len(dspark_full)} tokens in all, {(spec.stats.tokens + tail_timing[1]) / whole_s:.2f} tok/s/u overall; "
            f"{'equal to' if dspark_full == greedy_full[: len(dspark_full)] else 'differs from'} the greedy output"
        )
        print(
            f"\n===== DSpark output ({len(dspark_full)} tokens: {len(got)} DSpark, {tail} plain) =====\n"
            f"{text_of(dspark_conv, dspark_full)}\n",
            flush=True,
        )
        if REPLAY:
            _replay_rounds(model, prompt, got, history, rows, noise)

        lines = [
            f"DSpark block verify T={rows} vs plain greedy decode on the same model:",
            f"  greedy (no DSpark)  {greedy_tps:7.2f} tok/s/u  {1000 / greedy_tps:6.2f} ms/token",
            "  run      tokens/round  ms/round   tok/s/u  vs greedy  split (ms/round)",
        ]
        for name, (stats, _) in results.items():
            n = len(stats.accepted)
            split = ", ".join(f"{k} {1000 * v / n:.2f}" for k, v in stats.phases.items())
            tps = stats.tokens / stats.seconds
            lines.append(
                f"  {name:<7}  {stats.tokens / n:>12.2f}  {1000 * stats.seconds / n:>8.2f}  {tps:>8.2f}  "
                f"{tps / greedy_tps:>8.2f}x  {split}"
            )
        logger.info("\n".join(lines))
        block_ref = results["blockgr"][1]
        for name, (_, got) in results.items():
            check(f"{name} spec (block verify)", got)


def _dump_drafter_inputs(model, prompt: list[int], baseline: list[int], rows: int, path: str) -> None:
    """Teacher-force ``prompt + baseline`` from position 0 through eager verify blocks (every row accepted) and save
    to ``path``: ``tokens`` [N]; ``taps`` [N, targets * D], the rows the drafter receives; ``means`` [N, targets * D],
    the target layers' input stream means from the block's streams; ``argmax`` [N], the block path's pick for the
    next position; ``drafts`` {frontier: the device drafter's greedy proposals for frontier + 2 ..}, drafted right
    after the block holding the frontier (its later rows are in the rings, masked)."""
    stream = prompt + baseline
    targets = [li - 1 for li in model.config.dspark_target_layer_ids]  # input of layer l = output of layer l - 1
    model.dspark.set_sampling(0.0)
    taps, means, argmax, drafts = [], [], [], {}
    model._block_debug_tap_rows = []
    t0 = time.perf_counter()
    try:
        for p in range(0, len(stream), rows):
            block = stream[p : p + rows]
            n = len(block)
            block = block + [block[-1]] * (rows - n)
            model._block_debug_streams = []
            logits = model.debug_block_logits(block, p)
            st = model.block_debug_taps  # [layers, hc * T, D]
            hc = st.shape[1] // rows
            m = st[targets].reshape(len(targets), hc, rows, -1).mean(dim=1)  # [targets, T, D]
            means.append(m.permute(1, 0, 2).reshape(rows, -1)[:n])
            taps.append(model._block_debug_tap_rows.pop()[:n])
            argmax.append(logits[:n].argmax(dim=-1))
            for f in range(max(p, len(prompt) - 1), p + n):
                if f + 1 >= len(stream):
                    break
                model.spec_draft_async(stream[f + 1], f)
                drafts[f] = model.spec_read_draft(conf=False)[0]
    finally:
        model._block_debug_streams = None
        model._block_debug_tap_rows = None
    torch.save(
        {
            "tokens": torch.tensor(stream),
            "n_prompt": len(prompt),
            "taps": torch.cat(taps),
            "means": torch.cat(means),
            "argmax": torch.cat(argmax),
            "drafts": drafts,
        },
        path,
    )
    k = rows - 1
    hits = [0] * k
    for f, ids in drafts.items():
        for j in range(k):
            q = f + 2 + j
            if q >= len(stream) or ids[j] != stream[q]:
                break
            hits[j] += 1
    logger.info(
        f"dump: {len(stream)} positions, {len(drafts)} drafts in {time.perf_counter() - t0:.0f}s to {path}; "
        f"teacher-forced P(first j drafts right) {[round(h / max(len(drafts), 1), 3) for h in hits]}"
    )


def _replay_rounds(model, prompt, got, history, rows, noise) -> None:
    """Replay a block-verify spec run's rounds (``history``: (inputs, p, tokens) per verify) eagerly from a fresh
    prefill, capturing every layer's output streams, in two variants: A as run, B with every rejected row's token
    replaced by ``noise``. With causal rows and an exact rollback, every kept row is bit-identical in A and B: the first
    round / layer where they differ locates a row coupling or a rollback hole. Then the kept rows of A against the same
    positions teacher-forced in prompt-aligned blocks (C, no rejections): per layer, the max diff at the positions
    whose C argmax is not the run's token (the NEAR_TIE deficits), i.e. how block-alignment noise grows."""
    n = len(prompt)
    history = [h for h in history if h[1] >= n]  # not the prompt's prefill blocks
    rounds = []
    for i, (ids, p, tokens) in enumerate(history):
        m = (
            history[i + 1][1] - p - 1
            if i + 1 < len(history)
            else next((j for j in range(rows - 1) if ids[j + 1] != tokens[j]), rows - 1)
        )
        rounds.append((ids, p, tokens, m))

    def replay(variant: str) -> tuple[dict, dict]:
        """Kept rows' per-layer streams [layers, hc, D] and logits [V], by input position."""
        streams, logits = {}, {}
        _, rejected = model.prefill_blocks(prompt, 0)
        for ids, p, tokens, m in rounds:
            model.spec_fixup(rejected)
            if variant == "B":
                ids = ids[: m + 1] + [noise] * (rows - m - 1)
            model._block_debug_streams = []
            lg = model.debug_block_logits(ids, p)
            st = model.block_debug_taps  # [layers, hc * T, D]
            st = st.reshape(st.shape[0], -1, rows, st.shape[-1])
            for j in range(m + 1):
                streams[p + j] = st[:, :, j].to(torch.bfloat16)
                logits[p + j] = lg[j]
            if variant == "A":
                picks = [int(x) for x in lg.argmax(dim=-1)]
                if picks[: m + 1] != tokens[: m + 1]:
                    logger.warning(f"replay A at p={p}: eager picks {picks} vs the run's {tokens} (m={m})")
            rejected = list(range(p + m + 1, p + rows))
        model._block_debug_streams = None
        return streams, logits

    sa, la = replay("A")
    sb, lb = replay("B")
    first = None
    for q in sorted(sa):
        if not torch.equal(sa[q], sb[q]) or not torch.equal(la[q], lb[q]):
            d = (sa[q].float() - sb[q].float()).abs().amax(dim=(1, 2))
            bad = (d > 0).nonzero()
            first = (q, int(bad[0]) if len(bad) else None, [round(float(x), 4) for x in d[:40:4]])
            break
    logger.info(
        f"replay: {len(rounds)} rounds; kept rows as run (A) vs rejected rows as noise (B): "
        + (
            "bit-identical"
            if first is None
            else f"first differ at position {first[0]}, layer {first[1]}, per-layer max diff (every 4th) {first[2]}"
        )
    )
    # C: the same stream teacher-forced in prompt-aligned blocks, every row accepted.
    sc, lc = {}, {}
    _, padded = model.prefill_blocks(prompt, 0)
    model.spec_fixup(padded)
    seq = prompt + got
    for p in range(n, len(seq), rows):
        block = seq[p : p + rows]
        k = len(block)
        model._block_debug_streams = []
        lg = model.debug_block_logits(block + [block[-1]] * (rows - k), p)
        st = model.block_debug_taps
        st = st.reshape(st.shape[0], -1, rows, st.shape[-1])
        for j in range(k):
            sc[p + j] = st[:, :, j].to(torch.bfloat16)
            lc[p + j] = lg[j]
    model._block_debug_streams = None
    common = [q for q in sorted(sa) if q in sc and q + 1 - n < len(got)]
    deficits = []
    for q in common:
        want = seq[q + 1]
        top = float(lc[q].max())
        if int(lc[q].argmax()) != want:
            deficits.append((q, top - float(lc[q][want])))
    layer_diff = torch.stack([(sa[q].float() - sc[q].float()).abs().amax(dim=(1, 2)) for q in common])  # [Q, L]
    logit_diff = torch.tensor([float((la[q] - lc[q]).abs().max()) for q in common])
    logger.info(
        f"replay A vs teacher-forced C over {len(common)} positions: per-layer max stream diff, median over positions "
        f"(every 4th layer) {[round(float(x), 4) for x in layer_diff.median(dim=0).values[::4]]}, max "
        f"{[round(float(x), 4) for x in layer_diff.amax(dim=0)[::4]]}; logits max diff median "
        f"{float(logit_diff.median()):.3f} max {float(logit_diff.max()):.3f}"
    )
    for q, deficit in deficits:
        i = common.index(q)
        logger.info(
            f"  deficit at position {q} (token {q + 1 - n}): {deficit:.3f} logits; A's logits for the run's token vs "
            f"C's: {float(la[q][seq[q + 1]]):.3f} / {float(lc[q][seq[q + 1]]):.3f}, top {float(la[q].max()):.3f} / "
            f"{float(lc[q].max()):.3f}; per-layer stream diff (every 4th) {[round(float(x), 4) for x in layer_diff[i, ::4]]}"
        )


# Both parities below the sliding window (masked), across the causal switch, and past the ring's wrap.
_VERIFY_STARTS = (0, 5, 120, 125, 130, 135, 200, 205, 300, 305)


@pytest.mark.skipif(not _checkpoint_available(), reason=f"V4.1-Flash checkpoint not found under {DEFAULT_MODEL_DIR}")
@pytest.mark.timeout(7200)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [
        {
            "fabric_config": ttnn.FabricConfig.FABRIC_2D_TORUS_XY,
            "num_command_queues": 2,
            "l1_small_size": int(os.environ.get("DEEPSEEK_V4_L1_SMALL_SIZE", "16384")),
        }
    ],
    indirect=["device_params"],
)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_8stage"])
def test_parallel_verify_rows(mesh_device):
    """Each row of a parallel verify against the single-row decode of the same position, from the same caches.

    The greedy continuation of the prompt is the stream. At each of ``_VERIFY_STARTS`` the block ``stream[p :
    p + T]`` is verified in one step, its rows are rolled back with ``spec_fixup`` (as a round that rejects
    every draft), and the same block is decoded one row at a time. Logs per row the PCC, the argmax match and
    the single-row top1 - top2 gap, and both blocks' wall times; fails on a row below 0.99 PCC.
    """
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    nbytes = engram_ram_needed(loader, config.engram_layer_ids)
    if _available_bytes() < ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")
    t = DRAFTS + 1
    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)
    n_stream = max(max(_VERIFY_STARTS) + t, len(prompt))
    max_seq = _traced_max_seq(config, n_stream + t, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=DECODE_STAGES, verify_rows=t
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        model.prepare_spec_decode(rope["main"], drafter=False)
        sid = model.open_session()
        model.activate_session(sid)
        model.capture_spec_traces(prompt[0], 0)

        out = model.decode_prompt_traced(prompt, 0)
        stream = list(prompt)
        while len(stream) < n_stream:
            stream.append(int(out.reshape(-1).float().argmax()))
            out = model.decode_traced(stream[-1], len(stream) - 1)
        model.reset_session(sid)

        worst, done = 1.0, 0
        for p in _VERIFY_STARTS:
            if p > done:
                model.decode_prompt_traced(stream[done:p], done)
            block = stream[p : p + t]
            model.spec_fixup([])
            t0 = time.perf_counter()
            got = model.decode_verify(block, p).float()
            parallel_ms = (time.perf_counter() - t0) * 1e3
            model.spec_fixup(range(p, p + t))
            ref = []
            t0 = time.perf_counter()
            model.decode_prompt_traced(block, p, on_output=lambda i, o: ref.append(o.reshape(-1).float().clone()))
            pipelined_ms = (time.perf_counter() - t0) * 1e3
            done = p + t
            rows = []
            for j in range(t):
                pcc = float(torch.corrcoef(torch.stack([got[j], ref[j]]))[0, 1])
                worst = min(worst, pcc)
                same = int(got[j].argmax()) == int(ref[j].argmax())
                rows.append(f"row {j}: pcc {pcc:.5f} argmax {'=' if same else 'x'} gap {_top2_gap(ref[j]):.3f}")
            logger.info(
                f"verify at {p}: parallel {parallel_ms:.1f} ms, pipelined {pipelined_ms:.1f} ms; " + "; ".join(rows)
            )
        assert worst >= 0.99, f"a parallel-verify row is at PCC {worst:.5f} against its single-row decode"


def _draft_counts(model) -> list[int]:
    """The pipelined DSpark runs' drafts per round (``DEEPSEEK_V4_SPEC_DRAFTS``): ``all`` is every count
    :class:`DSparkSpeculator` takes on ``model``, 1 to the drafter's block, its ``write_main`` rows less the
    bonus row and the compressor windows less one; else the comma-separated counts, each checked against that."""
    most = min(model.dspark.block, model.dspark.main_rows - 1, model.spec_window_rows - 2)
    if DRAFT_COUNTS.strip() == "all":
        return list(range(1, most + 1))
    counts = [int(k) for k in DRAFT_COUNTS.split(",") if k.strip()]
    bad = [k for k in counts if not 1 <= k <= most]
    if bad or not counts:
        raise ValueError(f"DEEPSEEK_V4_SPEC_DRAFTS={DRAFT_COUNTS!r}: draft counts run from 1 to {most}")
    return counts


def _tokenize_turn(tokenizer, text: str) -> list[int]:
    """A follow-up user turn, as it goes after the model's end of sentence: no BOS, then the assistant prefix."""
    turn = encode_messages([{"role": "user", "content": text}], "chat", add_default_bos_token=False)
    return list(tokenizer(turn, add_special_tokens=False)["input_ids"])


def _top2_gap(logits) -> float:
    top = logits.reshape(-1).float().topk(2).values
    return float(top[0] - top[1])


def _first_diff(a, b) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
