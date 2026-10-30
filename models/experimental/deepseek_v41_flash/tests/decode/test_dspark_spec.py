# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Speculative decode on the full ``DeepSeekV41Model``.

Decode runs on the first seven 1x4 rows of the 8x4 mesh; DSpark runs alone on the eighth.

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

Runs 2 and 3 verify a round's ``DRAFTS + 1`` rows as pipelined single-token steps, then again with the
parallel verify, all rows through every layer in one traced step (``DSparkSpeculator(parallel_verify=True)``,
see ``DSPARK_PARALLEL_VERIFY.md``; eager with ``DEEPSEEK_V4_TRACED_DECODE=0``), and both rounds' splits
are logged.
``DEEPSEEK_V4_SPEC_VERIFY`` (``pipelined``, the default / ``parallel`` / ``both``) picks the modes; the model is
built with the parallel verify's traces only when a mode needs them. The parallel verify is off by default
until its rows match the single-row decode (``test_parallel_verify_rows``).
``DEEPSEEK_V4_SPEC_EXACT=0`` logs a divergence from the baseline instead of failing.

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
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import engram_table_nbytes
from models.experimental.deepseek_v41_flash.tt.spec import DSparkSpeculator
from models.experimental.deepseek_v4_flash.encoding_dsv4 import encode_messages, eos_token
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

DRAFTS = 4
CORRUPT = 1 / 3
VERIFY = os.environ.get("DEEPSEEK_V4_SPEC_VERIFY", "pipelined")
# 0: log a divergence from the baseline instead of failing (for throughput runs).
EXACT = os.environ.get("DEEPSEEK_V4_SPEC_EXACT", "1") != "0"
# The DSpark run's sampling (DSparkSpeculator.temperature / top_p); temperature 0 is greedy.
TEMPERATURE = float(os.environ.get("DEEPSEEK_V4_SPEC_TEMPERATURE", "1"))
TOP_P = float(os.environ.get("DEEPSEEK_V4_SPEC_TOP_P", "1"))
# Decode rows; the eighth row is the drafter's.
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
_ENGRAM_RAM_MARGIN = 1.25


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
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_8stage"])
def test_dspark_spec_decode(mesh_device):
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    nbytes = sum(engram_table_nbytes(loader, i) for i in config.engram_layer_ids)
    if _available_bytes() < _ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")

    n_tokens = int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "2000"))
    eos_id = tokenizer.convert_tokens_to_ids(eos_token)
    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)
    turns = [_tokenize_turn(tokenizer, text) for text in _FOLLOW_UPS]
    # Every follow-up asked, plus a verify block past the last position.
    needed = len(prompt) + n_tokens + sum(len(turn) for turn in turns) + 2 * (DRAFTS + 1)
    max_seq = _traced_max_seq(config, needed, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)
    modes = {"pipelined": (False,), "parallel": (True,), "both": (False, True)}[VERIFY]

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device,
            stack,
            loader=loader,
            config=config,
            num_stages=DECODE_STAGES,
            verify_rows=DRAFTS + 1 if True in modes else 1,
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        model.prepare_spec_decode(rope["main"])
        if model.use_prefetcher:
            stack.enter_context(tensor_prefetcher_session(model.dspark.device))
        model.activate_session(model.open_session())
        model.capture_spec_traces(prompt[0], 0)

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
                out = model.decode_prompt_traced(feed, len(stream))
                stream += feed
            else:
                t0 = time.perf_counter()
                out = model.decode_traced(baseline[-1], len(stream))
                plain_seconds += time.perf_counter() - t0
                plain_steps += 1
                stream.append(baseline[-1])
        greedy_tps = plain_steps / plain_seconds
        logger.info(
            f"greedy: {len(baseline)} tokens, {baseline.count(eos_id)} end(s) of sentence, {greedy_tps:.2f} tok/s; "
            f"{tokenizer.decode(stream[len(prompt):])!r}"
        )

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
            while len(got) < n_tokens:
                asked = got.count(eos_id)
                if got[-1] == eos_id:
                    if asked > len(turns):
                        break
                    got.append(spec.extend(turns[asked - 1]))
                else:
                    got += spec.generate(n_tokens - len(got), stop_ids=(eos_id,))
            return got

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
            logger.info(f"oracle, {verify}: {oracle.stats.summary(DRAFTS)}")
            check(f"oracle spec ({verify})", got, EXACT)

            # 3. The DSpark drafter. Sampling gives up equality with the baseline.
            spec = DSparkSpeculator(
                model, DRAFTS, parallel_verify=parallel_verify, temperature=TEMPERATURE, top_p=TOP_P
            )
            got = converse(spec)
            logger.info(
                f"DSpark, {verify}, temperature {TEMPERATURE}, top_p {TOP_P}: {spec.stats.summary(DRAFTS)} "
                f"(greedy {greedy_tps:.2f} tok/s)"
            )
            logger.info(f"DSpark output ({verify}): {tokenizer.decode(got)!r}")
            if TEMPERATURE == 0:
                check(f"DSpark spec ({verify})", got, EXACT)


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
    nbytes = sum(engram_table_nbytes(loader, i) for i in config.engram_layer_ids)
    if _available_bytes() < _ENGRAM_RAM_MARGIN * nbytes:
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


def _tokenize_turn(tokenizer, text: str) -> list[int]:
    """A follow-up user turn, as it goes after the model's end of sentence: no BOS, then the assistant prefix."""
    turn = encode_messages([{"role": "user", "content": text}], "chat", add_default_bos_token=False)
    return list(tokenizer(turn, add_special_tokens=False)["input_ids"])


def _top2_gap(logits) -> float:
    top = logits.reshape(-1).float().topk(2).values
    return float(top[0] - top[1])


def _first_diff(a, b) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
