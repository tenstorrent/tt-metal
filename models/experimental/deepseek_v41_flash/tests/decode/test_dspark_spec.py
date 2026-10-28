# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""Greedy speculative decode on the full ``DeepSeekV41Model``.

Decode runs on the first seven 1x4 rows of the 8x4 mesh; DSpark runs alone on the eighth.

One model serves three runs of the demo prompt, each from position 0:

1. plain greedy decode, the baseline;
2. speculative decode fed the baseline's own tokens with a third of them corrupted, so every
   round rejects somewhere -- the output must equal the baseline token for token, which holds only
   if the rejected rows leave no trace in the rings and compressor windows;
3. speculative decode with the DSpark drafter -- again equal to the baseline, and its acceptance
   and throughput are logged.

Runs 2 and 3 verify a round's ``DRAFTS + 1`` rows as pipelined single-token steps, then again with the
parallel verify, all rows through every layer in one step (``DSparkSpeculator(parallel_verify=True)``,
see ``DSPARK_PARALLEL_VERIFY.md``); both must equal the baseline, and both rounds' splits are logged.
``DEEPSEEK_V4_SPEC_VERIFY`` (``pipelined`` / ``parallel`` / ``both``, the default) picks the modes.
``DEEPSEEK_V4_SPEC_EXACT=0`` logs a divergence from the baseline instead of failing.
``DEEPSEEK_V4_SPEC_ACCEPT_MARGIN`` (logits, default 0) relaxes the DSpark run's accept rule
(``DSparkSpeculator.accept_margin``); its output is then not checked against the baseline.

The DSpark experts come from ``tools/fill_expert_cache.py --dspark``.

    DEEPSEEK_V4_SPEC_TOKENS=64 pytest -s models/experimental/deepseek_v41_flash/tests/decode/test_dspark_spec.py
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
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

DRAFTS = 4
CORRUPT = 1 / 3
VERIFY = os.environ.get("DEEPSEEK_V4_SPEC_VERIFY", "both")
# 0: log a divergence from the baseline instead of failing (for throughput runs).
EXACT = os.environ.get("DEEPSEEK_V4_SPEC_EXACT", "1") != "0"
# The DSpark run's relaxed accept rule, in logits (DSparkSpeculator.accept_margin); 0 is exact greedy.
ACCEPT_MARGIN = float(os.environ.get("DEEPSEEK_V4_SPEC_ACCEPT_MARGIN", "0"))
# Decode rows; the eighth row is the drafter's.
DECODE_STAGES = 7
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

    n_tokens = int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "64"))
    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)
    max_seq = _traced_max_seq(config, len(prompt) + n_tokens + 2 * (DRAFTS + 1), config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=DECODE_STAGES, verify_rows=DRAFTS + 1
        )
        model.prepare_static_decode(rope, max_seq, lm_head=lm_head)
        model.prepare_spec_decode(rope["main"])
        if model.use_prefetcher:
            stack.enter_context(tensor_prefetcher_session(model.dspark.device))
        model.activate_session(model.open_session())
        model.capture_spec_traces(prompt[0], 0)

        # 1. Plain greedy decode.
        out = model.decode_prompt_traced(prompt, 0)
        baseline = [int(out.reshape(-1).float().argmax())]
        # Greedy's top1 - top2 logit gap per token: a divergence at a near-tie is bf16 / bfp8 reduction
        # order, not a wrong row (DSPARK_PARALLEL_VERIFY.md, Numerics).
        gaps = [_top2_gap(out)]
        t0 = time.perf_counter()
        for i in range(n_tokens):
            out = model.decode_traced(baseline[-1], len(prompt) + i)
            baseline.append(int(out.reshape(-1).float().argmax()))
            gaps.append(_top2_gap(out))
        plain_seconds = time.perf_counter() - t0
        logger.info(f"greedy: {n_tokens / plain_seconds:.2f} tok/s; {tokenizer.decode(baseline)!r}")
        stream = prompt + baseline  # stream[q] is the token at position q

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

        modes = {"pipelined": (False,), "parallel": (True,), "both": (False, True)}[VERIFY]
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
            first = oracle.prefill(prompt)
            got = [first] + oracle.generate(n_tokens)
            logger.info(f"oracle, {verify}: {oracle.stats.summary(DRAFTS)}")
            check(f"oracle spec ({verify})", got, EXACT)

            # 3. The DSpark drafter. A relaxed accept rule gives up equality with the baseline.
            spec = DSparkSpeculator(model, DRAFTS, parallel_verify=parallel_verify, accept_margin=ACCEPT_MARGIN)
            first = spec.prefill(prompt)
            got = [first] + spec.generate(n_tokens)
            logger.info(
                f"DSpark, {verify}, accept margin {ACCEPT_MARGIN}: {spec.stats.summary(DRAFTS)} "
                f"(greedy {n_tokens / plain_seconds:.2f} tok/s)"
            )
            logger.info(f"DSpark output ({verify}): {tokenizer.decode(got)!r}")
            check(f"DSpark spec ({verify})", got, EXACT and ACCEPT_MARGIN == 0)


def _top2_gap(logits) -> float:
    top = logits.reshape(-1).float().topk(2).values
    return float(top[0] - top[1])


def _first_diff(a, b) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
