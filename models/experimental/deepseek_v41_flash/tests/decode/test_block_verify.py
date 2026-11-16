# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""DSpark parallel verify (T = 4 rows) on the persistent-MoE fast path, end to end.

The model is the ``persistent_moe`` layout (one TP4 stage on Galaxy row 4, the TP12 persistent MoE on rows 5-7, the
DSpark drafter on row 3) built with ``verify_rows=4``: every verify round is one traced step of every layer's
``(4, 32)``-tiled block ops (``DeepSeekV41Model.decode_verify_block``), the MoE server switching to 4-row requests
for it. Prefill goes through the same blocks.

1. plain greedy decode (device sampling, the one-row traces), the baseline and its tok/s/u;
2. DSpark rounds fed the baseline's own tokens with a third corrupted (``DEEPSEEK_V4_SPEC_ORACLE=1``, default on):
   the stream must equal the baseline, which holds only if the rejected rows leave no trace;
3. DSpark with the drafter at temperature 0: must equal the baseline; tokens/round, ms/round and its split.

``DEEPSEEK_V4_SPEC_TOKENS`` (default 300) tokens are generated after the demo prompt (128 tokens); the block verify
needs every compressed group in its dense select, so prompt + tokens stays under 508 positions.

    DEEPSEEK_V41_RUST_HOST=0 pytest -s models/experimental/deepseek_v41_flash/tests/decode/test_block_verify.py
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
from models.experimental.deepseek_v41_flash.tt import spec as spec_module
from models.experimental.deepseek_v41_flash.tt.config import DEFAULT_MODEL_DIR, load_config
from models.experimental.deepseek_v41_flash.tt.engram_lookup import (
    ENGRAM_RAM_MARGIN,
    engram_ram_needed,
    stage_engram_tables,
)
from models.experimental.deepseek_v41_flash.tt.spec import DSparkSpeculator
from models.experimental.deepseek_v4_flash.tt.weight_loader import DeepseekV4WeightLoader
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session

ROWS = 4
DRAFTS = ROWS - 1
CORRUPT = 1 / 3
N_TOKENS = int(os.environ.get("DEEPSEEK_V4_SPEC_TOKENS", "300"))
ORACLE = os.environ.get("DEEPSEEK_V4_SPEC_ORACLE", "1") not in ("0", "", "false", "False")
EXACT = os.environ.get("DEEPSEEK_V4_SPEC_EXACT", "1") != "0"
_VERBOSE = os.environ.get("DEEPSEEK_V4_SPEC_VERBOSE", "0") not in ("0", "", "false", "False")


def _first_diff(a: list[int], b: list[int]) -> int:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


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
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=["mesh_device"], ids=["tp4_1stage_persistent_moe12"])
def test_block_verify_spec(mesh_device, monkeypatch):
    monkeypatch.setattr(spec_module, "_VERBOSE", _VERBOSE)
    loader = DeepseekV4WeightLoader(DEFAULT_MODEL_DIR)
    config = load_config(loader.snapshot_dir)
    tokenizer = AutoTokenizer.from_pretrained(loader.snapshot_dir)
    stage_engram_tables(loader, config.engram_layer_ids)
    nbytes = engram_ram_needed(loader, config.engram_layer_ids)
    if _available_bytes() < ENGRAM_RAM_MARGIN * nbytes:
        pytest.skip(f"needs {nbytes / 1e9:.0f} GB of free memory for the Engram tables")

    prompt = _tokenize_chat(tokenizer, _DEFAULT_TEXT)
    needed = len(prompt) + N_TOKENS + 2 * ROWS
    max_seq = _traced_max_seq(config, needed, config.num_hidden_layers)
    rope = _build_rope(config, loader.snapshot_dir, max_seq)

    with contextlib.ExitStack() as stack:
        model, lm_head, loader, config = _construct_model(
            mesh_device, stack, loader=loader, config=config, num_stages=1, persistent_moe=True, verify_rows=ROWS
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

        # 1. Plain greedy decode on the one-row traces.
        out = model.decode_prompt_traced(prompt, 0)
        baseline = [token_of(out)]
        pos = len(prompt)
        seconds, steps = 0.0, 0
        while len(baseline) < N_TOKENS:
            t0 = time.perf_counter()
            out = model.decode_traced(baseline[-1], pos)
            if len(baseline) > 4:  # past the first steps' warmup
                seconds += time.perf_counter() - t0
                steps += 1
            baseline.append(token_of(out))
            pos += 1
        greedy_tps = steps / seconds
        logger.info(f"greedy: {len(baseline)} tokens, {greedy_tps:.2f} tok/s/u")
        if _VERBOSE:
            logger.info(f"greedy output: {tokenizer.decode(baseline)!r}")
        stream = prompt + baseline

        def check(run: str, got: list[int]) -> None:
            if got == baseline[: len(got)]:
                logger.info(f"{run}: {len(got)} tokens equal the greedy baseline")
                return
            diff = _first_diff(got, baseline)
            message = f"{run} diverged at token {diff}: {got[diff]} vs greedy {baseline[diff]}"
            assert not EXACT, message
            logger.warning(message)

        def converse(spec: DSparkSpeculator) -> list[int]:
            got = [spec.prefill(prompt)]
            got += spec.generate(N_TOKENS - 1)
            spec.finish()
            return got

        results = {}
        if ORACLE:
            rng = random.Random(0)

            def propose(token, frontier):
                return [
                    stream[q] if q < len(stream) and rng.random() > CORRUPT else rng.randrange(config.vocab_size)
                    for q in range(frontier + 2, frontier + 2 + DRAFTS)
                ]

            oracle = DSparkSpeculator(model, DRAFTS, propose=propose, parallel_verify=True, temperature=0.0)
            got = converse(oracle)
            logger.info(f"oracle ({DRAFTS} drafts, 1/3 corrupted), block verify: {oracle.stats.summary(DRAFTS)}")
            check("oracle spec (block verify)", got)
            results["oracle"] = oracle.stats

        spec = DSparkSpeculator(model, DRAFTS, parallel_verify=True, temperature=0.0)
        got = converse(spec)
        logger.info(f"DSpark, block verify, {DRAFTS} drafts, greedy: {spec.stats.summary(DRAFTS)}")
        if _VERBOSE:
            logger.info(f"DSpark output: {tokenizer.decode(got)!r}")
        check("DSpark spec (block verify)", got)
        results["dspark"] = spec.stats

        lines = [
            f"block verify T={ROWS} (greedy baseline {greedy_tps:.2f} tok/s/u, {1000 / greedy_tps:.2f} ms/token):",
            "  run      tokens/round  ms/round   tok/s/u  split (ms/round)",
        ]
        for name, stats in results.items():
            rounds = len(stats.accepted)
            split = ", ".join(f"{k} {1000 * v / rounds:.2f}" for k, v in stats.phases.items())
            lines.append(
                f"  {name:<7}  {stats.tokens / rounds:>12.2f}  {1000 * stats.seconds / rounds:>8.2f}  "
                f"{stats.tokens / stats.seconds:>8.2f}  {split}"
            )
        logger.info("\n".join(lines))
