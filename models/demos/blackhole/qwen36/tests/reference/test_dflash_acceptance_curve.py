# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does acceptance cross break-even? The number that decides whether DFlash ships.

Throughput is ``acceptance / step_time``, and a speculative step costs far more than a decode step:
the verify recomputes the whole span from the anchor -- up to ANCHOR=128 rows through 64 layers --
to validate a 16-token block, plus the drafter's own forward. Measured on the demo at 100 tokens:

    64.7 ms/token x 4.950 acceptance = 320 ms per speculative step
    production traced decode         =  56 ms per token, one token per forward

    break-even acceptance = 320 / 56 = 5.72

So 4.950 LOSES, at 0.86x production, even though it commits five tokens per forward. What is not
known -- and what this file measures -- is the SHAPE of acceptance against generation length. Two
figures exist and they disagree:

    64 new tokens    9 steps    63/9 = 7.000
    100 new tokens  20 steps   99/20 = 4.950

They come from different runs and configurations, so the difference between "acceptance genuinely
decays with length" and "those two runs differed for another reason" has never been settled. It
decides the shipping question: if acceptance crosses break-even at some length, DFlash is a win for
completions shorter than that and a regression beyond it, which is a workload policy rather than a
bug to fix.

ONE PROCESS, ONE MODEL LOAD, budgets swept in order. That is safe now and deliberately so: the
generation-parity defect that used to make every second traced generation collapse is fixed
(TtDFlashDrafter.reset keeps the fixed-capacity history instead of reallocating it under a parked
trace -- DFLASH_HANDOFF.md). So this sweep is ALSO a regression check on that fix: if acceptance
alternates with generation index rather than varying smoothly with length, the parity bug is back.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_acceptance_curve.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
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

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 250_000_000
PRODUCTION_MS_PER_TOK = 56.0  # text_demo.py traced_128 -- 17.87 tok/s
PROMPT = "The capital of France is"
BUDGETS = [16, 32, 64, 100, 150, 200, 256]
CAP = 320  # >= 5 + max(BUDGETS), rounded to a multiple of 32


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_acceptance_vs_generation_length(mesh_device, device_params, reset_seeds, ensure_gc):
    """Sweep the token budget on one prompt; report acceptance, throughput and break-even."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from transformers import AutoTokenizer

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    # The shipping configuration: fixed-capacity history (stable across generations), every block
    # width compiled before the capture, and the trace serving every bucket rather than only the
    # first. See DFLASH_HANDOFF.md for why each is required.
    drafter = TtDrafter(
        TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path), tt_ccl=model.tt_ccl, ctx_capacity=CAP),
        target,
    )
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    drafter.drafter.warm_block_widths()
    # One eager generation at the LARGEST budget: satisfies enable_traced_verify's precondition and
    # compiles every shape the sweep will touch, so nothing compiles under a parked trace.
    dflash_generate(drafter, target, prompt, max_new_tokens=max(BUDGETS))
    target.enable_traced_verify()
    # NOTE: this CONTAMINATES every budget whose span crosses 128. allow_trace_past_anchor is
    # not safe -- acceptance collapses after a crossing that lands mid-generation (4.357 before,
    # 1.090 after; see test_dflash_anchor_crossing.py's gen256 arm). Budgets 150/200/256 below
    # therefore measure THAT defect, not drafter quality. Only the 16/32/64/100 rows, which
    # never cross, describe the drafter. Left on deliberately so the contaminated rows stay
    # visible and comparable; set it False once the crossing defect is fixed and re-run.
    target.allow_trace_past_anchor = True

    rows = []
    for budget in BUDGETS:
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=budget, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        steps = len(stats.acceptance_lengths)
        acc = stats.mean_acceptance_length
        ms_tok = dt * 1000 / n
        step_ms = dt * 1000 / max(steps, 1)
        # What acceptance this step time would need in order to match plain decode.
        breakeven = step_ms / PRODUCTION_MS_PER_TOK
        rows.append((budget, n, steps, acc, ms_tok, n / dt, step_ms, breakeven))
        logger.info(
            f">>>>> budget {budget:3d}: {n:3d} tok in {dt:6.2f}s | acceptance {acc:5.3f} over {steps:3d} steps | "
            f"{ms_tok:6.1f} ms/tok ({n / dt:5.2f} tok/s) | step {step_ms:6.1f} ms | "
            f"break-even acc {breakeven:5.2f} | {n / dt / (1000 / PRODUCTION_MS_PER_TOK):4.2f}x production"
        )

    logger.info("=" * 100)
    logger.info(" budget  tokens  steps  acceptance  ms/tok   tok/s   step_ms  break-even  vs production  verdict")
    crossover = None
    for budget, n, steps, acc, ms_tok, tok_s, step_ms, be in rows:
        ratio = tok_s / (1000 / PRODUCTION_MS_PER_TOK)
        verdict = "WIN " if ratio > 1.0 else "lose"
        if ratio <= 1.0 and crossover is None and rows[0][3] > rows[0][7]:
            crossover = budget
        logger.info(
            f" {budget:6d}  {n:6d}  {steps:5d}  {acc:10.3f}  {ms_tok:6.1f}  {tok_s:6.2f}  {step_ms:7.1f}  "
            f"{be:10.2f}  {ratio:13.2f}x  {verdict}"
        )
    logger.info("=" * 100)

    print(
        "\n>>> acceptance curve: "
        + " | ".join(f"{b}:{a:.2f}({t / (1000 / PRODUCTION_MS_PER_TOK):.2f}x)" for b, _, _, a, _, t, _, _ in rows)
        + "\n"
    )

    wins = [b for b, _, _, _, _, t, _, _ in rows if t / (1000 / PRODUCTION_MS_PER_TOK) > 1.0]
    print(f">>> beats production at budgets: {wins if wins else 'NONE'}\n")

    # PARITY REGRESSION GUARD. These run back to back in one process, which is exactly the shape
    # that used to alternate (acceptance 4.950 / 1.021 / 4.950 by generation index). Acceptance may
    # legitimately DRIFT with length; it must not oscillate. A collapse to ~1.0 at any budget while
    # its neighbours are healthy means the reset-reallocation defect is back.
    accs = [a for _, _, _, a, _, _, _, _ in rows]
    assert min(accs) > 1.5, (
        f"acceptance collapsed to {min(accs):.3f} at some budget while others are healthy "
        f"({[f'{a:.2f}' for a in accs]}) -- the generation-parity defect has regressed"
    )
