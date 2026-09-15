# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does DFlash still beat production decode on ORDINARY PROSE? Every number so far used five tokens.

Throughput here is ``acceptance / step_time``, and step time is near-constant -- one verify forward
over a 128-row bucket plus one drafter forward, whatever the accept count. So acceptance IS the
throughput, and acceptance turns out to be strongly prompt-dependent:

    "The capital of France is"   acceptance 7.000   <- every benchmark in this work
    ordinary prose, 8 tokens     acceptance 2.300
    ordinary prose, 64 tokens    acceptance 1.917 / 2.091
                                 (tests/reference/test_dflash_prompt_length.py)

A continuation the drafter can nail is not a representative workload. If prose runs at ~2 tok/step
against a step built for 7, the measured 22 tok/s becomes roughly 6-7 -- BELOW production traced
decode at 17.87 tok/s / 56.0 ms/tok -- and the headline claim of this work ("1.19x of production")
would hold only for the one prompt it was measured on.

This measures it rather than inferring it: the same traced loop, same warm-up, two prompts.

Generation is capped so ``start`` stays below 127. At ``start % ANCHOR == ANCHOR - 1`` the loop
computes verify_size=1, SKIPS drafter.propose (generate.py guards ``if verify_size > 1``), and the
drafter's context silently stops advancing -- a real bug found by this file's sibling. Staying short
keeps that out of the timing.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_prose_throughput.py
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
NEW_TOKENS = 48
PRODUCTION_TOK_S = 17.87

BENCH_PROMPT = "The capital of France is"
PROSE = (
    "The history of computing hardware spans more than a century, beginning with mechanical "
    "calculators and progressing through relays, vacuum tubes, discrete transistors, and finally "
    "integrated circuits. Each transition reduced the cost of a single logical operation by orders "
    "of magnitude, and each one was driven less by a single invention than by the accumulation of "
    "manufacturing technique."
)


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
def test_prose_vs_benchmark_prompt(mesh_device, device_params, reset_seeds, ensure_gc):
    """tok/s and acceptance for the benchmark prompt and for prose, same traced loop."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from transformers import AutoTokenizer

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDrafter(
        TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path), tt_ccl=model.tt_ccl), target
    )
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())

    bench = tokenizer(BENCH_PROMPT, return_tensors="pt").input_ids
    prose = tokenizer(PROSE, return_tensors="pt", add_special_tokens=False).input_ids[:, :64]

    # ORDER IS LOAD-BEARING: an EAGER generation first, THEN the capture.
    #
    # capture_verify_trace warms only the programs its own dummy forward touches. Capturing before
    # any generation leaves the rest uncompiled, and the first traced generation then tries to
    # compile with a trace parked -- which does not raise, it HANGS: the process spins at ~110 % CPU
    # with no output and has to be killed, and killing it mid-trace wedges the Ethernet cores badly
    # enough to need `tt-smi -r`. That cost two device resets before the pattern was clear.
    #
    # test_dflash_traced_throughput.py satisfies this precondition by accident -- it runs an eager
    # arm for COMPARISON, which happens to compile everything first. Nothing documents or enforces
    # it. This warm-up is deliberate, and it is also the timing warm-up (charging compilation to a
    # measured run is the bias that made an A/B in this work read 0.85x when the truth was 1.00x).
    #
    # AND "AT LEAST ONE EAGER GENERATION" IS NOT ENOUGH -- IT MUST COVER THE SHAPES THE MEASURED
    # RUNS WILL USE. The rule above is necessary and was still insufficient: a warm-up compiles only
    # the programs ITS OWN shapes touch. Any shape first met AFTER the capture compiles with a trace
    # parked, which is the same hang the rule exists to prevent.
    #
    # This file hung on exactly that, twice, deterministically and byte-identically (2026-09-15,
    # T3K). The old warm-up was `bench, max_new_tokens=8`; the measured prose run is a 64-token
    # prompt with NEW_TOKENS=48, so max_length is 112 and its FINAL step has
    # `verify_size = min(16, 112 - 110) = 2`. A 2-wide block is a drafter shape the 8-token warm-up
    # never produces. py-spy pinned every hang at the same state -- drafter.py:585 `_kv_heads`,
    # layer_idx=4, start=110, hist_len=109, new_ctx=1, q_len=2, kv_seq=3 -- and gdb put the native
    # stack in `SystemMemoryManager::fetch_queue_reserve_back`, spinning on a dispatch fetch queue
    # that never drains: the host stuck PUSHING, the device having stopped consuming.
    #
    # That the shape itself is fine is measured separately: test_drafter_block_width.py runs the
    # drafter standalone at widths 16/8/4/3/2/1 and reproduces this exact state (109 rows of
    # history, new_ctx=1, q_len=2) in 0.20 s. What it does not have is a parked trace -- and it pays
    # ~3.3 s the first time it sees each new width, which is the compile that hangs here.
    #
    # So warm with the REAL prompts at the REAL budget. Each measured configuration is generated
    # once eagerly, which compiles every program it needs -- narrow tail included -- before anything
    # is captured.
    for warm_prompt in (bench, prose):
        dflash_generate(drafter, target, warm_prompt, max_new_tokens=NEW_TOKENS)
    target.enable_traced_verify()
    dflash_generate(drafter, target, bench, max_new_tokens=8)

    rows = []
    for label, prompt in (("benchmark (5 tok)", bench), ("prose (64 tok)", prose)):
        assert prompt.shape[1] + NEW_TOKENS < 127, "keep start below the verify_size==1 bug"
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=NEW_TOKENS, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
        rows.append((label, n / dt, dt * 1000 / n, stats.mean_acceptance_length, len(stats.acceptance_lengths), text))
        logger.info(
            f"[{label:18}] {n / dt:6.2f} tok/s ({dt * 1000 / n:5.1f} ms/tok), "
            f"acceptance {stats.mean_acceptance_length:.3f} over {len(stats.acceptance_lengths)} steps"
        )
        logger.info(f"[{label:18}] -> {text[:200]!r}")

    logger.info("=" * 78)
    for label, tps, mspt, acc, steps, _ in rows:
        logger.info(
            f"  {label:18} {tps:6.2f} tok/s  {mspt:5.1f} ms/tok  acceptance {acc:5.3f}  "
            f"{tps / PRODUCTION_TOK_S:.2f}x production"
        )
    logger.info(f"  production traced decode: {PRODUCTION_TOK_S} tok/s")
    logger.info("=" * 78)
    print("\n>>> " + " | ".join(f"{l}: {t:.2f} tok/s acc {a:.2f}" for l, t, _, a, _, _ in rows) + "\n")

    # No throughput assert -- this exists to establish whether the 1.19x claim generalises, and the
    # answer is the point. Assert only that both prompts actually speculated, so a degenerate run
    # cannot be mistaken for a slow one.
    for label, _, _, acc, _, _ in rows:
        assert acc > 1.0, f"{label}: acceptance {acc:.3f} means every draft was rejected"
