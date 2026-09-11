# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""What does tracing the VERIFY forward actually buy, end to end?

Everything up to here established that the traced verify is correct (see
``test_dflash_target_trace_replay.py``: replay is bit-identical to eager on hidden, on all 48 GDN
layers' recurrent state, and on the public ``verify_traced`` vs ``prefill_block_all_logits``
logits). None of that is a throughput number. This file runs the real speculative loop both ways
and reports tok/s.

READ THE RESULT AGAINST THE RIGHT BASELINE. Production traced decode is **17.87 tok/s / 56.0 ms per
token** (README-T3K-27B). The speculative loop was measured at **3.18 tok/s, 0.18x that** --
speculation currently LOSES to simply running the model, and the cause was that the whole path is
eager while production is traced. This measures how much of that gap the verify trace closes.

Tracing removes **dispatch**, not device time. The drafter was worth tracing because it ran at ~9 %
device utilization; a 128-row, 64-layer verify forward is far more device-bound, so the ceiling
here is whatever fraction of its ~2 s is host dispatch. The README's ~22 ms/tok projection is
arithmetic from the acceptance rate, not a measurement, and should not be treated as a target.

MEASURED (T3K, 64 new tokens, "The capital of France is", greedy, block 16, bucket 128):

    eager verify    64 tokens in 17.39 s = 3.68 tok/s (272 ms/tok), acceptance 7.000 tok/step
    TRACED verify   64 tokens in  8.88 s = 7.21 tok/s (139 ms/tok), acceptance 7.000 tok/step
                    -> 1.96x, tokens bit-identical, acceptance unchanged

So the verify trace roughly DOUBLES the speculative loop, from ~0.21x of production traced decode
to ~0.40x. Speculation still loses to simply running the model (17.87 tok/s) -- this closes about
half the gap, it does not close it.

What is left, now that the target is ~2x faster and therefore a smaller share of the step: at
139 ms/tok and 7 tok/step a step is ~970 ms, of which the still-EAGER drafter is ~120 ms (12 %, up
from 6 % when the target was slower). The other two levers are untouched by any of this -- the
128-row bucket verifies a 16-token block (8x more rows than needed), and ~89 % of steps take the
rollback path and pay a SECOND target forward.

The loop is greedy, so BOTH configurations must emit identical tokens -- the target verifies every
slot and accepts only its own argmax. That equality is asserted here: a throughput win that changed
the output would be a bug, not a win.
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
TRACE_REGION = 200_000_000
MAX_NEW_TOKENS = 64
PROMPT = "The capital of France is"


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
def test_traced_verify_throughput(mesh_device, device_params, reset_seeds, ensure_gc):
    """Eager vs traced verify: same tokens, and how many per second."""
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
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path)), target)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    def run(label):
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        logger.info(
            f"[{label}] {n} tokens in {dt:.2f}s = {n / dt:.2f} tok/s ({dt * 1000 / n:.0f} ms/tok), "
            f"acceptance {stats.mean_acceptance_length:.3f} tok/step over "
            f"{len(stats.acceptance_lengths)} steps"
        )
        return stats, dt, n

    eager_stats, eager_dt, eager_n = run("eager verify")

    target.enable_traced_verify()
    traced_stats, traced_dt, traced_n = run("TRACED verify")

    speedup = (eager_dt / eager_n) / (traced_dt / traced_n)
    logger.info(
        f"=== verify trace: {eager_n / eager_dt:.2f} -> {traced_n / traced_dt:.2f} tok/s "
        f"({speedup:.2f}x). Production traced decode is 17.87 tok/s. ==="
    )
    print(f"\n>>> eager {eager_n / eager_dt:.2f} tok/s -> traced {traced_n / traced_dt:.2f} tok/s ({speedup:.2f}x)\n")

    # Greedy + a verifying target: the tokens cannot change. If they did, the trace is wrong.
    assert torch.equal(eager_stats.output_ids, traced_stats.output_ids), (
        "traced verify changed the generated tokens; greedy speculation is exact, so this is a "
        "correctness bug in the traced path, not a speed/quality tradeoff"
    )
    assert traced_stats.mean_acceptance_length == pytest.approx(eager_stats.mean_acceptance_length, abs=1e-6)
