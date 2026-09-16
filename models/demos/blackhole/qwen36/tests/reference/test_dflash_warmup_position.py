# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How much does acceptance depend on WHERE the measured run sits in the sequence?

This isolates the one material difference between the two configurations in this suite that
disagree by 7x on the same prompt:

    test_dflash_traced_throughput.py   eager warm -> capture -> MEASURE          acceptance 7.000
    demo/dflash_demo.py                eager warm -> capture -> traced warm
                                                             -> MEASURE          acceptance 1.031

Everything else matches -- PAGED_BLOCK_SIZE 64, NUM_BLOCKS 64, max_seq_len 4096, device_taps=True,
same trace region, same prompt once DFLASH_PROMPT is set. The demo simply runs one more full
generation between the capture and the measurement. This file turns that into a parameter: N traced
generations between the capture and the measured one, N in {0, 1, 2}, everything else held at the
values the 7.000 run uses.

N=0 reproduces the known-good configuration, so it is the control and it must come out at 7.000. If
acceptance falls as N rises, the decay is a function of generations run, and the demo's number is
explained without invoking its prompt or its token budget (both already exonerated:
DFLASH_PROMPT="The capital of France is" in the demo still gives 1.031).

WHAT THIS IS NOT. It does not identify the mechanism. One specific mechanism has already been
tested and REFUTED: `TtTarget.reset()` allocating a fresh GDN snapshot per generation, which looked
like the obvious culprit (an allocation under a parked trace, which Metal warns about). Passing
`into=` to reuse the buffers did NOT restore acceptance (3.875, not 7.000) and SIGBUSed the drafter
on the next generation -- that allocation is load-bearing, because it follows
`_reset_gdn_state_for_new_sequence()`, which invalidates the previous snapshot. So the decay is not
simply "an allocation while a trace is parked".

Each N runs in its OWN process-level fixture instance, so the arms cannot contaminate each other --
which is the whole subject here.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_warmup_position.py
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
# 64, matching test_dflash_traced_throughput.py exactly, so N=0 is a true control.
MAX_NEW_TOKENS = 64
PROMPT = "The capital of France is"


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("n_traced_warm", [0, 1, 2], ids=lambda n: f"warm{n}")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_acceptance_vs_warmup_position(mesh_device, device_params, n_traced_warm, reset_seeds, ensure_gc):
    """N traced generations between the capture and the measured one. N=0 must give 7.000."""
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
    # No tt_ccl=model.tt_ccl: the known-good run lets the drafter build its own, and sharing was
    # tested and changes nothing. Held at the 7.000 run's value so it cannot be the variable.
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path)), target)
    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids

    def run(tag):
        t0 = time.perf_counter()
        stats = dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        steps = len(stats.acceptance_lengths)
        logger.info(
            f">>>>> {tag}: {n} tok in {dt:.2f}s = {n / dt:5.2f} tok/s, "
            f"acceptance {stats.mean_acceptance_length:.3f} over {steps} steps "
            f"({dt * 1000 / max(steps, 1):.0f} ms/step)"
        )
        return stats

    # The eager generation enable_traced_verify() requires, at the measured run's own shapes.
    run("eager warm")
    target.enable_traced_verify()
    for i in range(n_traced_warm):
        run(f"traced warm {i + 1}")

    stats = run(f"MEASURED (after {n_traced_warm} traced warm)")
    text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
    acc = stats.mean_acceptance_length
    steps = len(stats.acceptance_lengths)
    print(f"\n>>> n_traced_warm={n_traced_warm}: acceptance {acc:.3f} over {steps} steps")
    print(f">>> output: {text[:100]!r}\n")

    # Greedy decoding pins the tokens whatever the acceptance does, so this must hold in every arm;
    # if it ever fails, the problem is worse than a speed decay.
    assert text.startswith(" Paris."), f"expected ' Paris.' prefix, got {text[:80]!r}"

    if n_traced_warm == 0:
        # The control IS the known-good configuration. If it does not reproduce, nothing else in
        # this file means anything.
        assert acc > 6.0, f"control (N=0) gave acceptance {acc:.3f}, expected ~7.000"
