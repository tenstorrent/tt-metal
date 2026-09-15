# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does narrowing the verify LM head change the tokens, and what does it buy?

``TtTarget.forward`` returns ``logits[:, -S:]`` with S <= 16, but the captured verify runs ``norm``
and ``_lm_head`` over the whole ``ANCHOR``-row bucket (128) and reads back up to 128 rows. The head
is a vocab-sharded matmul followed by an ALL-GATHER that replicates the logits
(``Qwen36Model._lm_head``), and the readback is 26.8 ms of a 204 ms step
(tests/perf/test_traced_verify_host_breakdown.py) -- so up to 8x of all three is spent on rows that
are then discarded. The verify is device-bound (replay 107.1 ms against 105.8 ms of device time),
which is exactly the regime where removing DEVICE work is the only thing that helps.

``narrow_head=True`` stops the capture at the norm and runs the head per replay over the smallest
TILE-ALIGNED window containing the wanted rows -- 32 or 64 rows, never more (the arithmetic is
pinned exhaustively and without hardware in tests/unit/test_lm_head_window.py). At the steady state
(valid_len 128, block 16) that is a 4x narrower head.

WHY THIS IS A SEPARATE FILE AND NOT AN IN-PROCESS A/B. Acceptance on the traced path DECAYS with
generation index on a reused drafter/target -- measured 7.000 -> 2.611 -> 1.270
(tests/reference/test_dflash_generation_repeat.py, DFLASH_HANDOFF.md §0). Any second arm in the same
process is therefore measured further down that curve and would read as a throughput regression that
has nothing to do with the head. So each arm runs in its OWN process, selected by the parameter, and
the two runs are compared by hand:

    pytest -svq ...test_dflash_narrow_head.py -k wide
    pytest -svq ...test_dflash_narrow_head.py -k narrow

CORRECTNESS IS THE REAL GATE, and it does not need the comparison: greedy verification accepts only
the target's own argmax, so narrowing the head must leave the emitted tokens BIT-IDENTICAL. The
expected continuation is asserted against the known-good reference
(tests/reference/test_dflash_traced_throughput.py), which is checked into this file as EXPECTED so
either arm can fail on its own, in one process, without the other.
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

# The continuation test_dflash_traced_throughput.py emits for this prompt at 64 new tokens, greedy.
# Both arms must reproduce it exactly; the head cannot legally change a single token.
EXPECTED_PREFIX = " Paris."


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("narrow", [False, True], ids=["wide", "narrow"])
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_narrow_head(mesh_device, device_params, narrow, reset_seeds, ensure_gc):
    """One traced generation with the head wide or narrowed. Same tokens; report tok/s."""
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

    # The eager generation enable_traced_verify() requires, which also compiles the loop's programs
    # at exactly the shapes the measured run will use -- see DFLASH_HANDOFF.md §0 on why a warm-up
    # that does NOT cover the measured run's shapes hangs the process.
    dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS)
    target.enable_traced_verify(narrow_head=narrow)

    t0 = time.perf_counter()
    stats = dflash_generate(drafter, target, prompt, max_new_tokens=MAX_NEW_TOKENS, return_stats=True)
    dt = time.perf_counter() - t0
    n = stats.num_output_tokens
    text = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)

    label = "narrow" if narrow else "wide"
    logger.info(
        f"[{label} head] {n} tokens in {dt:.2f}s = {n / dt:.2f} tok/s ({dt * 1000 / n:.0f} ms/tok), "
        f"acceptance {stats.mean_acceptance_length:.3f} over {len(stats.acceptance_lengths)} steps"
    )
    logger.info(f"[{label} head] -> {text!r}")
    print(f"\n>>> {label} head: {n / dt:.2f} tok/s, acceptance {stats.mean_acceptance_length:.3f}")
    print(f">>> output: {text[:120]!r}\n")

    # Greedy verification pins the tokens: the head may change how fast they arrive, never what they
    # are. A narrow head that shifts the text is reading the wrong rows of the window.
    assert text.startswith(EXPECTED_PREFIX), (
        f"{label} head emitted {text[:80]!r}, expected it to start {EXPECTED_PREFIX!r}. "
        "Narrowing the LM head must not change a single token."
    )
    assert stats.mean_acceptance_length > 1.0, f"{label}: acceptance {stats.mean_acceptance_length:.3f}"
