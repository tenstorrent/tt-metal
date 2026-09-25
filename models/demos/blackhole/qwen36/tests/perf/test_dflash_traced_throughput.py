# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end throughput of the DFlash speculative loop with an eager vs a traced verify forward.

Runs the real speculative loop (27B target + ttnn drafter, greedy, block 16) on
"The capital of France is" for 64 new tokens, first with an eager verify and then with the traced
verify, and reports tok/s and acceptance for each. The target, drafter and verify are built in the
same configuration as demo/dflash_demo.py.

Greedy speculation is exact -- the target verifies every slot and accepts only its own argmax -- so
the test asserts that both arms emit identical tokens and identical acceptance.

The eager arm runs first and doubles as the warm-up, so it pays for compiling every program the loop
uses; read the traced tok/s, not the eager/traced ratio.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \
    DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_traced_throughput.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import (
    DFlashDrafterConfig,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.tt.dflash.config import (
    NUM_BLOCKS,
    PAGED_BLOCK_SIZE,
    PRODUCTION_DECODE_TOK_S_T3K,
    TRACE_REGION_SIZE,
    load_drafter_state_dict,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.speculative_drafter import TtDrafter
from models.demos.blackhole.qwen36.tt.dflash.target import TtTarget
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

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
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION_SIZE}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_traced_verify_throughput(mesh_device, device_params, reset_seeds, ensure_gc):
    """Eager vs traced verify: same tokens, how many per second, and what it actually said."""
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

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer(PROMPT, return_tensors="pt").input_ids
    # Same configuration as the demo:
    #   * anchor_for() sizes the bucket to the request.
    #   * ctx_capacity makes the drafter's KV history a fixed buffer at stable addresses.
    #   * narrow_head=True runs the verify's LM head over a 32/64-row window instead of all 128
    #     rows, and is the precondition for the device-side argmax (Qwen36Model._posterior_device).
    total_tokens = prompt.shape[1] + MAX_NEW_TOKENS
    anchor = TtTarget.anchor_for(total_tokens, new_tokens=MAX_NEW_TOKENS)
    ctx_capacity = -(-(total_tokens + 32) // 32) * 32
    logger.info(f"anchor {anchor} for {prompt.shape[1]} + {MAX_NEW_TOKENS} tokens, drafter ctx {ctx_capacity}")
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True, anchor=anchor)
    drafter = TtDrafter(
        TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(drafter_path), ctx_capacity=ctx_capacity),
        target,
    )
    # Every block width is compiled before any capture: a program first reached with a trace parked
    # hangs the process and wedges the device.
    drafter.drafter.warm_block_widths()

    def run(label):
        t0 = time.perf_counter()
        stats = dflash_generate(
            drafter,
            target,
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            return_stats=True,
        )
        dt = time.perf_counter() - t0
        n = stats.num_output_tokens
        logger.info(
            f"[{label}] {n} tokens in {dt:.2f}s = {n / dt:.2f} tok/s ({dt * 1000 / n:.0f} ms/tok), "
            f"acceptance {stats.mean_acceptance_length:.3f} tok/step over "
            f"{len(stats.acceptance_lengths)} steps"
        )
        # Log the decoded text: the equality assert below proves the two arms agree, not that either
        # is coherent, and a target left in a bad state (e.g. a stale GDN carry) still emits
        # fluent-looking tokens at full speed.
        cont = tokenizer.decode(stats.output_ids[0, stats.num_input_tokens :], skip_special_tokens=True)
        logger.info(f"[{label}] -> {cont!r}")
        return stats, dt, n, cont

    eager_stats, eager_dt, eager_n, eager_text = run("eager verify")

    # The eager arm above doubles as the warm-up this requires: capture_verify_trace only warms its
    # own forward, and compiling the rest under a parked trace hangs.
    target.enable_traced_verify(narrow_head=True)
    traced_stats, traced_dt, traced_n, traced_text = run("TRACED verify")

    speedup = (eager_dt / eager_n) / (traced_dt / traced_n)
    logger.info(
        f"=== verify trace: {eager_n / eager_dt:.2f} -> {traced_n / traced_dt:.2f} tok/s "
        f"({speedup:.2f}x). Production traced decode is {PRODUCTION_DECODE_TOK_S_T3K} tok/s. ==="
    )
    logger.info(f"prompt: {PROMPT!r}")
    logger.info(f"output: {traced_text!r}")
    if eager_text != traced_text:
        # The id-level assert below is the real gate; this only makes the difference readable.
        logger.warning(f"eager output differs: {eager_text!r}")

    # Greedy + a verifying target: the tokens cannot change. If they did, the trace is wrong.
    assert torch.equal(eager_stats.output_ids, traced_stats.output_ids), (
        "traced verify changed the generated tokens; greedy speculation is exact, so this is a "
        "correctness bug in the traced path, not a speed/quality tradeoff"
    )
    assert traced_stats.mean_acceptance_length == pytest.approx(eager_stats.mean_acceptance_length, abs=1e-6)
