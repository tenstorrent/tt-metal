# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Does DFlash speculation actually make decode faster on device?

Acceptance length (tokens per target forward) is not throughput. This measures wall clock against
the model's OWN decode path, which is the only honest reference:

* **decode_tp** — the bespoke eager single-token decode loop. What "decode tok/s" normally means.
* **block_size=1** — the speculative loop with speculation disabled, so every token still costs one
  128-token bucket prefill. Separates the cost of the harness from the benefit of speculation.
* **ttnn drafter** — real speculation, drafter on the mesh. Taps never leave the device.
* **host drafter** — real speculation, drafter in PyTorch on the CPU. Kept as a measured A/B
  because it was the original implementation.

Every variant runs in ONE process, back to back, and the split of each run's wall clock is reported.
That matters: this is a shared machine, and host load has moved the absolute numbers by 1.7x between
sessions (``decode_tp`` has measured 0.93 and 1.92 tok/s on identical code). Only within-run ratios
are trustworthy; check ``uptime`` before believing an absolute.

Opt-in (loads the full 27B and generates four times):

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      DFLASH_RUN_TARGET=1 \\
      pytest -svq models/demos/blackhole/qwen36/tests/reference/test_dflash_throughput.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import HostDrafter, TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.generate import dflash_generate
from models.demos.blackhole.qwen36.reference.dflash.loader import load_drafter
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
    resolve_target_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
MAX_NEW = 32


@pytest.mark.timeout(0)  # loads the 27B and runs four generations; the 300s default is not enough
@torch.no_grad()
@parametrize_mesh_tp()
def test_tt_decode_throughput(mesh_device, reset_seeds, ensure_gc):
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to measure throughput on the real 27B")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(resolve_target_path())
    prompt = tokenizer("The capital of France is", return_tensors="pt").input_ids
    prompt_len = prompt.shape[1]

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)

    # ---- reference: the model's own eager decode loop (bespoke concat-KV path) ----
    model.reset_tp()
    pad = max(128, ((prompt_len + 127) // 128) * 128)
    padded = torch.cat([prompt, torch.zeros(1, pad - prompt_len, dtype=torch.long)], dim=1)
    logits = model.prefill_tp(padded, valid_len=prompt_len)
    token = int(torch.argmax(logits))
    ttnn.synchronize_device(mesh_device)

    t0 = time.perf_counter()
    for step in range(MAX_NEW):
        logits = model.decode_tp(token, prompt_len + step)
        token = int(torch.argmax(logits))
    ttnn.synchronize_device(mesh_device)
    decode_s = time.perf_counter() - t0

    # ---- the speculative harness ----
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    drafter_path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(drafter_path)
    sd = load_drafter_state_dict(drafter_path)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    # One target per tap mode over the SAME model: the ttnn drafter wants the taps left on the mesh,
    # the host drafter wants them read back. Re-armed per variant below.
    dev_target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    host_target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=False)

    tt_drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, sd, tt_ccl=model.tt_ccl), dev_target)
    host_drafter = HostDrafter(load_drafter(drafter_path), host_target)
    host_target.embed(prompt)  # force the lazy 2.5 GB embedding load out of the timed region

    timers = {"target.forward": 0.0, "drafter.propose": 0.0}

    def _timed(name, fn):
        def wrapper(*a, **kw):
            t = time.perf_counter()
            out = fn(*a, **kw)
            timers[name] += time.perf_counter() - t
            return out

        return wrapper

    variants = (
        ("bucket-only (block=1)", 1, dev_target, tt_drafter),
        (f"ttnn drafter (block={cfg.block_size})", None, dev_target, tt_drafter),
        (f"host drafter (block={cfg.block_size})", None, host_target, host_drafter),
    )

    results, splits = {}, {}
    for label, block_size, target, drafter in variants:
        # Re-arm the model's taps for this variant's mode (both targets share one model).
        model.set_residual_taps(cfg.target_layer_ids, keep_on_device=target.device_taps)
        target.forward = _timed("target.forward", type(target).forward.__get__(target))
        drafter.propose = _timed("drafter.propose", type(drafter).propose.__get__(drafter))

        # WARM UP FIRST. Every distinct matmul shape compiles a kernel on first use, and without
        # this the first variant to touch the drafter absorbs all of that compile time inside the
        # timed region — which silently reports as "this drafter is slow". An earlier version of
        # this benchmark had exactly that bug: the JIT cache logged 961/1007 hits, i.e. 46 programs
        # built mid-measurement. Steady state is what a serving path would see.
        dflash_generate(drafter, target, prompt, max_new_tokens=2 * cfg.block_size, block_size=block_size)
        ttnn.synchronize_device(mesh_device)

        for k in timers:
            timers[k] = 0.0
        t0 = time.perf_counter()
        stats = dflash_generate(
            drafter, target, prompt, max_new_tokens=MAX_NEW, block_size=block_size, return_stats=True
        )
        ttnn.synchronize_device(mesh_device)
        results[label] = (time.perf_counter() - t0, stats)
        splits[label] = dict(timers)

    logger.info("")
    logger.info(f"{'path':<34} {'tok/s':>8} {'s/tok':>8}   detail")
    logger.info(f"{'-' * 78}")
    logger.info(f"{'decode_tp (model decode path)':<34} {MAX_NEW / decode_s:>8.2f} {decode_s / MAX_NEW:>8.3f}")
    for label, (elapsed, stats) in results.items():
        n = stats.num_output_tokens
        detail = f"{stats.mean_acceptance_length:.2f} tok/step over {len(stats.acceptance_lengths)} steps"
        logger.info(f"{label:<34} {n / elapsed:>8.2f} {elapsed / n:>8.3f}   {detail}")
    logger.info("")

    logger.info("where each run's wall clock goes:")
    for label, (elapsed, _) in results.items():
        parts = splits[label]
        rest = elapsed - sum(parts.values())
        detail = "  ".join(f"{k} {v:.1f}s ({100 * v / elapsed:.0f}%)" for k, v in parts.items())
        logger.info(f"  {label:<32} total {elapsed:.1f}s = {detail}  loop {rest:.1f}s ({100 * rest / elapsed:.0f}%)")
    logger.info("")

    def tps(label):
        elapsed, stats = results[label]
        return stats.num_output_tokens / elapsed

    tt_tps = tps(f"ttnn drafter (block={cfg.block_size})")
    logger.info(
        f"ttnn drafter vs host drafter:            {tt_tps / tps(f'host drafter (block={cfg.block_size})'):.2f}x"
    )
    logger.info(f"ttnn drafter vs its own bucket baseline: {tt_tps / tps('bucket-only (block=1)'):.2f}x")
    logger.info(f"ttnn drafter vs the real decode path:    {tt_tps / (MAX_NEW / decode_s):.2f}x")
