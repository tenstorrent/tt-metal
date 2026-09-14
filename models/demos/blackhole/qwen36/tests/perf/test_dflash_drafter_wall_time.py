# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WALL time per drafter step, un-profiled — the number that decides dispatch-count changes.

``DFLASH_DRAFTER_OP_MAPPING.md`` reports **device** time (10.5 ms/step) and notes the drafter runs
at roughly 13 % device utilization, but the wall figure behind that came from
``test_dflash_throughput`` (``propose`` 0.6 s / 6 steps), which also includes the target's LM head
and a host argmax. So there was no clean per-step wall number, and without one you cannot judge any
change that trades device time for dispatch count — and the drafter has two of those:

* the tied K/V head-split (:meth:`~...drafter.TtDFlashDrafter._kv_heads`), −4 ops/layer, device wash
* the packer-fused SiLU (:meth:`~...drafter.TtDFlashDrafter._layer_mlp`), −1 op/layer, +21 μs device

Both are only worth having if host dispatch, not the device, sets the pace. This file measures that
directly: warm every program first (so no compile lands in the window), then time N steps of
``project_taps`` + ``forward`` with one ``synchronize_device`` at the end.

MEASURED (T3K, 30 steps, block 16, n_new 16): **120 ms/step**, against 10.5 ms of device time —
so the drafter runs at **~9 % device utilization** and is **host-dispatch-bound**. At 180 dispatches
per step that is ~0.6 ms of wall per op, which is the exchange rate that matters: **one op removed
is worth ~30x more wall time than one microsecond of device time saved.** Every dispatch-count
change in the drafter should be read against that, not against the device total.

TWO TRAPS, both hit while writing this file:

* **Sync every step, not once at the end.** Without a per-step ``synchronize_device`` the host runs
  30 steps x 180 ops ahead of the device and the measurement degrades catastrophically —
  3,267 ms/step, 27x the real figure. The per-step sync is also what the real loop does (it reads
  the drafted tokens back to host before building the next block), so it is the honest shape.
* **Warm up first.** Un-warmed steps measure JIT compilation, not dispatch.

Not asserted against a threshold — wall time depends on host CPU and is not CI-stable, and the
per-step spread is wide (58–177 ms in the reference run). It prints, and the caller reads it. That
spread means this cannot resolve a difference of a few percent: use it to establish the REGIME
(host- vs device-bound), not to A/B a handful of ops.

THAT WARNING WAS TESTED, 2026-09-14, and it holds. Swapping the attention head-concat from
transpose+reshape to nlp_concat_heads removes ~5 dispatches/step, which the 0.69 ms/op exchange rate
above predicts as ~3.4 ms. Five interleaved A/B pairs:

    pair       1        2        3        4        5     mean    sd
    baseline  131.66  128.48  130.22  129.40  124.98   128.9   2.6
    concat    128.30  116.58  106.54  106.68  131.18   117.9  11.4

No demonstrated benefit: the baseline is tight, the concat arm swings 106-131, and pair 5 lands
ABOVE the baseline. The first pair alone read -3.36 ms, agreeing with the prediction almost exactly
-- and that agreement was coincidence. A prediction matching a SINGLE measurement is weak evidence
when the measurement's own spread is four times the effect.

Pairs 2-4 then declined monotonically, and cache warming was the obvious story; pair 5 falsified it,
because warming does not reverse. Run A/B pairs interleaved, and repeat until they converge or
contradict.

The wider consequence: the drafter's entire available op-count reduction is ~25 ops (a fused QKV
projection, -20, plus this one) ~= 17 ms ~= 1.06x end to end, which sits under this instrument's
noise floor AND under the end-to-end test's. Op trimming is not a lever that can be validated here,
whatever its true sign.

Run::

    MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_drafter_wall_time.py
"""

from __future__ import annotations

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.perf.test_profile_dflash_drafter import _upload_noise, _upload_taps
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter

STEPS = 30
N_NEW = 16


@pytest.mark.timeout(0)
@torch.no_grad()
@parametrize_mesh_tp()
def test_dflash_drafter_wall_time(mesh_device, device_params, reset_seeds, ensure_gc):
    """Print un-profiled ms/step for project_taps + 5 layers."""
    del device_params
    try:
        path = resolve_drafter_path()
    except Exception as e:  # noqa: BLE001 — a missing checkpoint is a skip
        pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")
    cfg = DFlashDrafterConfig.from_pretrained(path)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))
    gen = torch.Generator().manual_seed(0)
    q_len = cfg.block_size

    def one_step(start):
        taps = _upload_taps(mesh_device, torch.randn(1, N_NEW, cfg.target_feature_size, generator=gen) * 0.05, cfg)
        noise = _upload_noise(mesh_device, torch.randn(1, q_len, cfg.hidden_size, generator=gen) * 0.05)
        hidden = drafter.forward(drafter.project_taps(taps), noise, start)
        ttnn.deallocate(noise)
        ttnn.deallocate(hidden)

    # Warm-up: every program must be compiled before the timed window, or the first steps measure
    # the JIT rather than dispatch. Two steps, because step 1 has no KV history and step 2 does.
    drafter.reset()
    one_step(N_NEW)
    one_step(2 * N_NEW)
    ttnn.synchronize_device(mesh_device)

    drafter.reset()
    per_step = []
    t0 = time.perf_counter()
    for i in range(STEPS):
        t1 = time.perf_counter()
        one_step(N_NEW * (i + 1))
        ttnn.synchronize_device(mesh_device)
        per_step.append((time.perf_counter() - t1) * 1000)
    ms = (time.perf_counter() - t0) * 1000 / STEPS
    logger.info("per-step ms: " + " ".join(f"{v:.0f}" for v in per_step))
    logger.info(f"drafter WALL: {ms:.1f} ms/step over {STEPS} steps (device time is ~10.5 ms/step)")
    print(f"\n>>> WALL {ms:.2f} ms/step <<<\n")
