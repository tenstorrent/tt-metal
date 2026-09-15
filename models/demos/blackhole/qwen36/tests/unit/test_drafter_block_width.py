# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the drafter survive a NARROW final block (``q_len < block_size``)?

Every drafter test in this suite drives ``q_len = 16``. ``test_drafter_fixed_capacity.py``'s STEPS
vary ``new_ctx`` (0/7/16/3/11) but pin ``q_len`` at 16 in all five, so no test has ever run the
drafter at a narrower block. The real loop does, on the LAST step of a generation:

    generate.py:  verify_size = min(block_size, max_length - start, target.max_block(start))

``max_length - start`` shrinks to the tokens still owed. ``verify_size == 1`` is safe -- the loop
guards ``if verify_size > 1`` and skips ``drafter.propose`` entirely -- but 2..15 go straight
through to ``forward()`` as a narrow block. Which of those widths a generation ends on depends on
its acceptance pattern, which is why this is intermittent rather than deterministic.

WHY THIS FILE EXISTS (measured 2026-09-15, T3K):

``test_dflash_prose_throughput.py`` hung. py-spy put the main thread in ``_kv_heads``
(drafter.py:585, ``nlp_create_qkv_heads``) at ``layer_idx=4``, with locals ``start=110``,
``hist_len=109``, ``new_ctx=1`` and **``q_len=2``** -- the final step of a 64-token prompt with
NEW_TOKENS=48 (max_length 112, so ``max_length - start == 2``). gdb put the native stack in::

    tt::umd::memcpy_from_device
      SiliconTlbWindow::read_block
        Cluster::read_core
          SystemMemoryManager::fetch_queue_reserve_back        <- spinning here
            program_dispatch::write_program_command_sequence
              FDMeshCommandQueue::enqueue_mesh_workload
                NlpCreateHeadsDeviceOperation

``fetch_queue_reserve_back`` polls the dispatch fetch queue for space. Spinning there means the
DEVICE STOPPED CONSUMING COMMANDS and the queue backed up -- the host is stuck pushing, not stuck
computing. So ``nlp_create_qkv_heads`` is where the queue happened to fill, NOT necessarily the op
at fault. That matches the standing warning in test_dflash_hang_repro.py that the Python-level
location is wherever the host first blocks, dispatch being asynchronous.

It also CORRECTS a premise recorded there. That file dismissed the drafter-attention localisation
on the grounds that the shapes are constant -- "k there is always ctx_pad(16) + block(16) = 32
rows". They are constant only while ``q_len == block_size``. ``_layer_attention`` computes
``kv_seq = kv_src.shape[-2]`` from ``new_ctx + q_len``, so on the hung step it was **3**, not 32.
The shapes are constant across every step but the last.

(For the record, the matmul program config is NOT the degenerate part: ``_proj_pc`` keys on
``ceil(m/32)``, so m=3, m=16 and m=32 all collapse to ``m_tiles=1`` and share one cached config.
Whatever the mechanism is, it is not a per_core_M of zero.)

This test isolates block width from everything else -- no 27B target, no verify trace, no
speculative loop. ``forward()`` takes the tap projection and the noise embedding as plain tensors,
so both are synthesised, exactly as test_drafter_fixed_capacity.py does it.

Two arms, cheapest first:

* ``width_sweep`` -- a fresh drafter, short history, one step at each q_len in 16..1. Isolates
  width with nothing else varying.
* ``tail_after_history`` -- builds ~109 rows of history with 16-wide steps and THEN takes one
  2-wide step, reproducing the observed failure's actual state.

Each step logs before and after, so a hang names the width that caused it rather than leaving it
to be inferred.

Run::

    MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash pytest -svq \\
      models/demos/blackhole/qwen36/tests/unit/test_drafter_block_width.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter

# 16 is the control (every existing test runs here). 1 should never reach forward() from the real
# loop -- generate.py guards it -- but it is cheap to check and a hang there would mean the width
# guard is the only thing holding the shipping path up.
WIDTHS = (16, 8, 4, 3, 2, 1)

# The hung step's own state: 109 rows of history, 1 newly accepted row, a 2-wide block.
HIST_TARGET = 109
TAIL_NEW_CTX = 1
TAIL_Q_LEN = 2


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _mk(drafter, rows, gen):
    """A synthetic [1, 1, rows, hidden] activation on the mesh, as the real taps/noise arrive."""
    t = torch.randn(1, 1, rows, drafter.cfg.hidden_size, generator=gen, dtype=torch.float32) * 0.05
    return ttnn.from_torch(
        t.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=drafter.device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(drafter.device)) if drafter.multi else {}),
    )


def _step(drafter, new_ctx, q_len, start, gen, tag):
    """One drafter forward, logged either side so a hang names the step that caused it."""
    kv_source = _mk(drafter, new_ctx, gen) if new_ctx else None
    noise = _mk(drafter, q_len, gen)
    logger.info(f">>>>> {tag}: START new_ctx={new_ctx} q_len={q_len} start={start} kv_seq={new_ctx + q_len}")
    t0 = time.perf_counter()
    hidden = drafter.forward(kv_source, noise, start)
    # forward() returns as soon as the work is dispatched; synchronize so the timing (and any
    # failure) belongs to THIS step rather than landing on whichever step next forces a wait.
    ttnn.synchronize_device(drafter.device)
    dt = time.perf_counter() - t0
    logger.info(f">>>>> {tag}: DONE in {dt:.2f}s, hidden {tuple(hidden.shape)}")
    ttnn.deallocate(hidden)
    if kv_source is not None:
        ttnn.deallocate(kv_source)
    return dt


def _build(mesh_device):
    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    return TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    # project_taps' tap all-gather is the drafter's one collective, so this needs the fabric.
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_width_sweep(mesh_device, device_params, reset_seeds, ensure_gc):
    """One step at each block width, fresh drafter, short history. Width is the only variable."""
    del device_params
    drafter = _build(mesh_device)
    gen = torch.Generator().manual_seed(0)

    timings = {}
    for q_len in WIDTHS:
        drafter.reset()
        # A 16-row context first, so the narrow block is the ONLY unusual shape in the step (a
        # zero-context first step is its own special case -- see test_drafter_fixed_capacity STEPS).
        _step(drafter, 0, 16, 0, gen, f"warm q_len=16 (before width {q_len})")
        timings[q_len] = _step(drafter, 16, q_len, 16, gen, f"width {q_len}")

    logger.info("=" * 70)
    for q_len, dt in timings.items():
        logger.info(f"  q_len {q_len:2d}  {dt:6.2f}s")
    logger.info("=" * 70)
    print("\n>>> widths OK: " + ", ".join(f"{w}({t:.2f}s)" for w, t in timings.items()) + "\n")


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_tail_after_history(mesh_device, device_params, reset_seeds, ensure_gc):
    """Reproduce the hung step's state: ~109 rows of history, then a 2-wide block.

    If test_width_sweep passes and this hangs, width alone is not the trigger -- the accumulated
    history is part of it, which is the correlate test_dflash_hang_repro.py was chasing.
    """
    del device_params
    drafter = _build(mesh_device)
    gen = torch.Generator().manual_seed(0)
    drafter.reset()

    # Grow the history in 16-wide steps, exactly as a fully-accepted run would.
    start = 0
    _step(drafter, 0, 16, start, gen, "hist seed")
    while start + 16 <= HIST_TARGET:
        start += 16
        _step(drafter, 16, 16, start, gen, f"hist grow -> {start}")
    if start < HIST_TARGET:
        rem = HIST_TARGET - start
        start += rem
        _step(drafter, rem, 16, start, gen, f"hist top-up -> {start}")

    logger.info(f">>>>> history at {start} rows; now the narrow tail")
    _step(drafter, TAIL_NEW_CTX, TAIL_Q_LEN, start + TAIL_NEW_CTX, gen, "TAIL q_len=2")
    print(f"\n>>> tail q_len={TAIL_Q_LEN} after {start} rows of history: OK\n")
