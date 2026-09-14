# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Where does a TRACED drafter step's wall time go? The capture works and is currently SLOWER.

The drafter trace is correct -- identical tokens and identical acceptance against eager dispatch
(tests/reference/test_dflash_drafter_trace.py) -- but it measured **0.85x**, i.e. tracing the
drafter made the loop slower. That is a real result and it needs an attribution, not a theory: the
same question about the verify path was answered by timing the phases, which found 486 ms sitting in
a readback nobody suspected.

A traced drafter step, in order:

    project_taps           EAGER, outside the trace (its input address moves every step)
    stage_step             6 host->device buffers: cos, sin, q_cos, q_sin, and both masks
    stage_taps             clear the context buffer, then slice_write the real rows in
    stage_tokens           the block's ids
    execute_trace          the part that is actually traced
    commit_staged_context  5 layers x 2 x (slice + slice_write), EAGER -- the offset varies
    lm_head + readback     already narrowed to one device (test_dflash_lm_head_readback.py)

The two suspects are the ones tracing did not remove. ``stage_step`` uploads two [1,1,q_len,C+32]
masks per step, which at C=512 is most of the staged bytes and scales with CAPACITY rather than with
the real context. ``commit_staged_context`` is 20 dispatches that a capture cannot hold. If those
two dominate, the fix is a smaller C and a cheaper commit, not a different trace.

Prints a breakdown; asserts nothing about time.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_drafter_trace_breakdown.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 250_000_000
BLOCK = 16
ITERS = 10
CAPACITIES = (512, 128)


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
def test_traced_drafter_breakdown(mesh_device, device_params, reset_seeds, ensure_gc):
    """Per-phase wall time of one traced drafter step, at two capacities."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    state_dict = load_drafter_state_dict(path)
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)

    g = torch.Generator().manual_seed(17)
    ids = torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.long)

    for cap in CAPACITIES:
        tt = TtDFlashDrafter(mesh_device, cfg, state_dict, tt_ccl=model.tt_ccl, ctx_capacity=cap)
        drafter = TtDrafter(tt, target)
        drafter.enable_traced_draft(q_len=BLOCK, ctx_pad=16)

        ctx = (torch.randn(1, 1, 16, cfg.hidden_size, generator=g, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        kv_source = ttnn.from_torch(
            ctx,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
        )

        phases = {k: 0.0 for k in ("stage_step", "stage_taps", "stage_tokens", "replay", "commit", "lm_head")}

        def _t(key, fn):
            t0 = time.perf_counter()
            r = fn()
            phases[key] += (time.perf_counter() - t0) * 1000
            return r

        tt.reset()
        tt.alloc_step_buffers(q_len=BLOCK, ctx_pad=16)
        # Each iteration commits 16 context rows, so a small capacity supports fewer of them before
        # the buffer is full. Per-step cost is what is being timed, not how far the context grows.
        iters = min(ITERS, cap // 16 - 1)
        for i in range(iters):
            start = 16 * (i + 1)
            _t("stage_step", lambda: tt.stage_step(start, 16))
            _t("stage_taps", lambda: tt.stage_taps(kv_source, 16))
            _t("stage_tokens", lambda: tt.stage_tokens(ids))

            def _replay():
                ttnn.execute_trace(mesh_device, drafter._trace_id, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh_device)

            _t("replay", _replay)
            _t("commit", lambda: tt.commit_staged_context(16))
            # draft_ids_device, not lm_head_device: greedy propose argmaxes on device now, so this
            # must time the call the product makes. Timing the old one would report a cost the
            # drafter has stopped paying -- the exact defect that made an earlier verify breakdown
            # report 609 ms for a readback that had already been narrowed.
            _t("lm_head", lambda: target.draft_ids_device(drafter._trace_hidden, keep_rows=BLOCK - 1))

        total = sum(phases.values()) / iters
        mask_kb = 2 * BLOCK * (cap + 32) * 2 / 1024
        logger.info("=" * 78)
        logger.info(f"  C={cap}  (staged masks {mask_kb:.0f} KB/step)")
        for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
            logger.info(f"    {k:14} {v / iters:7.1f} ms  ({100 * v / sum(phases.values()):5.1f} %)")
        logger.info(f"    {'TOTAL':14} {total:7.1f} ms per traced drafter step")
        logger.info("=" * 78)
        print(f">>> C={cap}: {total:.1f} ms/step  " + "  ".join(f"{k} {v / iters:.1f}" for k, v in phases.items()))
        drafter.release_draft_trace()
        ttnn.deallocate(kv_source)
