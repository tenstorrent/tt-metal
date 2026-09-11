# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Tracy target for the ttnn DFlash drafter ALONE — no 27B target in the capture.

Runs exactly one speculative step's worth of drafter work with ``start``/``stop`` signposts around
it: :meth:`TtDFlashDrafter.project_taps` (the tap all-gather, ``fc``, ``hidden_norm``) then
:meth:`TtDFlashDrafter.forward` (5 layers). Weight upload, the RoPE table build and a warm-up step
all sit OUTSIDE the window.

The draft-logit projection is deliberately NOT in the window: that runs on the *target's* LM head
(``TtTarget.lm_head_device``), so it belongs to a target report, not a drafter one.

WHY THIS IS ITS OWN TEST
------------------------
An end-to-end DFlash capture is dominated by the 27B verify forward — measured at 94% of a
speculative step, against 6% for drafting. A report of that tells you nothing about the drafter. This
file is the Tracy target for the drafter by itself, so a report of it is a drafter report.

The drafter is **replicated** across the mesh (see ``tt/dflash/weights.py``), so the only collective
in the window is the tap all-gather in ``project_taps``. Everything else is a local matmul, norm,
RoPE or SDPA.

SHAPES
------
One step at the production block size: ``n_new`` newly-accepted context rows (their taps arrive
fractured on the hidden dim, exactly as the target hands them over) plus a ``block_size``-slot noise
block. ``ctx`` is the drafter's already-committed KV history, which changes the SDPA k/v length —
parametrized because that is the one shape in the drafter that grows with the sequence.

Real checkpoint weights: matmul cost does not depend on the values, but the drafter's shapes come
from its ``config.json``, so the real config is loaded. PCC lives in
``tests/test_dflash_drafter_tp.py``.

Standalone Tracy capture (T3K)::

    MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      python -m tracy -p --op-support-count 100000 -r -v -m \\
        pytest "models/demos/blackhole/qwen36/tests/perf/test_profile_dflash_drafter.py::test_profile_dflash_drafter[wormhole_b0-device_params0-1x8-ctx64]"

    Note the ``-m`` and the full node id with NO trailing pytest flags: tracy parses argv with
    optparse, so a later ``-v`` is taken as tracy's own verbose, and without ``-m`` argv[0] is opened
    as a script path ("FileNotFoundError: 'pytest'").

Then::

    D=<new report dir>
    tt-perf-report generated/profiler/reports/$D/ops_perf_results_$D.csv \\
      --start-signpost start --end-signpost stop
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.tests.test_factory import parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.config import (
    DFlashDrafterConfig,
    load_drafter_state_dict,
    resolve_drafter_path,
)
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter


def _tracy_signpost_available() -> bool:
    try:
        from tracy import signpost  # noqa: F401

        return True
    except ImportError:
        return False


def _upload_taps(mesh_device, ctx: torch.Tensor, cfg: DFlashDrafterConfig):
    """The taps as the target hands them over: one tensor per tap, fractured on the hidden dim."""
    hidden = cfg.hidden_size
    multi = mesh_device.get_num_devices() > 1
    return [
        ttnn.from_torch(
            ctx[:, :, j * hidden : (j + 1) * hidden].reshape(1, 1, ctx.shape[1], hidden).to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1)) if multi else {}),
        )
        for j in range(len(cfg.target_layer_ids))
    ]


def _upload_noise(mesh_device, noise: torch.Tensor):
    multi = mesh_device.get_num_devices() > 1
    return ttnn.from_torch(
        noise.reshape(1, 1, noise.shape[-2], noise.shape[-1]).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if multi else {}),
    )


@torch.no_grad()
@pytest.mark.timeout(0)
@pytest.mark.parametrize("ctx", [64, 512], ids=lambda c: f"ctx{c}")
@parametrize_mesh_tp()
def test_profile_dflash_drafter(mesh_device, ctx, reset_seeds, ensure_gc):
    """One drafter step, signposted. ``ctx`` is the committed KV history the block attends over."""
    try:
        path = resolve_drafter_path()
    except Exception as e:  # noqa: BLE001 — a missing checkpoint is a skip, not a failure
        pytest.skip(f"drafter checkpoint unavailable ({type(e).__name__}: {e})")

    cfg = DFlashDrafterConfig.from_pretrained(path)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))

    q_len, n_new = cfg.block_size, cfg.block_size
    gen = torch.Generator().manual_seed(0)

    def one_step():
        """project_taps + 5 layers. Mirrors what TtDrafter.propose does per speculative step.

        ``start`` is the block's FIRST SLOT, which sits after the ``n_new`` context rows this step
        hands over — so it is ``context_len + n_new``, not ``context_len``. The drafter asserts that
        invariant.
        """
        start = drafter.context_len + n_new
        taps = _upload_taps(mesh_device, torch.randn(1, n_new, cfg.target_feature_size, generator=gen) * 0.05, cfg)
        noise = _upload_noise(mesh_device, torch.randn(1, q_len, cfg.hidden_size, generator=gen) * 0.05)
        hidden = drafter.forward(drafter.project_taps(taps), noise, start)
        ttnn.deallocate(noise)
        return hidden

    # Prime the drafter's KV history to `ctx` rows, and compile every program in the step. Both are
    # outside the measured window; without the warm-up the capture is mostly kernel compilation.
    while drafter.context_len < ctx:
        ttnn.deallocate(one_step())
    ttnn.synchronize_device(mesh_device)
    logger.info(f"warmed: drafter context {drafter.context_len} rows, block {q_len}, mesh {tuple(mesh_device.shape)}")

    # Upload the measured step's inputs OUTSIDE the window. In production nothing is uploaded at
    # all -- TtDrafter.propose takes the taps on-device from the target's _record_tap and the noise
    # block from target.embed_device -- so leaving the six ttnn.from_torch tilizes inside the
    # signposts (5 taps x 33 us + the 15 us block = 180 us, 1.7 % of the step) measures the harness
    # and not the drafter. Neither project_taps nor forward deallocates these, so one upload serves.
    m_start = drafter.context_len + n_new
    m_taps = _upload_taps(mesh_device, torch.randn(1, n_new, cfg.target_feature_size, generator=gen) * 0.05, cfg)
    m_noise = _upload_noise(mesh_device, torch.randn(1, q_len, cfg.hidden_size, generator=gen) * 0.05)
    ttnn.synchronize_device(mesh_device)

    signposted = _tracy_signpost_available()
    if signposted:
        from tracy import signpost

        signpost("start")

    out = drafter.forward(drafter.project_taps(m_taps), m_noise, m_start)
    ttnn.synchronize_device(mesh_device)

    if signposted:
        from tracy import signpost

        signpost("stop")

    assert out.shape[-2] == q_len and out.shape[-1] == cfg.hidden_size
    ttnn.deallocate(out)
    ttnn.deallocate(m_noise)
