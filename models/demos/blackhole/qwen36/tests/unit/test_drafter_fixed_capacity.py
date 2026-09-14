# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 1 gate for tracing the drafter: does fixed-capacity KV reproduce the growing history EXACTLY?

Tracing the drafter (lever A) requires every per-step shape to be constant and every buffer address
to be stable -- a trace bakes both. ``ctx_capacity=C`` converts the drafter to that form:

* a PERSISTENT ``[1, nkv, C, hd]`` K/V history per layer, written through with slice_write instead
  of rebound to a fresh concat each step;
* the newly accepted context LEFT-padded to a constant 16 rows, so the fused kv_proj's M stops
  varying with the accept count;
* the accept count moved out of shapes and into MASK CONTENTS, which a capture can tolerate.

None of that is supposed to change the arithmetic. Because the append is exact -- slice_write takes
an arbitrary row offset (measured in test_drafter_kv_write_primitives.py), so there is no gap, no
eviction and no truncation -- the bar here is EQUALITY, not "close enough". If this drops to
0.99-something, the most likely cause is the bidirectional layer's new validity mask: that layer
took ``attn_mask=None`` before, and masking the not-yet-written rows is the one place where the
conversion touches arithmetic rather than plumbing.

ONE drafter instance is run both ways, flipping ``_cap`` between passes, so the two paths share
their weights exactly. Building two drafters would price the comparison against load-time dtype
variance instead of against the change under test.

This needs the drafter checkpoint but NOT the 27B: forward() takes the tap projection and the noise
embedding as plain tensors, so both are synthesised here.

Run::

    MESH_DEVICE=T3K DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash pytest -svq \\
      models/demos/blackhole/qwen36/tests/unit/test_drafter_fixed_capacity.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter

CAPACITIES = (64, 256, 1024)  # all >= 37, the context the STEPS above accumulate
# (new_ctx, q_len) per step -- the accept counts a real run produces: a first step with no context,
# then partial accepts of varying width. Every one of these is a DIFFERENT shape on the legacy path
# and the SAME shape on the fixed one, which is the whole point.
STEPS = ((0, 16), (7, 16), (16, 16), (3, 16), (11, 16))


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _run(drafter, steps, gen):
    """Drive `steps` through the drafter, returning each step's hidden output on host."""
    drafter.reset()
    out, start = [], 0
    hidden_size = drafter.cfg.hidden_size
    for new_ctx, q_len in steps:

        def _mk(rows):
            t = torch.randn(1, 1, rows, hidden_size, generator=gen, dtype=torch.float32) * 0.05
            return ttnn.from_torch(
                t.to(torch.bfloat16),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=drafter.device,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(drafter.device)) if drafter.multi else {}),
            )

        kv_source = _mk(new_ctx) if new_ctx else None
        noise = _mk(q_len)
        start += new_ctx
        hidden = drafter.forward(kv_source, noise, start)
        composer = dict(mesh_composer=ttnn.ConcatMeshToTensor(drafter.device, dim=0)) if drafter.multi else {}
        out.append(ttnn.to_torch(hidden, **composer)[:1].float())
        ttnn.deallocate(hidden)
        if kv_source is not None:
            ttnn.deallocate(kv_source)
        # `start` advances by the ACCEPTED rows only: the block's own q_len slots are speculative
        # and are never committed to the context by the drafter.
    return out


@pytest.mark.timeout(0)
@torch.no_grad()
# ctx_len must be <= block_size: this test hands the whole context to ONE forward, and the fixed
# path pads a step's accepted context to 16 rows. That is not a limitation of the scheme -- in the
# real loop new_ctx IS the accept count, which never exceeds block_size -- but it does bound what a
# single-step comparison may ask for. kv_len is ctx_len + 16, so 16 is the tile-aligned control and
# 8 / 4 are the padded cases.
@pytest.mark.parametrize("ctx_len", [16, 8, 4], ids=lambda n: f"ctx{n}")
@pytest.mark.parametrize(
    # project_taps' tap all-gather is the drafter's one collective, so this test needs the fabric.
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_fixed_capacity_against_host_oracle(mesh_device, device_params, ctx_len, reset_seeds, ensure_gc):
    """Score BOTH KV schemes against the fp32 host ``DFlashDraftModel`` -- the actual oracle.

    The first version of this gate compared fixed-capacity against the growing-history path and
    called the difference a regression. That was backwards. The growing-history path attends its
    key sequence's TILE PADDING whenever ``ctx_len + q_len`` is not a multiple of 32, because the
    additive mask is itself padded out with zeros and zero means VISIBLE
    (tests/unit/test_sdpa_tile_padding.py: at kv_len=16 the first query row keeps 6 % of its
    magnitude). So the legacy path is the one that needs explaining, and only a host reference can
    say which is right.

    ``ctx_len`` is chosen to straddle the boundary. kv_len is ``ctx_len + 16``:

        ctx 16 -> kv 32, tile-ALIGNED    both paths should agree with the oracle (control)
        ctx  8 -> kv 24,  8 pad rows     legacy dilutes, fixed does not
        ctx  4 -> kv 20, 12 pad rows     more padding, so a larger gap

    MEASURED 2026-09-14 (T3K, real drafter checkpoint, fp32 host oracle):

        ctx  kv_len  pad rows    legacy      fixed       delta
         16      32         0    0.995341    0.995341    +0.000000
          8      24         8    0.991290    0.994357    +0.003066
          4      20        12    0.979809    0.994045    +0.014236

    The fixed path holds ~0.994 whatever the padding; legacy decays as pad rows accumulate, and with
    ZERO padding the two are identical to six decimals. That is the conversion proved twice over: it
    changes nothing when there is nothing to change, and it recovers exactly what the tile padding
    was costing.

    Note the existing host-oracle test runs at ctx 64 and passes at PCC 0.99 despite the dilution:
    with 80 real keys the deficit is nearly uniform across query rows (80/96 vs 65/81), and PCC is
    correlation, so it barely registers a near-uniform scale factor. That is how this survived.
    """
    del device_params
    from models.common.utility_functions import comp_pcc
    from models.demos.blackhole.qwen36.tests.test_dflash_drafter_tp import (
        _download,
        _host_drafter,
        _inputs,
        _upload_replicated,
        _upload_taps,
    )

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    sd = load_drafter_state_dict(path)
    host = _host_drafter(cfg, sd)
    tt = TtDFlashDrafter(mesh_device, cfg, sd)

    q_len = cfg.block_size
    ctx, noise = _inputs(cfg, ctx_len, q_len)
    golden = host(
        target_hidden=ctx,
        noise_embedding=noise,
        position_ids=torch.arange(ctx_len + q_len)[None],
    )

    def _score(cap):
        tt._cap = cap
        tt.reset()
        kv = tt.project_taps(_upload_taps(mesh_device, ctx, cfg))
        got = _download(mesh_device, tt.forward(kv, _upload_replicated(mesh_device, noise), start=ctx_len))
        _, pcc = comp_pcc(golden, got, 0.99)
        return float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc

    legacy = _score(None)
    fixed = _score(CAPACITIES[-1])
    pad = (-(ctx_len + q_len)) % 32
    logger.info(
        f"ctx_len={ctx_len:3d} (kv_len={ctx_len + q_len:3d}, {pad:2d} tile-pad rows)  "
        f"legacy {legacy:.6f}   fixed {fixed:.6f}   delta {fixed - legacy:+.6f}"
    )
    assert fixed > 0.99, f"fixed-capacity drafter diverged from the host oracle: pcc {fixed:.6f}"
    if pad:
        assert fixed > legacy, (
            f"with {pad} tile-pad rows the fixed path ({fixed:.6f}) should beat the diluted legacy "
            f"path ({legacy:.6f}) against the host oracle; if it does not, the padding is not the "
            "mechanism after all"
        )


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_masked_history_alone_is_inert(mesh_device, reset_seeds, ensure_gc):
    """A step with ``new_ctx == 16`` and no prior context pads NOTHING, so the two schemes build the
    identical 32-row kv_src and the identical causal mask over it. The ONLY difference left is the C
    fully-masked history rows the fixed path prepends.

    MEASURED: pcc 1.000000 -- prepending masked history is exactly inert, which is what isolated the
    disagreement to the tile padding instead.
    """
    from models.common.utility_functions import comp_pcc

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))

    steps = ((16, 16),)  # hist_len 0, new_ctx 16 -> zero pad rows in EITHER path
    drafter._cap = None
    legacy = _run(drafter, steps, torch.Generator().manual_seed(23))
    drafter._cap = 256
    fixed = _run(drafter, steps, torch.Generator().manual_seed(23))

    _, pcc = comp_pcc(legacy[0], fixed[0], 0.99)
    value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
    logger.info(f"no-padding step (new_ctx=16, hist_len=0), masked history is the ONLY delta: pcc {value:.6f}")
    assert value > 0.9999, f"masked history alone changed the attention: pcc {value:.6f}"


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_staged_step_matches_unstaged(mesh_device, reset_seeds, ensure_gc):
    """Stage 2 gate: a step reading PERSISTENT buffers must equal one that builds its inputs.

    Staging is what makes a capture possible. ``forward`` normally builds this step's rope slices
    and masks fresh (``ttnn.from_torch`` -- a host upload, illegal inside a capture) and reads the
    tap projection at whatever address ``project_taps`` just returned. ``alloc_step_buffers`` +
    ``stage_step`` + ``stage_taps`` move all of that to fixed addresses refilled in place, so a
    replay has something stable to read.

    None of it is supposed to change the arithmetic -- the staged buffers hold the SAME values, just
    somewhere permanent -- so the bar is equality. The interesting failure is not a big divergence
    but a small one on a later step, which would mean a buffer is carrying a previous step's
    contents (a stale mask, an uncleared context tail) rather than this step's.
    """
    from models.common.utility_functions import comp_pcc

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path), ctx_capacity=256)

    # Varying accept counts, so a stale buffer from the previous step cannot pass unnoticed.
    steps = ((0, 16), (7, 16), (16, 16), (3, 16), (11, 16))
    gen_seed, hidden_size = 31, cfg.hidden_size

    def _mk(rows, gen):
        t = (torch.randn(1, 1, rows, hidden_size, generator=gen, dtype=torch.float32) * 0.05).to(torch.bfloat16)
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=drafter.device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(drafter.device)) if drafter.multi else {}),
        )

    def _run_mode(staged):
        gen = torch.Generator().manual_seed(gen_seed)
        drafter.reset()
        if staged:
            drafter.alloc_step_buffers(q_len=16, ctx_pad=16)
        out, start = [], 0
        for new_ctx, q_len in steps:
            kv_source = _mk(new_ctx, gen) if new_ctx else None
            noise = _mk(q_len, gen)
            start += new_ctx
            if staged:
                drafter.stage_step(start, new_ctx)
                drafter.stage_taps(kv_source, new_ctx)
                hidden = drafter.forward(None, noise, start, staged=True)
                # The staged path defers its append: the step wrote its context K/V to fixed
                # staging buffers (constant offsets, so a capture can record them) and this moves
                # them into the history at the offset that actually varies.
                drafter.commit_staged_context(new_ctx)
            else:
                hidden = drafter.forward(kv_source, noise, start)
            composer = dict(mesh_composer=ttnn.ConcatMeshToTensor(drafter.device, dim=0)) if drafter.multi else {}
            out.append(ttnn.to_torch(hidden, **composer)[:1].float())
            ttnn.deallocate(hidden)
            if kv_source is not None:
                ttnn.deallocate(kv_source)
        return out

    unstaged = _run_mode(False)
    staged = _run_mode(True)

    worst = 1.0
    for i, ((new_ctx, _), a, b) in enumerate(zip(steps, unstaged, staged)):
        assert a.shape == b.shape, f"step {i}: shape changed {tuple(a.shape)} -> {tuple(b.shape)}"
        _, pcc = comp_pcc(a, b, 0.99)
        value = float(str(pcc).split()[-1]) if not isinstance(pcc, float) else pcc
        worst = min(worst, value)
        logger.info(f"step {i} (new_ctx={new_ctx:2d}) staged vs unstaged: pcc {value:.6f}")
    logger.info(f"worst pcc across {len(steps)} staged steps: {worst:.6f}")
    assert worst > 0.9999, (
        f"a staged step diverged from the equivalent unstaged one (worst pcc {worst:.6f}); the "
        "buffers are supposed to hold identical values, so suspect one carrying a previous step's "
        "contents -- an uncleared context tail or a mask that stage_step did not refill"
    )
