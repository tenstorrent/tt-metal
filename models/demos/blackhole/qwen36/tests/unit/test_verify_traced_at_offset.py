# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the traced verify still correct at a NON-ZERO chunk_start -- on logits AND on taps?

``test_dflash_target_trace_replay.py`` proves replay == eager, but only at ``chunk_start=0``, which
is the one offset the capture itself was taken at. The speculative loop spends most of a long
generation somewhere else: ``TtTarget`` re-anchors every ANCHOR tokens, so every verify after the
first bucket runs at ``chunk_start=128``, 256, ... against a trace captured at 0.

WHY THIS FILE EXISTS. Acceptance collapses to exactly 1 -- every draft rejected, for the rest of the
generation -- the moment a run crosses the anchor, and the collapse is caused by the TRACE, not by
re-anchoring (tests/reference/test_dflash_anchor_crossing.py):

    traced verify   before the anchor 4.241, after 1.118, 37/728 non-ascii in the output
    EAGER verify    before the anchor 4.241, after 4.471,  0/810 non-ascii

Eager crosses the boundary with acceptance intact. So re-anchoring is sound and the replay is not.

LOGITS AND TAPS ARE CHECKED SEPARATELY, and that separation is the point. The emitted tokens stay
plausible (clean English once the re-anchor snapshot is allocated fresh rather than reused), so the
target's ARGMAX survives whatever is wrong -- while the drafter, which eats the taps, fails
completely. ``Qwen36Model.take_taps`` already documents that exact asymmetry from a previous bug:
handing the drafter the wrong rows "reads as a plausible-looking tap (PCC ~0.38 vs the host taps)
and drafts pure garbage". A defect that moves logits a little and taps a lot is therefore expected,
and a test that only compared logits would miss it.

The eager path is the reference: it takes ``chunk_start`` as a plain int and rebuilds everything per
call, and ``TtTarget``'s own docstring records it as exact at these offsets ("PCC against a one-shot
prefill is exactly 1.0 at offsets 128 and 256").

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_verify_traced_at_offset.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
ANCHOR = 128
BLOCK = 16
# Tap layers: any few mid-stack layers will do -- this file is about WHERE the rows come from, not
# about which layers the drafter was trained on.
TAP_LAYERS = [10, 20, 30]


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _to_host(taps):
    """Device taps are per-layer, hidden-fractured across the mesh; gather to one host tensor each."""
    out = []
    for t in taps:
        out.append(ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=-1)).float())
    return out


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize("chunk_start", [0, ANCHOR], ids=lambda n: f"start{n}")
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_traced_matches_eager_at_offset(mesh_device, device_params, chunk_start, reset_seeds, ensure_gc):
    """Replay vs eager at chunk_start 0 (the control) and ANCHOR (where the loop actually lives)."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    model.set_residual_taps(TAP_LAYERS, keep_on_device=True)

    g = torch.Generator().manual_seed(7)
    total = chunk_start + BLOCK
    tokens = torch.randint(1000, 2000, (1, max(total, ANCHOR)), generator=g, dtype=torch.int32)
    block = tokens[:, chunk_start : chunk_start + BLOCK]

    def _prime():
        """Put the model in the state the loop would be in just before this block: everything
        before chunk_start already run, as whole ANCHOR buckets, exactly as TtTarget.forward does."""
        model._reset_gdn_state_for_new_sequence()
        for lo in range(0, chunk_start, ANCHOR):
            model.prefill_block_all_logits(
                tokens[:, lo : lo + ANCHOR], page_table, actual_len=ANCHOR, chunk_start=lo, bucket=ANCHOR
            )
            model.take_taps(ANCHOR)
        return model.save_gdn_state()

    # ---- EAGER reference. Run twice: the first compiles, and compiling anything after the capture
    # below would hang the process rather than raise.
    anchor_state = _prime()
    for _ in range(2):
        model.restore_gdn_state(anchor_state)
        eager_logits = model.prefill_block_all_logits(
            block, page_table, actual_len=BLOCK, chunk_start=chunk_start, bucket=ANCHOR
        )
        eager_taps = _to_host(model.take_taps(BLOCK))

    # ---- capture at chunk_start=0, as the shipping path does, then replay at `chunk_start`.
    model.capture_verify_trace(page_table, ANCHOR)

    model.restore_gdn_state(anchor_state)
    token_buf = torch.zeros(1, ANCHOR, dtype=torch.int32)
    token_buf[:, :BLOCK] = block
    traced_logits = model.verify_traced(token_buf, BLOCK, chunk_start, page_table, ANCHOR)
    traced_taps = _to_host(model.take_taps(BLOCK))

    lg_ok, lg_pcc = comp_pcc(eager_logits, traced_logits, 0.99)
    logger.info(f"chunk_start={chunk_start}: LOGITS pcc {lg_pcc}")
    tap_pccs = []
    for i, (e, t) in enumerate(zip(eager_taps, traced_taps)):
        ok, pcc = comp_pcc(e, t, 0.99)
        tap_pccs.append((TAP_LAYERS[i], pcc, ok))
        logger.info(f"chunk_start={chunk_start}: TAP layer {TAP_LAYERS[i]} pcc {pcc}")

    # Argmax is what the tokens depend on; taps are what the drafter depends on. Report both, since
    # the whole point is that they can diverge.
    agree = (eager_logits.argmax(-1) == traced_logits.argmax(-1)).float().mean().item()
    logger.info(f"chunk_start={chunk_start}: argmax agreement {agree:.4f}")
    print(f"\n>>> chunk_start={chunk_start}: logits pcc {lg_pcc}, argmax agree {agree:.4f}")
    print(">>> taps: " + ", ".join(f"L{l}={p}" for l, p, _ in tap_pccs) + "\n")

    assert lg_ok, f"chunk_start={chunk_start}: traced logits diverge from eager, pcc {lg_pcc}"
    for layer, pcc, ok in tap_pccs:
        assert ok, f"chunk_start={chunk_start}: tap layer {layer} diverges, pcc {pcc}"
