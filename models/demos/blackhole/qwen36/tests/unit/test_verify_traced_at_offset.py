# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Is the traced verify still correct at a NON-ZERO chunk_start -- on logits AND on taps?

ANSWER: YES. All three arms are pcc 1.0 with argmax agreement 1.0000 and every tap exact:

    capture@0   replay@0     logits pcc 1.0   argmax 1.0000   all taps 1.0
    capture@0   replay@128   logits pcc 1.0   argmax 1.0000   all taps 1.0
    capture@128 replay@128   logits pcc 1.0   argmax 1.0000   all taps 1.0

``test_dflash_target_trace_replay.py`` only ever checked ``chunk_start=0``, the offset the capture
is taken at, so this closes that gap: ONE capture serves every chunk_start as well as every
valid_len, and the replay machinery is not offset-specific.

READ THE FIXTURE NOTE BEFORE TRUSTING ANY RESULT FROM THIS FILE. An earlier version of this test
reported capture@0 replay@128 at logits pcc 0.843 with argmax agreement 0.0625 -- one row in
sixteen -- and that number was entirely an artifact of the fixture. ``capture_verify_trace`` runs
real forwards (two warm-up passes plus the captured one), and those include ``paged_fill_cache``,
so capturing at offset 0 writes its dummy all-zero tokens over KV pages 0..1 -- exactly the prefix
this test had primed. The replay then attended a zeroed prefix. The capture@128 arm looked healthy
for the mirror-image reason: it clobbered pages 2..3, which the replay immediately rewrote.

Re-priming after the capture removes the artifact and all three arms pass. The real loop never had
this problem: ``TtTarget.forward`` re-runs the whole bucket eagerly when it crosses an anchor, which
rewrites those pages before any traced tail replays.

WHAT THIS DOES NOT EXPLAIN, and what is therefore still open. Restricting the trace to ``lo == 0``
demonstrably fixes the loop -- demo acceptance 1.138 -> 4.950, and on the 200-token crossing arm
1.118 -> 4.471, matching pure-eager exactly (tests/reference/test_dflash_anchor_crossing.py). Those
are end-to-end measurements, not fixture artifacts, so replaying past the anchor IN THE LOOP really
is broken. But it is not because the capture bakes chunk_start, because this file shows it does not.

The leading remaining hypothesis is an interaction between the anchor GDN snapshot and the parked
trace, which this file cannot see because it takes a FRESH snapshot (``_prime`` calls
``save_gdn_state()`` with no ``into=``) while ``TtTarget.forward`` re-anchors with
``save_gdn_state(into=self._anchor_gdn)``. That reuse is independently implicated: DFLASH_FRESH_ANCHOR=1
moves post-anchor acceptance 1.118 -> 1.617 and removes the output corruption outright, and adding
the same reuse to ``reset()`` SIGBUSed the drafter. To test it, re-run this file with the snapshot
reused rather than freshly allocated.

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
# Early layers, to find the FIRST one that diverges: the tap PCCs at 10/20/30 fall 0.979 / 0.938 /
# 0.880, i.e. the error is introduced early and compounds with depth. Whether the first bad layer is
# full-attention or GDN decides where the baked chunk_start lives.
TAP_LAYERS = [0, 1, 2, 3, 4, 5, 10, 20, 30]


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
# (capture_at, replay_at). The first two are the original question: one capture at 0 replayed at 0
# and at ANCHOR. The third asks whether the trace is valid at an offset it was CAPTURED at -- if it
# is, the capture is simply offset-specific and the fix is to stage what it bakes (or capture per
# anchor); if it is not, something deeper than chunk_start is wrong.
@pytest.mark.parametrize(
    "capture_at, chunk_start",
    [(0, 0), (0, ANCHOR), (ANCHOR, ANCHOR)],
    ids=["cap0_run0", "cap0_run128", "cap128_run128"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_traced_matches_eager_at_offset(mesh_device, device_params, capture_at, chunk_start, reset_seeds, ensure_gc):
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

    # ---- capture at `capture_at` (0 is what the shipping path does), then replay at `chunk_start`.
    model.capture_verify_trace(page_table, ANCHOR, capture_chunk_start=capture_at)

    # THE CAPTURE WRITES TO THE PAGED KV. capture_verify_trace runs real forwards (two warm-up
    # passes plus the captured one) and those include paged_fill_cache, so it fills the pages for
    # `capture_at` with its dummy all-zero tokens. At capture_at=0 that clobbers the very prefix
    # this test primed, and a replay at chunk_start=128 would then attend a zeroed prefix -- which
    # looks exactly like "the trace is invalid at a different offset" while actually being a
    # fixture artifact. Re-prime so the prefix pages hold the real prefix again.
    #
    # The real loop does not have this problem: TtTarget.forward re-runs the whole bucket eagerly
    # when it crosses an anchor, which rewrites those pages before any traced tail replays.
    anchor_state = _prime()

    model.restore_gdn_state(anchor_state)
    token_buf = torch.zeros(1, ANCHOR, dtype=torch.int32)
    token_buf[:, :BLOCK] = block
    traced_logits = model.verify_traced(token_buf, BLOCK, chunk_start, page_table, ANCHOR)
    traced_taps = _to_host(model.take_taps(BLOCK))

    lg_ok, lg_pcc = comp_pcc(eager_logits, traced_logits, 0.99)
    logger.info(f"capture@{capture_at} replay@{chunk_start}: LOGITS pcc {lg_pcc}")
    tap_pccs = []
    for i, (e, t) in enumerate(zip(eager_taps, traced_taps)):
        ok, pcc = comp_pcc(e, t, 0.99)
        kind = "attn" if model.args.is_full_attention_layer(TAP_LAYERS[i]) else "gdn "
        tap_pccs.append((TAP_LAYERS[i], pcc, ok, kind))
        logger.info(f"capture@{capture_at} replay@{chunk_start}: TAP L{TAP_LAYERS[i]:<2} [{kind}] pcc {pcc}")

    # Argmax is what the tokens depend on; taps are what the drafter depends on. Report both, since
    # the whole point is that they can diverge.
    agree = (eager_logits.argmax(-1) == traced_logits.argmax(-1)).float().mean().item()
    logger.info(f"chunk_start={chunk_start}: argmax agreement {agree:.4f}")
    print(f"\n>>> capture@{capture_at} replay@{chunk_start}: logits pcc {lg_pcc}, argmax agree {agree:.4f}")
    print(">>> taps: " + ", ".join(f"L{l}[{k.strip()}]={p:.4f}" for l, p, _, k in tap_pccs) + "\n")

    assert lg_ok, f"chunk_start={chunk_start}: traced logits diverge from eager, pcc {lg_pcc}"
    for layer, pcc, ok, _k in tap_pccs:
        assert ok, f"chunk_start={chunk_start}: tap layer {layer} diverges, pcc {pcc}"
