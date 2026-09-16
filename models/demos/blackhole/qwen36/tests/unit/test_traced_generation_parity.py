# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Which step of the SECOND traced generation stops matching the first?

Acceptance alternates with traced-generation index -- odd generations healthy, even ones ruined,
the third recovering completely:

    demo, reference prompt   1st traced gen  17.54 tok/s  acceptance 4.950  0.98x production
                             2nd traced gen   2.99 tok/s  acceptance 1.021  0.17x
                             3rd traced gen  17.74 tok/s  acceptance 4.950  0.99x

Four hypotheses have been refuted by measurement (shared TT_CCL, fixed-capacity drafter,
per-generation snapshot allocation, and -- for the sibling anchor bug -- trace-at-offset and
snapshot reuse, both pcc 1.0). Every isolated construction comes back clean while the loop stays
broken, so this file stops guessing at a cause and just asks the loop where it first diverges.

HOW IT AVOIDS THE TRAP THE EARLIER TESTS FELL INTO. There is no drafter here and nothing adaptive:
both generations are driven through the SAME fixed token blocks. The target is deterministic, so
generation 2 must reproduce generation 1 exactly, logit for logit and tap for tap. Any difference is
the target's, it needs no reference implementation to compare against, and the step index of the
first mismatch is the whole answer. (A previous test of mine measured a fixture artifact and
produced a confident, wrong root cause -- see DFLASH_HANDOFF.md. Comparing the loop against ITSELF
cannot do that.)

WHY TAPS, NOT JUST LOGITS. Acceptance 1.021 means EVERY draft is rejected -- total drafter failure,
not degradation. The drafter's only input from the target is the taps, and ``verify_traced`` re-arms
them per replay from ``_vt_taps``. ``take_taps`` already documents that wrong tap rows "read as a
plausible-looking tap ... and draft pure garbage" while leaving the argmax intact, so the two can
diverge and only the taps may move. Both are reported per step.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_traced_generation_parity.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
ANCHOR = 128
BLOCK = 16
PROMPT_LEN = 5
# 5 + 16*5 = 85, comfortably inside the first bucket, so nothing here crosses an anchor and every
# verify stays on the traced path. The anchor bug is a SEPARATE defect; keep it out of this one.
N_STEPS = 5
N_GENS = 3
TAP_LAYERS = [10, 20, 30]


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


def _taps_host(target, taps):
    """Device taps are per-layer and hidden-fractured; gather each to one host tensor."""
    if not target.device_taps:
        return [taps.float()]
    return [ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(t.device(), dim=-1)).float() for t in taps]


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_which_step_of_gen2_diverges(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same fixed blocks, three traced generations. Generation 1 is the reference for 2 and 3."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    target = TtTarget(model, TAP_LAYERS, page_table, device_taps=True)

    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(1000, 2000, (1, PROMPT_LEN), generator=g, dtype=torch.long)
    # Fixed blocks, identical in every generation -- this is what makes the comparison exact.
    blocks = [torch.randint(1000, 2000, (1, BLOCK), generator=g, dtype=torch.long) for _ in range(N_STEPS)]

    def run_generation():
        """Prompt + N fixed blocks, exactly as the loop drives the target, minus the drafter."""
        target.reset()
        target.forward(prompt, 0, all_logits=False)
        rows, start = [], PROMPT_LEN
        for blk in blocks:
            lg, tp = target.forward(blk, start)
            rows.append((start, lg.float().cpu(), _taps_host(target, tp)))
            start += blk.shape[1]
        return rows

    # One eager generation first: enable_traced_verify() requires it, and it compiles the loop's
    # programs before the trace is parked.
    run_generation()
    target.enable_traced_verify()

    gens = [run_generation() for _ in range(N_GENS)]

    ref = gens[0]
    logger.info("=" * 78)
    first_bad = {}
    for gi in range(1, N_GENS):
        for si, ((start, lg_r, tp_r), (_, lg_g, tp_g)) in enumerate(zip(ref, gens[gi])):
            lg_ok, lg_pcc = comp_pcc(lg_r, lg_g, 0.999)
            argmax_agree = (lg_r.argmax(-1) == lg_g.argmax(-1)).float().mean().item()
            tap_pccs = [comp_pcc(a, b, 0.999)[1] for a, b in zip(tp_r, tp_g)]
            bad = (not lg_ok) or argmax_agree < 1.0
            if bad and gi not in first_bad:
                first_bad[gi] = si
            logger.info(
                f"gen{gi + 1} vs gen1  step {si} (start {start:3d})  "
                f"logits pcc {lg_pcc}  argmax {argmax_agree:.4f}  "
                f"taps " + " ".join(f"L{l}={p}" for l, p in zip(TAP_LAYERS, tap_pccs))
            )
        logger.info("-" * 78)

    logger.info("=" * 78)
    for gi in range(1, N_GENS):
        where = first_bad.get(gi)
        logger.info(
            f"generation {gi + 1}: "
            + (f"FIRST DIVERGENCE at step {where}" if where is not None else "matches generation 1 exactly")
        )
    print(
        "\n>>> "
        + " | ".join(
            f"gen{gi + 1}: " + (f"diverges at step {first_bad[gi]}" if gi in first_bad else "exact")
            for gi in range(1, N_GENS)
        )
        + "\n"
    )

    # The target is deterministic and every generation ran the SAME tokens, so this must hold. If it
    # fails, the step index above is the answer the parity hunt has been missing.
    assert not first_bad, (
        f"traced generations diverge from generation 1 at steps {first_bad} despite identical "
        "inputs -- the target is not reproducing itself across generations"
    )
