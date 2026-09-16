# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the DRAFTER reproduce itself across generations while a target trace is parked?

The mirror of ``test_traced_generation_parity.py``, and the probe that follows directly from it.
That file drove the target through the same fixed blocks for three traced generations and found it
BIT-EXACT every time -- logits pcc 1.0, argmax 1.0000, taps 1.0. Acceptance is how many drafted
tokens match the target's argmax, so if the target's argmax is identical in generation 2 and
acceptance still collapses there (4.950 -> 1.021, a 5.9x throughput swing), the drafts must be what
changed. This file asks the drafter the same question the target already passed.

FIVE HYPOTHESES HAVE DIED, all of them aimed at the target: shared TT_CCL, fixed-capacity drafter,
the per-generation reset snapshot allocation, trace-at-offset, and snapshot reuse. They kept
measuring clean because they were pointed at the wrong component.

THE MECHANISM THIS IS BUILT TO CATCH. The drafter runs EAGERLY while the target's trace is parked,
allocating and freeing device buffers on every step, and Metal warns that buffers allocated under an
active trace "may be corrupted once a trace is executed". An alternating allocate/free pattern
across generations is exactly how that produces a period-2 effect, and it casts the drafter as the
VICTIM of the parked trace's memory rather than a participant in it. That also explains why
``ctx_capacity`` alone does not fix it (measured 1.031): fixed capacity pins the KV history buffers
but not the per-step temporaries.

WHAT IS HELD FIXED, and why that matters. The drafter's inputs are SYNTHETIC and identical in every
generation -- same seed, same tensors -- so its output is a pure function of its own state. It needs
no reference implementation, and a fixture cannot manufacture a divergence between a thing and
itself. (Two earlier probes of mine produced confident, wrong root causes from fixture artifacts;
see DFLASH_HANDOFF.md. Self-comparison is what closed that door.)

WHAT IS DELIBERATELY NOT HELD FIXED: a real target verify replays between drafter steps, exactly as
the loop does it. The drafter's inputs stay synthetic, but the surrounding device traffic --
allocations, the parked trace executing -- is the real thing, because that traffic is the suspect.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_drafter_generation_parity.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
TRACE_REGION = 200_000_000
BLOCK = 16
PROMPT_LEN = 5
N_GENS = 3
# (new_ctx, q_len) per step -- a plausible accept pattern. Nothing here crosses the 128 anchor, so
# the separate anchor defect stays out of this measurement.
STEPS = ((0, BLOCK), (7, BLOCK), (7, BLOCK), (7, BLOCK))


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
def test_drafter_reproduces_itself_across_generations(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same synthetic inputs, three generations, target trace parked and replaying in between."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path))

    tg = torch.Generator().manual_seed(23)
    prompt = torch.randint(1000, 2000, (1, PROMPT_LEN), generator=tg, dtype=torch.long)
    blocks = [torch.randint(1000, 2000, (1, BLOCK), generator=tg, dtype=torch.long) for _ in STEPS]

    def _mk(rows, gen):
        t = torch.randn(1, 1, rows, cfg.hidden_size, generator=gen, dtype=torch.float32) * 0.05
        return ttnn.from_torch(
            t.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if drafter.multi else {}),
        )

    def run_generation(drive_target):
        """One generation's worth of drafter steps, with identical synthetic inputs every time.

        `drive_target` replays a real target verify between drafter steps, so the surrounding
        device traffic matches the loop even though the drafter's own inputs are pinned.
        """
        gen = torch.Generator().manual_seed(101)  # SAME seed every generation -> same inputs
        drafter.reset()
        if drive_target:
            target.reset()
            target.forward(prompt, 0, all_logits=False)
        out, start, t_start = [], 0, PROMPT_LEN
        for (new_ctx, q_len), blk in zip(STEPS, blocks):
            kv_source = _mk(new_ctx, gen) if new_ctx else None
            noise = _mk(q_len, gen)
            start += new_ctx
            hidden = drafter.forward(kv_source, noise, start)
            composer = dict(mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0)) if drafter.multi else {}
            out.append(ttnn.to_torch(hidden, **composer)[:1].float())
            ttnn.deallocate(hidden)
            ttnn.deallocate(noise)
            if kv_source is not None:
                ttnn.deallocate(kv_source)
            if drive_target:
                target.forward(blk, t_start)
                t_start += blk.shape[1]
        return out

    # One eager pass of each before the capture, so nothing compiles with the trace parked.
    run_generation(drive_target=True)
    target.enable_traced_verify()

    gens = [run_generation(drive_target=True) for _ in range(N_GENS)]

    ref = gens[0]
    first_bad = {}
    logger.info("=" * 78)
    for gi in range(1, N_GENS):
        for si, (a, b) in enumerate(zip(ref, gens[gi])):
            ok, pcc = comp_pcc(a, b, 0.999)
            if not ok and gi not in first_bad:
                first_bad[gi] = si
            logger.info(f"gen{gi + 1} vs gen1  step {si}  hidden pcc {pcc}{'   <-- DIVERGES' if not ok else ''}")
        logger.info("-" * 78)

    for gi in range(1, N_GENS):
        where = first_bad.get(gi)
        logger.info(
            f"generation {gi + 1}: "
            + (f"FIRST DIVERGENCE at step {where}" if where is not None else "matches generation 1 exactly")
        )
    print(
        "\n>>> drafter "
        + " | ".join(
            f"gen{gi + 1}: " + (f"diverges at step {first_bad[gi]}" if gi in first_bad else "exact")
            for gi in range(1, N_GENS)
        )
        + "\n"
    )

    # The drafter saw identical inputs in every generation, so this must hold. If it fails, the
    # drafter is the component that alternates and the step index is where to look.
    assert not first_bad, (
        f"drafter diverges from generation 1 at steps {first_bad} despite identical inputs -- "
        "it is not reproducing itself across generations while a target trace is parked"
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_drafter_reproduces_itself_on_the_real_tap_path(mesh_device, device_params, reset_seeds, ensure_gc):
    """Same question, but the drafter eats the TARGET'S OWN TAPS instead of synthetic tensors.

    The synthetic-input test above pins the drafter's inputs to rule its own state in or out. This
    one closes the remaining gap: the real path hands the drafter ``target.forward``'s taps, which
    arrive as DEVICE tensors that the traced replay refreshes in place (``verify_traced`` re-arms
    ``self._taps = dict(self._vt_taps)`` every replay). Synthetic tensors are freshly allocated host
    uploads and therefore cannot expose a defect that lives in those trace-owned tap buffers.

    Determinism is kept by driving the target with FIXED token blocks rather than the drafter's own
    proposals, so the sequence, the taps and hence the drafter's inputs are identical in every
    generation. The drafted tokens are recorded but never fed back -- that is what keeps two
    generations comparable at all.

    test_traced_generation_parity.py already proved the taps themselves are bit-exact across
    generations, so if the DRAFTS still differ here, the fault is in how the drafter consumes them,
    not in their contents.
    """
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    from models.demos.blackhole.qwen36.reference.dflash.drafters import TtDrafter

    path = resolve_drafter_path()
    cfg = DFlashDrafterConfig.from_pretrained(path)
    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDrafter(TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(path)), target)

    tg = torch.Generator().manual_seed(23)
    prompt = torch.randint(1000, 2000, (1, PROMPT_LEN), generator=tg, dtype=torch.long)
    blocks = [torch.randint(1000, 2000, (1, BLOCK), generator=tg, dtype=torch.long) for _ in STEPS]

    def run_generation():
        """Prompt, then fixed blocks: propose from the real taps, then advance on the FIXED block."""
        target.reset()
        drafter.reset()
        _, taps = target.forward(prompt, 0, all_logits=False)
        pending, drafts, start = [taps], [], PROMPT_LEN
        for blk in blocks:
            drafted, _ = drafter.propose(
                pending[0] if len(pending) == 1 else target._taps_cat(pending),
                blk.clone(),
                start,
                temperature=0.0,
                top_p=1.0,
                top_k=0,
            )
            drafts.append(drafted.cpu().clone())
            _, tp = target.forward(blk, start)
            pending = [tp]
            start += blk.shape[1]
        return drafts

    run_generation()  # eager: satisfies enable_traced_verify and compiles before the trace parks
    target.enable_traced_verify()

    gens = [run_generation() for _ in range(N_GENS)]

    ref, first_bad = gens[0], {}
    logger.info("=" * 78)
    for gi in range(1, N_GENS):
        for si, (a, b) in enumerate(zip(ref, gens[gi])):
            same = bool(torch.equal(a, b))
            n_diff = int((a != b).sum().item())
            if not same and gi not in first_bad:
                first_bad[gi] = si
            logger.info(
                f"gen{gi + 1} vs gen1  step {si}  drafted tokens "
                f"{'IDENTICAL' if same else f'DIFFER in {n_diff}/{a.numel()} slots   <-- DIVERGES'}"
            )
        logger.info("-" * 78)

    for gi in range(1, N_GENS):
        where = first_bad.get(gi)
        logger.info(
            f"generation {gi + 1}: "
            + (f"FIRST DIVERGENCE at step {where}" if where is not None else "drafts match generation 1 exactly")
        )
    print(
        "\n>>> real-tap drafter "
        + " | ".join(
            f"gen{gi + 1}: " + (f"diverges at step {first_bad[gi]}" if gi in first_bad else "exact")
            for gi in range(1, N_GENS)
        )
        + "\n"
    )

    assert not first_bad, (
        f"drafted tokens diverge from generation 1 at steps {first_bad} on the real tap path, "
        "despite identical prompts, identical blocks and provably identical taps -- the drafter is "
        "the component that alternates"
    )
