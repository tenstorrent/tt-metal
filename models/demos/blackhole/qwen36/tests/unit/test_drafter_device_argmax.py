# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Does the on-device argmax pick the same draft tokens as the host one?

MEASURED 2026-09-14 (T3K, block 16, vocab 248,320): **yes, and it is 4.50x faster.**

    host argmax over read-back logits   23.2 ms/step
    DEVICE argmax, ids only              5.2 ms/step

BUT ONLY IN ROW_MAJOR. The first version of this reduced in TILE layout and measured **0.26x** --
four times SLOWER than shipping the logits to host -- which looked like a dead lever. It was not;
it was the wrong layout. Decomposing the cost (test_where_the_device_argmax_time_goes below):

    lm_head alone                  2.0 ms
    + TILE argmax                106.3 ms   the reduction itself ~104.3 ms
    + ROW_MAJOR argmax             4.3 ms   reduction + untilize ~2.3 ms

A 45x difference in the same op on the same data. ttnn's docs point at it in passing --
``sub_core_grids`` is documented as supported on ROW_MAJOR last-dim reductions -- so the tiled path
is simply not the tuned one at this width. Worth remembering generally: a slow ttnn reduction is
worth re-timing in the other layout before it is written off.

Under greedy decoding the host does nothing with the drafter's logits except ``argmax`` them, so
``TtTarget.draft_ids_device`` does that on the device and moves ``q_len - 1`` uint32s instead of a
``[1, 15, vocab]`` block. The readback is the largest phase of a drafter step (~33 ms of 75,
tests/perf/test_dflash_drafter_trace_breakdown.py), and this is the part of it that is pure waste.

The failure this is really guarding is silent. ``_lm_head`` returns a row PADDED past ``vocab_size``,
and the host path only trimmed after reading back. An argmax taken over the padded columns needs
only to find a padding value greater than the real logits -- which are negative more often than
not -- to return an index that is not a token at all. Nothing downstream would reject it: the target
verifies drafted slots and would simply reject them, so the symptom would be an acceptance rate that
quietly sags, not an error.

So this compares ids, and it also checks the argmax's chosen index lies inside the vocabulary.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_drafter_device_argmax.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
BLOCK = 16
KEEP = BLOCK - 1
ITERS = 10


def _mesh_shape():
    name = (os.environ.get("MESH_DEVICE") or "").upper()
    return {"P150": (1, 1), "N150": (1, 1), "N300": (1, 2), "T3K": (1, 8)}.get(name, (1, 8))


MESH_SHAPE = _mesh_shape()


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_device_argmax_matches_host(mesh_device, device_params, reset_seeds, ensure_gc):
    """Device argmax ids == host argmax ids, over several different hidden states."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    cfg = DFlashDrafterConfig.from_pretrained(resolve_drafter_path())
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)

    g = torch.Generator().manual_seed(19)

    def _hidden():
        t = (torch.randn(1, 1, BLOCK, model.args.dim, generator=g) * 0.05).to(torch.bfloat16)
        return ttnn.from_torch(
            t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if model.num_devices > 1 else {}),
        )

    for trial in range(3):
        h = _hidden()
        want = torch.argmax(target.lm_head_device(h, keep_rows=KEEP).float(), dim=-1)
        got = target.draft_ids_device(h, keep_rows=KEEP)
        assert got.shape == want.shape, f"trial {trial}: shape {tuple(want.shape)} -> {tuple(got.shape)}"
        assert got.min() >= 0 and got.max() < model.vocab_size, (
            f"trial {trial}: device argmax returned id {got.max().item()} outside the "
            f"{model.vocab_size}-token vocabulary -- it is reducing over the head's PADDED width"
        )
        assert torch.equal(got, want), (
            f"trial {trial}: device argmax picked different tokens than the host argmax\n"
            f"  device {got.tolist()}\n  host   {want.tolist()}"
        )
        logger.info(f"trial {trial}: device argmax matches host, ids in [0, {model.vocab_size})")

    # Price it against the readback it replaces.
    h = _hidden()
    for fn in (lambda: target.lm_head_device(h, keep_rows=KEEP), lambda: target.draft_ids_device(h, keep_rows=KEEP)):
        fn()
    ttnn.synchronize_device(mesh_device)

    def _time(fn):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
            ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) * 1000 / ITERS

    logits_ms = _time(lambda: torch.argmax(target.lm_head_device(h, keep_rows=KEEP).float(), dim=-1))
    ids_ms = _time(lambda: target.draft_ids_device(h, keep_rows=KEEP))
    logger.info("=" * 78)
    logger.info(f"  host argmax over read-back logits {logits_ms:7.1f} ms/step")
    logger.info(f"  DEVICE argmax, ids only           {ids_ms:7.1f} ms/step   {logits_ms / max(ids_ms, 1e-6):.2f}x")
    logger.info("=" * 78)
    print(
        f"\n>>> drafter head+readback: {logits_ms:.1f} -> {ids_ms:.1f} ms/step ({logits_ms / max(ids_ms, 1e-6):.2f}x)\n"
    )


@pytest.mark.timeout(0)
@torch.no_grad()
@pytest.mark.parametrize(
    "device_params",
    [{"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [MESH_SHAPE], indirect=True)
def test_where_the_device_argmax_time_goes(mesh_device, device_params, reset_seeds, ensure_gc):
    """The device argmax measured 0.26x. Is the REDUCTION slow, or the plumbing around it?

    Three things happen between the LM head and the ids, and only one of them is the argmax: a slice
    to trim the head's padded width, the reduction itself, and (in the variant below) a layout
    conversion. ttnn's own docs suggest the tiled path is not the tuned one -- ``sub_core_grids`` is
    documented as supported on ROW_MAJOR last-dim reductions -- so the question is whether a
    ROW_MAJOR argmax beats shipping the logits to host, even after paying to untilize a
    [15, vocab] block.

    If neither form beats ~27 ms, the lever is dead and the drafter keeps its host argmax.
    """
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)

    g = torch.Generator().manual_seed(23)
    t = (torch.randn(1, 1, BLOCK, model.args.dim, generator=g) * 0.05).to(torch.bfloat16)
    h = ttnn.from_torch(
        t,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if model.num_devices > 1 else {}),
    )
    vocab = model.vocab_size
    probe = model._lm_head(h)
    width = probe.shape[-1]
    logger.info(
        f"lm_head output width {width} vs vocab_size {vocab} -> trim {'NEEDED' if width > vocab else 'not needed'}"
    )
    ttnn.deallocate(probe)

    def _time(fn, iters=ITERS):
        fn()
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        for _ in range(iters):
            r = fn()
            ttnn.synchronize_device(mesh_device)
            if r is not None and not isinstance(r, torch.Tensor):
                ttnn.deallocate(r)
        return (time.perf_counter() - t0) * 1000 / iters

    head_ms = _time(lambda: model._lm_head(h))

    def _argmax_tile():
        lg = model._lm_head(h)
        keep = lg if width == vocab else ttnn.slice(lg, (0, 0, 0, 0), (1, 1, BLOCK, vocab))
        ids = ttnn.argmax(keep, dim=-1)
        if keep is not lg:
            ttnn.deallocate(keep)
        ttnn.deallocate(lg)
        return ids

    def _argmax_rm():
        lg = model._lm_head(h)
        keep = lg if width == vocab else ttnn.slice(lg, (0, 0, 0, 0), (1, 1, BLOCK, vocab))
        rm = ttnn.to_layout(keep, ttnn.ROW_MAJOR_LAYOUT)
        ids = ttnn.argmax(rm, dim=-1)
        ttnn.deallocate(rm)
        if keep is not lg:
            ttnn.deallocate(keep)
        ttnn.deallocate(lg)
        return ids

    tile_ms = _time(_argmax_tile)
    try:
        rm_ms = _time(_argmax_rm)
    except Exception as e:  # noqa: BLE001 -- an unsupported layout is a result
        rm_ms = float("nan")
        logger.info(f"ROW_MAJOR argmax unavailable: {type(e).__name__}: {str(e).splitlines()[0][:140]}")

    logger.info("=" * 78)
    logger.info(f"  lm_head alone                {head_ms:7.1f} ms")
    logger.info(f"  + TILE argmax (total)        {tile_ms:7.1f} ms   argmax itself ~{tile_ms - head_ms:.1f} ms")
    logger.info(f"  + ROW_MAJOR argmax (total)   {rm_ms:7.1f} ms   argmax+untilize ~{rm_ms - head_ms:.1f} ms")
    logger.info(f"  host readback baseline          26.8 ms  (measured in the test above)")
    logger.info("=" * 78)
    print(f"\n>>> head {head_ms:.1f} | tile argmax {tile_ms:.1f} | rm argmax {rm_ms:.1f} | host baseline 26.8 ms\n")
