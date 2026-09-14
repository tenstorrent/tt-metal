# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""How much does the drafter's per-step logits readback cost, and what did narrowing it buy?

MEASURED 2026-09-14 (T3K, block 16, 10 iters): **97.9 -> 25.8 ms/step, 3.80x**, logits identical
(``torch.equal``). End to end that moved the traced loop from **18.82 to 21.30 tok/s** (53 -> 47
ms/tok, acceptance unchanged at 7.000, tokens identical) -- past production traced decode at 17.87
tok/s, from 1.05x to 1.19x of it.

The end-to-end gain is smaller than the isolated one (about 42 ms/step against 72 ms measured here)
and that discount is real, not noise: this A/B puts a ``synchronize_device`` around every call, which
serializes a transfer the real loop partly overlaps with device work. Treat the isolated figure as
the size of the transfer, and the end-to-end one as what the loop actually keeps.

Note what this was NOT: it is not gated on ``ctx_capacity`` or on tracing, so the shipped drafter
gets it as-is. It also corrected the drafter's cost accounting -- the 120 ms/step in
test_dflash_drafter_wall_time.py covers ``project_taps + forward`` only, and this readback was a
further ~98 ms on top, so the drafter was ~59 % of a speculative step rather than the ~32 % that
figure suggested on its own.

Every speculative step ends by projecting the drafter's hidden through the target's resident LM head
and bringing the logits back to host to argmax them. That readback is the drafter's only large
device->host transfer, and it was written the naive way:

    ttnn.to_torch(logits, mesh_composer=ConcatMeshToTensor(device, dim=0))[0]

which gathers the logits from ALL 8 devices and then keeps one of them. The LM head all-gathers its
vocab shards, so every device already holds the identical full-width row -- 7/8 of that transfer is
moving bytes in order to discard them. The same mistake, in the same shape, cost 486 ms per step in
the traced verify path before ``Qwen36Model._read_verify_logits`` narrowed it to ~10 ms.

This times the two forms against each other and checks they agree. The OLD form is inlined here
deliberately, as the baseline -- it is no longer anywhere in the product, so there is nothing to
drift away from. The NEW form calls the shipping method, which is the rule that matters (an earlier
perf test inlined a copy of shipping code and kept reporting a cost the product had stopped paying).

Not asserted on time -- host transfer rates are machine-dependent. It asserts EQUALITY, because an
8x cheaper readback that returns different logits is not an optimization.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/perf/test_dflash_lm_head_readback.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
Q_LEN = 16
KEEP_ROWS = Q_LEN - 1  # slot 0 is the confirmed anchor; only the drafted slots are read
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
def test_lm_head_readback_narrowing(mesh_device, device_params, reset_seeds, ensure_gc):
    """Naive whole-mesh readback vs the shipping narrowed one: same logits, less wall."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)
    target = TtTarget(model, [1, 16, 31, 46, 61], page_table, device_taps=True)

    g = torch.Generator().manual_seed(7)
    hidden_t = (torch.randn(1, 1, Q_LEN, model.args.dim, generator=g) * 0.05).to(torch.bfloat16)

    def _hidden():
        return ttnn.from_torch(
            hidden_t,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            **(dict(mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device)) if model.num_devices > 1 else {}),
        )

    def _old_form(keep_rows):
        """The readback as it was written before narrowing. Baseline only -- not shipping code."""
        logits = model._lm_head(_hidden())
        if model.num_devices > 1:
            host = ttnn.to_torch(logits, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[0]
        else:
            host = ttnn.to_torch(logits)
        ttnn.deallocate(logits)
        host = host.reshape(-1, host.shape[-1])[:, : model.vocab_size].float()
        return (host if keep_rows is None else host[-keep_rows:]).unsqueeze(0)

    # Warm both paths: the first call through either compiles programs, which is not the transfer.
    _old_form(KEEP_ROWS)
    target.lm_head_device(_hidden(), keep_rows=KEEP_ROWS)
    ttnn.synchronize_device(mesh_device)

    def _time(fn):
        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = fn()
            ttnn.synchronize_device(mesh_device)
        return (time.perf_counter() - t0) * 1000 / ITERS, out

    old_ms, old_out = _time(lambda: _old_form(KEEP_ROWS))
    new_ms, new_out = _time(lambda: target.lm_head_device(_hidden(), keep_rows=KEEP_ROWS))

    logger.info("=" * 78)
    logger.info(f"  whole-mesh readback (old) {old_ms:8.1f} ms/step")
    logger.info(f"  narrowed readback  (new)  {new_ms:8.1f} ms/step   {old_ms / max(new_ms, 1e-6):.2f}x")
    logger.info("=" * 78)
    print(f"\n>>> lm_head readback: {old_ms:.1f} -> {new_ms:.1f} ms/step ({old_ms / max(new_ms, 1e-6):.2f}x)\n")

    assert new_out.shape == old_out.shape, f"shape changed: {tuple(old_out.shape)} -> {tuple(new_out.shape)}"
    assert torch.equal(new_out, old_out), (
        "the narrowed readback returned different logits than the whole-mesh one; the LM head's "
        "vocab all-gather is supposed to make every device's copy identical, so this means it is not"
    )
