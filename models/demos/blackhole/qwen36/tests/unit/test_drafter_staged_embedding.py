# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stage 2 gate, target side: the staged noise embedding must equal the uploaded one.

Of everything a drafter step does, this is the piece that makes a trace capture outright ILLEGAL
rather than merely wrong. ``TtTarget.embed_device`` builds the block's token tensor with
``ttnn.from_torch`` on every step -- a host->device upload, which a capture rejects. Rope offsets
and mask shapes only BAKE (recorded once, then silently reused); this one cannot be recorded at all.

``TtDFlashDrafter.stage_tokens`` DMAs the ids into a persistent buffer outside the capture, and
``TtTarget.embed_device_staged`` runs the embedding and its vocab all-gather from that buffer, which
is ordinary device work a capture can record. The values must not change: same ids, same embedding,
same gather.

Both paths are exercised at a SHORT id run as well as a full block, because the staged buffer is
fixed-width and a short block leaves trailing slots holding whatever the previous step wrote. The
drafter always fills every slot (unwritten ones carry the mask token), so those trailing rows are
real inputs rather than padding -- this checks the staging reproduces them rather than reusing
stale ones.

Run::

    DFLASH_RUN_TARGET=1 MESH_DEVICE=T3K HF_MODEL=Qwen/Qwen3.6-27B \\
      DFLASH_HF_MODEL=z-lab/Qwen3.6-27B-DFlash \\
      TT_CACHE_PATH=$HOME/.cache/tt_cache/Qwen3.6-27B \\
      pytest -svq models/demos/blackhole/qwen36/tests/unit/test_drafter_staged_embedding.py
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.demos.blackhole.qwen36.reference.dflash.loader import DFlashDrafterConfig, resolve_drafter_path
from models.demos.blackhole.qwen36.reference.dflash.targets import TtTarget
from models.demos.blackhole.qwen36.tt.dflash.config import load_drafter_state_dict
from models.demos.blackhole.qwen36.tt.dflash.drafter import TtDFlashDrafter
from models.demos.blackhole.qwen36.tt.model import Qwen36Model

PAGED_BLOCK_SIZE = 64
NUM_BLOCKS = 64
BLOCK = 16


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
def test_staged_embedding_matches_upload(mesh_device, device_params, reset_seeds, ensure_gc):
    """stage_tokens + embed_device_staged == embed_device, for a full block and a short one."""
    del device_params
    if os.environ.get("DFLASH_RUN_TARGET") != "1":
        pytest.skip("set DFLASH_RUN_TARGET=1 to run the full 27B")

    model = Qwen36Model.from_pretrained(mesh_device, max_batch_size=1, max_seq_len=NUM_BLOCKS * PAGED_BLOCK_SIZE)
    kv_shape = [NUM_BLOCKS, model.args.n_local_kv_heads, PAGED_BLOCK_SIZE, model.args.head_dim]
    model.allocate_kv_caches(kv_shape, ttnn.bfloat16, batch_size=1)
    page_table = torch.arange(NUM_BLOCKS, dtype=torch.int32).unsqueeze(0)

    cfg = DFlashDrafterConfig.from_pretrained(resolve_drafter_path())
    target = TtTarget(model, cfg.target_layer_ids, page_table, device_taps=True)
    drafter = TtDFlashDrafter(mesh_device, cfg, load_drafter_state_dict(resolve_drafter_path()), ctx_capacity=256)
    drafter.alloc_step_buffers(q_len=BLOCK, ctx_pad=16)

    def _download(t):
        if model.num_devices > 1:
            return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:1].float()
        return ttnn.to_torch(t).float()

    g = torch.Generator().manual_seed(13)
    # A full block first, then a SHORT one -- the short case must not pick up the first block's
    # trailing ids out of the fixed-width buffer.
    for label, n in (("full block", BLOCK), ("short block", 5)):
        ids = torch.randint(1000, 2000, (1, n), generator=g, dtype=torch.long)
        padded = torch.zeros(1, BLOCK, dtype=torch.long)
        padded[:, :n] = ids

        want = _download(target.embed_device(padded))
        tok = drafter.stage_tokens(padded)
        got = _download(target.embed_device_staged(tok))

        assert want.shape == got.shape, f"{label}: shape {tuple(want.shape)} -> {tuple(got.shape)}"
        same = torch.equal(want, got)
        logger.info(f"{label} ({n} real ids): staged embedding {'EQUAL' if same else 'DIFFERS'}")
        assert same, (
            f"{label}: the staged embedding differs from the uploaded one. Same ids and same "
            "embedding table, so suspect stage_tokens leaving the buffer's tail from a previous "
            "step, or the all-gather being handed a different layout"
        )
