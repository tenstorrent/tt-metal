# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""M8: the whole drafter vs ``DFlashDraftModel.forward`` -- the Milestone 1 gate.

All 58 real weights, and the captured fixture's real inputs: the Qwen3.6-27B residual stream
at layers [1, 16, 31, 46, 61] and the target's real embedding of ``[anchor, MASK x15]``.

Two things are graded, and the difference matters:

* **final-hidden PCC** -- the strict numeric gate.
* **candidate agreement** -- reported, and floored well below 15/15 rather than asserted
  exact. Muse-Glimmer work_log F2 records why: DFlash is not output-lossless in bf16, and
  near-tied logits argmax differently under different reduction orders. That port measured
  0.900 candidate agreement between its TTNN drafter and the genuine HF drafter *while being
  correct*, and swapping the real drafter in changed accepted-tokens-per-block only
  2.50 -> 2.57. Asserting exact token equality here would be a flaky gate on a correct port
  -- F25 records that mistake costing a day.

Run:
    MESH_DEVICE=T3K pytest models/demos/blackhole/qwen36/tests/dflash/test_drafter_pcc.py -v -s
"""

from __future__ import annotations

import os

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.blackhole.qwen36.tests.dflash.conftest import FIXTURE_CTX_LENS, load_fixture
from models.demos.blackhole.qwen36.tests.test_factory import get_pcc_threshold, parametrize_mesh_tp
from models.demos.blackhole.qwen36.tt.dflash.drafter import DFlashDrafter
from models.demos.blackhole.qwen36.tt.dflash.weights import permute_fc_input_activation

#: Floor, not a target. See the module docstring for why this is not 15/15.
MIN_CANDIDATE_AGREEMENT = 13


def _target_lm_head() -> torch.Tensor | None:
    """The target's ``lm_head.weight``, read as a single tensor from its sharded checkpoint.

    The drafter ships no LM head -- it borrows the target's -- so turning hidden states into
    candidates needs this. Returns ``None`` if the target checkpoint is not available, so the
    PCC gate still runs without it.
    """
    import json

    from safetensors.torch import safe_open

    try:
        from models.demos.blackhole.qwen36.tests.dflash.capture_fixtures import _resolve

        d = _resolve(os.environ.get("HF_MODEL", "Qwen/Qwen3.6-27B"))
        index = json.load(open(os.path.join(d, "model.safetensors.index.json")))
        shard = index["weight_map"]["lm_head.weight"]
        with safe_open(os.path.join(d, shard), framework="pt") as f:
            return f.get_tensor("lm_head.weight")
    except Exception as exc:
        logger.warning(f"target lm_head unavailable ({type(exc).__name__}: {exc}); skipping candidate check")
        return None


@torch.no_grad()
@parametrize_mesh_tp()
@pytest.mark.parametrize("ctx_len", FIXTURE_CTX_LENS)
def test_drafter_pcc(mesh_device, ctx_len, reset_seeds, ensure_gc, request, drafter_cfg):
    cfg = drafter_cfg
    tp = mesh_device.get_num_devices()
    fx = load_fixture(ctx_len)
    block = fx["block_size"]

    target_hidden = fx["target_hidden"].float()
    noise = fx["noise_embedding"].float()
    expected = fx["reference_hidden"]  # [1, block, 5120], reference fp32

    drafter = DFlashDrafter(mesh_device, cfg, max_position=ctx_len + block)
    # Pre-warm so the timed path never builds a mask on host.
    drafter.prewarm_masks([ctx_len], block)

    # Permute into per-chip tap order, then shard. On device the taps already arrive in this
    # order; this only undoes the fixture's host-side concatenation.
    permuted = permute_fc_input_activation(target_hidden, tp, cfg.hidden_size, len(cfg.target_layer_ids))
    tt_ctx = ttnn.from_torch(
        permuted.reshape(1, 1, ctx_len, cfg.target_feature_size).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=-1),
    )
    tt_noise = ttnn.from_torch(
        noise.reshape(1, 1, block, cfg.hidden_size).to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    tt_out = drafter.forward(tt_ctx, tt_noise)

    stacked = ttnn.to_torch(tt_out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))
    actual = stacked[:1].reshape(1, block, cfg.hidden_size).float()
    spread = max((stacked[d : d + 1].float() - stacked[:1].float()).abs().max().item() for d in range(1, tp))
    logger.info(f"ctx {ctx_len}: replication spread {spread:.3e}")

    passing, pcc = comp_pcc(expected, actual, get_pcc_threshold(request))
    logger.info(f"ctx {ctx_len}: drafter final-hidden PCC {pcc}")

    # Candidate agreement, using the TARGET's LM head over the last block-1 rows.
    lm_head = _target_lm_head()
    if lm_head is not None:
        draft_hidden = actual[:, 1 - block :, :]
        cand = torch.nn.functional.linear(draft_hidden, lm_head.float()).argmax(-1)[0]
        ref_cand = fx["reference_candidates"]
        agree = int((cand == ref_cand).sum())
        leading = int((cand == ref_cand).to(torch.int32).cumprod(0).sum())
        logger.info(
            f"ctx {ctx_len}: candidates {agree}/{cfg.num_draft_tokens} match reference "
            f"({leading} leading); golden accepted {fx['golden_acceptance']}"
        )
        assert agree >= MIN_CANDIDATE_AGREEMENT, (
            f"only {agree}/{cfg.num_draft_tokens} candidates match the torch reference "
            f"(floor {MIN_CANDIDATE_AGREEMENT})"
        )

    assert passing, f"ctx {ctx_len} drafter PCC {pcc}"
