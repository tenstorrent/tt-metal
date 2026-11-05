# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import time

import pytest
import torch
from loguru import logger
from safetensors import safe_open

from models.demos.muse_glimmer_30b.reference.dflash2 import CandidateSelector, DFlash2Config
from models.demos.muse_glimmer_30b.tt.dflash.selector import HostCandidateSelector, merge_device_topk
from models.demos.muse_glimmer_30b.tt.model_config import dflash_checkpoint_dir

CKPT = os.path.join(dflash_checkpoint_dir(), "model.safetensors")
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="DFlash2 checkpoint not found")

VOCAB = 202_048
VOCAB_LOCAL = 50_688


def test_merge_device_topk():
    torch.manual_seed(0)
    full = torch.randn(32, 4 * VOCAB_LOCAL) * 5
    # Only a few padding columns are hot: if all 704 were, shard 3's top-32 would hold no valid ids at all.
    full[:, VOCAB : VOCAB + 8] = 100.0
    vals, ids = zip(*(torch.topk(full[:, d * VOCAB_LOCAL : (d + 1) * VOCAB_LOCAL], 32) for d in range(4)))
    scores, cands = merge_device_topk(
        torch.stack(vals), torch.stack(ids), vocab_local=VOCAB_LOCAL, vocab_size=VOCAB, k=16
    )
    exp_scores, exp_cands = torch.topk(full[:, :VOCAB], 16)
    assert scores.dtype == torch.float32 and cands.dtype == torch.int64
    for r in range(32):
        assert set(cands[r].tolist()) == set(exp_cands[r].tolist())
    assert torch.allclose(scores, exp_scores)


@pytest.fixture(scope="module")
def tensors():
    with safe_open(CKPT, framework="pt") as f:
        return {
            "pred": f.get_tensor("candidate_selector.predecessor_codebook"),
            "succ": f.get_tensor("candidate_selector.successor_codebook"),
            "hp": f.get_tensor("candidate_selector.hidden_projection.weight"),
        }


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_selector_matches_reference(tensors, seed):
    ref = CandidateSelector(DFlash2Config())
    ref.predecessor_codebook.weight.data = tensors["pred"].clone()
    ref.successor_codebook.weight.data = tensors["succ"].clone()
    ref.hidden_projection.weight.data = tensors["hp"].clone()
    ref = ref.float().eval()

    torch.manual_seed(seed)
    hidden = torch.randn(1, 15, 6656)
    logits = torch.randn(1, 15, VOCAB) * 5
    anchor = 1234
    with torch.no_grad():
        ref_path = ref.select(hidden, logits, torch.tensor([anchor]))[0][0].tolist()

    hp = hidden[0] @ tensors["hp"].float().T
    scores, cands = torch.topk(logits[0], 16)
    path = HostCandidateSelector(tensors["pred"], tensors["succ"]).select(scores, cands, hp, anchor)
    assert path == ref_path


def test_selector_speed(tensors):
    sel = HostCandidateSelector(tensors["pred"], tensors["succ"])
    torch.manual_seed(0)
    scores = torch.randn(15, 16)
    cands = torch.randint(0, VOCAB, (15, 16))
    hp = torch.randn(15, 256)
    for _ in range(5):
        sel.select(scores, cands, hp, 1234)
    t0 = time.perf_counter()
    for _ in range(50):
        sel.select(scores, cands, hp, 1234)
    mean_ms = (time.perf_counter() - t0) / 50 * 1000
    logger.info(f"HostCandidateSelector.select mean {mean_ms:.3f} ms")
    assert mean_ms < 20
