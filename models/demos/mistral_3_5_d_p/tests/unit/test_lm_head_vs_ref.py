# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""LM-head projection vs a torch reference, with the vocab shard layout the model uses.
Pattern: ``gemma4/tests/unit/test_lm_head.py``.

The head is column-parallel over the VOCAB dim: device ``c`` produces logits
``[c*per_device_vocab, (c+1)*per_device_vocab)``, and the host concatenates them (there is no
collective — the shards are disjoint outputs, not partial sums). Getting the shard order wrong
produces logits that are a permutation of the right ones, which a PCC over the flattened tensor
would NOT catch, so this test reassembles by column index and also checks the argmax token id.

Mistral needs no vocab padding — 131072 / 8 = 16384 is tile-aligned and already a power of two — and
the test asserts that, because the padding path exists for other TP factors and a stray pad column
would shift every logit index above it.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.mistral_3_5_d_p.reference.mistral_config import MistralMedium35Config as C
from models.demos.mistral_3_5_d_p.spec import SPEC
from models.demos.mistral_3_5_d_p.tt.model import compute_per_device_vocab
from models.demos.mistral_3_5_d_p.utils.general_utils import get_matmul_compute_config

from ..test_factory import build_mesh_and_ccl, parametrize_mesh, to_device


def test_no_vocab_padding_at_the_spec_tp():
    """131072 over TP=8 lands on 16384 per device: tile-aligned and a power of two already."""
    per_device = compute_per_device_vocab(C.VOCAB_SIZE, SPEC.tp)
    assert per_device == C.VOCAB_SIZE // SPEC.tp == 16384
    assert per_device * SPEC.tp == C.VOCAB_SIZE, "no padding columns should be added at this TP"
    assert per_device % ttnn.TILE_SIZE == 0 and per_device & (per_device - 1) == 0


@parametrize_mesh()
@pytest.mark.parametrize("tokens", [32, 128], ids=["t32", "t128"])
@pytest.mark.parametrize("vocab", [4096], ids=["v4096"])
def test_lm_head_vs_ref(mesh_device, device_params, tokens, vocab, reset_seeds):
    """The column-parallel head vs a torch matmul, reassembled in shard order.

    ``vocab`` is reduced from 131072 so the host can hold the reference weight
    (``[hidden, vocab]`` at the real hidden of 12288); the sharding, the dtype and the reassembly
    are identical at the real vocab, which ``test_model_sp_vs_ref.py`` exercises end to end.
    """
    _rows, cols = tuple(mesh_device.shape)
    hidden = C.HIDDEN_SIZE
    per_device = vocab // cols
    assert vocab % cols == 0, f"vocab {vocab} must split across tp={cols}"

    x = torch.randn(1, 1, tokens, hidden) * 0.1
    weight = torch.randn(hidden, vocab) * 0.02  # [hidden, vocab], as the model stores it

    ref = (x.float().reshape(tokens, hidden) @ weight.float()).reshape(1, 1, tokens, vocab)

    mesh_config, _ = build_mesh_and_ccl(mesh_device)
    lm_head = ttnn.as_tensor(
        weight.unsqueeze(0).unsqueeze(0),
        device=mesh_device,
        layout=ttnn.TILE_LAYOUT,
        dtype=ttnn.bfloat8_b,  # the dtype the model uses for the head
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mesh_config.column_parallel(mesh_device),
    )
    logits = ttnn.matmul(
        to_device(x, mesh_device),
        lm_head,
        dtype=ttnn.bfloat8_b,
        compute_kernel_config=get_matmul_compute_config(mesh_device),
    )
    assert logits.shape[-1] == per_device, f"each device must hold {per_device} vocab columns, got {logits.shape[-1]}"

    # Reassemble by SHARD ORDER: device c holds columns [c*per_device, (c+1)*per_device).
    shards = ttnn.get_device_tensors(logits)
    full = torch.cat([ttnn.to_torch(shards[c]).float().reshape(1, 1, tokens, per_device) for c in range(cols)], dim=-1)

    ok, pcc = comp_pcc(ref, full, SPEC.pcc)
    logger.info(f"lm_head tokens={tokens} vocab={vocab} tp={cols}: pcc={pcc}")
    assert ok, f"lm_head PCC fail: {pcc}"

    # A shard-order mistake permutes the vocab axis, which PCC over the whole tensor tolerates.
    # Compare the argmax token id per position instead — that is what a caller actually consumes.
    ref_argmax = ref[0, 0].argmax(dim=-1)
    got_argmax = full[0, 0].argmax(dim=-1)
    agree = (ref_argmax == got_argmax).float().mean().item()
    logger.info(f"lm_head argmax agreement: {agree:.4f}")
    assert agree > 0.95, f"top-1 token disagrees on {100 * (1 - agree):.1f}% of positions (shard order?)"
