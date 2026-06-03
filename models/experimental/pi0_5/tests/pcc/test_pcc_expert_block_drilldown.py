# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""
Block-level PCC drilldown for the Pi0 action-expert (Gemma 300M) transformer block.

The existing on-device PCC suite covers the expert *MLP* in isolation
(``test_pcc_gemma``) and a full *VLM* block (``test_pcc_paligemma``), but not a
full *expert* block. This drilldown fills that gap: it runs a single expert
``GemmaBlock`` (attention + gated residual + MLP + norms) through both the
PyTorch reference and the TTNN implementation and compares them, so a
full-model PCC regression can be localized to the expert stack rather than the
VLM stack.

It mirrors the proven ``test_pcc_paligemma_vlm_block`` drilldown (identical
construction, cos/sin handling and PCC threshold) but targets
``expert_blocks[0]``. Helpers are reused from ``test_pcc_paligemma`` to avoid
config/weight drift between the two block tests.

This exercises the non-adaRMS expert path (the default ``lerobot/pi0_base``
config). The adaRMS expert path is covered at the reference level by
``tests/unit/test_adarms.py`` and on-device by ``test_pcc_pi05_per_step.py``.

Usage:
    pytest models/experimental/pi0_5/tests/pcc/test_pcc_expert_block_drilldown.py -v
"""

import pytest
import torch
import ttnn

from models.experimental.pi0_5.tests.pcc.test_pcc_paligemma import (
    PaliGemmaBackboneTorch,
    PaliGemmaBackboneTTNN,
    compute_pcc,
    create_config,
    create_small_config,
    get_paligemma_weights,
)

SEED = 42
EXPERT_BLOCK_PCC_THRESHOLD = 0.85


@pytest.mark.parametrize("device_params", [{"l1_small_size": 24576}], indirect=True)
@pytest.mark.parametrize("use_pretrained", [True, False], ids=["pretrained", "random"])
def test_pcc_expert_block_drilldown(device, use_pretrained):
    """Single action-expert transformer block: TTNN vs PyTorch reference (PCC)."""
    torch.manual_seed(SEED)

    config = create_config() if use_pretrained else create_small_config()
    weights = get_paligemma_weights(use_pretrained, config)

    model_torch = PaliGemmaBackboneTorch(config, weights)
    model_ttnn = PaliGemmaBackboneTTNN(config, weights, device)

    seq_len = 64
    hidden = torch.randn(1, seq_len, config.expert_config.width)

    # PyTorch expert block[0]. Expert and VLM share head_dim/rope_base, so the
    # backbone's cos/sin table applies to the expert block unchanged.
    block_torch = model_torch.expert_blocks[0]
    cos_torch, sin_torch = model_torch.cos[:seq_len], model_torch.sin[:seq_len]
    out_torch, _ = block_torch.forward(hidden, cos_torch, sin_torch)

    # TTNN expert block[0]
    hidden_ttnn = ttnn.from_torch(hidden, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    cos_ttnn = ttnn.from_torch(cos_torch)
    sin_ttnn = ttnn.from_torch(sin_torch)
    block_ttnn = model_ttnn.expert_blocks[0]
    out_ttnn, _ = block_ttnn.forward(hidden_ttnn, cos_ttnn, sin_ttnn)

    if isinstance(out_ttnn, ttnn.Tensor):
        out_ttnn = ttnn.to_torch(out_ttnn)

    pcc = compute_pcc(out_torch, out_ttnn)
    weight_type = "pretrained" if use_pretrained else "random"
    print(f"\n✅ Expert Block[0] PCC ({weight_type}): {pcc:.6f}")
    assert pcc >= EXPERT_BLOCK_PCC_THRESHOLD, f"PCC {pcc:.6f} < threshold {EXPERT_BLOCK_PCC_THRESHOLD}"
