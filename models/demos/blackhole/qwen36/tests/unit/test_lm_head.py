# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Component PCC: LM head logits (TTNN vs torch), bfloat8_b vs bfloat16 weights.

``device`` and ``setup`` come from tests/unit/conftest.py.
"""
import pytest
import torch
import torch.nn.functional as F
from loguru import logger

import ttnn
from models.common.utility_functions import run_for_blackhole
from models.demos.blackhole.qwen36.tests.test_factory import compute_pcc, get_pcc_threshold

from .conftest import DEVICE_PARAMS

pytestmark = [run_for_blackhole(), pytest.mark.parametrize("device_params", DEVICE_PARAMS, indirect=True)]


def test_lm_head_precision(device, setup, request):
    """LM head logits match the torch reference; also reports bf8/bf16 top-10 overlap."""
    args, sd, raw = setup
    lm_w = sd["output.weight"]  # [vocab_size, hidden_size]

    x_cpu = torch.randn(1, 1, 4096, dtype=torch.bfloat16)

    # Torch reference
    ref = F.linear(x_cpu, lm_w.to(torch.bfloat16))  # [1, 1, vocab_size]

    # TTNN with bfloat8_b (current)
    lm_w_t = lm_w.T.contiguous()  # [4096, vocab_size]
    lm_ttnn_bf8 = ttnn.from_torch(lm_w_t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device)
    x_ttnn = ttnn.from_torch(x_cpu, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_bf8 = ttnn.to_torch(ttnn.linear(x_ttnn, lm_ttnn_bf8))

    pcc_bf8 = compute_pcc(ref, out_bf8)
    logger.info(f"LM Head PCC (bfloat8_b): {pcc_bf8:.6f}")

    # TTNN with bfloat16 (higher precision)
    lm_ttnn_bf16 = ttnn.from_torch(lm_w_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out_bf16 = ttnn.to_torch(ttnn.linear(x_ttnn, lm_ttnn_bf16))

    pcc_bf16 = compute_pcc(ref, out_bf16)
    logger.info(f"LM Head PCC (bfloat16): {pcc_bf16:.6f}")

    # Check top-k agreement
    ref_top10 = ref.squeeze().topk(10).indices.tolist()
    bf8_top10 = out_bf8.squeeze().topk(10).indices.tolist()
    bf16_top10 = out_bf16.squeeze().topk(10).indices.tolist()
    logger.info(f"Top-10 overlap (bf8): {len(set(ref_top10) & set(bf8_top10))}/10")
    logger.info(f"Top-10 overlap (bf16): {len(set(ref_top10) & set(bf16_top10))}/10")

    assert pcc_bf16 > get_pcc_threshold(request), f"LM head bf16 PCC too low: {pcc_bf16}"
