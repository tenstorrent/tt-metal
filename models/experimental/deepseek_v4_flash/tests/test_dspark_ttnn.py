# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the ttnn DSpark drafter against the PyTorch reference.

Uses :meth:`DSparkConfig.ttnn_tiny` so every projection is tile-aligned for
``matmul_decode`` + the DRISC prefetcher. Skips when programmable DRAM cores
are unavailable. Batch=1, context length = block size = 32 (one tile).
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.experimental.deepseek_v4_flash.dspark import DSparkConfig, DSparkModel
from models.experimental.deepseek_v4_flash.tt.dspark import DSparkModel as TtDSparkModel
from tests.ttnn.unit_tests.operations.prefetcher_common import tensor_prefetcher_session
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.fixture(autouse=True)
def _require_tensor_prefetcher(device):
    """Skip unless programmable DRAM cores are available on this device."""
    if not ttnn.experimental.is_tensor_prefetcher_supported(device):
        pytest.skip(
            "programmable DRAM cores unavailable (need Blackhole, firmware >= 19.12.0.0, "
            "and either no harvested DRAM channels or a single device)"
        )


@torch.no_grad()
def test_dspark_ttnn_prefetcher_pcc(device):
    """ttnn DSpark (prefetched LinearDecode) matches the PyTorch drafter at PCC 0.99."""
    torch.manual_seed(0)
    cfg = DSparkConfig.ttnn_tiny()
    pt_model = DSparkModel(cfg).eval()
    tt_model = TtDSparkModel.from_torch(pt_model, device, num_prefetch_pages=2)

    batch, ctx = 1, 32
    target_hiddens = torch.randn(batch, ctx, cfg.num_target_layers, cfg.hidden_size)
    anchor_ids = torch.randint(0, cfg.vocab_size - 1, (batch,))

    ref = pt_model(target_hiddens, anchor_ids, greedy=True)

    with tensor_prefetcher_session(device):
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        out = tt_model(target_hiddens, anchor_ids, greedy=True, hoist_prefetch=True)

    assert_with_pcc(ref.context, out.context, 0.99)
    assert_with_pcc(ref.hidden_states, out.hidden_states, 0.99)
    assert_with_pcc(ref.base_logits, out.base_logits, 0.99)
    assert_with_pcc(ref.logits, out.logits, 0.99)
    assert_with_pcc(ref.confidence, out.confidence, 0.99)
    assert torch.equal(out.block_input_ids, ref.block_input_ids)
