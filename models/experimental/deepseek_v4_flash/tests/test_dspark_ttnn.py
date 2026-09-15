# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0
"""PCC test for the ttnn DSpark drafter against the PyTorch reference.

Uses :meth:`DSparkConfig.ttnn_tiny` so every projection is tile-aligned for
``matmul_decode`` + the DRISC prefetcher. Skips when programmable DRAM cores
are unavailable. Batch=1, context length = block size = 32 (one tile).
"""

from __future__ import annotations

import contextlib
import time

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


@torch.no_grad()
def test_dspark_ttnn_traced_socket_draft_matches_eager(device):
    """The traced socket path returns the same greedy draft block as eager DSpark."""
    torch.manual_seed(1)
    cfg = DSparkConfig.ttnn_tiny()
    pt_model = DSparkModel(cfg).eval()
    tt_model = TtDSparkModel.from_torch(pt_model, device, num_prefetch_pages=2)

    target_hiddens = torch.randn(1, 1, cfg.num_target_layers, cfg.hidden_size)
    anchor = torch.randint(0, cfg.vocab_size - 1, (1,))
    target_hiddens_2 = torch.randn_like(target_hiddens)
    anchor_2 = torch.randint(0, cfg.vocab_size - 1, (1,))

    with contextlib.ExitStack() as stack:
        stack.enter_context(tensor_prefetcher_session(device))
        # Registered after the session so it unwinds first (LIFO). The traced path's daemon
        # replay thread closes over the model, so without this the model and its tensors
        # stay reachable until interpreter shutdown and nanobind reports them as leaked.
        stack.callback(tt_model.shutdown)
        ttnn.experimental.wait_for_cq_on_tensor_prefetcher(device, cq_id=0)
        eager = tt_model(target_hiddens, anchor, greedy=True, hoist_prefetch=True)
        eager_2 = tt_model(target_hiddens_2, anchor_2, greedy=True, hoist_prefetch=True)
        steps = [(target_hiddens, int(anchor.item())), (target_hiddens_2, int(anchor_2.item()))]
        tt_model.replay_traced_ahead(steps)
        tt_model.write_step_packet(*steps[0])
        tt_model.write_step_packet(*steps[1])
        draft = tt_model.read_decoded_output()
        draft_2 = tt_model.read_decoded_output()

        bench_steps = steps * 2
        eager_start = time.perf_counter()
        for target, anchor_id in bench_steps:
            tt_model(target, torch.tensor([anchor_id]), greedy=True, hoist_prefetch=True)
        eager_elapsed = time.perf_counter() - eager_start

        traced_start = time.perf_counter()
        for target, anchor_id in bench_steps:
            tt_model.replay_traced()
            tt_model.write_step_packet(target, anchor_id)
            tt_model.read_decoded_output()
        traced_elapsed = time.perf_counter() - traced_start
        gamma = cfg.dspark_block_size
        print(
            f"DSPARK_TTNN_EAGER={len(bench_steps) * gamma / eager_elapsed:.2f} "
            f"DSPARK_TTNN_TRACED={len(bench_steps) * gamma / traced_elapsed:.2f} "
            f"speedup={eager_elapsed / traced_elapsed:.2f}x",
            flush=True,
        )

    assert torch.equal(draft, eager.draft_ids)
    assert torch.equal(draft_2, eager_2.draft_ids)
