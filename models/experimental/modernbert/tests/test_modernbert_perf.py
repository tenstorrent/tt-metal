# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""End-to-end inference timing for ModernBERT."""

import time

import pytest
import torch
from loguru import logger

import ttnn
from models.experimental.modernbert.common import build_inputs, load_config, load_torch_model
from models.experimental.modernbert.reference.modernbert import ModernBertModel
from models.experimental.modernbert.tt.modernbert_model import TtnnModernBertModel
from models.experimental.modernbert.tt.weights import deallocate_weights, prepare_weights

WARMUP_ITERS = 3
TIMED_ITERS = 20


def _build(device, seq_len, batch_size):
    config = load_config()
    hf = load_torch_model()
    ref = ModernBertModel(config)
    ref.load_state_dict(hf.state_dict(), strict=True)
    ref.eval()

    ids, attention_mask = build_inputs(seq_len=seq_len, batch_size=batch_size)
    params = prepare_weights(ref, device)
    model = TtnnModernBertModel(params, config, device, seq_len, attention_mask=attention_mask)
    tt_ids = ttnn.from_torch(ids.to(torch.int32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT, device=device)
    return model, params, tt_ids


@pytest.mark.models_performance_bare_metal
@pytest.mark.parametrize("seq_len, batch_size", [(256, 1), (512, 1), (256, 4), (256, 8)])
def test_modernbert_inference_time(device, seq_len, batch_size):
    # Released in a finally block so a failing case cannot leak ~300 MB of
    # weights, masks and rotary caches into whatever runs next in this process.
    #
    # It is belt-and-braces for cross-case contamination: tt-metal's `device`
    # fixture is function-scoped unless a test opts in with
    # @pytest.mark.use_module_device, so each parametrised case here already
    # gets its own CreateDevice/close_device pair and starts from a clean
    # allocator. That is why these four cases reproduce the README table inside
    # a full-suite run to within 0.03 ms. Keep the release anyway - the guarantee
    # is a property of the fixture, not of this file.
    model, params, tt_ids = _build(device, seq_len, batch_size)
    try:
        for _ in range(WARMUP_ITERS):
            out = model(tt_ids)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)

        t0 = time.time()
        for _ in range(TIMED_ITERS):
            out = model(tt_ids)
            ttnn.deallocate(out)
        ttnn.synchronize_device(device)
        elapsed = time.time() - t0
    finally:
        model.deallocate()
        deallocate_weights(params)
        ttnn.deallocate(tt_ids)

    per_iter = elapsed / TIMED_ITERS
    seqs_per_sec = batch_size / per_iter
    tokens_per_sec = batch_size * seq_len / per_iter

    logger.info(
        f"ModernBERT batch={batch_size} seq={seq_len}: "
        f"{per_iter * 1000:.2f} ms/inference, "
        f"{seqs_per_sec:.1f} sequences/s, "
        f"{tokens_per_sec:,.0f} tokens/s"
    )
    assert per_iter > 0
