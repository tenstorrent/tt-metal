# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Vision pooler (3x3 position pooling x sqrt(768), padded outputs stripped) and embed_vision
(scale-free RMSNorm + Linear 768 -> 512) vs HF, each fed the HF input of that module.

The pooler check also asserts the host-built pooling matrix reproduces HF's kept-row mask
(soft-token count per image: 280 x valid / 2520 for images, 140 x valid / 1260 per video frame).
"""

import pytest
import torch

from models.demos.blackhole.embedding_gemma2.tests.common import PCC_LAYER, log_result, pcc
from models.demos.blackhole.embedding_gemma2.tests.pcc.vision.vision_common import ALL_CASES, upload_rows
from models.demos.blackhole.embedding_gemma2.tt.vision.vision_encoder import VisionEncoder


@pytest.mark.parametrize("case", ALL_CASES)
def test_vision_pooler(case, eg2_device, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    enc = tt_vision_encoder
    host = enc.prepare_inputs(pv, pid)
    x = enc.to_device(host)
    hf_counts = ref.pooled_mask.sum(-1).tolist()
    assert host.num_soft_tokens == hf_counts, f"soft-token counts {host.num_soft_tokens} != HF {hf_counts}"
    for b, n in enumerate(hf_counts):
        assert bool(ref.pooled_mask[b, :n].all()), "HF kept rows are not a prefix"
    pooled = enc.model.pooler(upload_rows(eg2_device, ref.encoder_out, host.seq_len), x.pool)
    rows = VisionEncoder.read_rows(pooled, host.num_soft_tokens)
    ok = True
    for b, n in enumerate(hf_counts):
        value = pcc(ref.pooled[b, :n], rows[b])
        ok &= log_result("vision_pooler_pcc", value, PCC_LAYER, case=case, image=b, soft_tokens=n)
    assert ok


@pytest.mark.parametrize("case", ALL_CASES)
def test_embed_vision(case, eg2_device, tt_vision_encoder, vision_reference_for):
    pv, pid, ref = vision_reference_for(case)
    enc = tt_vision_encoder
    host = enc.prepare_inputs(pv, pid)
    S_out = enc.config.pooled_rows(host.seq_len)
    pooled = torch.zeros(host.batch, S_out, enc.config.hidden_size)
    for b, n in enumerate(host.num_soft_tokens):
        pooled[b, :n] = ref.pooled[b, :n]
    import ttnn

    soft = enc.model.embedder(upload_rows(eg2_device, pooled, S_out, dtype=ttnn.float32))
    rows = VisionEncoder.read_rows(soft, host.num_soft_tokens)
    ok = True
    for b, n in enumerate(host.num_soft_tokens):
        value = pcc(ref.soft_tokens[b], rows[b])
        ok &= log_result("vision_embed_pcc", value, PCC_LAYER, case=case, image=b)
    assert ok
