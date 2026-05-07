# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import ttnn

from models.experimental.voxtraltts.tests.common import create_real_voxtral_text_model_or_skip


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_inference(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    assert model.inner.vocab_size > 0
    assert model.inner.args.n_layers > 0


@torch.no_grad()
@pytest.mark.timeout(3600)
def test_text_model_prefill_inference(device, reset_seeds):
    model = create_real_voxtral_text_model_or_skip(device, max_seq_len=256, dtype=ttnn.bfloat8_b)

    seq_len = 128
    tokens = torch.randint(0, model.inner.vocab_size, (1, seq_len), dtype=torch.int64)
    tt_x, rot_mats_global, rot_mats_local, _, _ = model.prepare_inputs_prefill(tokens, start_pos=0)
    tt_logits = model.inner.ttnn_prefill_forward(
        tt_x,
        rot_mats_global=rot_mats_global,
        rot_mats_local=rot_mats_local,
        # get_last_token is tile start (multiple of 32), not absolute token index.
        get_last_token=((seq_len - 1) // 32) * 32,
    )
    logits = model.inner.process_output_prefill(
        tt_logits.cpu(),
        last_token_idx=((seq_len - 1) % 32),
    ).float()

    assert list(logits.shape) == [model.inner.vocab_size]
    assert torch.isfinite(logits).all()
