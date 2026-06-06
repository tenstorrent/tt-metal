# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""TP4 model head (final norm + column-parallel LM head + argmax) vs torch."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, from_replicated_to_torch, to_replicated
from models.experimental.dots_ocr_tp4.tt.lm_head import DotsOCRLMHeadTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape
from models.experimental.dots_ocr_tp4.tests.torch_reference import TorchLMHead


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [2816])
@pytest.mark.parametrize("vocab_size", [151936])
def test_dots_ocr_lm_head_tp4(mesh_device, seq_len, vocab_size):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    config = DotsOCRConfig()
    H = config.hidden_size

    torch_head = TorchLMHead(config, vocab_size=vocab_size).eval()  # float32 reference
    hidden = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    torch_logits = torch_head(hidden.to(torch.float32), last_token_only=True)  # [1, 1, vocab]
    torch_token = int(torch_logits[0, -1].argmax())

    tt_head = DotsOCRLMHeadTP4.from_torch(mesh_device, config, torch_head.norm, torch_head.lm_head)
    hidden_tt = to_replicated(hidden, mesh_device, dtype=ttnn.bfloat16)

    logits_tt, token_ids = tt_head.forward(hidden_tt, last_token_only=True, return_token=True)
    ttnn.synchronize_device(mesh_device)

    logits_torch = from_replicated_to_torch(logits_tt, mesh_device).to(torch.float32).reshape(torch_logits.shape)

    passed, msg = assert_with_pcc(torch_logits.to(torch.float32), logits_torch, pcc=0.99)
    tt_token = int(logits_torch[0, -1].argmax())
    device_token = int(token_ids.flatten()[0])
    print(
        f"\n[dots_ocr_tp4] LM head PCC: {msg} | torch_token={torch_token} tt_token={tt_token} device_argmax={device_token}"
    )
    assert tt_token == torch_token, f"top-1 token mismatch: torch {torch_token} vs tt {tt_token}"
    assert device_token == torch_token, f"device argmax {device_token} != torch {torch_token}"
