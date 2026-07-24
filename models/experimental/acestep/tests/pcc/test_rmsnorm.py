# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""PCC: ACE-Step v1.5 RMSNorm vs TTTv2 RMSNorm1D (pure reuse, no custom class).

ACE-Step uses Qwen3RMSNorm everywhere (input_layernorm, post_attention_layernorm,
self_attn_norm, cross_attn_norm, mlp_norm, and per-head q_norm/k_norm). It is a
standard RMSNorm, so TTTv2 RMSNorm1D covers it directly. We validate against the
real HF Qwen3RMSNorm at ACE-Step's hidden_size and head_dim with eps=1e-6.
"""

import pytest
import torch

import ttnn
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

from models.common.modules.rmsnorm.rmsnorm_1d import RMSNorm1D, RMSNorm1DConfig
from models.experimental.acestep.tests.test_utils import (
    HEAD_DIM,
    HIDDEN_SIZE,
    RMS_NORM_EPS,
    SEQUENCE_LENGTHS,
    assert_pcc,
    make_lazy_weight,
    require_single_device,
    to_torch,
    to_ttnn_tensor,
)

BATCH_SIZE = 1


@pytest.mark.parametrize("seq_len", SEQUENCE_LENGTHS, ids=[f"S{seq_len}" for seq_len in SEQUENCE_LENGTHS])
@pytest.mark.parametrize("dim", [HIDDEN_SIZE, HEAD_DIM], ids=["hidden2048", "head128"])
def test_rmsnorm_vs_qwen3(device, seq_len, dim):
    require_single_device(device)
    torch.manual_seed(42)

    reference_layer = Qwen3RMSNorm(dim, eps=RMS_NORM_EPS).eval()
    with torch.no_grad():
        # Non-trivial weights (default is all ones) to exercise the scale path.
        reference_layer.weight.copy_(1.0 + 0.02 * torch.randn_like(reference_layer.weight))

    x = torch.randn((BATCH_SIZE, 1, seq_len, dim), dtype=torch.float32)

    tt_model = RMSNorm1D.from_config(
        RMSNorm1DConfig(
            weight=make_lazy_weight(
                reference_layer.weight.detach().clone(),
                device,
                dtype=ttnn.bfloat16,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            ),
            eps=RMS_NORM_EPS,
        )
    )

    tt_output = tt_model.forward(to_ttnn_tensor(x, device), mode="prefill")
    tt_output_torch = to_torch(tt_output, expected_shape=(BATCH_SIZE, 1, seq_len, dim))

    with torch.no_grad():
        reference_output = reference_layer(x.to(torch.float32)).to(torch.float32)

    assert_pcc(reference_output, tt_output_torch, 0.999)
