# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN ``bilstm_nlc`` vs PyTorch 1-layer bidirectional ``nn.LSTM``.

Stand-alone coverage for the LSTM primitive used by the Kokoro predictor (both the prosody
``predictor.lstm`` and the shared LSTM). Until this test existed, LSTM precision was only checked
implicitly via the predictor duration/full tests — making it hard to attribute precision drops to a
specific layer.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.tt.ttnn_kokoro_lstm import bilstm_nlc, preprocess_pytorch_lstm_1layer


def _hifi3_fp32_dest(device):
    """Match the predictor compute config so the LSTM matmuls run at HiFi3 + fp32 dest acc."""
    return ttnn.init_device_compute_kernel_config(
        device.arch(),
        math_fidelity=ttnn.MathFidelity.HiFi3,
        math_approx_mode=False,
        fp32_dest_acc_en=True,
    )


@pytest.mark.parametrize(
    "batch,seq_len,input_size,hidden_size,sequence_lengths",
    [
        (1, 32, 512, 256, None),  # predictor.lstm shape (matches preprocess_predictor_duration)
        (1, 32, 512, 256, [32]),  # explicit length mask path (matches predictor's lengths_list)
        (1, 16, 64, 32, None),  # small smoke-test shape
    ],
)
def test_bilstm_nlc_matches_torch(device, batch, seq_len, input_size, hidden_size, sequence_lengths):
    torch.manual_seed(0)
    lstm = torch.nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        batch_first=True,
        bidirectional=True,
    )
    lstm.eval()

    x = torch.randn(batch, seq_len, input_size, dtype=torch.float32)
    with torch.no_grad():
        y_ref, _ = lstm(x)

    fwd, rev = preprocess_pytorch_lstm_1layer(lstm, device, weights_dtype=ttnn.bfloat16)
    assert rev is not None

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    y_tt = bilstm_nlc(
        x_nlc=x_tt,
        fwd=fwd,
        rev=rev,
        compute_kernel_config=_hifi3_fp32_dest(device),
        sequence_lengths=sequence_lengths,
    )
    y_hat = ttnn.to_torch(y_tt).to(torch.float32).reshape(y_ref.shape)
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_hat.shape == y_ref.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"bilstm_nlc PCC: {pcc:.6f}  shape={tuple(y_ref.shape)}  lengths={sequence_lengths}")
    # bf16 weights + bf16 inputs through a sequential LSTM compound to ~0.99 vs CPU fp32. Tighten when
    # the LSTM is rewritten to fp32 inputs / weights.
    assert pcc >= 0.99, f"bilstm_nlc PCC too low: {pcc}"
