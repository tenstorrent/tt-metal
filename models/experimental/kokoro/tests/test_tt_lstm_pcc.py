# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC: :func:`~models.experimental.kokoro.tt.tt_lstm.tt_bilstm_nlc` vs PyTorch BiLSTM.

Uses loaded Kokoro-82M layers from ``reference/reference.txt``::

    ProsodyPredictor.(lstm|shared): LSTM(640, 256, batch_first=True, bidirectional=True)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.tests.kokoro_checkpoint import (
    assert_bilstm_config,
    capture_predictor_lstm_input_nlc,
    device_compute_config,
    load_kmodel,
    torch_bilstm_nlc,
)
from models.experimental.kokoro.tt.tt_lstm import preprocess_tt_lstm_1layer, tt_bilstm_nlc


@pytest.fixture(scope="module")
def kmodel():
    try:
        return load_kmodel()
    except FileNotFoundError:
        pytest.skip("Kokoro-82M checkpoint not found locally.")


def test_tt_bilstm_640_256_lstm_full_sequence_matches_torch(device, kmodel):
    """``ProsodyPredictor.lstm`` weights — full valid length."""
    lstm = kmodel.predictor.lstm
    assert_bilstm_config(lstm, name="predictor.lstm", input_size=640, hidden_size=256)
    lstm.eval()
    fwd, rev = preprocess_tt_lstm_1layer(lstm, device)
    assert rev is not None

    x_nlc, _ = capture_predictor_lstm_input_nlc(kmodel, seq_len=48, seed=0)
    y_ref = torch_bilstm_nlc(lstm, x_nlc)

    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_bilstm_nlc(
        x_nlc=x_tt,
        fwd=fwd,
        rev=rev,
        compute_kernel_config=device_compute_config(device),
    )
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_ref.shape == y_hat.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TT BiLSTM predictor.lstm full-seq PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_bilstm_640_256_lstm_variable_length_matches_torch(device, kmodel):
    """``ProsodyPredictor.lstm`` with packed-length semantics on captured ``d``."""
    lstm = kmodel.predictor.lstm
    assert_bilstm_config(lstm, name="predictor.lstm", input_size=640, hidden_size=256)
    lstm.eval()
    fwd, rev = preprocess_tt_lstm_1layer(lstm, device)
    assert rev is not None

    x_nlc, _ = capture_predictor_lstm_input_nlc(kmodel, seq_len=40, seed=1)
    b, t, _ = x_nlc.shape
    x_nlc = x_nlc.repeat(3, 1, 1)
    lengths = [min(t, 38), min(t, 12), min(t, 25)]

    y_ref = torch_bilstm_nlc(lstm, x_nlc, sequence_lengths=lengths)

    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_bilstm_nlc(
        x_nlc=x_tt,
        fwd=fwd,
        rev=rev,
        sequence_lengths=lengths,
        compute_kernel_config=device_compute_config(device),
    )
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    assert y_ref.shape == y_hat.shape
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TT BiLSTM predictor.lstm variable-length PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_bilstm_640_256_shared_fp32_state_matches_torch(device, kmodel):
    """``ProsodyPredictor.shared`` weights — ``fp32_state`` path used in ``F0Ntrain``."""
    lstm = kmodel.predictor.shared
    assert_bilstm_config(lstm, name="predictor.shared", input_size=640, hidden_size=256)
    lstm.eval()
    fwd, rev = preprocess_tt_lstm_1layer(lstm, device)
    assert rev is not None

    x_nlc, _ = capture_predictor_lstm_input_nlc(kmodel, seq_len=32, seed=2)
    y_ref = torch_bilstm_nlc(lstm, x_nlc)

    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_bilstm_nlc(
        x_nlc=x_tt,
        fwd=fwd,
        rev=rev,
        fp32_state=True,
        compute_kernel_config=device_compute_config(device),
    )
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)

    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TT BiLSTM predictor.shared fp32_state PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
