# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""PCC for Kokoro math-decomposed ops (no single TTNN op) with Kokoro-82M weights.

These modules appear in ``reference/reference.txt`` and are implemented in ``tt/`` via
composed ``ttnn`` primitives (mean/rsqrt, LSTM gate loop, ``repeat_interleave``, etc.).

Already covered elsewhere (do not duplicate here):

- ``test_tt_lstm_pcc.py`` — ``predictor.lstm``, ``predictor.shared`` (640→256)
- ``test_tt_adain_1d_pcc.py`` — generic InstanceNorm; ``predictor.F0[0].norm1`` (C=512);
  decoder AdaIN shapes with random weights
- ``test_tt_ada_layer_norm_pcc.py`` — AdaLayerNorm random weights
- ``test_tt_upsample_1d_pcc.py`` — UpSample1d random weights
- ``test_tt_sinegen_pcc.py``, ``test_tt_torch_stft_pcc.py`` — no learnable weights
- ``test_tt_linear_norm_pcc.py`` — ``ttnn.linear`` only (direct op)
- Composite stacks: ``test_tt_text_encoder_pcc.py``, ``test_tt_duration_encoder_pcc.py``,
  ``test_tt_decoder_pcc.py``, ``test_tt_generator_pcc.py``, etc.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

_TT_METAL_ROOT = Path(__file__).resolve().parents[4]
if str(_TT_METAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_TT_METAL_ROOT))

import ttnn

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.modules import AdaLayerNorm
from models.experimental.kokoro.tests.kokoro_checkpoint import (
    KOKORO_STYLE_DIM,
    assert_bilstm_config,
    capture_duration_encoder_lstm0_input_nlc,
    capture_text_encoder_lstm_input_nlc,
    device_compute_config,
    get_module_attr,
    load_kmodel,
    torch_bilstm_nlc,
)
from models.experimental.kokoro.tt.tt_adain_1d import (
    TTAdaIN1d,
    preprocess_tt_adain_1d,
    preprocess_tt_instance_norm_1d,
    tt_instance_norm_1d_nlc,
)
from models.experimental.kokoro.tt.tt_ada_layer_norm import TTAdaLayerNorm, preprocess_tt_ada_layer_norm
from models.experimental.kokoro.tt.tt_lstm import preprocess_tt_lstm_1layer, tt_bilstm_nlc
from models.experimental.kokoro.tt.tt_upsample_1d import TTUpSample1d

# (pytest id, dotted path on KModel, expected InstanceNorm channel count)
_INSTANCE_NORM_CASES = [
    ("predictor.F0[1].norm2", "predictor.F0[1].norm2.norm", 256),
    ("predictor.N[0].norm1", "predictor.N[0].norm1.norm", 512),
    ("decoder.encode.norm1", "decoder.encode.norm1.norm", 514),
    ("decoder.encode.norm2", "decoder.encode.norm2.norm", 1024),
    ("decoder.decode[0].norm1", "decoder.decode[0].norm1.norm", 1090),
    ("generator.noise_res[0].adain1[0]", "decoder.generator.noise_res[0].adain1[0].norm", 256),
    ("generator.noise_res[1].adain1[0]", "decoder.generator.noise_res[1].adain1[0].norm", 128),
]

# Full AdaIN1d modules (InstanceNorm + style linear) with trained weights
_ADAIN1D_CASES = [
    ("predictor.F0[0].norm1", "predictor.F0[0].norm1"),
    ("predictor.F0[1].norm1", "predictor.F0[1].norm1"),
    ("decoder.encode.norm1", "decoder.encode.norm1"),
    ("decoder.encode.norm2", "decoder.encode.norm2"),
]


@pytest.fixture(scope="module")
def kmodel():
    try:
        return load_kmodel()
    except FileNotFoundError:
        pytest.skip("Kokoro-82M checkpoint not found locally.")


def _run_instance_norm_pcc(device, inn: nn.InstanceNorm1d, *, b: int = 2, seq_len: int = 96) -> float:
    inn.eval()
    p = preprocess_tt_instance_norm_1d(inn, device)
    torch.manual_seed(0)
    x_bcl = torch.randn(b, inn.num_features, seq_len)
    with torch.no_grad():
        y_ref = inn(x_bcl)
    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_instance_norm_1d_nlc(x_nlc=x_tt, params=p)
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    return float(pcc)


@pytest.mark.parametrize("case_id,path,channels", _INSTANCE_NORM_CASES, ids=[c[0] for c in _INSTANCE_NORM_CASES])
def test_tt_instance_norm_1d_kmodel_reference_configs(device, kmodel, case_id, path, channels):
    """``tt_instance_norm_1d_nlc`` vs trained ``InstanceNorm1d`` from ``reference.txt``."""
    inn = get_module_attr(kmodel, path)
    assert isinstance(inn, nn.InstanceNorm1d)
    assert inn.num_features == channels
    assert inn.eps == pytest.approx(1e-05)
    assert inn.affine and not inn.track_running_stats
    pcc = _run_instance_norm_pcc(device, inn)
    print(f"TTInstanceNorm1d ({case_id}, C={channels}) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low for {case_id}: {pcc}"


@pytest.mark.parametrize("case_id,path", _ADAIN1D_CASES, ids=[c[0] for c in _ADAIN1D_CASES])
def test_tt_adain_1d_kmodel_reference_configs(device, kmodel, case_id, path):
    """``TTAdaIN1d`` vs trained ``AdaIN1d`` (``InstanceNorm1d`` + style ``fc``)."""
    ref_mod = get_module_attr(kmodel, path)
    ref_mod.eval()
    params = preprocess_tt_adain_1d(ref_mod, device)
    tt_mod = TTAdaIN1d(params)
    c = ref_mod.norm.num_features
    torch.manual_seed(1)
    b, l = 2, 64
    x_bcl = torch.randn(b, c, l)
    s = torch.randn(b, KOKORO_STYLE_DIM)
    with torch.no_grad():
        y_ref = ref_mod(x_bcl, s)
    x_nlc = x_bcl.transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=device_compute_config(device))
    y_hat = ttnn.to_torch(y_tt).float().transpose(1, 2).contiguous()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaIN1d ({case_id}) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low for {case_id}: {pcc}"


def test_tt_bilstm_text_encoder_lstm_512_256_kmodel(device, kmodel):
    """``text_encoder.lstm``: ``LSTM(512, 256, batch_first=True, bidirectional=True)``."""
    lstm = kmodel.text_encoder.lstm
    assert_bilstm_config(lstm, name="text_encoder.lstm", input_size=512, hidden_size=256)
    lstm.eval()
    fwd, rev = preprocess_tt_lstm_1layer(lstm, device)
    assert rev is not None
    x_nlc, lengths = capture_text_encoder_lstm_input_nlc(kmodel, seq_len=48, seed=0)
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
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TT BiLSTM text_encoder.lstm PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_bilstm_duration_encoder_lstm0_640_256_kmodel(device, kmodel):
    """``predictor.text_encoder.lstms[0]``: ``LSTM(640, 256, ..., dropout=0.2)``."""
    lstm = kmodel.predictor.text_encoder.lstms[0]
    assert_bilstm_config(lstm, name="DurationEncoder.lstm[0]", input_size=640, hidden_size=256)
    lstm.eval()
    fwd, rev = preprocess_tt_lstm_1layer(lstm, device)
    assert rev is not None
    x_nlc, lengths = capture_duration_encoder_lstm0_input_nlc(kmodel, seq_len=40, seed=1)
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
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TT BiLSTM DurationEncoder.lstm[0] PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_ada_layer_norm_duration_encoder_kmodel(device, kmodel):
    """``predictor.text_encoder.lstms[1]``: ``AdaLayerNorm`` (``fc``: 128→1024, ``C=512``)."""
    ref_mod = kmodel.predictor.text_encoder.lstms[1]
    assert isinstance(ref_mod, AdaLayerNorm)
    assert ref_mod.channels == 512
    ref_mod.eval()
    params = preprocess_tt_ada_layer_norm(ref_mod, device)
    tt_mod = TTAdaLayerNorm(params)
    torch.manual_seed(2)
    b, t = 1, 40
    x = torch.randn(b, t, 512)
    s = torch.randn(b, KOKORO_STYLE_DIM)
    with torch.no_grad():
        y_ref = ref_mod(x, s)
    x_tt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    s_tt = ttnn.from_torch(s, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_mod(x_tt, s_tt, compute_kernel_config=device_compute_config(device))
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    ttnn.deallocate(s_tt)
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTAdaLayerNorm DurationEncoder.lstms[1] PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"


def test_tt_upsample_1d_predictor_f0_block1_kmodel(device, kmodel):
    """``predictor.F0[1].upsample`` — nearest ×2 (``UpSample1d`` not ``none``)."""
    ref_up = kmodel.predictor.F0[1].upsample
    layer_type = str(ref_up.layer_type)
    assert layer_type != "none"
    tt_up = TTUpSample1d(layer_type)
    torch.manual_seed(3)
    b, c, l = 1, 512, 32
    x_bcl = torch.randn(b, c, l)
    x_nlc = x_bcl.transpose(1, 2).contiguous()
    with torch.no_grad():
        y_ref = ref_up(x_bcl).transpose(1, 2).contiguous()
    x_tt = ttnn.from_torch(x_nlc, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    y_tt = tt_up(x_tt)
    y_hat = ttnn.to_torch(y_tt).float()
    ttnn.deallocate(y_tt)
    ttnn.deallocate(x_tt)
    _, pcc = comp_pcc(y_ref, y_hat, pcc=0.0)
    print(f"TTUpSample1d (predictor.F0[1], type={layer_type!r}) PCC: {pcc:.6f}")
    assert pcc > 0.99, f"PCC too low: {pcc}"
