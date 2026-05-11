# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN generator upsampling (``conv_transpose2d``) vs ``nn.ConvTranspose1d``."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_generator_preprocess import _safe_remove_weight_norm
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt.ttnn_kokoro_generator import _UpsConvTranspose1d


def _ups_spec_from_module(m, device):
    w = m.weight.data.unsqueeze(-1).contiguous()
    op = int(m.output_padding[0]) if isinstance(m.output_padding, tuple) else int(m.output_padding)
    return {
        "weight": ttnn.from_torch(w, dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT),
        "bias": None
        if m.bias is None
        else ttnn.from_torch(
            torch.reshape(m.bias.data, (1, 1, 1, m.out_channels)),
            dtype=ttnn.float32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
        ),
        "stride": int(m.stride[0]),
        "kernel_size": int(m.kernel_size[0]),
        "padding": int(m.padding[0]),
        "output_padding": op,
        "in_channels": int(m.in_channels),
        "out_channels": int(m.out_channels),
    }


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.mark.parametrize("ups_idx", [0, 1])
def test_generator_ups_conv_transpose_matches_torch(ttnn_device, ups_idx: int):
    """Each Kokoro ``generator.ups`` layer matches PyTorch ``ConvTranspose1d`` on random input."""
    gen = load_decoder_from_huggingface(device="cpu", disable_complex=True).decoder.generator
    for m in gen.ups:
        _safe_remove_weight_norm(m)
    m = gen.ups[ups_idx]
    spec = _ups_spec_from_module(m, ttnn_device)
    tt_ups = _UpsConvTranspose1d(ttnn_device, spec)

    torch.manual_seed(ups_idx + 7)
    t_in = 8
    x = torch.randn(1, m.in_channels, t_in, dtype=torch.float32)
    with torch.no_grad():
        y_ref = m(x)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y_tt = tt_ups(x_tt, 1, t_in)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)

    ok, p = comp_pcc(y_ref, y_hat, pcc=0.99)
    print(f"ups[{ups_idx}] PCC={p:.6f} pass={ok} shape_ref={tuple(y_ref.shape)}")
    assert ok, f"ups[{ups_idx}] PCC {p} expected >= 0.99"
