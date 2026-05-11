# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""PCC: TTNN ``AdaINResBlock1`` vs PyTorch (generator ``noise_res`` / ``resblocks``)."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

ttnn = pytest.importorskip("ttnn")

from models.common.utility_functions import comp_pcc
from models.experimental.kokoro.reference.kokoro_istftnet import load_decoder_from_huggingface
from models.experimental.kokoro.tt import AdaINResBlock1, preprocess_adain_resblock1_parameters


@pytest.fixture
def ttnn_device():
    device = ttnn.open_device(device_id=0, l1_small_size=24576)
    yield device
    ttnn.close_device(device)


@pytest.fixture
def kokoro_decoder_cpu_disable_complex():
    return load_decoder_from_huggingface(device="cpu", disable_complex=True)


@pytest.mark.parametrize("which", ["noise_res0", "resblock0"])
def test_adain_resblock1_pcc(ttnn_device, kokoro_decoder_cpu_disable_complex, which: str):
    gen = kokoro_decoder_cpu_disable_complex.decoder.generator
    if which == "noise_res0":
        block = gen.noise_res[0]
    else:
        block = gen.resblocks[0]

    ch = int(block.convs1[0].weight.shape[0])
    style_dim = int(block.adain1[0].fc.weight.shape[1])
    torch.manual_seed(1)
    x = torch.randn(1, ch, 24, dtype=torch.float32)
    s = torch.randn(1, style_dim, dtype=torch.float32)

    with torch.no_grad():
        y_ref = block(x, s)

    params = preprocess_adain_resblock1_parameters(block, ttnn_device)
    tt_block = AdaINResBlock1(ttnn_device, params)

    x_tt = ttnn.from_torch(
        x,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    s_tt = ttnn.from_torch(
        s,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=ttnn_device,
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    y_tt = tt_block(x_tt, s_tt)
    y_hat = ttnn.to_torch(y_tt).reshape(y_ref.shape)

    ok, p = comp_pcc(y_ref, y_hat, pcc=0.97)
    print(f"{which} PCC={p:.6f} pass={ok}")
    assert ok, f"PCC {p} expected >= 0.97"
