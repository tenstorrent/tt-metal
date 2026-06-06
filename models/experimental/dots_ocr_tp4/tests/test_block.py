# SPDX-FileCopyrightText: (C) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Single dots.ocr decoder block (norm/attn/res/norm/mlp/res) TP4 vs torch."""

import pytest
import torch

import ttnn
from tests.ttnn.utils_for_testing import assert_with_pcc

from models.experimental.dots_ocr_tp4.tt.common import DotsOCRConfig, from_replicated_to_torch, to_replicated
from models.experimental.dots_ocr_tp4.tt.decoder_block import DotsOCRDecoderBlockTP4
from models.experimental.dots_ocr_tp4.tests.common import device_params, resolve_mesh_shape
from models.experimental.dots_ocr_tp4.tests.torch_reference import TorchDecoderBlock


@pytest.mark.parametrize("device_params", [device_params()], indirect=True)
@pytest.mark.parametrize("mesh_device", [resolve_mesh_shape()], indirect=True)
@pytest.mark.parametrize("seq_len", [2816])
def test_dots_ocr_block_tp4(mesh_device, seq_len):
    torch.manual_seed(0)
    torch.set_grad_enabled(False)

    config = DotsOCRConfig()
    H = config.hidden_size

    torch_block = TorchDecoderBlock(config).eval()
    x = torch.randn(1, seq_len, H, dtype=torch.bfloat16)
    torch_out = torch_block(x.to(torch.float32))

    tt_block = DotsOCRDecoderBlockTP4.from_torch(mesh_device, config, torch_block, layer_idx=0)
    x_tt = to_replicated(x, mesh_device, dtype=ttnn.bfloat16)

    out_tt = tt_block.forward(x_tt)
    ttnn.synchronize_device(mesh_device)

    out_torch = from_replicated_to_torch(out_tt, mesh_device).to(torch.float32).reshape(torch_out.shape)

    assert_with_pcc(torch_out.to(torch.float32), out_torch, pcc=0.99)
