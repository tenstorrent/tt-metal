# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

import ttnn
from models.demos.rvc.torch_impl.synthesizer.models import SynthesizerTrnMsNSF as TorchSynthesizerTrnMsNSF
from models.demos.rvc.tt_impl.synthesizer.models import SynthesizerTrnMsNSF as TTSynthesizerTrnMsNSF
from tests.ttnn.utils_for_testing import assert_with_pcc


@pytest.mark.parametrize("device_params", [{"l1_small_size": 65536}], indirect=True)
def test_synthesizer_trn_ms_nsf(device):
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 64
    embedding_dims = 96
    inter_channels = 8
    hidden_channels = 64
    filter_channels = 32
    num_heads = 2
    num_layers = 1
    kernel_size = 3
    resblock = "2"
    resblock_kernel_sizes = [3]
    resblock_dilation_sizes = [(1, 3)]
    upsample_rates = [2]
    upsample_initial_channel = 16
    upsample_kernel_sizes = [4]
    spk_embed_dim = 4
    gin_channels = 8
    sr = "32k"

    torch_model = TorchSynthesizerTrnMsNSF(
        embedding_dims=embedding_dims,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        kernel_size=kernel_size,
        resblock=resblock,
        resblock_kernel_sizes=resblock_kernel_sizes,
        resblock_dilation_sizes=resblock_dilation_sizes,
        upsample_rates=upsample_rates,
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=upsample_kernel_sizes,
        spk_embed_dim=spk_embed_dim,
        gin_channels=gin_channels,
        sr=sr,
    ).eval()

    tt_model = TTSynthesizerTrnMsNSF(
        device=device,
        embedding_dims=embedding_dims,
        inter_channels=inter_channels,
        hidden_channels=hidden_channels,
        filter_channels=filter_channels,
        num_heads=num_heads,
        num_layers=num_layers,
        kernel_size=kernel_size,
        resblock=resblock,
        resblock_kernel_sizes=list(resblock_kernel_sizes),
        resblock_dilation_sizes=[list(x) for x in resblock_dilation_sizes],
        upsample_rates=list(upsample_rates),
        upsample_initial_channel=upsample_initial_channel,
        upsample_kernel_sizes=list(upsample_kernel_sizes),
        spk_embed_dim=spk_embed_dim,
        gin_channels=gin_channels,
        sr=sr,
    )
    parameters = {f"net_g.{k}": v for k, v in torch_model.state_dict().items()}
    tt_model.load_state_dict(parameters, module_prefix="net_g.")

    torch_phone = torch.randn(batch_size, seq_len, embedding_dims, dtype=torch.float32)
    torch_pitch = torch.randint(0, 255, (batch_size, seq_len), dtype=torch.int64)
    torch_nsff0 = torch.rand(batch_size, seq_len, dtype=torch.float32) * 300.0
    torch_speaker_id = torch.randint(0, spk_embed_dim, (batch_size,), dtype=torch.int64)

    torch.manual_seed(1234)
    torch_output = torch_model(torch_phone, torch_pitch, torch_nsff0, torch_speaker_id)

    tt_phone = ttnn.from_torch(
        torch_phone,  # .permute(0, 2, 1),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )
    tt_pitch = ttnn.from_torch(
        torch_pitch,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_nsff0 = ttnn.from_torch(
        torch_nsff0.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )
    tt_speaker_id = ttnn.from_torch(
        torch_speaker_id,
        dtype=ttnn.uint32,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
    )

    torch.manual_seed(1234)
    tt_output = tt_model(tt_phone, tt_pitch, tt_nsff0, tt_speaker_id)
    tt_output_torch = ttnn.to_torch(tt_output).to(torch.float32).permute(0, 2, 1)

    assert tuple(tt_output_torch.shape) == tuple(torch_output.shape)
    assert_with_pcc(torch_output, tt_output_torch, pcc=0.9)
