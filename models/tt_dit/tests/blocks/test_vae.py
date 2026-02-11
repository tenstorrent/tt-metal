# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

import diffusers.models.autoencoders.autoencoder_kl_qwenimage as reference
import pytest
import torch

import ttnn

from ...blocks.vae import VaeAttention, VaeContext, VaeNormDescRms, VaeRmsNorm
from ...parallel.manager import CCLManager
from ...utils import tensor
from ...utils.check import assert_quality


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis"),
    [
        pytest.param((1, 1), None, id="1x1"),
        pytest.param((1, 2), None, id="1x2"),
        # pytest.param((1, 2), 1, id="1x2tp"),  # hangs, resolved by https://github.com/tenstorrent/tt-metal/pull/33254
        pytest.param((1, 4), 1, id="1x4tp"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width", "channels"),
    [
        (2, 128, 128, 384),
    ],
)
def test_vae_rms_norm(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
    channels: int,
    tp_axis: int | None,
) -> None:
    torch.manual_seed(0)

    torch_model = reference.QwenImageRMS_norm(channels)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    ctx = VaeContext(tp_axis=tp_axis, device=mesh_device, ccl_manager=ccl_manager)
    tt_model = VaeRmsNorm(channels, eps=1e-12, ctx=ctx)

    tt_model.load_torch_state_dict(torch_model.state_dict())

    inp = torch.randn(batch_size, channels, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), mesh_axes=[..., tp_axis], device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model.forward(inp)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[..., tp_axis]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.99999, relative_rmse=0.008)


@pytest.mark.parametrize(
    ("mesh_device", "tp_axis"),
    [
        pytest.param((1, 1), None, id="1x1"),
        pytest.param((1, 2), None, id="1x2"),
        # pytest.param((1, 2), 1, id="1x2tp"),  # hangs, resolved by https://github.com/tenstorrent/tt-metal/pull/33254
        pytest.param((1, 4), 1, id="1x4tp"),
    ],
    indirect=["mesh_device"],
)
@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "l1_small_size": 65536, "trace_region_size": 20000000}],
    indirect=True,
)
@pytest.mark.parametrize(
    ("batch_size", "height", "width", "dim"),
    [
        (2, 128, 128, 384),
    ],
)
def test_vae_qwenimage_attention(
    *,
    mesh_device: ttnn.Device,
    batch_size: int,
    height: int,
    width: int,
    dim: int,
    tp_axis: int | None,
) -> None:
    torch.manual_seed(0)

    torch_model = reference.QwenImageAttentionBlock(dim)
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)

    ctx = VaeContext(tp_axis=tp_axis, device=mesh_device, ccl_manager=ccl_manager)
    tt_model = VaeAttention(num_channels=dim, norm=VaeNormDescRms(eps=1e-12), ctx=ctx)

    state = torch_model.state_dict()
    tt_model.load_torch_state_dict(
        {
            "to_out.0.weight": state["proj.weight"].squeeze(2, 3),
            "to_out.0.bias": state["proj.bias"],
            "to_q.weight": state["to_qkv.weight"].squeeze(2, 3).chunk(3)[0],
            "to_k.weight": state["to_qkv.weight"].squeeze(2, 3).chunk(3)[1],
            "to_v.weight": state["to_qkv.weight"].squeeze(2, 3).chunk(3)[2],
            "to_q.bias": state["to_qkv.bias"].chunk(3)[0],
            "to_k.bias": state["to_qkv.bias"].chunk(3)[1],
            "to_v.bias": state["to_qkv.bias"].chunk(3)[2],
            "norm.gamma": state["norm.gamma"],
        }
    )

    inp = torch.randn(batch_size, dim, height, width)

    tt_inp = tensor.from_torch(inp.permute(0, 2, 3, 1), mesh_axes=[..., tp_axis], device=mesh_device)

    with torch.no_grad():
        torch_output = torch_model.forward(inp.unsqueeze(2)).squeeze(2)

    tt_out = tt_model.forward(tt_inp)

    tt_out_torch = tensor.to_torch(tt_out, mesh_axes=[..., tp_axis]).permute(0, 3, 1, 2)
    assert_quality(torch_output, tt_out_torch, pcc=0.99996, relative_rmse=0.009)
