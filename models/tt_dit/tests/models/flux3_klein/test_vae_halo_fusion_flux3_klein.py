# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""The decoder with set_halo_fusion on and off under FLUX 3 Klein's conv tables: both match the torch decoder, and the
fused paths (external-halo convs, norm2 from conv1 statistics) actually run."""

from __future__ import annotations

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch

import ttnn

from ....layers.conv2d import configure_conv2d
from ....models.vae.vae import VaeConv2d, VaeResnetBlock, set_halo_fusion
from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....models.vae.vae_flux3_klein import decoder_conv_blocking, decoder_slice_params
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....pipelines.flux3_klein.weights_flux3_klein import resolve_weights_dir
from ....utils import tensor
from ....utils.check import assert_quality
from ..flux2.device_params import line_params_flux2
from ..flux2.test_vae_flux2 import _LAYERS_PER_BLOCK, _torch_decode_reference, prep_data


def _load_pruned_torch_model() -> reference.AutoencoderKLFlux2:
    model = reference.AutoencoderKLFlux2.from_pretrained(resolve_weights_dir("distilled"), subfolder="vae")
    model.eval()
    for up_block in model.decoder.up_blocks:
        up_block.resnets = torch.nn.ModuleList(list(up_block.resnets)[: _LAYERS_PER_BLOCK + 1])
    return model


@pytest.mark.parametrize("mesh_device, device_params", [[(4, 8), line_params_flux2]], ids=["4x8"], indirect=True)
@pytest.mark.parametrize("size", [1024], ids=["1024"])
def test_vae_halo_fusion_flux3_klein(*, mesh_device: ttnn.MeshDevice, size: int, monkeypatch) -> None:
    torch.manual_seed(0)
    torch_model = _load_pruned_torch_model()
    h_axis, w_axis = 1, 0
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=None, h_axis=h_axis, w_axis=w_axis)
    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=_LAYERS_PER_BLOCK,
        z_channels=torch_model.config.latent_channels,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear),
        use_conv3d=False,
        group_norm_2d=True,
        local_attention=True,
    )
    table = decoder_slice_params((size,), int(mesh_device.shape[h_axis]), int(mesh_device.shape[w_axis]))
    configure_conv2d(
        tt_model, slice_params={tuple(mesh_device.shape): table}, shape_blocking=decoder_conv_blocking(table)
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    calls = {"halo": 0, "stats": 0}

    def counted(name, fn):
        def wrapper(*args, **kwargs):
            calls[name] += 1
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(VaeConv2d, "_forward_external_halo", counted("halo", VaeConv2d._forward_external_halo))
    monkeypatch.setattr(
        VaeResnetBlock, "_conv1_norm2_from_stats", counted("stats", VaeResnetBlock._conv1_norm2_from_stats)
    )

    f = 8 * 2
    inp = torch.randn(1, torch_model.config.latent_channels * 4, size // f, size // f)
    torch_output = _torch_decode_reference(torch_model, inp)

    outputs = {}
    for on in (False, True):
        set_halo_fusion(tt_model, on)
        tt_inp = prep_data(parallel_config, inp, mesh_device, tt_model=tt_model)
        tt_out = tt_model.forward(tt_inp)
        ttnn.synchronize_device(mesh_device)
        outputs[on] = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
        assert_quality(torch_output, outputs[on], pcc=0.9978, relative_rmse=0.034)
        if not on:
            assert calls == {"halo": 0, "stats": 0}

    assert calls["halo"] > 0 and calls["stats"] > 0, calls
    assert_quality(outputs[False], outputs[True], pcc=0.9995)
