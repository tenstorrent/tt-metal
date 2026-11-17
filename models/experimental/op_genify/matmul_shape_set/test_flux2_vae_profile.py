# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Profile the stock FLUX.2-dev VAE decoder (tt_dit Flux2VaeDecoder) for the #65148 matmul shape set.

Its only matmuls are post_quant_conv (a 1x1 conv run as Linear 32->32) and the mid-block attention's
to_qkv / to_out; everything else is a 3x3 conv (#65144). Mesh axes follow the QB2 (2, 2) preset of
pipeline_flux2.py (VAE TP on axis 1, latent rows split on axis 0), 1024x1024 output. The checkpoint is
gated, so the weights are random at diffusers' AutoencoderKLFlux2 defaults (FLUX.2's VAE config), with
the up blocks pruned to 2 resnets as in tests/models/flux2/test_vae_flux2.py (no matmuls there).

Run (needs a Tracy build):
    scripts/run_safe_pytest.sh --profile models/experimental/op_genify/matmul_shape_set/test_flux2_vae_profile.py
"""

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch
from tracy import signpost

import ttnn
from models.tt_dit.models.vae.vae_flux2 import Flux2VaeDecoder
from models.tt_dit.parallel.config import Flux2VaeParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.flux2.device_params import line_params_flux2
from models.tt_dit.tests.models.flux2.test_vae_flux2 import prep_data

LAYERS_PER_BLOCK = 1


@pytest.mark.parametrize(
    "mesh_device, tp_axis, h_axis, num_links, device_params",
    [[(2, 2), 1, 0, 2, line_params_flux2]],
    ids=["bh_2x2_pipeline_axes"],
    indirect=["mesh_device", "device_params"],
)
def test_flux2_vae_decoder_profile(mesh_device, tp_axis, h_axis, num_links):
    torch.manual_seed(0)
    torch_model = reference.AutoencoderKLFlux2()
    for up_block in torch_model.decoder.up_blocks:
        up_block.resnets = torch.nn.ModuleList(list(up_block.resnets)[: LAYERS_PER_BLOCK + 1])
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, num_links=num_links, topology=ttnn.Topology.Linear)
    parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)
    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=LAYERS_PER_BLOCK,
        z_channels=torch_model.config.latent_channels,
        device=mesh_device,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    tt_model.load_torch_state_dict(torch_model.state_dict())

    patch, scale = 2, 8
    inp = torch.randn(
        1, torch_model.config.latent_channels * patch**2, 1024 // (scale * patch), 1024 // (scale * patch)
    )

    tt_model.forward(prep_data(parallel_config, inp, mesh_device, tt_model=tt_model))  # warm-up
    ttnn.synchronize_device(mesh_device)
    ttnn.ReadDeviceProfiler(mesh_device)

    tt_inp = prep_data(parallel_config, inp, mesh_device, tt_model=tt_model)
    signpost("VAE_DECODE")
    out = tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)
    signpost("VAE_DECODE_END")
    ttnn.ReadDeviceProfiler(mesh_device)
    assert out.shape[-1] == 3
