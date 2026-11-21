# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 VAE decode with its 3x3 convs on ttnn.conv2d vs ttnn.operations.conv2d_3x3.

The full decoder (layers_per_block 2) with diffusers' AutoencoderKLFlux2 default config, which is FLUX.2's
VAE architecture, with seeded random weights: the gated FLUX.2-dev checkpoint is not needed to compare the
two conv implementations, and both see the same weights and the same torch reference. Meshes: one device,
and the Blackhole (2, 4) preset of pipeline_flux2.py (channels split on mesh axis 1, rows on axis 0).

Each case checks quality against torch, then times MEASURED_DECODES untraced decodes (wall clock after
synchronize, after one warm-up decode that compiles). Results go to FLUX_E2E_RESULTS (JSON lines).
"""

import json
import os
import statistics
import time

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch

import ttnn

from ....layers.conv2d import Conv2d, configure_conv2d
from ....models.vae.vae_flux2 import Flux2VaeDecoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from .device_params import line_params_flux2, single_device_params_flux2
from .test_vae_flux2 import _torch_decode_reference, prep_data

MEASURED_DECODES = 3
RESULTS = os.environ.get("FLUX_E2E_RESULTS", "/tmp/flux_e2e_results.jsonl")


def _conv_paths(module) -> tuple[int, int]:
    """(convs that ran on conv2d_3x3, convs that ran on ttnn.conv2d) in the last decode."""
    if isinstance(module, Conv2d):
        on_3x3 = any(k and k[0] == "conv2d_3x3" for k in module._prepared)
        return int(on_3x3), int(not on_3x3)
    on_3x3 = on_ttnn = 0
    for _, child in module.named_children():
        a, b = _conv_paths(child)
        on_3x3, on_ttnn = on_3x3 + a, on_ttnn + b
    return on_3x3, on_ttnn


@pytest.mark.parametrize(
    "mesh_device, device_params, tp_axis, h_axis",
    [
        [(1, 1), single_device_params_flux2, 0, 1],
        [(2, 4), line_params_flux2, 1, 0],
    ],
    ids=["1x1", "bh2x4"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024), (2048, 2048)], ids=["1024", "2048"])
@pytest.mark.parametrize("conv", ["ttnn", "conv2d_3x3"])
def test_vae_flux2_decoder_conv2d_3x3(
    *, mesh_device: ttnn.MeshDevice, tp_axis: int, h_axis: int, height: int, width: int, conv: str
) -> None:
    torch.manual_seed(0)
    torch_model = reference.AutoencoderKLFlux2()
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)

    z_channels = torch_model.config.latent_channels
    patch_size = 2
    vae_scale_factor = 8

    tt_model = Flux2VaeDecoder(
        out_channels=torch_model.config.out_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=torch_model.config.layers_per_block,
        z_channels=z_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
        use_conv3d=False,
    )
    configure_conv2d(tt_model, use_conv2d_3x3=conv == "conv2d_3x3")
    tt_model.load_torch_state_dict(torch_model.state_dict())

    f = vae_scale_factor * patch_size
    inp = torch.randn(1, z_channels * patch_size**2, height // f, width // f)
    torch_output = _torch_decode_reference(torch_model, inp)

    tt_inp = prep_data(vae_parallel_config, inp, mesh_device, tt_model=tt_model)
    tt_out = tt_model.forward(tt_inp)
    ttnn.synchronize_device(mesh_device)
    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    ttnn.deallocate(tt_out)

    wall_ms = []
    for _ in range(MEASURED_DECODES):
        t0 = time.perf_counter()
        tt_out = tt_model.forward(tt_inp)
        ttnn.synchronize_device(mesh_device)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(tt_out)

    on_3x3, on_ttnn = _conv_paths(tt_model)
    pcc = torch.corrcoef(torch.stack([torch_output.flatten().double(), tt_out_torch.flatten().double()]))[0, 1].item()
    record = {
        "mesh": list(mesh_device.shape),
        "image": [height, width],
        "conv": conv,
        "decode_ms_median": statistics.median(wall_ms),
        "decode_ms": wall_ms,
        "pcc": pcc,
        "convs_on_conv2d_3x3": on_3x3,
        "convs_on_ttnn_conv2d": on_ttnn,
    }
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print("FLUX_E2E:", json.dumps(record))

    # PCC only: test_vae_flux2.py's relative-RMSE bound is calibrated on the pretrained, pruned decoder; the
    # full-depth random-weight decoder reads RMSE/sigma ~6% on ttnn.conv2d itself.
    assert_quality(torch_output, tt_out_torch, pcc=0.995)
