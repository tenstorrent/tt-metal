# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""FLUX.2 VAE decode and encode with their 3x3 convs on ttnn.conv2d vs ttnn.operations.conv2d_3x3.

The full decoder and encoder (layers_per_block 2) with diffusers' AutoencoderKLFlux2 default config, which is
FLUX.2's VAE architecture, with seeded random weights: the gated FLUX.2-dev checkpoint is not needed to compare
the two conv implementations, and both see the same weights and the same torch reference. Meshes: one device,
and the Blackhole (2, 2) and (2, 4) presets of pipeline_flux2.py (channels split on mesh axis 1, rows on axis 0).

Each case checks quality against torch, then times MEASURED_DECODES untraced passes (wall clock after
synchronize, after one warm-up pass that compiles). With conv2d_3x3 every 3x3 conv must run on it. Results go
to FLUX_E2E_RESULTS (JSON lines).
"""

import json
import os
import statistics
import sys
import time

import diffusers.models.autoencoders.autoencoder_kl_flux2 as reference
import pytest
import torch

import ttnn

from ....layers.conv2d import Conv2d, configure_conv2d
from ....models.vae.vae_flux2 import Flux2VaeDecoder, Flux2VaeEncoder
from ....parallel.config import Flux2VaeParallelConfig
from ....parallel.manager import CCLManager
from ....utils import tensor
from ....utils.check import assert_quality
from ....utils.tracing import Tracer
from .device_params import TRACE_REGION_SIZE, line_params_flux2, single_device_params_flux2
from .test_vae_flux2 import _torch_decode_reference, prep_data

MEASURED_DECODES = 3
TRACED_DECODES = 20
RESULTS = os.environ.get("FLUX_E2E_RESULTS", "/tmp/flux_e2e_results.jsonl")


def _in_collect_pass():
    """True during run_safe_pytest's up-front collect pass, which runs test bodies to record programs."""
    plugin = sys.modules.get("tests.plugins.up_front_collect")
    return plugin is not None and getattr(plugin, "_PASS", "real") == "collect"


def _conv_paths(module) -> tuple[int, int]:
    """(3x3 convs that ran on conv2d_3x3, 3x3 convs that ran on ttnn.conv2d) in the last pass."""
    if isinstance(module, Conv2d):
        if tuple(module.kernel_size) != (3, 3):
            return 0, 0
        on_3x3 = any(k and k[0] == "conv2d_3x3" for k in module._prepared)
        return int(on_3x3), int(not on_3x3)
    on_3x3 = on_ttnn = 0
    for _, child in module.named_children():
        a, b = _conv_paths(child)
        on_3x3, on_ttnn = on_3x3 + a, on_ttnn + b
    return on_3x3, on_ttnn


def _auto_slice_strided(module, mesh_shape) -> None:
    """Let ttnn pick the slicing of the stride-2 convs. Conv2d.slice_params is keyed without stride, so a stride-2
    downsampler gets the slice count of the stride-1 conv of the same input shape, which ttnn rejects for its
    halved output width (num_slices <= max_num_slices)."""
    if isinstance(module, Conv2d):
        if module.stride != (1, 1):
            table = dict(module.slice_params.get(mesh_shape, module.slice_default))
            for key in list(table):
                table[key] = None
            module.slice_params = {**module.slice_params, mesh_shape: table}
        return
    for _, child in module.named_children():
        _auto_slice_strided(child, mesh_shape)


def _decoder_case(mesh_device, tp_axis, h_axis, height, width, conv):
    """FLUX.2's VAE decoder with seeded random weights on `conv`, its torch reference output for a random latent,
    and that latent prepared on device as pipeline_flux2.py hands it to the decoder."""
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
    return tt_model, tt_inp, torch_output, ccl_manager


@pytest.mark.parametrize(
    "mesh_device, device_params, tp_axis, h_axis",
    [
        [(1, 1), single_device_params_flux2, 0, 1],
        [(2, 2), line_params_flux2, 1, 0],
        [(2, 4), line_params_flux2, 1, 0],
        # The Galaxy preset splits channels on axis 0 (4) and rows on axis 1 (8), not 1/0 like the smaller meshes.
        [(4, 8), line_params_flux2, 0, 1],
    ],
    ids=["1x1", "bh2x2", "bh2x4", "bh4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024), (2048, 2048)], ids=["1024", "2048"])
@pytest.mark.parametrize("conv", ["ttnn", "conv2d_3x3"])
def test_vae_flux2_decoder_conv2d_3x3(
    *, mesh_device: ttnn.MeshDevice, tp_axis: int, h_axis: int, height: int, width: int, conv: str
) -> None:
    tt_model, tt_inp, torch_output, _ccl_manager = _decoder_case(mesh_device, tp_axis, h_axis, height, width, conv)
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
        "part": "decoder",
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
    if conv == "conv2d_3x3":
        assert on_ttnn == 0, f"{on_ttnn} 3x3 convs ran on ttnn.conv2d"


@pytest.mark.parametrize(
    "mesh_device, device_params, tp_axis, h_axis",
    [
        # On one chip ttnn.conv2d DRAM-slices the 2048² decoder's convs, and its trace outgrows TRACE_REGION_SIZE.
        [(1, 1), {**single_device_params_flux2, "trace_region_size": 4 * TRACE_REGION_SIZE}, 0, 1],
        [(2, 2), {**line_params_flux2, "trace_region_size": TRACE_REGION_SIZE}, 1, 0],
        [(2, 4), {**line_params_flux2, "trace_region_size": TRACE_REGION_SIZE}, 1, 0],
        [(4, 8), {**line_params_flux2, "trace_region_size": TRACE_REGION_SIZE}, 0, 1],
    ],
    ids=["1x1", "bh2x2", "bh2x4", "bh4x8"],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(("height", "width"), [(1024, 1024), (2048, 2048)], ids=["1024", "2048"])
@pytest.mark.parametrize("conv", ["ttnn", "conv2d_3x3"])
def test_vae_flux2_decoder_conv2d_3x3_traced(
    *, mesh_device: ttnn.MeshDevice, tp_axis: int, h_axis: int, height: int, width: int, conv: str
) -> None:
    """The production decode: pipeline_flux2.py replays a trace of the decoder's forward (Tracer). Latency is
    TRACED_DECODES replays timed one at a time; throughput is the same replays issued back to back."""
    tt_model, tt_inp, torch_output, _ccl_manager = _decoder_case(mesh_device, tp_axis, h_axis, height, width, conv)
    # As in the pipeline, the size runs untraced before the capture: buffers the first run allocates and keeps (conv
    # weights, CCL buffers) would otherwise be allocated after the capture and overwritten by its replays.
    ttnn.deallocate(tt_model.forward(tt_inp))
    ttnn.synchronize_device(mesh_device)
    if _in_collect_pass():
        return

    tracer = Tracer(tt_model.forward, device=mesh_device, clone_prep_inputs=False)
    tt_out = tracer(tt_inp)
    ttnn.synchronize_device(mesh_device)
    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)

    latency_ms = []
    for _ in range(TRACED_DECODES):
        t0 = time.perf_counter()
        tracer(tt_inp)
        ttnn.synchronize_device(mesh_device)
        latency_ms.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    for _ in range(TRACED_DECODES):
        tracer(tt_inp, tracer_blocking_execution=False)
    ttnn.synchronize_device(mesh_device)
    fps = TRACED_DECODES / (time.perf_counter() - t0)
    tracer.release_trace()

    on_3x3, on_ttnn = _conv_paths(tt_model)
    pcc = torch.corrcoef(torch.stack([torch_output.flatten().double(), tt_out_torch.flatten().double()]))[0, 1].item()
    record = {
        "part": "decoder-traced",
        "mesh": list(mesh_device.shape),
        "image": [height, width],
        "conv": conv,
        "traced_ms_median": statistics.median(latency_ms),
        "traced_ms": latency_ms,
        "traced_fps": fps,
        "pcc": pcc,
        "convs_on_conv2d_3x3": on_3x3,
        "convs_on_ttnn_conv2d": on_ttnn,
    }
    with open(RESULTS, "a") as fh:
        fh.write(json.dumps(record) + "\n")
    print("FLUX_E2E:", json.dumps(record))

    assert_quality(torch_output, tt_out_torch, pcc=0.995)
    if conv == "conv2d_3x3":
        assert on_ttnn == 0, f"{on_ttnn} 3x3 convs ran on ttnn.conv2d"


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
def test_vae_flux2_encoder_conv2d_3x3(
    *, mesh_device: ttnn.MeshDevice, tp_axis: int, h_axis: int, height: int, width: int, conv: str
) -> None:
    torch.manual_seed(0)
    torch_model = reference.AutoencoderKLFlux2()
    torch_model.eval()

    ccl_manager = CCLManager(mesh_device, topology=ttnn.Topology.Linear)
    vae_parallel_config = Flux2VaeParallelConfig.from_axes(mesh_device, tp_axis=tp_axis, h_axis=h_axis, w_axis=None)

    tt_model = Flux2VaeEncoder(
        in_channels=torch_model.config.in_channels,
        block_out_channels=list(torch_model.config.block_out_channels),
        layers_per_block=torch_model.config.layers_per_block,
        z_channels=torch_model.config.latent_channels,
        device=mesh_device,
        parallel_config=vae_parallel_config,
        ccl_manager=ccl_manager,
    )
    configure_conv2d(tt_model, use_conv2d_3x3=conv == "conv2d_3x3")
    _auto_slice_strided(tt_model, tuple(mesh_device.shape))
    tt_model.load_torch_state_dict(torch_model.state_dict())

    image = torch.rand(1, 3, height, width) * 2 - 1
    with torch.no_grad():
        torch_output = torch_model.encode(image).latent_dist.mode()

    h_mesh_axis = vae_parallel_config.h_parallel.mesh_axis if vae_parallel_config.h_parallel is not None else None
    tt_image = tensor.from_torch(
        image.permute(0, 2, 3, 1),
        device=mesh_device,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        mesh_axes=[None, h_mesh_axis, None, None],
    )
    tt_out = tt_model.forward(tt_image)
    ttnn.synchronize_device(mesh_device)
    tt_out_torch = tensor.to_torch(tt_out).permute(0, 3, 1, 2)
    ttnn.deallocate(tt_out)

    wall_ms = []
    for _ in range(MEASURED_DECODES):
        t0 = time.perf_counter()
        tt_out = tt_model.forward(tt_image)
        ttnn.synchronize_device(mesh_device)
        wall_ms.append((time.perf_counter() - t0) * 1000)
        ttnn.deallocate(tt_out)

    on_3x3, on_ttnn = _conv_paths(tt_model)
    pcc = torch.corrcoef(torch.stack([torch_output.flatten().double(), tt_out_torch.flatten().double()]))[0, 1].item()
    record = {
        "part": "encoder",
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

    assert_quality(torch_output, tt_out_torch, pcc=0.995)
    if conv == "conv2d_3x3":
        assert on_ttnn == 0, f"{on_ttnn} 3x3 convs ran on ttnn.conv2d"
