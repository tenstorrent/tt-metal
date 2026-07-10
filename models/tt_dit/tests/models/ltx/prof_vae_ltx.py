# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Traced device-time profiling for the LTX-2 video VAE decoder at production length.

The standalone parity test (test_vae_ltx.py) builds a torch reference and runs it on host; at 145
frames / 1080p that forward needs tens of GB of host RAM and is impractical. These harnesses build the
torch decoder only to source random weights, load them into the TT decoder, and run the TT
``decode_device`` alone — the device-only region of ``LTXVideoDecoder.forward`` (denorm → conv_in →
up_blocks → norm_out → conv_out), with the host upload/gather kept outside so it can be captured as one
ttnn trace.

Defaults: 145 frames @ 1088x1920 on a 4x8 mesh, the distilled-pipeline production config. Override via
NUM_FRAMES / HEIGHT / WIDTH.
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.vae_ltx import LTXVideoDecoder
from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
from models.tt_dit.parallel.manager import CCLManager
from models.tt_dit.tests.models.ltx.test_vae_ltx import (
    _LTX_PROD_DECODER_BLOCKS,
    _diffusers_decoder_state_to_tt,
    _require_diffusers_ltx_vae,
    _TorchLTXVideoDecoder,
)

_NUM_FRAMES = int(os.environ.get("NUM_FRAMES", "145"))
_HEIGHT = int(os.environ.get("HEIGHT", "1088"))
_WIDTH = int(os.environ.get("WIDTH", "1920"))
_TRACE_ITERS = int(os.environ.get("TRACE_ITERS", "10"))


def _build_tt_decoder(mesh):
    """Production decoder with random weights; torch module built for its state_dict only (no host
    forward)."""
    vae_mods = _require_diffusers_ltx_vae()
    if not vae_mods["ltx2"]:
        pytest.skip("LTX-2 decoder profiling requires diffusers autoencoder_kl_ltx2")

    torch.manual_seed(42)
    torch_decoder = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=vae_mods,
    )
    torch_decoder.eval()
    state = torch_decoder.state_dict()
    # Identity per-channel stats so denorm is a no-op (random weights give garbage stats).
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    torch_decoder.load_state_dict(state)

    # Match the BH production config (create_pipeline device_configs): num_links=2.
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear, num_links=int(os.environ.get("NP_LINKS", "2")))
    parallel_config = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=tuple(mesh.shape)[0], mesh_axis=0),
        width_parallel=ParallelFactor(factor=tuple(mesh.shape)[1], mesh_axis=1),
    )
    tt_decoder = LTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        num_frames=_NUM_FRAMES,
        height=_HEIGHT,
        width=_WIDTH,
        mesh_device=mesh,
        parallel_config=parallel_config,
        ccl_manager=ccl_manager,
    )
    tt_decoder.load_torch_state_dict(_diffusers_decoder_state_to_tt(torch_decoder.state_dict()))
    return tt_decoder


def _latent():
    latent_frames = (_NUM_FRAMES - 1) // 8 + 1
    return torch.randn(1, 128, latent_frames, _HEIGHT // 32, _WIDTH // 32, dtype=torch.float32)


def _prepare_decode_input(tt_decoder, latent):
    """Replicate LTXVideoDecoder.forward's host upload → device-sharded sample_tt + logical dims, so the
    device-only decode (decode_device) can be captured as a trace with the host I/O kept outside."""
    from models.tt_dit.utils.conv3d import conv_pad_height, conv_pad_width
    from models.tt_dit.utils.tensor import typed_tensor_2dshard

    pc = tt_decoder.parallel_config
    sample = latent.permute(0, 2, 3, 4, 1)  # (B, T, H, W, C)
    sample, logical_h = conv_pad_height(sample, pc.height_parallel.factor)
    sample, logical_w = conv_pad_width(sample, pc.width_parallel.factor)
    sample_tt = typed_tensor_2dshard(
        sample,
        tt_decoder.mesh_device,
        shard_mapping={pc.height_parallel.mesh_axis: 2, pc.width_parallel.mesh_axis: 3},
        layout=ttnn.ROW_MAJOR_LAYOUT,
        dtype=ttnn.bfloat16,
    )
    return sample_tt, logical_h, logical_w


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prof_vae_ltx_trace(mesh_device, device_params):
    """TRUE traced-decode WALL: capture the device-only decode as one ttnn trace and replay it, so the
    wall/iter is the real e2e device time with no host-dispatch gap — the untraced pipeline decode pays
    that gap on every op."""
    mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 8))
    tt_decoder = _build_tt_decoder(mesh)

    latent = _latent()
    sample_tt, logical_h, logical_w = _prepare_decode_input(tt_decoder, latent)

    # Warmup: cold-compile every program in the decode (trace capture requires cached programs).
    _ = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.synchronize_device(mesh)

    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)

    t0 = time.perf_counter()
    for _ in range(_TRACE_ITERS):
        ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
    ttnn.synchronize_device(mesh)
    wall_ms = (time.perf_counter() - t0) * 1000 / _TRACE_ITERS
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(out)

    print(
        f"\nTRACED_DECODE_WALL_MS={wall_ms:.2f}  frames={_NUM_FRAMES}  {_HEIGHT}x{_WIDTH}  " f"iters={_TRACE_ITERS}",
        flush=True,
    )


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 200000000}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_prof_vae_ltx_trace_pcc(mesh_device, device_params):
    """Trace replay must be bit-identical to eager execution: decode the SAME input once untraced and once
    via capture/replay, gather both to host, and compare. execute_trace replays the exact captured ops, so
    PCC must be ~1.0 — this guards the decode_device split + trace wiring, not model numerics. Set a small
    NUM_FRAMES to keep the one untraced decode cheap."""
    from models.common.utility_functions import comp_pcc
    from models.tt_dit.utils.tensor import fast_device_to_host

    mesh = mesh_device.create_submesh(ttnn.MeshShape(4, 8))
    tt_decoder = _build_tt_decoder(mesh)
    torch.manual_seed(123)
    latent = _latent()
    sample_tt, logical_h, logical_w = _prepare_decode_input(tt_decoder, latent)
    concat_dims = [None, None]
    concat_dims[tt_decoder.parallel_config.height_parallel.mesh_axis] = 2
    concat_dims[tt_decoder.parallel_config.width_parallel.mesh_axis] = 3

    # Eager reference: one untraced decode_device + gather.
    ref_out = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.synchronize_device(mesh)
    ref = fast_device_to_host(ref_out, mesh, concat_dims, ccl_manager=tt_decoder.ccl_manager)
    ttnn.deallocate(ref_out)

    # Traced: warmup (programs already cached), capture, replay, gather.
    _ = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.synchronize_device(mesh)
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    trace_out = tt_decoder.decode_device(sample_tt, logical_h, logical_w)
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    ttnn.synchronize_device(mesh)
    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
    traced = fast_device_to_host(trace_out, mesh, concat_dims, ccl_manager=tt_decoder.ccl_manager)
    ttnn.release_trace(mesh, tid)
    ttnn.deallocate(trace_out)

    ok, pcc = comp_pcc(ref, traced, 0.999)
    print(f"DECODE-TRACE-PCC: pcc={pcc} ({'OK' if ok else 'FAIL'})", flush=True)
    assert ok, f"trace replay diverged from eager decode: {pcc}"
