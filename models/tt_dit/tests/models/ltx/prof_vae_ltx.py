# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-time profiling for the LTX-2 video VAE decoder at production length.

The standalone parity test (test_vae_ltx.py) builds a torch reference and runs
it on host; at 145 frames / 1080p that forward needs tens of GB of host RAM and
is impractical. These harnesses build the torch decoder only to source random
weights, load them into the TT decoder, and run the TT forward alone — so the
shape matches production (decode_latents calls the decoder once with all 19
latent frames) without paying for the host reference.

Defaults: 145 frames @ 1088x1920 on a 2x4 mesh (h_axis=0, w_axis=1), the
distilled-pipeline production config. Override via NUM_FRAMES / HEIGHT / WIDTH.
"""

import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.vae_ltx import LTXCausalConv3d, LTXDepthToSpaceUpsample, LTXResnetBlock3D, LTXVideoDecoder
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
# LTX_USE_FUSED=0 forces the all-standalone baseline; 1 (default) lets the per-shape
# win-table route the conv-heavy s3/s4 layers through the fused neighbor_pad_conv3d op.
_USE_FUSED = os.environ.get("LTX_USE_FUSED", "1") != "0"


def _walk(m):
    yield m
    for _, c in m.named_children():
        yield from _walk(c)


def _build_tt_decoder(mesh):
    """Production decoder with random weights; torch module built for its
    state_dict only (no host forward)."""
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

    # Match the BH (2,4) production config (create_pipeline device_configs): num_links=2.
    ccl_manager = CCLManager(mesh, topology=ttnn.Topology.Linear, num_links=2)
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
        use_fused=_USE_FUSED,
    )
    tt_decoder.load_torch_state_dict(_diffusers_decoder_state_to_tt(torch_decoder.state_dict()))
    return tt_decoder


def _latent():
    latent_frames = (_NUM_FRAMES - 1) // 8 + 1
    return torch.randn(1, 128, latent_frames, _HEIGHT // 32, _WIDTH // 32, dtype=torch.float32)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_per_conv(mesh_device, device_params):
    # Per-conv device time (sync-wrapped wall), with the C_in_block each conv3d got.
    # Identifies the heaviest conv shapes and whether their blocking is the generic fallback.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)

    rows = {}

    def wrap(mod):
        orig = mod.forward
        key0 = (mod.unpadded_in_channels, mod.unpadded_out_channels, mod.kernel_size, mod.stride)
        cib = getattr(mod.conv_config, "C_in_block", None)

        def timed(x, *a, **k):
            ttnn.synchronize_device(mod.mesh_device)
            t = time.perf_counter()
            r = orig(x, *a, **k)
            ttnn.synchronize_device(mod.mesh_device)
            dt = (time.perf_counter() - t) * 1000
            rows.setdefault(key0 + (int(x.shape[1]), cib), []).append(dt)
            return r

        mod.forward = timed

    for mod in _walk(tt_decoder):
        if isinstance(mod, LTXCausalConv3d):
            wrap(mod)

    latent = _latent()
    _ = tt_decoder(latent)  # warm program cache
    rows.clear()
    _ = tt_decoder(latent)

    print(f"\n===== LTX VAE per-conv device time ({_NUM_FRAMES}f@{_HEIGHT}x{_WIDTH}) =====", flush=True)
    print(f"{'ms/fwd':>8} {'Cin':>5} {'Cout':>5} {'kernel':>10} {'stride':>10} {'Tin':>4} {'Cblk':>5}", flush=True)
    agg = []
    for (cin, cout, k, s, tin, cib), v in rows.items():
        agg.append((sum(v) / len(v), cin, cout, k, s, tin, cib))
    total = 0.0
    for ms, cin, cout, k, s, tin, cib in sorted(agg, reverse=True):
        total += ms
        print(f"{ms:8.2f} {cin:5} {cout:5} {str(k):>10} {str(s):>10} {tin:4} {str(cib):>5}", flush=True)
    print(f"\nConv total: {total:.1f} ms/fwd", flush=True)


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_devicetime(mesh_device, device_params):
    # Run under: python -m tracy -p -r -m pytest <this>::test_prof_vae_ltx_devicetime
    # Flushes the on-device profiler after each resnet/upsample block to stay under the
    # 1000-zone tracy buffer; the CSV then holds every op's DEVICE FW DURATION. Sum over one
    # forward = true device-active time. Host wall is printed for the dispatch-bound fraction.
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    tt_decoder = _build_tt_decoder(mesh)

    for mod in _walk(tt_decoder):
        if isinstance(mod, (LTXResnetBlock3D, LTXDepthToSpaceUpsample)):
            orig = mod.forward

            def timed(*a, _orig=orig, **k):
                r = _orig(*a, **k)
                ttnn.ReadDeviceProfiler(mesh)
                return r

            mod.forward = timed

    latent = _latent()
    _ = tt_decoder(latent)  # warm program cache
    ttnn.synchronize_device(mesh)
    ttnn.ReadDeviceProfiler(mesh)
    # tracy signpost
    from tracy import signpost

    signpost("start")
    t0 = time.perf_counter()
    _ = tt_decoder(latent)
    ttnn.synchronize_device(mesh)
    host_wall = (time.perf_counter() - t0) * 1000
    ttnn.ReadDeviceProfiler(mesh)
    signpost("stop")
    print(f"\nSINGLE_FORWARD_HOST_WALL_MS={host_wall:.2f}", flush=True)


_TRACE_ITERS = int(os.environ.get("TRACE_ITERS", "10"))


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
@pytest.mark.parametrize("mesh_device", [(2, 4)], indirect=True)
def test_prof_vae_ltx_trace(mesh_device, device_params):
    """TRUE traced-decode WALL: capture the device-only decode as one ttnn trace and replay it, so the
    wall/iter is the real e2e device time with no host-dispatch gap. This is the metric the per-shape
    routing should be derived from (device-FW MIN is optimistic, the isolated-op bench is pessimistic).

    LTX_USE_FUSED=1 (default) = hybrid per-shape routing; =0 = all-standalone. Run both to compare.
    """
    mesh = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
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

    mode = "hybrid(fused)" if _USE_FUSED else "all-standalone"
    print(
        f"\nTRACED_DECODE_WALL_MS={wall_ms:.2f}  mode={mode}  frames={_NUM_FRAMES}  {_HEIGHT}x{_WIDTH}  "
        f"iters={_TRACE_ITERS}",
        flush=True,
    )
