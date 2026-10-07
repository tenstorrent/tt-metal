# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One arm of the #56023 A/B: traced conv VAE decode on the full 4x8 mesh at 1088x1920/145f (production shards).

The arm is the tree (TT_METAL_HOME), so firmware and dispatch kernels build from that tree's headers. With no
stored reference and LTX_VAE_REF_RECORD=1, a halo-off decode records it first. Then the default decoder runs
warmup, eager decodes, capture and traced replays; prints AB lines, checks traced against eager and the
reference, and saves $AB_OUT_DIR/yuv_traced.pt.
"""

import gc
import os
import time

import pytest
import torch

import ttnn

NF, H, W = 145, 1088, 1920
EAGER_DECODES = 3
TRACED_DECODES = 8
TRACE_REGION = int(os.environ.get("LTX_TRACE_REGION", "500000000"))


def _latent():
    # Saved 2.5 stage-2 latent at 1080p: T=19, H=34, W=60, packed tokens.
    v = torch.load(os.environ["AB_LATENT"], map_location="cpu")["video"].float().reshape(1, 19, 34, 60, 128)
    return v.permute(0, 4, 1, 2, 3).contiguous()


def _timed(mesh_device, fn, n):
    times = []
    for _ in range(n):
        ttnn.synchronize_device(mesh_device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(mesh_device)
        times.append(time.perf_counter() - t0)
    return times


def _line(arm, ts):
    med = sorted(ts)[len(ts) // 2]
    return f"AB arm={arm} decode_s={' '.join(f'{t:.4f}' for t in ts)} min={min(ts):.4f} med={med:.4f}"


@pytest.mark.parametrize(
    "device_params",
    [{"fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": TRACE_REGION}],
    indirect=True,
)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_t133_vae_4x8(mesh_device, device_params):
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )
    from models.tt_dit.tests.models.ltx.tools.vae_ref_check import (
        check_against_reference,
        latent_md5,
        reference_path,
    )

    torch.manual_seed(42)
    tdec = _TorchLTXVideoDecoder(
        decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
        in_channels=128,
        out_channels=3,
        patch_size=4,
        base_channels=128,
        causal=False,
        spatial_padding_mode="zeros",
        vae_mods=_require_diffusers_ltx_vae(),
    ).eval()
    state = tdec.state_dict()
    state["per_channel_statistics.mean-of-means"] = torch.zeros(128)
    state["per_channel_statistics.std-of-means"] = torch.ones(128)
    tt_state = _diffusers_decoder_state_to_tt(state)
    lat = _latent()
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=4, mesh_axis=0),
        width_parallel=ParallelFactor(factor=8, mesh_axis=1),
    )

    def build():
        dec = vae_ltx.LTXVideoDecoder(
            decoder_blocks=_LTX_PROD_DECODER_BLOCKS,
            in_channels=128,
            out_channels=3,
            patch_size=4,
            base_channels=128,
            causal=False,
            num_frames=NF,
            height=H,
            width=W,
            mesh_device=mesh_device,
            parallel_config=pc,
            ccl_manager=CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2),
        )
        dec.load_torch_state_dict(tt_state)
        return dec

    def check(out, arm, dec):
        check_against_reference(
            out, arm, latent=lat, num_frames=NF, height=H, width=W, mesh_shape=(4, 8), exact_shard=dec.exact_shard
        )

    # The halo path is fixed at conv construction, so the env switch must precede build().
    if not os.path.exists(reference_path(H, W, NF, latent_md5(lat))) and os.environ.get("LTX_VAE_REF_RECORD") == "1":
        os.environ["LTX_VAE_HALO_ONLY"] = "0"
        dec = build()
        check(torch.as_tensor(dec(lat, output_type="yuv")).clone(), "halo_off", dec)
        del dec
        gc.collect()
        os.environ.pop("LTX_VAE_REF_RECORD")

    os.environ["LTX_VAE_HALO_ONLY"] = "1"
    dec = build()
    dec(lat, output_type="yuv")  # compile + program cache
    t_eager = _timed(mesh_device, lambda: dec(lat, output_type="yuv"), EAGER_DECODES)
    print(_line("eager", t_eager), flush=True)
    eager = torch.as_tensor(dec(lat, output_type="yuv")).clone()
    t_capture = _timed(mesh_device, lambda: dec(lat, output_type="yuv", traced=True), 1)
    print(f"AB capture_s={t_capture[0]:.4f}", flush=True)
    t_traced = _timed(mesh_device, lambda: dec(lat, output_type="yuv", traced=True), TRACED_DECODES)
    print(_line("traced", t_traced), flush=True)
    traced = torch.as_tensor(dec(lat, output_type="yuv", traced=True)).clone()
    dec.release_trace()

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(traced, os.path.join(out_dir, "yuv_traced.pt"))
    same = traced.shape == eager.shape and torch.equal(traced, eager)
    print(f"AB_CMP traced_vs_eager identical={same} shape={tuple(traced.shape)} dtype={traced.dtype}", flush=True)
    check(traced, "traced", dec)
    assert same
