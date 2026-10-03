# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ceiling of a fused neighbor_pad+conv3d on the halo-only conv VAE decode, 2x4 submesh at 544x960/145f.

A fused op can at best hide the halo exchange behind conv3d, and it gives up conv cores to the fabric
kernels. This times one decoder in four interleaved arms:
  base         halo-only path as shipped
  skip_routed  no halo exchange on the convs the fused router would take (_HALO_LAST_KEYS/_FORCE_SPATIAL_KEYS);
               conv3d reads the stale halo buffer from the last real exchange of that shape
  skip_all     no halo exchange on any conv
  grid_routed  routed convs on one compute-grid column fewer (the fused op reserves 8 cores for the fabric)
Skip arms output wrong border pixels by design; only their wall time means anything. Run with
LTX_CONV3D_BLOCKING_MESH=4,8 so the per-chip shapes hit the production blockings and routing keys.
"""

import hashlib
import time

import pytest
import torch

import ttnn
from models.tt_dit.tests.models.ltx.test_vae_ltx_fold_time_pad_ab import NF, TIMED_DECODES, H, W, _latent

ARMS = ("base", "skip_routed", "skip_all", "grid_routed")


def _md5(t: torch.Tensor) -> str:
    return hashlib.md5(t.contiguous().numpy().tobytes()).hexdigest()


def _psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (a.double() - b.double()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0**2 / mse)).item()


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_np_ceiling_ab(mesh_device, device_params):
    # A bare 2x4 open on the BH galaxy fails fabric init; open the system mesh and carve the 2x4.
    mesh_device = mesh_device.create_submesh(ttnn.MeshShape(2, 4))
    from models.tt_dit.models.vae import vae_ltx
    from models.tt_dit.parallel.config import ParallelFactor, VaeHWParallelConfig
    from models.tt_dit.parallel.manager import CCLManager
    from models.tt_dit.tests.models.ltx.test_vae_ltx import (
        _LTX_PROD_DECODER_BLOCKS,
        _diffusers_decoder_state_to_tt,
        _require_diffusers_ltx_vae,
        _TorchLTXVideoDecoder,
    )
    from models.tt_dit.utils import conv3d as conv3d_utils

    # Record each conv's effective blocking key (after the LTX_CONV3D_BLOCKING_MESH override) by config id.
    keys_by_cfg = {}
    orig_get_cfg = vae_ltx.get_conv3d_config

    def get_cfg(in_channels, out_channels, kernel_size, weights_dtype, grid_size, *, h_factor=1, w_factor=1, **dims):
        cfg = orig_get_cfg(
            in_channels,
            out_channels,
            kernel_size,
            weights_dtype,
            grid_size,
            h_factor=h_factor,
            w_factor=w_factor,
            **dims,
        )
        T, Hd, Wd = dims.get("T", 0), dims.get("H", 0), dims.get("W", 0)
        hf, wf = conv3d_utils._blocking_mesh_override(
            h_factor, w_factor, in_channels, out_channels, kernel_size, T, Hd, Wd
        )
        keys_by_cfg[id(cfg)] = (hf, wf, in_channels, out_channels, kernel_size, T, Hd, Wd)
        return cfg

    vae_ltx.get_conv3d_config = get_cfg

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
    pc = VaeHWParallelConfig(
        height_parallel=ParallelFactor(factor=2, mesh_axis=0),
        width_parallel=ParallelFactor(factor=4, mesh_axis=1),
    )
    ccl = CCLManager(mesh_device, topology=ttnn.Topology.Linear, num_links=2)
    try:
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
            ccl_manager=ccl,
        )
    finally:
        vae_ltx.get_conv3d_config = orig_get_cfg
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))

    routed_keys = set(conv3d_utils._HALO_LAST_KEYS) | set(conv3d_utils._FORCE_SPATIAL_KEYS)
    convs = [m for m in conv3d_utils._walk_conv3d_modules(dec) if isinstance(m, vae_ltx.LTXCausalConv3d)]
    for m in convs:
        m._t98_key = keys_by_cfg.get(id(m.conv_config))
        m._t98_routed = m._t98_key in routed_keys
    print(f"NP_CEIL convs={len(convs)} routed={sum(m._t98_routed for m in convs)}")
    for k in sorted({m._t98_key for m in convs}, key=str):
        n = sum(m._t98_key == k for m in convs)
        print(f"NP_CEIL key={k} n={n} routed={k in routed_keys}")

    gx, gy = mesh_device.compute_with_storage_grid_size().x, mesh_device.compute_with_storage_grid_size().y
    full_grid, small_grid = ttnn.CoreCoord(gx, gy), ttnn.CoreCoord(gx - 1, gy)
    st = {"arm": "base", "routed": False}
    halo_cache = {}
    orig_np = ccl.neighbor_pad_halo_only

    def np_halo(tensor, **kw):
        key = (tuple(tensor.shape), tuple(kw["dims"]), tuple(kw["pad_left"]), tuple(kw["pad_right"]))
        skip = st["arm"] == "skip_all" or (st["arm"] == "skip_routed" and st["routed"])
        if skip and key in halo_cache:
            return halo_cache[key]
        halo = orig_np(tensor, **kw)
        halo_cache[key] = halo
        return halo

    ccl.neighbor_pad_halo_only = np_halo
    orig_fwd = vae_ltx.LTXCausalConv3d.forward

    def conv_fwd(self, *a, **k):
        st["routed"] = self._t98_routed
        return orig_fwd(self, *a, **k)

    vae_ltx.LTXCausalConv3d.forward = conv_fwd

    def set_arm(arm):
        st["arm"] = arm
        grid = small_grid if arm == "grid_routed" else full_grid
        for m in convs:
            if m._t98_routed:
                m.conv_config.compute_with_storage_grid_size = grid

    lat = _latent()
    outs, times, md5s = {}, {a: [] for a in ARMS}, {a: set() for a in ARMS}
    try:
        for arm in ARMS:  # warmup per arm; base first so every halo shape has a real buffer cached
            set_arm(arm)
            outs[arm] = torch.as_tensor(dec(lat, output_type="yuv")).clone()
            md5s[arm].add(_md5(outs[arm]))
        for _ in range(TIMED_DECODES):
            for arm in ARMS:
                set_arm(arm)
                ttnn.synchronize_device(mesh_device)
                t0 = time.perf_counter()
                out = dec(lat, output_type="yuv")
                ttnn.synchronize_device(mesh_device)
                times[arm].append(time.perf_counter() - t0)
                outs[arm] = torch.as_tensor(out).clone()
                md5s[arm].add(_md5(outs[arm]))
    finally:
        set_arm("base")
        vae_ltx.LTXCausalConv3d.forward = orig_fwd
        ccl.neighbor_pad_halo_only = orig_np

    base = outs["base"]
    for arm in ARMS:
        t = times[arm]
        same = outs[arm].shape == base.shape and torch.equal(outs[arm], base)
        print(
            f"AB arm={arm} decode_s={' '.join(f'{x:.4f}' for x in t)} min={min(t):.4f}"
            f" delta_min_ms={(min(t) - min(times['base'])) * 1e3:.1f} identical_to_base={same}"
            f" psnr_vs_base={_psnr(outs[arm], base):.2f} md5={sorted(md5s[arm])}"
        )
    assert len(md5s["base"]) == 1, "run-to-run base output changed"
