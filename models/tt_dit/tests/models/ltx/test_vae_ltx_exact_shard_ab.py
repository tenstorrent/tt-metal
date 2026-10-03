# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Padded vs exact-shard (LTX_VAE_EXACT_SHARD) conv VAE decode on a 2x4 submesh at 544x960/145f, one process.

On 2x4 this gives each chip the same shard as 1088x1920/145f on the production 4x8. One decoder; the arm is
toggled through ``dec.exact_shard``. Both arms are warmed up, then timed interleaved (pad, exact, ...). Prints
AB timing lines and the md5 of each arm's yuv output, saves $AB_OUT_DIR/yuv_{pad,exact}.pt and asserts the
outputs are identical.
"""

import hashlib
import os
import time

import pytest
import torch

import ttnn
from models.tt_dit.tests.models.ltx.test_vae_ltx_fold_time_pad_ab import NF, TIMED_DECODES, H, W, _latent

ARMS = (("pad", False), ("exact", True))


def _md5(t: torch.Tensor) -> str:
    return hashlib.md5(t.contiguous().numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("device_params", [{"fabric_config": ttnn.FabricConfig.FABRIC_1D}], indirect=True)
@pytest.mark.parametrize("mesh_device", [(4, 8)], indirect=True)
def test_vae_ltx_exact_shard_ab(mesh_device, device_params):
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
    dec.load_torch_state_dict(_diffusers_decoder_state_to_tt(state))

    lat = _latent()
    outs, times, md5s = {}, {a: [] for a, _ in ARMS}, {a: set() for a, _ in ARMS}
    for arm, exact in ARMS:  # warmup: kernel compile + program cache per arm
        dec.exact_shard = exact
        outs[arm] = torch.as_tensor(dec(lat, output_type="yuv")).clone()
        md5s[arm].add(_md5(outs[arm]))
    for _ in range(TIMED_DECODES):
        for arm, exact in ARMS:
            dec.exact_shard = exact
            ttnn.synchronize_device(mesh_device)
            t0 = time.perf_counter()
            out = dec(lat, output_type="yuv")
            ttnn.synchronize_device(mesh_device)
            times[arm].append(time.perf_counter() - t0)
            outs[arm] = torch.as_tensor(out).clone()
            md5s[arm].add(_md5(outs[arm]))
    for arm, _ in ARMS:
        t = times[arm]
        print(f"AB arm={arm} decode_s={' '.join(f'{x:.4f}' for x in t)} min={min(t):.4f} md5={sorted(md5s[arm])}")

    out_dir = os.environ.get("AB_OUT_DIR")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        for arm, _ in ARMS:
            torch.save(outs[arm], os.path.join(out_dir, f"yuv_{arm}.pt"))
    pad, exact = outs["pad"], outs["exact"]
    same = pad.shape == exact.shape and torch.equal(pad, exact)
    diff = (pad.int() - exact.int()).abs().max().item() if pad.shape == exact.shape else "shape"
    print(
        f"AB_CMP exact_vs_pad identical={same} max_abs_diff={diff} shape={tuple(exact.shape)} dtype={exact.dtype}"
        f" delta_min_ms={(min(times['exact']) - min(times['pad'])) * 1e3:.1f}"
    )
    assert len(md5s["pad"]) == 1 and len(md5s["exact"]) == 1, "run-to-run output changed"
    assert same
