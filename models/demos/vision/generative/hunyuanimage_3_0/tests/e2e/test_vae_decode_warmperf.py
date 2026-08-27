# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated WARM-decode A/B harness for the on-device VAE decode (norm-lever A/B).

Times ``VaeDecoder._forward`` at real 1024^2 resolution on the (8,4) mesh, 3 iterations:
iter 1 = cold (compile), iters 2..N = WARM (the number to compare). Also PCC-gates the last
warm output vs the torch decoder so a perf run doubles as a correctness check.

Gather-baseline (pristine vae_decode.py) warm floor was ~16.2s. Run this against pristine and
against the distributed-GroupNorm version, same box/session, for an apples-to-apples A/B.

Run (on box):
  python -m pytest -o timeout=0 -s \
    models/demos/vision/generative/hunyuanimage_3_0/tests/e2e/test_vae_decode_warmperf.py
"""
from __future__ import annotations

import time

import pytest
import torch

import ttnn
from models.demos.vision.generative.hunyuanimage_3_0.tt.vae_decode import (
    MeshCtx,
    VaeDecoder,
    gather_output_ncthw,
    reference_autoencoder_path,
    shard_input_nthwc,
)
from tests.ttnn.utils_for_testing import check_with_pcc

# same fixture params as the PCC ladder's mesh tests
MESH_DEV_PARAMS = [
    {"l1_small_size": 24576, "fabric_config": ttnn.FabricConfig.FABRIC_1D, "trace_region_size": 20000000}
]

# HF ref loaded lazily from the same snapshot the PCC test uses
import importlib.util
import sys


def _load_hf_ref():
    path = reference_autoencoder_path()
    spec = importlib.util.spec_from_file_location("akl3d_ref_perf", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["akl3d_ref_perf"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("device_params", MESH_DEV_PARAMS, indirect=True)
@pytest.mark.parametrize("mesh_device", [(8, 4)], indirect=True)
def test_warm_decode_fullres(device_params, mesh_device):
    HF = _load_hf_ref()
    dev = mesh_device
    hf, wf = int(dev.shape[0]), int(dev.shape[1])
    torch.manual_seed(0)
    ctx = MeshCtx(dev, h_factor=hf, h_axis=0, w_factor=wf, w_axis=1, num_links=1)
    cfg = dict(
        in_channels=3,
        out_channels=3,
        latent_channels=32,
        block_out_channels=[128, 256, 512, 1024, 1024],
        layers_per_block=2,
        ffactor_spatial=16,
        ffactor_temporal=4,
        sample_size=384,
        sample_tsize=96,
        scaling_factor=0.562679178327931,
    )
    vae = HF.AutoencoderKLConv3D(**cfg).eval()
    dec = vae.decoder
    z = torch.randn(1, 32, 1, 64, 64)
    with torch.no_grad():
        gt = dec(z)  # [1,3,4,1024,1024]

    tt = VaeDecoder(dec, dev, latent_t=1, latent_h=64, latent_w=64, ctx=ctx)
    zt = shard_input_nthwc(z, ctx)

    N = 3
    times = []
    yt = None
    for i in range(N):
        ttnn.synchronize_device(dev)
        t0 = time.time()
        yt = tt._forward(zt)
        ttnn.synchronize_device(dev)
        dt = time.time() - t0
        times.append(dt)
        print(f"WARMPERF iter{i} forward_s={dt:.2f} ({'COLD' if i == 0 else 'warm'})", flush=True)

    warm = min(times[1:]) if len(times) > 1 else times[0]
    y = gather_output_ncthw(yt, ctx, 3, 1024, 1024)
    ok, msg = check_with_pcc(gt, y, pcc=0.99)
    print(
        f"\nWARMPERF_RESULT warm_forward_s={warm:.2f} cold_s={times[0]:.2f} all={['%.2f' % t for t in times]} "
        f"PCC {msg} -> {'OK' if ok else 'FAIL'}",
        flush=True,
    )
    assert ok, msg
