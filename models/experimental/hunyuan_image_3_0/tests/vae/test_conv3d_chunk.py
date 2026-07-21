# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fast equivalence test: H-chunked conv3d == unchunked, same weights.

Isolates the chunked-conv math (used to fit the full-res VAE decode on device)
from the rest of the decoder. Builds one HunyuanSymmetricConv3d, runs it
unchunked and chunked on the same input/weights, compares.

Run:
  python_env/bin/python -m pytest \
    models/experimental/hunyuan_image_3_0/tests/vae/test_conv3d_chunk.py -v -s --timeout=600
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
import models.experimental.hunyuan_image_3_0.tt.vae.conv3d as conv3d_mod
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import (
    HunyuanSymmetricConv3d,
    _TAIL_CONV_OUT_CHUNK_ELEMS,
    conv3d_h_chunk_size,
    conv3d_h_chunk_size_for_conv,
)
from models.experimental.hunyuan_image_3_0.tt.vae.decoder import ConvOutTTNN
from models.experimental.hunyuan_image_3_0.tt.vae.spatial import enable_vae_spatial
from models.tt_dit.parallel.manager import CCLManager


def test_tail_conv_out_chunk_pins_1024m_cap():
    """Tail conv_out keeps 1024M chunk strips even when global threshold is raised."""
    in_ch = 128
    t, h, w = 4, 512, 512
    hc_tail = conv3d_h_chunk_size(
        t=t,
        h=h,
        w=w,
        in_channels=in_ch,
        valid_conv=True,
        chunk_elems=_TAIL_CONV_OUT_CHUNK_ELEMS,
    )
    hc_global_1280 = conv3d_h_chunk_size(
        t=t,
        h=h,
        w=w,
        in_channels=in_ch,
        valid_conv=True,
        chunk_elems=1280 * 1024 * 1024,
    )
    assert hc_tail == 128
    assert hc_global_1280 == 170
    assert hc_tail < hc_global_1280


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv_out_sharded_pins_tail_chunk_override(mesh_device):
    """ConvOutTTNN sets _h_chunk_override from the tail chunk cap before conv."""
    mesh_device.enable_program_cache()
    ccl = CCLManager(mesh_device, num_links=1, topology=ttnn.Topology.Linear)
    tail_t, tail_h, tail_w, tail_c = 4, 1024, 1024, 128
    tt = ConvOutTTNN(tail_c, mesh_device, t=tail_t, h=tail_h, w=tail_w)
    torch.manual_seed(0)
    tt.conv.load_torch_state_dict({"weight": torch.randn(3, tail_c, 3, 3, 3) * 0.01, "bias": torch.randn(3) * 0.01})
    enable_vae_spatial(tt, ccl, h_mesh_axis=0, w_mesh_axis=1)

    x = torch.randn(1, tail_t, tail_h, tail_w, tail_c) * 0.1
    x_tt = ttnn.from_torch(
        x,
        dtype=tt.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=tuple(mesh_device.shape), dims=[2, 3]),
    )

    saved_global = conv3d_mod._CONV3D_CHUNK_ELEMS
    try:
        conv3d_mod._CONV3D_CHUNK_ELEMS = 1280 * 1024 * 1024
        tt(x_tt)
        override = tt.conv._h_chunk_override
    finally:
        conv3d_mod._CONV3D_CHUNK_ELEMS = saved_global

    expected = conv3d_h_chunk_size_for_conv(
        tt.conv,
        t=tail_t,
        h=tail_h // 2,
        w=tail_w // 2,
        chunk_elems=_TAIL_CONV_OUT_CHUNK_ELEMS,
    )
    assert override == expected
    assert override == 128


@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_conv3d_chunk_equiv(mesh_device):
    mesh_device.enable_program_cache()
    B, T, H, W, Cin, Cout = 1, 4, 64, 64, 32, 32

    conv = HunyuanSymmetricConv3d(Cin, Cout, kernel_size=3, stride=1, padding=1, mesh_device=mesh_device, t=T, h=H, w=W)

    # Random but FIXED weights (same for both paths), in PyTorch Conv3d format.
    torch.manual_seed(0)
    conv.load_torch_state_dict({"weight": torch.randn(Cout, Cin, 3, 3, 3) * 0.05, "bias": torch.randn(Cout) * 0.05})

    x = torch.randn(B, T, H, W, Cin) * 0.1
    x_tt = ttnn.from_torch(
        x,
        dtype=conv.dtype,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )

    def run():
        out = conv(x_tt)
        return ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:B].float()

    saved = conv3d_mod._CONV3D_CHUNK_ELEMS
    try:
        conv3d_mod._CONV3D_CHUNK_ELEMS = 10**18  # no chunk
        out_full = run()
        conv3d_mod._CONV3D_CHUNK_ELEMS = 1  # force chunk over H
        out_chunk = run()
    finally:
        conv3d_mod._CONV3D_CHUNK_ELEMS = saved

    assert out_full.shape == out_chunk.shape, f"{out_full.shape} vs {out_chunk.shape}"
    passing, pcc = comp_pcc(out_full, out_chunk, 0.999)
    logger.info(f"conv3d chunked-vs-full PCC: {pcc:.6f}  shape={tuple(out_chunk.shape)}")
    assert passing, f"PCC {pcc:.6f} < 0.999"
