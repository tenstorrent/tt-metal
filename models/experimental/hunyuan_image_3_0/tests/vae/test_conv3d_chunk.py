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
from models.experimental.hunyuan_image_3_0.tt.vae.conv3d import HunyuanSymmetricConv3d


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


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
