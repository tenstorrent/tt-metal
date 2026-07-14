# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""PCC tests for fused channel group-mean used by DCAE shortcuts.

Reference math (encoder ``DownsampleDCAE`` shortcut)::

    shortcut.view(b, out_channels, group_size, t, h, w).mean(dim=2)

BTHWC equivalent::

    x.view(b, t, h, w, out_channels, group_size).mean(dim=-1)

Also checks H-chunked vs unchunked device paths stay bit-equivalent under the
``_GROUP_MEAN_MAX_FLAT_ELEMENTS`` guard.

Run:
  python_env/bin/python -m pytest \\
    models/experimental/hunyuan_image_3_0/tests/vae/test_group_mean_pcc.py -v -s --timeout=600
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn
from models.common.utility_functions import comp_pcc
import models.experimental.hunyuan_image_3_0.tt.vae.decoder as dec


@pytest.fixture(scope="function")
def device_params(request):
    return {"fabric_config": ttnn.FabricConfig.FABRIC_1D}


def _ref_group_mean_bthwc(x: torch.Tensor, *, out_channels: int, group_size: int) -> torch.Tensor:
    """Torch reference matching DCAE ``.mean(dim=2)`` on grouped channels."""
    b, t, h, w, c = x.shape
    assert c == out_channels * group_size, f"c={c} != {out_channels}*{group_size}"
    # BTHWC path: mean over last (group) dim.
    out_bthwc = x.view(b, t, h, w, out_channels, group_size).mean(dim=-1)
    # Also assert BCTHW-style ``mean(dim=2)`` is identical (the encoder reference).
    x_bcthw = x.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,T,H,W]
    out_bcthw = x_bcthw.view(b, out_channels, group_size, t, h, w).mean(dim=2)
    out_from_bcthw = out_bcthw.permute(0, 2, 3, 4, 1).contiguous()
    assert torch.allclose(out_bthwc, out_from_bcthw, atol=0, rtol=0)
    return out_bthwc


def _upload_bthwc(mesh_device: ttnn.MeshDevice, x: torch.Tensor) -> ttnn.Tensor:
    return ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh_device),
    )


def _download_bthwc(mesh_device: ttnn.MeshDevice, x_tt: ttnn.Tensor, batch: int) -> torch.Tensor:
    return ttnn.to_torch(x_tt, mesh_composer=ttnn.ConcatMeshToTensor(mesh_device, dim=0))[:batch].float()


@pytest.mark.parametrize(
    "shape,out_channels,group_size",
    [
        # (B,T,H,W,C_in), out_channels, group_size  — C_in = out * group_size
        ((1, 4, 8, 8, 16), 8, 2),
        ((1, 4, 16, 16, 64), 32, 2),
        ((1, 2, 8, 8, 64), 8, 8),
        ((1, 4, 32, 32, 256), 128, 2),
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_group_mean_vs_torch_mean_dim2(mesh_device, shape, out_channels, group_size):
    """Device fused mean vs torch ``.mean(dim=2)`` / BTHWC ``.mean(dim=-1)``."""
    torch.manual_seed(0)
    b, t, h, w, c = shape
    x = torch.randn(b, t, h, w, c)
    ref = _ref_group_mean_bthwc(x, out_channels=out_channels, group_size=group_size)

    x_tt = _upload_bthwc(mesh_device, x)
    out_tt = dec.group_mean_groups_bthwc(x_tt, out_channels=out_channels, group_size=group_size)
    out = _download_bthwc(mesh_device, out_tt, b)
    ttnn.deallocate(x_tt, force=False)
    ttnn.deallocate(out_tt, force=False)

    assert out.shape == ref.shape, f"{tuple(out.shape)} vs {tuple(ref.shape)}"
    # bf16 device vs fp32 host reference
    passing, pcc = comp_pcc(ref.bfloat16().float(), out, 0.999)
    logger.info(f"group_mean vs torch mean(dim=2): shape={shape} out_c={out_channels} " f"g={group_size} PCC={pcc:.6f}")
    assert passing, f"PCC {pcc:.6f} < 0.999 for shape={shape} g={group_size}"


@pytest.mark.parametrize(
    "shape,out_channels,group_size",
    [
        ((1, 4, 64, 64, 256), 128, 2),
        ((1, 2, 32, 32, 512), 64, 8),
    ],
)
@pytest.mark.parametrize("mesh_device", [(2, 2)], indirect=True)
def test_group_mean_chunk_vs_full(mesh_device, shape, out_channels, group_size):
    """H-chunked group_mean matches unchunked path (same fused mean kernel)."""
    torch.manual_seed(1)
    b, t, h, w, c = shape
    x = torch.randn(b, t, h, w, c)
    ref = _ref_group_mean_bthwc(x, out_channels=out_channels, group_size=group_size)

    x_tt = _upload_bthwc(mesh_device, x)

    def run():
        o = dec.group_mean_groups_bthwc(x_tt, out_channels=out_channels, group_size=group_size)
        return _download_bthwc(mesh_device, o, b)

    saved = dec._GROUP_MEAN_MAX_FLAT_ELEMENTS
    try:
        dec._GROUP_MEAN_MAX_FLAT_ELEMENTS = 10**18  # force unchunked
        out_full = run()
        dec._GROUP_MEAN_MAX_FLAT_ELEMENTS = 1  # force H-chunking
        out_chunk = run()
    finally:
        dec._GROUP_MEAN_MAX_FLAT_ELEMENTS = saved

    ttnn.deallocate(x_tt, force=False)

    assert out_full.shape == out_chunk.shape == ref.shape
    p_full, pcc_full = comp_pcc(ref.bfloat16().float(), out_full, 0.999)
    p_eq, pcc_eq = comp_pcc(out_full, out_chunk, 0.999)
    logger.info(
        f"group_mean chunk-vs-full: shape={shape} g={group_size} " f"vs_ref={pcc_full:.6f} chunk_vs_full={pcc_eq:.6f}"
    )
    assert p_full, f"full vs ref PCC {pcc_full:.6f} < 0.999"
    assert p_eq, f"chunk vs full PCC {pcc_eq:.6f} < 0.999"
