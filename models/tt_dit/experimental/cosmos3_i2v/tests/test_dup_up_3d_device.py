# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""On-device DupUp3D PCC sandbox.

`DupUp3D.host_forward` is the diffusers-exact reference; the residual decoder
shortcut currently round-trips host↔device per call (≥47% of VAE-decode wall
at production shape). This file iterates candidate on-device formulations
against the host reference across single-device + 2D-sharded submeshes.

The prior on-device attempt (commit 2572f2c7527^) failed because rank-changing
reshape+permute sequences transiently shift the sharded H/W axes and ttnn's
shard-axis metadata desyncs. This sandbox starts from `_candidate_resurrected`
(verbatim from that commit) on a single-device submesh to confirm the math,
then escalates to sharded submeshes to either reproduce the desync or
demonstrate that the BTHWC variant (sharded axes at fixed positions 2/3) is
viable.
"""

from __future__ import annotations

import pytest
import torch

import ttnn
from models.tt_dit.models.vae.vae_wan2_1 import AvgDown3D, DupUp3D
from models.tt_dit.utils.check import assert_quality
from models.tt_dit.utils.tensor import to_torch, typed_tensor_2dshard
from models.tt_dit.utils.test import line_params


# Candidate A: resurrected from commit 2572f2c7527^ (the original on-device forward
# kept here verbatim so the test can iterate on it without touching vae_wan2_1.py
# until something passes PCC under sharding).
def _candidate_resurrected(
    x_BTHWC: ttnn.Tensor,
    *,
    in_channels: int,
    out_channels: int,
    factor_t: int,
    factor_s: int,
    first_chunk: bool = False,
) -> ttnn.Tensor:
    ft = factor_t
    fs = factor_s
    B, T, H, W, C = x_BTHWC.shape
    factor = ft * fs * fs
    repeats = out_channels * factor // in_channels

    if ft == 1 and fs == 1 and repeats == 1:
        return x_BTHWC

    if repeats != 1:
        x = ttnn.repeat_interleave(x_BTHWC, repeats, dim=4)
    else:
        x = x_BTHWC

    out_C = out_channels

    if fs > 1:
        # un-tile fs_w
        x = ttnn.reshape(x, (B, T, H, W, out_C * ft * fs, fs))
        x = ttnn.permute(x, (0, 1, 2, 3, 5, 4))
        x = ttnn.reshape(x, (B, T, H, W * fs, out_C * ft * fs))

        # un-tile fs_h
        x = ttnn.reshape(x, (B, T, H, W * fs, out_C * ft, fs))
        x = ttnn.permute(x, (0, 1, 2, 5, 3, 4))
        x = ttnn.reshape(x, (B, T, H * fs, W * fs, out_C * ft))
        W = W * fs
        H = H * fs

    if ft > 1:
        x = ttnn.reshape(x, (B, T, H, W, out_C, ft))
        x = ttnn.permute(x, (0, 1, 5, 2, 3, 4))
        x = ttnn.reshape(x, (B, T * ft, H, W, out_C))
        if first_chunk:
            x = x[:, ft - 1 :, :, :, :]

    return x


_CONFIGS = [
    # (in_C, out_C, ft, fs) — decoder residual up levels (post-quant, channel-shrinking)
    pytest.param(1024, 512, 2, 2, id="dec1_1024_to_512_st2"),
    pytest.param(512, 256, 2, 2, id="dec2_512_to_256_st2"),
    pytest.param(256, 256, 1, 2, id="dec3_256_id_s2"),
    # smallest: pure spatial s2 with channel identity
    pytest.param(160, 160, 1, 2, id="small_160_id_s2"),
]


def _shape_for(in_C: int) -> tuple[int, int, int, int, int]:
    """Pick a tiny BTHWC shape that respects sharding alignment for 1x2 / 2x2 meshes.
    H and W chosen multiples of 4 so that any axis ÷ 2 stays ≥ 2."""
    B, T, H, W = 1, 4, 16, 16
    return (B, T, H, W, in_C)


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), line_params, id="bh_galaxy_4x8")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [(1, 1), (1, 2), (2, 2)],
    ids=["sub_1x1", "sub_1x2", "sub_2x2"],
)
@pytest.mark.parametrize("chunks", [1, 2, 4], ids=["k1", "k2", "k4"])
@pytest.mark.parametrize("first_chunk", [False, True], ids=["full_chunk", "first_chunk"])
@pytest.mark.parametrize("in_C, out_C, ft, fs", _CONFIGS)
def test_dup_up_3d_on_device_pcc(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    first_chunk: bool,
    chunks: int,
    in_C: int,
    out_C: int,
    ft: int,
    fs: int,
    monkeypatch,
) -> None:
    monkeypatch.setenv("TT_DIT_VAE_DUPUP_CHUNKS", str(chunks))
    """Candidate A (resurrected) against DupUp3D.host_forward.

    Submesh (1,1) verifies math without sharding interaction.
    Submesh (1,2) sharded on W axis reproduces (or refutes) the prior desync.
    """
    parent_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > parent_shape[0] or submesh_shape[1] > parent_shape[1]:
        pytest.skip(f"submesh {submesh_shape} doesn't fit in parent {parent_shape}")
    if submesh_shape == parent_shape:
        submesh = mesh_device
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sub_shape = tuple(submesh.shape)
    # Sharding axes (BTHWC): H @ row-axis (0), W @ col-axis (1). Same convention
    # WanResidualUpBlock uses via VaeHWParallelConfig.
    h_axis, w_axis = 0, 1

    B, T, H, W, _ = _shape_for(in_C)
    # Ensure H and W are divisible by the per-axis mesh extent.
    if H % sub_shape[h_axis] or W % sub_shape[w_axis]:
        pytest.skip(f"H={H}, W={W} not divisible by sub_shape={sub_shape}")

    torch.manual_seed(0)
    x_BCTHW_host = torch.randn(B, in_C, T, H, W, dtype=torch.float32)

    # Host reference.
    ref_BCTHW = DupUp3D.host_forward(
        x_BCTHW_host,
        in_channels=in_C,
        out_channels=out_C,
        factor_t=ft,
        factor_s=fs,
        first_chunk=first_chunk,
    )
    ref_BTHWC = ref_BCTHW.permute(0, 2, 3, 4, 1).contiguous()

    # Upload as BTHWC with H/W sharded.
    x_BTHWC_host = x_BCTHW_host.permute(0, 2, 3, 4, 1).contiguous().to(torch.bfloat16)
    x_BTHWC_dev = typed_tensor_2dshard(
        x_BTHWC_host,
        submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=ttnn.bfloat16,
    )

    dup_up = DupUp3D(in_channels=in_C, out_channels=out_C, factor_t=ft, factor_s=fs)
    y_dev = dup_up(x_BTHWC_dev, first_chunk=first_chunk)

    # Gather back. Output is (B, T*ft, H*fs, W*fs, out_C); H/W still sharded.
    mesh_axes_for_output: list[int | None] = [None] * 5
    mesh_axes_for_output[2] = h_axis
    mesh_axes_for_output[3] = w_axis
    y_host = to_torch(y_dev, mesh_axes=tuple(mesh_axes_for_output), composer_device=submesh)

    assert_quality(ref_BTHWC.to(torch.float32), y_host.to(torch.float32), pcc=0.9999, relative_rmse=0.01)


_DOWN_CONFIGS = [
    # encoder residual down levels (Cosmos3 base_dim×dim_mult: [256, 512, 1024, 1024])
    pytest.param(256, 512, 1, 2, id="enc_lvl0_s2"),
    pytest.param(512, 1024, 2, 2, id="enc_lvl1_st2"),
    pytest.param(1024, 1024, 2, 2, id="enc_lvl2_st2"),
    # identity-shortcut at the bottom (down_flag=False) — exercised via factor=1
    pytest.param(1024, 1024, 1, 1, id="enc_lvl3_id"),
]


@pytest.mark.parametrize(
    "mesh_device, device_params",
    [pytest.param((4, 8), line_params, id="bh_galaxy_4x8")],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    "submesh_shape",
    [(1, 1), (1, 2), (2, 2)],
    ids=["sub_1x1", "sub_1x2", "sub_2x2"],
)
@pytest.mark.parametrize("in_C, out_C, ft, fs", _DOWN_CONFIGS)
def test_avg_down_3d_on_device_pcc(
    mesh_device: ttnn.MeshDevice,
    submesh_shape: tuple[int, int],
    in_C: int,
    out_C: int,
    ft: int,
    fs: int,
) -> None:
    """On-device AvgDown3D.forward against AvgDown3D.host_forward.

    Inputs that aren't a multiple of ft on T exercise the front-pad path.
    """
    parent_shape = tuple(mesh_device.shape)
    if submesh_shape[0] > parent_shape[0] or submesh_shape[1] > parent_shape[1]:
        pytest.skip(f"submesh {submesh_shape} doesn't fit in parent {parent_shape}")
    if submesh_shape == parent_shape:
        submesh = mesh_device
    else:
        submesh = mesh_device.create_submesh(ttnn.MeshShape(*submesh_shape))

    sub_shape = tuple(submesh.shape)
    h_axis, w_axis = 0, 1

    # Shape: T=5 forces front-pad when ft=2 (encoder Cosmos3 typical first chunk).
    B, T, H, W = 1, 5, 16, 16
    if H % sub_shape[h_axis] or W % sub_shape[w_axis]:
        pytest.skip(f"H={H}, W={W} not divisible by sub_shape={sub_shape}")
    if (H // fs) % sub_shape[h_axis] or (W // fs) % sub_shape[w_axis]:
        pytest.skip(f"post-down H/W not divisible by sub_shape")

    torch.manual_seed(0)
    x_BCTHW_host = torch.randn(B, in_C, T, H, W, dtype=torch.float32)

    ref_BCTHW = AvgDown3D.host_forward(
        x_BCTHW_host,
        in_channels=in_C,
        out_channels=out_C,
        factor_t=ft,
        factor_s=fs,
    )
    ref_BTHWC = ref_BCTHW.permute(0, 2, 3, 4, 1).contiguous()

    avg_down = AvgDown3D(in_channels=in_C, out_channels=out_C, factor_t=ft, factor_s=fs)

    x_BTHWC_host = x_BCTHW_host.permute(0, 2, 3, 4, 1).contiguous().to(torch.bfloat16)
    x_BTHWC_dev = typed_tensor_2dshard(
        x_BTHWC_host,
        submesh,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        shard_mapping={h_axis: 2, w_axis: 3},
        dtype=ttnn.bfloat16,
    )

    y_dev = avg_down(x_BTHWC_dev)

    mesh_axes_for_output: list[int | None] = [None] * 5
    mesh_axes_for_output[2] = h_axis
    mesh_axes_for_output[3] = w_axis
    y_host = to_torch(y_dev, mesh_axes=tuple(mesh_axes_for_output), composer_device=submesh)

    assert_quality(ref_BTHWC.to(torch.float32), y_host.to(torch.float32), pcc=0.9999, relative_rmse=0.01)
