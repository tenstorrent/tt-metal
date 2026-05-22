# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Sub-component parity test for :class:`FramePackMotionerWan`.

Compares the motion-token output of the on-device ``FramePackMotionerWan``
against a CPU port of the WAN 2.2 reference's ``FramePackMotioner``.
Exercises all three ``zip_frame_buckets`` projections (post / 2x / 4x).

Production config: inner_dim=5120, num_heads=40, lat at 480p (60, 104).
PCC ≥ 0.99 on the concatenated motion tokens.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.motioner import FramePackMotionerWan
from .....models.transformers.wan2_2.s2v.rope_s2v import rope_params, rope_precompute
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....utils.check import assert_quality
from .....utils.tensor import to_torch
from .....utils.test import line_params, ring_params

# Production Wan-AI/Wan2.2-S2V-14B model config (resolution-independent).
IN_CHANNELS = 16
INNER_DIM = 5120
NUM_HEADS = 40
HEAD_DIM = INNER_DIM // NUM_HEADS  # 128
ZIP_FRAME_BUCKETS = (1, 2, 16)
T_MOTION = sum(ZIP_FRAME_BUCKETS)  # 19 (fixed by zip_frame_buckets)


# ---------------------------------------------------------------------------
# Torch reference — inlined port of wan/modules/s2v/motioner.py:FramePackMotioner.
# Only the production code path (add_last_motion=2, drop_mode="padd").
# ---------------------------------------------------------------------------


class TorchFramePackMotioner(nn.Module):
    def __init__(
        self,
        *,
        in_channels: int,
        inner_dim: int,
        num_heads: int,
        zip_frame_buckets: tuple[int, int, int],
    ) -> None:
        super().__init__()
        self.proj = nn.Conv3d(in_channels, inner_dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = nn.Conv3d(in_channels, inner_dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = nn.Conv3d(in_channels, inner_dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))
        self.zip_frame_buckets = torch.tensor(zip_frame_buckets, dtype=torch.long)
        self.inner_dim = inner_dim
        self.num_heads = num_heads
        self.in_channels = in_channels
        d = inner_dim // num_heads
        self.freqs = torch.cat(
            [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
            dim=1,
        )

    def forward(self, motion_latents: torch.Tensor) -> torch.Tensor:
        m = motion_latents[0]
        lat_h, lat_w = m.shape[2], m.shape[3]
        padd_lat = torch.zeros(self.in_channels, int(self.zip_frame_buckets.sum()), lat_h, lat_w, dtype=m.dtype)
        overlap = min(padd_lat.shape[1], m.shape[1])
        if overlap > 0:
            padd_lat[:, -overlap:] = m[:, -overlap:]
        padd_lat = padd_lat.unsqueeze(0)
        clean_4x, clean_2x, clean_post = padd_lat.split(list(self.zip_frame_buckets)[::-1], dim=2)
        clean_post = self.proj(clean_post).flatten(2).transpose(1, 2)
        clean_2x = self.proj_2x(clean_2x).flatten(2).transpose(1, 2)
        clean_4x = self.proj_4x(clean_4x).flatten(2).transpose(1, 2)
        return torch.cat([clean_post, clean_2x, clean_4x], dim=1)


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("LAT_H", "LAT_W"),
    [
        pytest.param(60, 104, id="480p"),
        pytest.param(90, 160, id="720p"),
    ],
)
def test_frame_packer(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    LAT_H: int,
    LAT_W: int,
) -> None:
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )

    ref = (
        TorchFramePackMotioner(
            in_channels=IN_CHANNELS,
            inner_dim=INNER_DIM,
            num_heads=NUM_HEADS,
            zip_frame_buckets=ZIP_FRAME_BUCKETS,
        )
        .eval()
        .to(torch.float32)
    )

    tt = FramePackMotionerWan(
        in_channels=IN_CHANNELS,
        inner_dim=INNER_DIM,
        num_heads=NUM_HEADS,
        zip_frame_buckets=ZIP_FRAME_BUCKETS,
        drop_mode="padd",
        mesh_device=mesh_device,
        parallel_config=parallel_config,
    )
    incompat = tt.load_torch_state_dict(ref.state_dict(), strict=False)
    logger.info(
        f"FramePackMotionerWan load: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}"
    )

    motion_latents = torch.randn(1, IN_CHANNELS, T_MOTION, LAT_H, LAT_W, dtype=torch.float32)

    with torch.no_grad():
        ref_tokens = ref(motion_latents)
    logger.info(f"reference tokens: {tuple(ref_tokens.shape)}")

    tt_tokens_dev = tt.forward(motion_latents)
    # Tokens are TP-fractured on the last (D) axis; SP axis is replicated.
    tt_tokens_torch = to_torch(tt_tokens_dev, mesh_axes=[None, None, None, tp_axis])
    assert tt_tokens_torch.dim() == 4, f"expected [1, B, N, dim], got {tt_tokens_torch.shape}"
    tt_tokens_torch = tt_tokens_torch.squeeze(0).float()
    logger.info(f"TT tokens: {tuple(tt_tokens_torch.shape)}")

    assert (
        tt_tokens_torch.shape == ref_tokens.shape
    ), f"motion-token shape mismatch: tt={tuple(tt_tokens_torch.shape)} ref={tuple(ref_tokens.shape)}"
    assert_quality(tt_tokens_torch, ref_tokens.float(), pcc=0.99)

    # Sanity: TT and reference build their rope freqs identically, and
    # rope_precompute is callable from the same grid + freqs on host.
    d = HEAD_DIM
    expected_freqs = torch.cat(
        [rope_params(1024, d - 4 * (d // 6)), rope_params(1024, 2 * (d // 6)), rope_params(1024, 2 * (d // 6))],
        dim=1,
    )
    assert torch.equal(tt.freqs.real, expected_freqs.real), "freqs table real part mismatch"
    assert torch.equal(tt.freqs.imag, expected_freqs.imag), "freqs table imag part mismatch"
    placeholder = torch.zeros(1, tt_tokens_torch.shape[1], NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    zb = ZIP_FRAME_BUCKETS
    grid_post = [
        torch.tensor([[-int(zb[0]), 0, 0]], dtype=torch.long),
        torch.tensor([[0, LAT_H // 2, LAT_W // 2]], dtype=torch.long),
        torch.tensor([[int(zb[0]), LAT_H // 2, LAT_W // 2]], dtype=torch.long),
    ]
    grid_2x = [
        torch.tensor([[-int(zb[0] + zb[1]), 0, 0]], dtype=torch.long),
        torch.tensor([[-int(zb[0] + zb[1]) + int(zb[1]) // 2, LAT_H // 4, LAT_W // 4]], dtype=torch.long),
        torch.tensor([[int(zb[1]), LAT_H // 2, LAT_W // 2]], dtype=torch.long),
    ]
    grid_4x = [
        torch.tensor([[-int(zb[0] + zb[1] + zb[2]), 0, 0]], dtype=torch.long),
        torch.tensor([[-int(zb[0] + zb[1] + zb[2]) + int(zb[2]) // 4, LAT_H // 8, LAT_W // 8]], dtype=torch.long),
        torch.tensor([[int(zb[2]), LAT_H // 2, LAT_W // 2]], dtype=torch.long),
    ]
    motion_rope = rope_precompute(placeholder, [grid_post, grid_2x, grid_4x], tt.freqs, start=None)
    assert motion_rope.shape[1] == tt_tokens_torch.shape[1], "motion rope length mismatch"
