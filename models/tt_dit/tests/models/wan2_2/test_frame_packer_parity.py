# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test for :class:`FramePackMotionerWan`.

Compares the motion-token output of our on-device ``FramePackMotionerWan``
against an inlined CPU port of ``wan/modules/s2v/motioner.py:FramePackMotioner``
from the WAN 2.2 reference. Exercises all three ``zip_frame_buckets``
projections (post / 2x / 4x) end-to-end.

Test bar: PCC ≥ 0.99 on the concatenated motion tokens; byte-identity on the
rope grid (host-side construction).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.motioner import FramePackMotionerWan
from ....models.transformers.wan2_2.s2v_rope import rope_params, rope_precompute
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....utils.check import assert_quality
from ....utils.tensor import to_torch
from ....utils.test import line_params

# Reduced config that still exercises all three buckets and the smallest
# spatial stride (proj_4x kernel (4, 8, 8)). production: inner_dim=5120,
# num_heads=40, lat at 480p (60, 104).
IN_CHANNELS = 16
INNER_DIM = 128
NUM_HEADS = 4
HEAD_DIM = INNER_DIM // NUM_HEADS  # 32 — satisfies rope_params split (d - 4*(d//6) + 2*(d//6) + 2*(d//6) = 32)
ZIP_FRAME_BUCKETS = (1, 2, 16)

# Latent spatial size must be divisible by 8 (largest kernel) for proj_4x.
LAT_H = 16
LAT_W = 32
T_MOTION = sum(ZIP_FRAME_BUCKETS)  # 19 — exactly fills the pad buffer, no zero rows.


class _RefFramePackMotioner(nn.Module):
    """Inlined CPU port of ``wan/modules/s2v/motioner.py:FramePackMotioner``.

    Only the production code path (``add_last_motion=2``, ``drop_mode="padd"``)
    is reproduced; the drop-mode branches that exist in the reference for
    historical multi-clip partial-context paths are skipped.
    """

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
        d = inner_dim // num_heads
        self.freqs = torch.cat(
            [
                rope_params(1024, d - 4 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
                rope_params(1024, 2 * (d // 6)),
            ],
            dim=1,
        )

    def forward(self, motion_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``motion_latents`` shape ``[1, C, T, H, W]``. Returns ``(tokens, rope)``."""
        m = motion_latents[0]  # → [C, T, H, W]
        lat_h, lat_w = m.shape[2], m.shape[3]
        padd_lat = torch.zeros(IN_CHANNELS, int(self.zip_frame_buckets.sum()), lat_h, lat_w, dtype=m.dtype)
        overlap = min(padd_lat.shape[1], m.shape[1])
        if overlap > 0:
            padd_lat[:, -overlap:] = m[:, -overlap:]
        padd_lat = padd_lat.unsqueeze(0)
        clean_4x, clean_2x, clean_post = padd_lat.split(list(self.zip_frame_buckets)[::-1], dim=2)

        clean_post = self.proj(clean_post).flatten(2).transpose(1, 2)
        clean_2x = self.proj_2x(clean_2x).flatten(2).transpose(1, 2)
        clean_4x = self.proj_4x(clean_4x).flatten(2).transpose(1, 2)

        motion_lat = torch.cat([clean_post, clean_2x, clean_4x], dim=1)

        zb = self.zip_frame_buckets
        start_post = -int(zb[0])
        end_post = start_post + int(zb[0])
        grid_post = [
            torch.tensor([[start_post, 0, 0]], dtype=torch.long),
            torch.tensor([[end_post, lat_h // 2, lat_w // 2]], dtype=torch.long),
            torch.tensor([[int(zb[0]), lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        start_2x = -int(zb[0] + zb[1])
        end_2x = start_2x + int(zb[1]) // 2
        grid_2x = [
            torch.tensor([[start_2x, 0, 0]], dtype=torch.long),
            torch.tensor([[end_2x, lat_h // 4, lat_w // 4]], dtype=torch.long),
            torch.tensor([[int(zb[1]), lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        start_4x = -int(zb[0] + zb[1] + zb[2])
        end_4x = start_4x + int(zb[2]) // 4
        grid_4x = [
            torch.tensor([[start_4x, 0, 0]], dtype=torch.long),
            torch.tensor([[end_4x, lat_h // 8, lat_w // 8]], dtype=torch.long),
            torch.tensor([[int(zb[2]), lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        grid_sizes = [grid_post, grid_2x, grid_4x]

        rope_input = motion_lat.detach().view(1, motion_lat.shape[1], self.num_heads, self.inner_dim // self.num_heads)
        motion_rope = rope_precompute(rope_input, grid_sizes, self.freqs, start=None)
        return motion_lat, motion_rope


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param(
            (2, 4),
            (2, 4),
            1,
            0,
            2,
            line_params,
            ttnn.Topology.Linear,
            False,
            id="bh_2x4sp1tp0",
        ),
    ],
    indirect=["mesh_device", "device_params"],
)
def test_frame_packer_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )

    # ---- Reference (CPU) ----
    ref = (
        _RefFramePackMotioner(
            in_channels=IN_CHANNELS,
            inner_dim=INNER_DIM,
            num_heads=NUM_HEADS,
            zip_frame_buckets=ZIP_FRAME_BUCKETS,
        )
        .eval()
        .to(torch.float32)
    )

    # ---- TT model ----
    tt = FramePackMotionerWan(
        in_channels=IN_CHANNELS,
        inner_dim=INNER_DIM,
        num_heads=NUM_HEADS,
        zip_frame_buckets=ZIP_FRAME_BUCKETS,
        drop_mode="padd",
        mesh_device=mesh_device,
        parallel_config=parallel_config,
    )
    # WanPatchEmbed._prepare_torch_state translates Conv3d weight/bias into
    # the unfolded-matmul ``proj_weight`` / ``proj_bias`` parameters.
    incompat = tt.load_torch_state_dict(ref.state_dict(), strict=False)
    logger.info(
        f"FramePackMotionerWan load: missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}"
    )
    if incompat.missing_keys:
        logger.warning(f"missing keys (first 5): {incompat.missing_keys[:5]}")
    if incompat.unexpected_keys:
        logger.warning(f"unexpected keys (first 5): {incompat.unexpected_keys[:5]}")

    # ---- Inputs ----
    motion_latents = torch.randn(1, IN_CHANNELS, T_MOTION, LAT_H, LAT_W, dtype=torch.float32)

    # ---- Reference forward ----
    with torch.no_grad():
        ref_tokens, ref_rope = ref(motion_latents)
    logger.info(f"reference tokens: {tuple(ref_tokens.shape)}; rope: {tuple(ref_rope.shape)}")

    # ---- TT forward ----
    tt_tokens_dev, tt_rope = tt.forward(motion_latents)
    # Tokens are TP-fractured on the last (D) axis; the SP axis is replicated.
    # Pass ``mesh_axes`` to to_torch so the composer concatenates the TP shards
    # back to the full embedding dim.
    tt_tokens_torch = to_torch(tt_tokens_dev, mesh_axes=[None, None, None, tp_axis])
    assert tt_tokens_torch.dim() == 4, f"expected [1, B, N, dim], got {tt_tokens_torch.shape}"
    tt_tokens_torch = tt_tokens_torch.squeeze(0).float()  # [B, N, dim]
    logger.info(f"TT tokens: {tuple(tt_tokens_torch.shape)}")

    # ---- Compare ----
    assert (
        tt_tokens_torch.shape == ref_tokens.shape
    ), f"motion-token shape mismatch: tt={tuple(tt_tokens_torch.shape)} ref={tuple(ref_tokens.shape)}"
    assert (
        tt_rope.shape == ref_rope.shape
    ), f"motion-rope shape mismatch: tt={tuple(tt_rope.shape)} ref={tuple(ref_rope.shape)}"
    # Rope is host-built in both paths from identical grid_sizes + freqs → byte-identical.
    assert torch.equal(tt_rope.real, ref_rope.real), "motion-rope real part not byte-identical"
    assert torch.equal(tt_rope.imag, ref_rope.imag), "motion-rope imag part not byte-identical"
    # Tokens go through bf16 device matmul → tolerate PCC noise.
    assert_quality(tt_tokens_torch, ref_tokens.float(), pcc=0.99)
