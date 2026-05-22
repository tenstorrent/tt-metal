# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S2V grid-based ``rope_precompute`` parity."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.rope_s2v import rope_precompute
from .....models.transformers.wan2_2.s2v.transformer_wan_s2v import WanS2VTransformer3DModel
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.tensor import local_device_to_torch
from .....utils.test import line_params, ring_params

# Production Wan-AI/Wan2.2-S2V-14B model config (resolution-independent).
DIM = 5120
NUM_HEADS = 40
HEAD_DIM = DIM // NUM_HEADS  # 128
FFN_DIM = 13824
ZIP_FRAME_BUCKETS = (1, 2, 16)


def build_ref_grid_sizes(
    *,
    ppf: int,
    pph: int,
    ppw: int,
    lat_h: int,
    lat_w: int,
    include_motion: bool,
) -> list:
    """Reference grid construction matching ``WanModel_S2V.forward`` (noisy + ref)
    and ``FramePackMotioner.forward`` (motion 3-bucket)."""
    noisy = [
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[ppf, pph, ppw]], dtype=torch.long),
        torch.tensor([[ppf, pph, ppw]], dtype=torch.long),
    ]
    ref = [
        torch.tensor([[30, 0, 0]], dtype=torch.long),
        torch.tensor([[31, pph, ppw]], dtype=torch.long),
        torch.tensor([[1, pph, ppw]], dtype=torch.long),
    ]
    grids = [noisy, ref]
    if include_motion:
        zb = ZIP_FRAME_BUCKETS
        s1 = -int(zb[0])
        grids.append(
            [
                torch.tensor([[s1, 0, 0]], dtype=torch.long),
                torch.tensor([[s1 + int(zb[0]), lat_h // 2, lat_w // 2]], dtype=torch.long),
                torch.tensor([[zb[0], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
        s2 = -int(zb[0] + zb[1])
        grids.append(
            [
                torch.tensor([[s2, 0, 0]], dtype=torch.long),
                torch.tensor([[s2 + int(zb[1]) // 2, lat_h // 4, lat_w // 4]], dtype=torch.long),
                torch.tensor([[zb[1], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
        s3 = -int(zb[0] + zb[1] + zb[2])
        grids.append(
            [
                torch.tensor([[s3, 0, 0]], dtype=torch.long),
                torch.tensor([[s3 + int(zb[2]) // 4, lat_h // 8, lat_w // 8]], dtype=torch.long),
                torch.tensor([[zb[2], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
    return grids


@pytest.mark.parametrize(
    ("mesh_device", "mesh_shape", "sp_axis", "tp_axis", "num_links", "device_params", "topology", "is_fsdp"),
    [
        pytest.param((2, 4), (2, 4), 1, 0, 2, line_params, ttnn.Topology.Linear, False, id="bh_2x4sp1tp0"),
        pytest.param((4, 8), (4, 8), 1, 0, 2, ring_params, ttnn.Topology.Ring, False, id="bh_4x8sp1tp0"),
    ],
    indirect=["mesh_device", "device_params"],
)
@pytest.mark.parametrize(
    ("F_noisy", "H_latent", "W_latent"),
    [
        pytest.param(20, 60, 104, id="480p"),
        pytest.param(20, 90, 160, id="720p"),
    ],
)
@pytest.mark.parametrize("include_motion", [False, True], ids=["noisy_ref", "noisy_ref_motion"])
def test_rope_s2v_precompute(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    F_noisy: int,
    H_latent: int,
    W_latent: int,
    include_motion: bool,
) -> None:
    """Per-device-interleaved rope parity (with/without motion)."""
    torch.manual_seed(0)
    parent_mesh = mesh_device
    mesh_device = parent_mesh.create_submesh(ttnn.MeshShape(*mesh_shape))

    pT, pH, pW = 1, 2, 2
    ppf = F_noisy
    pph = H_latent // pH
    ppw = W_latent // pW
    lat_h, lat_w = H_latent, W_latent
    n_noisy = ppf * pph * ppw
    n_ref = pph * ppw

    if include_motion:
        n_post = ZIP_FRAME_BUCKETS[0] * (lat_h // 2) * (lat_w // 2)
        n_2x = (ZIP_FRAME_BUCKETS[1] // 2) * (lat_h // 4) * (lat_w // 4)
        n_4x = (ZIP_FRAME_BUCKETS[2] // 4) * (lat_h // 8) * (lat_w // 8)
        n_motion = n_post + n_2x + n_4x
    else:
        n_motion = 0
    n_total = n_noisy + n_ref + n_motion

    parallel_config = DiTParallelConfig(
        tensor_parallel=ParallelFactor(mesh_axis=tp_axis, factor=tuple(mesh_device.shape)[tp_axis]),
        sequence_parallel=ParallelFactor(mesh_axis=sp_axis, factor=tuple(mesh_device.shape)[sp_axis]),
        cfg_parallel=None,
    )
    ccl_manager = CCLManager(mesh_device=mesh_device, num_links=num_links, topology=topology)
    model = WanS2VTransformer3DModel(
        patch_size=(pT, pH, pW),
        num_heads=NUM_HEADS,
        dim=DIM,
        in_channels=16,
        out_channels=16,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=FFN_DIM,
        num_layers=2,
        eps=1e-6,
        audio_dim=1024,
        num_audio_layers=25,
        num_audio_token=4,
        audio_inject_layers=(),
        enable_adain=False,
        enable_framepack=True,
        cond_dim=16,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    model.original_seq_len = n_noisy
    model._cached_total_seq_len = n_total
    hidden_states = torch.zeros(1, 16, F_noisy * pT, H_latent, W_latent, dtype=torch.float32)

    cos_tt, sin_tt, _trans_mat = model.prepare_rope_features(hidden_states)

    # Production builds rope per-segment and concats on device, matching the
    # spatial sequence's per-device layout. After SP-gather the result is
    # per-device-interleaved (noisy_0|const_0|noisy_1|const_1|...), NOT global
    # [noisy|const]. Reconstruct the expected layout from the reference.
    sp_factor = parallel_config.sequence_parallel.factor
    if sp_factor > 1:
        cos_gathered = ccl_manager.all_gather_persistent_buffer(cos_tt, dim=2, mesh_axis=sp_axis)
        sin_gathered = ccl_manager.all_gather_persistent_buffer(sin_tt, dim=2, mesh_axis=sp_axis)
    else:
        cos_gathered, sin_gathered = cos_tt, sin_tt
    cos_tt_torch = local_device_to_torch(cos_gathered).float()
    sin_tt_torch = local_device_to_torch(sin_gathered).float()

    freqs_ref = model.frame_packer.freqs
    grid_sizes_ref = build_ref_grid_sizes(
        ppf=ppf, pph=pph, ppw=ppw, lat_h=lat_h, lat_w=lat_w, include_motion=include_motion
    )
    placeholder = torch.zeros(1, n_total, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    freqs_complex_ref = rope_precompute(placeholder, grid_sizes_ref, freqs_ref, start=None)
    cos_ref = freqs_complex_ref.real[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
    sin_ref = freqs_complex_ref.imag[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)

    def expected_interleaved(rope_global: torch.Tensor) -> torch.Tensor:
        noisy_seg = rope_global[:, :, :n_noisy, :]
        const_seg = rope_global[:, :, n_noisy:n_total, :] if n_total > n_noisy else None

        padded_pn = ((noisy_seg.shape[2] + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
        if padded_pn > noisy_seg.shape[2]:
            noisy_seg = F.pad(noisy_seg, (0, 0, 0, padded_pn - noisy_seg.shape[2]))
        pn_per_dev = padded_pn // sp_factor

        if const_seg is not None and const_seg.shape[2] > 0:
            padded_pc = ((const_seg.shape[2] + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
            if padded_pc > const_seg.shape[2]:
                const_seg = F.pad(const_seg, (0, 0, 0, padded_pc - const_seg.shape[2]))
            pc_per_dev = padded_pc // sp_factor
        else:
            const_seg = None
            pc_per_dev = 0

        chunks = []
        for d in range(sp_factor):
            chunks.append(noisy_seg[:, :, d * pn_per_dev : (d + 1) * pn_per_dev, :])
            if const_seg is not None:
                chunks.append(const_seg[:, :, d * pc_per_dev : (d + 1) * pc_per_dev, :])
        return torch.cat(chunks, dim=2)

    cos_expected = expected_interleaved(cos_ref)
    sin_expected = expected_interleaved(sin_ref)

    assert cos_tt_torch.shape == cos_expected.shape, f"cos shape: tt={cos_tt_torch.shape} ref={cos_expected.shape}"
    assert sin_tt_torch.shape == sin_expected.shape, f"sin shape: tt={sin_tt_torch.shape} ref={sin_expected.shape}"
    assert_quality(cos_tt_torch, cos_expected, pcc=0.99)
    assert_quality(sin_tt_torch, sin_expected, pcc=0.99)
    logger.info(f"rope parity ok: cos/sin {tuple(cos_tt_torch.shape)}")
