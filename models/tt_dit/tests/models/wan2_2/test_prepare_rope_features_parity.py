# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Parity test for :meth:`WanS2VTransformer3DModel.prepare_rope_features`.

Compares our per-token (cos, sin) rope tensors against the reference's
grid construction in ``WanModel_S2V.forward`` plus
``FramePackMotioner.forward``'s motion-bucket grid. The vendored
:func:`rope_precompute` is already byte-for-byte vs the reference (see
``rope_s2v.py`` cleanup) — this test verifies our **grid_sizes construction**.

Test bar: PCC ≥ 0.99.
"""

from __future__ import annotations

import pytest
import torch
from loguru import logger

import ttnn

from ....models.transformers.wan2_2.rope_s2v import rope_precompute
from ....models.transformers.wan2_2.transformer_wan_s2v import WanS2VTransformer3DModel
from ....parallel.config import DiTParallelConfig, ParallelFactor
from ....parallel.manager import CCLManager
from ....utils.check import assert_quality
from ....utils.tensor import local_device_to_torch
from ....utils.test import line_params

# Reduced config. patch_size=(1,2,2), so pph = H_lat // 2, ppw = W_lat // 2.
# Production 480p: F=21, H=480, W=832 → ppf=21, pph=30, ppw=52.
# We shrink F so the test is fast but keep the H/W spatial extent so the
# motion 4x bucket exercises the lat_h // 8 vs pph // 8 distinction
# (the bug fix this test was designed to catch).
DIM = 128
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 32
FFN_DIM = 256

F_NOISY = 8  # patched temporal: ppf = 8
H_LATENT = 16  # patched spatial: pph = H_LATENT // 2 = 8
W_LATENT = 32  # patched spatial: ppw = W_LATENT // 2 = 16
ZIP_FRAME_BUCKETS = (1, 2, 16)


def _build_ref_grid_sizes(
    *,
    ppf: int,
    pph: int,
    ppw: int,
    lat_h: int,
    lat_w: int,
    zip_frame_buckets: tuple[int, int, int],
    include_motion: bool,
) -> list:
    """Reproduce the reference's grid_sizes for noisy + ref + motion.

    Mirrors ``WanModel_S2V.forward`` lines 704-734 (noisy + ref) and
    ``FramePackMotioner.forward`` lines 722-749 (motion-3-bucket).
    """
    # Noisy: zeros / [ppf, pph, ppw] / [ppf, pph, ppw].
    noisy = [
        torch.zeros(1, 3, dtype=torch.long),
        torch.tensor([[ppf, pph, ppw]], dtype=torch.long),
        torch.tensor([[ppf, pph, ppw]], dtype=torch.long),
    ]
    # Ref: temporal slot 30, single frame.
    ref = [
        torch.tensor([[30, 0, 0]], dtype=torch.long),
        torch.tensor([[31, pph, ppw]], dtype=torch.long),
        torch.tensor([[1, pph, ppw]], dtype=torch.long),
    ]
    grids = [noisy, ref]
    if include_motion:
        zb = zip_frame_buckets
        # post bucket — conv kernel (1,2,2), output (1, lat_h//2, lat_w//2) = (1, pph, ppw).
        start1 = -int(zb[0])
        end1 = start1 + int(zb[0])
        motion_post = [
            torch.tensor([[start1, 0, 0]], dtype=torch.long),
            torch.tensor([[end1, lat_h // 2, lat_w // 2]], dtype=torch.long),
            torch.tensor([[zb[0], lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        # 2x bucket — conv kernel (2,4,4), output (1, lat_h//4, lat_w//4).
        start2 = -int(zb[0] + zb[1])
        end2 = start2 + int(zb[1]) // 2
        motion_2x = [
            torch.tensor([[start2, 0, 0]], dtype=torch.long),
            torch.tensor([[end2, lat_h // 4, lat_w // 4]], dtype=torch.long),
            torch.tensor([[zb[1], lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        # 4x bucket — conv kernel (4,8,8), output (4, lat_h//8, lat_w//8).
        start3 = -int(zb[0] + zb[1] + zb[2])
        end3 = start3 + int(zb[2]) // 4
        motion_4x = [
            torch.tensor([[start3, 0, 0]], dtype=torch.long),
            torch.tensor([[end3, lat_h // 8, lat_w // 8]], dtype=torch.long),
            torch.tensor([[zb[2], lat_h // 2, lat_w // 2]], dtype=torch.long),
        ]
        grids += [motion_post, motion_2x, motion_4x]
    return grids


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
@pytest.mark.parametrize("include_motion", [False, True], ids=["noisy_ref", "noisy_ref_motion"])
def test_prepare_rope_features_parity(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
    include_motion: bool,
) -> None:
    """Compares our rope cos/sin tensors against a reference grid construction.

    The test was specifically designed to exercise the motion-bucket H/W
    extents: prior to the fix, motion 4x used ``pph // 8`` instead of
    ``pph // 4``, producing rope for only ~20% of the 4x motion tokens.
    """
    torch.manual_seed(0)

    pT, pH, pW = 1, 2, 2
    ppf = F_NOISY  # F_NOISY is already the patched temporal extent in this test
    pph = H_LATENT // pH
    ppw = W_LATENT // pW
    lat_h = H_LATENT
    lat_w = W_LATENT
    N_noisy = ppf * pph * ppw
    N_ref = pph * ppw

    # Compute reference motion token count (matches FramePackMotioner conv outputs).
    if include_motion:
        N_post = ZIP_FRAME_BUCKETS[0] * (lat_h // 2) * (lat_w // 2)
        N_2x = (ZIP_FRAME_BUCKETS[1] // 2) * (lat_h // 4) * (lat_w // 4)
        N_4x = (ZIP_FRAME_BUCKETS[2] // 4) * (lat_h // 8) * (lat_w // 8)
        N_motion = N_post + N_2x + N_4x
    else:
        N_motion = 0
    N_total = N_noisy + N_ref + N_motion

    # ---- Build our model ----
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
        text_dim=512,
        freq_dim=64,
        ffn_dim=FFN_DIM,
        num_layers=2,
        eps=1e-6,
        audio_dim=128,
        num_audio_layers=3,
        num_audio_token=4,
        audio_inject_layers=(),
        enable_adain=False,
        enable_framepack=True,
        cond_dim=16,
        mesh_device=mesh_device,
        ccl_manager=ccl_manager,
        parallel_config=parallel_config,
    )
    # Set up state that prepare_rope_features reads.
    model.original_seq_len = N_noisy
    model._cached_total_seq_len = N_total
    # Hidden_states shape carries F, H, W for prepare_rope_features.
    hidden_states = torch.zeros(1, 16, F_NOISY * pT, H_LATENT, W_LATENT, dtype=torch.float32)

    # ---- Our path ----
    cos_tt, sin_tt, _trans_mat = model.prepare_rope_features(hidden_states)
    # Production now builds rope per-segment and ttnn.concat([noisy, const])
    # on device, matching the spatial sequence's per-device layout. After
    # SP-gather the result is per-device-interleaved (noisy_0|const_0|noisy_1|
    # const_1|...), NOT the global [noisy|const] order the reference produces.
    sp_factor = parallel_config.sequence_parallel.factor
    if sp_factor > 1:
        cos_gathered = ccl_manager.all_gather_persistent_buffer(cos_tt, dim=2, mesh_axis=sp_axis)
        sin_gathered = ccl_manager.all_gather_persistent_buffer(sin_tt, dim=2, mesh_axis=sp_axis)
    else:
        cos_gathered, sin_gathered = cos_tt, sin_tt
    cos_tt_torch = local_device_to_torch(cos_gathered).float()  # [1, 1, padded_N_noisy + padded_const, head_dim]
    sin_tt_torch = local_device_to_torch(sin_gathered).float()
    logger.info(f"TT rope (per-device-interleaved): cos {tuple(cos_tt_torch.shape)} sin {tuple(sin_tt_torch.shape)}")

    # ---- Reference grid construction + rope_precompute ----
    freqs_ref = model.frame_packer.freqs
    grid_sizes_ref = _build_ref_grid_sizes(
        ppf=ppf,
        pph=pph,
        ppw=ppw,
        lat_h=lat_h,
        lat_w=lat_w,
        zip_frame_buckets=ZIP_FRAME_BUCKETS,
        include_motion=include_motion,
    )
    placeholder = torch.zeros(1, N_total, NUM_HEADS, HEAD_DIM, dtype=torch.float32)
    freqs_complex_ref = rope_precompute(placeholder, grid_sizes_ref, freqs_ref, start=None)
    # Convert complex → real (cos, sin) with the same repeat_interleave layout
    # the production code uses.
    cos_ref = freqs_complex_ref.real[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
    sin_ref = freqs_complex_ref.imag[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
    logger.info(f"Ref rope (global): cos {tuple(cos_ref.shape)} sin {tuple(sin_ref.shape)}")

    # ---- Build the expected per-device-interleaved layout from the reference ----
    # Mirrors the production: split into noisy / const, pad each, then
    # interleave by SP device:
    #   expected[d_n_chunk:(d+1)_n_chunk] = cos_noisy_padded[d*pn_per_dev:(d+1)*pn_per_dev]
    #   expected[(N_noisy_padded_total + d_c_chunk):...] = cos_const_padded[d*pc_per_dev:(d+1)*pc_per_dev]
    # Simpler: cat per-device [noisy_chunk, const_chunk] in device order.
    from torch.nn.functional import pad as _pad

    def _build_expected(rope_global: torch.Tensor) -> torch.Tensor:
        cos_noisy = rope_global[:, :, :N_noisy, :]
        cos_const = rope_global[:, :, N_noisy:N_total, :] if N_total > N_noisy else None
        # Pad each segment to sp_factor*TILE multiple (mirrors production).
        padded_pn = ((cos_noisy.shape[2] + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
        pn_pad = padded_pn - cos_noisy.shape[2]
        if pn_pad > 0:
            cos_noisy = _pad(cos_noisy, (0, 0, 0, pn_pad))
        pn_per_dev = padded_pn // sp_factor

        if cos_const is not None and cos_const.shape[2] > 0:
            padded_pc = ((cos_const.shape[2] + 32 * sp_factor - 1) // (32 * sp_factor)) * (32 * sp_factor)
            pc_pad = padded_pc - cos_const.shape[2]
            if pc_pad > 0:
                cos_const = _pad(cos_const, (0, 0, 0, pc_pad))
            pc_per_dev = padded_pc // sp_factor
        else:
            cos_const = None
            pc_per_dev = 0

        # Build per-device-interleaved layout.
        chunks = []
        for d in range(sp_factor):
            chunks.append(cos_noisy[:, :, d * pn_per_dev : (d + 1) * pn_per_dev, :])
            if cos_const is not None:
                chunks.append(cos_const[:, :, d * pc_per_dev : (d + 1) * pc_per_dev, :])
        return torch.cat(chunks, dim=2)

    cos_expected = _build_expected(cos_ref)
    sin_expected = _build_expected(sin_ref)
    logger.info(f"Expected per-device-interleaved: cos {tuple(cos_expected.shape)} sin {tuple(sin_expected.shape)}")

    # ---- Compare ----
    assert (
        cos_tt_torch.shape == cos_expected.shape
    ), f"cos shape mismatch: tt={cos_tt_torch.shape} expected={cos_expected.shape}"
    assert (
        sin_tt_torch.shape == sin_expected.shape
    ), f"sin shape mismatch: tt={sin_tt_torch.shape} expected={sin_expected.shape}"
    assert_quality(cos_tt_torch, cos_expected, pcc=0.99)
    assert_quality(sin_tt_torch, sin_expected, pcc=0.99)
