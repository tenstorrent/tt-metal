# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""S2V prepare-stage parity tests: audio embeddings + rope features.

Both compare on-device output against an inlined CPU reference of the
upstream Wan-Video/Wan2.2 math, so the tests run without the reference
repo on disk. Test bar: PCC >= 0.99.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from loguru import logger

import ttnn

from .....models.transformers.wan2_2.s2v.audio_utils import CausalAudioEncoder
from .....models.transformers.wan2_2.s2v.rope_s2v import rope_precompute
from .....models.transformers.wan2_2.s2v.transformer_wan_s2v import WanS2VTransformer3DModel
from .....parallel.config import DiTParallelConfig, ParallelFactor
from .....parallel.manager import CCLManager
from .....utils.check import assert_quality
from .....utils.tensor import local_device_to_torch
from .....utils.test import line_params

# ---------------------------------------------------------------------------
# test_prepare_audio_emb -- CausalAudioEncoder end-to-end
# ---------------------------------------------------------------------------
# Reduced config matches the production shape contract but with smaller
# out_dim for fast CPU reference. Production: audio_dim=1024, num_layers=25,
# out_dim=5120, num_token=4, motion_frames=(73, 19).
AUDIO_DIM = 1024
NUM_LAYERS = 25
OUT_DIM = 256
NUM_TOKEN = 4
MOTION_FRAMES = (17, 5)
T_AUDIO = 80


class _RefCausalConv1d(nn.Module):
    """``nn.Conv1d`` with replicate left-pad by ``kernel_size - 1`` (causal).

    Inlined from ``wan/modules/s2v/auxi_blocks.py:CausalConv1d``.
    """

    def __init__(self, chan_in: int, chan_out: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.time_causal_padding, mode="replicate")
        return self.conv(x)


class _RefMotionEncoderTC(nn.Module):
    """Three causal-conv stages (stride 1, 2, 2) -> 4x temporal downsample.

    Inlined from ``wan/modules/s2v/auxi_blocks.py:MotionEncoder_tc`` with
    ``need_global=False``. Output: ``[B, T/4, num_heads + 1, hidden]``
    (final +1 head is the learned padding token appended on the token axis).
    """

    def __init__(self, in_dim: int, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.conv1_local = _RefCausalConv1d(in_dim, hidden_dim // 4 * num_heads, 3, stride=1)
        self.norm1 = nn.LayerNorm(hidden_dim // 4, elementwise_affine=False, eps=1e-6)
        self.conv2 = _RefCausalConv1d(hidden_dim // 4, hidden_dim // 2, 3, stride=2)
        self.norm2 = nn.LayerNorm(hidden_dim // 2, elementwise_affine=False, eps=1e-6)
        self.conv3 = _RefCausalConv1d(hidden_dim // 2, hidden_dim, 3, stride=2)
        self.norm3 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.act = nn.SiLU()
        self.padding_tokens = nn.Parameter(torch.zeros(1, 1, 1, hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, T, in_dim]
        b, t, _ = x.shape
        x = x.permute(0, 2, 1)
        x = self.conv1_local(x)
        x = x.reshape(b, self.num_heads, self.hidden_dim // 4, t).permute(0, 1, 3, 2)
        x = x.reshape(b * self.num_heads, t, self.hidden_dim // 4)
        x = self.act(self.norm1(x))
        x = x.permute(0, 2, 1)
        x = self.conv2(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.norm2(x))
        x = x.permute(0, 2, 1)
        x = self.conv3(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.norm3(x))
        x = x.reshape(b, self.num_heads, -1, self.hidden_dim).permute(0, 2, 1, 3)
        pad = self.padding_tokens.expand(b, x.shape[1], 1, self.hidden_dim)
        return torch.cat([x, pad], dim=2)


class _RefCausalAudioEncoder(nn.Module):
    """Per-layer learned weighted sum of wav2vec2 hidden states -> MotionEncoder_tc.

    Inlined from ``wan/modules/s2v/audio_utils.py:CausalAudioEncoder``
    (``need_global=False`` branch).
    """

    def __init__(self, *, dim: int, num_layers: int, out_dim: int, num_token: int) -> None:
        super().__init__()
        self.encoder = _RefMotionEncoderTC(in_dim=dim, hidden_dim=out_dim, num_heads=num_token)
        self.weights = nn.Parameter(torch.ones(1, num_layers, 1, 1) * 0.01)
        self.act = nn.SiLU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: [B, num_layers, dim, T]
        w = self.act(self.weights)
        weighted = (features * w / w.sum(dim=1, keepdim=True)).sum(dim=1)
        return self.encoder(weighted.permute(0, 2, 1))


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
def test_prepare_audio_emb(
    mesh_device: ttnn.MeshDevice,
    mesh_shape: tuple[int, int],
    sp_axis: int,
    tp_axis: int,
    num_links: int,
    is_fsdp: bool,
    topology: ttnn.Topology,
) -> None:
    """End-to-end audio path: CausalAudioEncoder + slice off motion_frames[1]."""
    torch.manual_seed(0)

    ref = (
        _RefCausalAudioEncoder(dim=AUDIO_DIM, num_layers=NUM_LAYERS, out_dim=OUT_DIM, num_token=NUM_TOKEN)
        .eval()
        .to(torch.float32)
    )
    tt = CausalAudioEncoder(
        dim=AUDIO_DIM,
        num_layers=NUM_LAYERS,
        out_dim=OUT_DIM,
        num_token=NUM_TOKEN,
        need_global=False,
        mesh_device=mesh_device,
    )
    tt.load_torch_state_dict(ref.state_dict(), strict=False)

    audio_input_torch = torch.randn(1, NUM_LAYERS, AUDIO_DIM, T_AUDIO, dtype=torch.float32)
    # Left-pad by motion_frames[0] (wan/modules/s2v/model_s2v.py:683).
    pre = audio_input_torch[..., :1].expand(-1, -1, -1, MOTION_FRAMES[0])
    audio_input_padded = torch.cat([pre, audio_input_torch], dim=-1)

    with torch.no_grad():
        ref_out = ref(audio_input_padded)
        ref_merged = ref_out[:, MOTION_FRAMES[1] :, :]

    tt_out_torch = local_device_to_torch(tt(audio_input_padded))
    tt_merged = tt_out_torch[:, MOTION_FRAMES[1] :, :, :]

    assert (
        tt_merged.shape == ref_merged.shape
    ), f"shape mismatch: tt={tuple(tt_merged.shape)} ref={tuple(ref_merged.shape)}"
    assert_quality(tt_merged.float(), ref_merged.float(), pcc=0.99)


# ---------------------------------------------------------------------------
# test_prepare_rope_features -- rope cos/sin grid construction
# ---------------------------------------------------------------------------
# Reduced config matches the production shape contract while still exercising
# the motion 4x bucket's ``lat_h//8`` extent.
DIM = 128
NUM_HEADS = 4
HEAD_DIM = DIM // NUM_HEADS  # 32
FFN_DIM = 256
F_NOISY = 8
H_LATENT = 16
W_LATENT = 32
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
    """Reproduce ``WanModel_S2V.forward`` (lines 704-734, noisy + ref) and
    ``FramePackMotioner.forward`` (lines 722-749, motion 3-bucket) grids.
    """
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
        zb = zip_frame_buckets
        # post (kernel (1,2,2)), 2x (kernel (2,4,4)), 4x (kernel (4,8,8))
        start1 = -int(zb[0])
        grids.append(
            [
                torch.tensor([[start1, 0, 0]], dtype=torch.long),
                torch.tensor([[start1 + int(zb[0]), lat_h // 2, lat_w // 2]], dtype=torch.long),
                torch.tensor([[zb[0], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
        start2 = -int(zb[0] + zb[1])
        grids.append(
            [
                torch.tensor([[start2, 0, 0]], dtype=torch.long),
                torch.tensor([[start2 + int(zb[1]) // 2, lat_h // 4, lat_w // 4]], dtype=torch.long),
                torch.tensor([[zb[1], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
        start3 = -int(zb[0] + zb[1] + zb[2])
        grids.append(
            [
                torch.tensor([[start3, 0, 0]], dtype=torch.long),
                torch.tensor([[start3 + int(zb[2]) // 4, lat_h // 8, lat_w // 8]], dtype=torch.long),
                torch.tensor([[zb[2], lat_h // 2, lat_w // 2]], dtype=torch.long),
            ]
        )
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
def test_prepare_rope_features(
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

    ``include_motion=False`` covers clip 0 with ``drop_first_motion=True``;
    ``include_motion=True`` covers subsequent clips with motion threading.
    """
    torch.manual_seed(0)

    pT, pH, pW = 1, 2, 2
    ppf = F_NOISY
    pph = H_LATENT // pH
    ppw = W_LATENT // pW
    lat_h, lat_w = H_LATENT, W_LATENT
    N_noisy = ppf * pph * ppw
    N_ref = pph * ppw

    if include_motion:
        N_post = ZIP_FRAME_BUCKETS[0] * (lat_h // 2) * (lat_w // 2)
        N_2x = (ZIP_FRAME_BUCKETS[1] // 2) * (lat_h // 4) * (lat_w // 4)
        N_4x = (ZIP_FRAME_BUCKETS[2] // 4) * (lat_h // 8) * (lat_w // 8)
        N_motion = N_post + N_2x + N_4x
    else:
        N_motion = 0
    N_total = N_noisy + N_ref + N_motion

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
    model.original_seq_len = N_noisy
    model._cached_total_seq_len = N_total
    hidden_states = torch.zeros(1, 16, F_NOISY * pT, H_LATENT, W_LATENT, dtype=torch.float32)

    cos_tt, sin_tt, _trans_mat = model.prepare_rope_features(hidden_states)

    # Production builds rope per-segment and ttnn.concat([noisy, const]) on
    # device, matching the spatial sequence's per-device layout. After
    # SP-gather the result is per-device-interleaved
    # (noisy_0|const_0|noisy_1|const_1|...), NOT global [noisy|const].
    sp_factor = parallel_config.sequence_parallel.factor
    if sp_factor > 1:
        cos_gathered = ccl_manager.all_gather_persistent_buffer(cos_tt, dim=2, mesh_axis=sp_axis)
        sin_gathered = ccl_manager.all_gather_persistent_buffer(sin_tt, dim=2, mesh_axis=sp_axis)
    else:
        cos_gathered, sin_gathered = cos_tt, sin_tt
    cos_tt_torch = local_device_to_torch(cos_gathered).float()
    sin_tt_torch = local_device_to_torch(sin_gathered).float()

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
    cos_ref = freqs_complex_ref.real[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)
    sin_ref = freqs_complex_ref.imag[:, :, 0:1, :].float().repeat_interleave(2, dim=-1).permute(0, 2, 1, 3)

    # Build expected per-device-interleaved layout from the reference.
    def _build_expected(rope_global: torch.Tensor) -> torch.Tensor:
        noisy_seg = rope_global[:, :, :N_noisy, :]
        const_seg = rope_global[:, :, N_noisy:N_total, :] if N_total > N_noisy else None

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

    cos_expected = _build_expected(cos_ref)
    sin_expected = _build_expected(sin_ref)

    assert (
        cos_tt_torch.shape == cos_expected.shape
    ), f"cos shape mismatch: tt={cos_tt_torch.shape} expected={cos_expected.shape}"
    assert (
        sin_tt_torch.shape == sin_expected.shape
    ), f"sin shape mismatch: tt={sin_tt_torch.shape} expected={sin_expected.shape}"
    assert_quality(cos_tt_torch, cos_expected, pcc=0.99)
    assert_quality(sin_tt_torch, sin_expected, pcc=0.99)
    logger.info(f"rope parity ok: cos/sin {tuple(cos_tt_torch.shape)}")
