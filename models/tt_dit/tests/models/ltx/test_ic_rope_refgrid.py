# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Host-only tests for IC-LoRA reference-sheet two-grid RoPE in ``rope_ltx``.

These exercise the additive ``ref_num_frames`` branch of ``prepare_video_rope`` /
``prepare_av_cross_pe`` purely on the host: the builders are CPU position/cos/sin generators whose
only device touch is the final ``bf16_tensor*`` cast/shard, which is monkeypatched to a pass-through
so the host cos/sin are directly comparable. No mesh device is created or required.

Checks:
  * ``ref_num_frames=None`` is an exact no-op (identical cos/sin to the call without the kwarg).
  * with ``ref_num_frames=R`` (downscale/temporal factors = 1) the appended reference block's
    positions/cos/sin are bit-identical to the target grid's first ``min(T, R)`` frames' worth
    (the reference grid == a target grid of ``R`` frames), and the total token length is
    ``sp_pad((T + R) * hw)``.
"""

from __future__ import annotations

import pytest
import torch

from models.tt_dit.models.transformers.ltx import rope_ltx
from models.tt_dit.parallel.config import DiTParallelConfig

TILE_SIZE = 32


@pytest.fixture(autouse=True)
def _passthrough_device(monkeypatch):
    """Intercept the final device cast/shard so the host cos/sin tensors flow straight through."""
    monkeypatch.setattr(rope_ltx, "bf16_tensor_2dshard", lambda t, device=None, shard_mapping=None: t)
    monkeypatch.setattr(rope_ltx, "bf16_tensor", lambda t, device=None, mesh_axis=None, shard_dim=None: t)


def _parallel_config(sp_factor: int) -> DiTParallelConfig:
    return DiTParallelConfig.from_tuples(cfg=(1, 0), sp=(sp_factor, 1), tp=(1, 0))


def _sp_pad(n_real: int, sp_factor: int) -> int:
    divisor = TILE_SIZE * sp_factor
    return ((n_real + divisor - 1) // divisor) * divisor


def _video_kwargs(sp_factor: int) -> dict:
    return dict(
        inner_dim=48,
        num_attention_heads=4,
        theta=10000.0,
        max_pos=[20, 2048, 2048],
        mesh_device=None,
        parallel_config=_parallel_config(sp_factor),
        fps=24.0,
    )


def _av_kwargs(sp_factor: int) -> dict:
    return dict(
        theta=10000.0,
        mesh_device=None,
        parallel_config=_parallel_config(sp_factor),
        fps=24.0,
        cross_pe_max_pos=20,
    )


# (latent_frames T, ref_num_frames R, latent_height H, latent_width W, sp_factor)
CASES = [
    (4, 2, 2, 2, 1),
    (4, 4, 2, 2, 1),
    (5, 3, 2, 3, 2),
    (3, 6, 2, 2, 1),  # R > T: identity holds for the first min(T, R) frames
]


@pytest.mark.parametrize("T, R, H, W, sp", CASES)
def test_prepare_video_rope_ref_noop(T, R, H, W, sp):
    kw = _video_kwargs(sp)
    cos_base, sin_base = rope_ltx.prepare_video_rope(T, H, W, **kw)
    cos_none, sin_none = rope_ltx.prepare_video_rope(T, H, W, ref_num_frames=None, **kw)
    assert torch.equal(cos_base, cos_none)
    assert torch.equal(sin_base, sin_none)


@pytest.mark.parametrize("T, R, H, W, sp", CASES)
def test_prepare_video_rope_two_grid_identity(T, R, H, W, sp):
    kw = _video_kwargs(sp)
    cos_base, _ = rope_ltx.prepare_video_rope(T, H, W, **kw)
    cos_ref, sin_ref = rope_ltx.prepare_video_rope(T, H, W, ref_num_frames=R, **kw)

    hw = H * W
    # Total length is (T + R) * hw rounded up to the SP boundary.
    assert cos_ref.shape[2] == _sp_pad((T + R) * hw, sp)

    # Appending the reference block must not perturb the target rows.
    assert torch.equal(cos_ref[:, :, : T * hw, :], cos_base[:, :, : T * hw, :])

    # The reference block (rows [T*hw, (T+R)*hw)) is a fresh R-frame grid == the target's first R
    # frames, so cos/sin over the first min(T, R) frames' worth are bit-identical.
    m = min(T, R) * hw
    assert torch.equal(cos_ref[:, :, :m, :], cos_ref[:, :, T * hw : T * hw + m, :])
    assert torch.equal(sin_ref[:, :, :m, :], sin_ref[:, :, T * hw : T * hw + m, :])


@pytest.mark.parametrize("T, R, H, W, sp", CASES)
def test_prepare_av_cross_pe_ref_noop(T, R, H, W, sp):
    kw = _av_kwargs(sp)
    audio_N = 4
    base = rope_ltx.prepare_av_cross_pe(T, H, W, audio_N, audio_N, **kw)
    none = rope_ltx.prepare_av_cross_pe(T, H, W, audio_N, audio_N, ref_num_frames=None, **kw)
    for b, n in zip(base, none):
        assert torch.equal(b, n)


@pytest.mark.parametrize("T, R, H, W, sp", CASES)
def test_prepare_av_cross_pe_two_grid_identity(T, R, H, W, sp):
    kw = _av_kwargs(sp)
    audio_N = 4
    base = rope_ltx.prepare_av_cross_pe(T, H, W, audio_N, audio_N, **kw)
    ref = rope_ltx.prepare_av_cross_pe(T, H, W, audio_N, audio_N, ref_num_frames=R, **kw)

    v_q_cos_base, v_q_sin_base = base[0], base[1]
    v_q_cos_ref, v_q_sin_ref = ref[0], ref[1]

    hw = H * W
    assert v_q_cos_ref.shape[2] == _sp_pad((T + R) * hw, sp)

    # Target video rows unchanged by the reference append.
    assert torch.equal(v_q_cos_ref[:, :, : T * hw, :], v_q_cos_base[:, :, : T * hw, :])

    # Reference block's temporal cross-PE == the target's first min(T, R) frames' worth.
    m = min(T, R) * hw
    assert torch.equal(v_q_cos_ref[:, :, :m, :], v_q_cos_ref[:, :, T * hw : T * hw + m, :])
    assert torch.equal(v_q_sin_ref[:, :, :m, :], v_q_sin_ref[:, :, T * hw : T * hw + m, :])

    # Audio-side rope (a_q, a_k) is independent of the video reference block.
    for i in (2, 3, 4, 5):
        assert torch.equal(base[i], ref[i])


def test_reference_and_keyframe_mutually_exclusive(expect_error):
    kw = _video_kwargs(1)
    with expect_error(AssertionError, r"ref_num_frames.*keyframe conditioning.*anchor_frames.*mutually exclusive"):
        rope_ltx.prepare_video_rope(4, 2, 2, anchor_frames=[1], ref_num_frames=2, **kw)
    av = _av_kwargs(1)
    with expect_error(AssertionError, r"ref_num_frames.*keyframe conditioning.*anchor_frames.*mutually exclusive"):
        rope_ltx.prepare_av_cross_pe(4, 2, 2, 4, 4, anchor_frames=[1], ref_num_frames=2, **av)
