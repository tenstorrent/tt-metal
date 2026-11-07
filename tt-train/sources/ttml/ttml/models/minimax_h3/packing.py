# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Host-side (numpy) geometry of a MiniMax-H3 packed sequence: row layout, rotary tables, modality tags, per-row
timesteps, the timestep MLP and the selection matrices that pick AdaLN rows on device."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

VIDEO_TAG, TEXT_TAG, AUDIO_TAG = 0, 1, 2
MODALITY_NUM = 3
AUDIO_CHANNELS = 2
AUDIO_LATENTS = 2  # round(1/24 s * 40 latents/s)
PATCH_SIZE = (1, 2, 2)
TILE = 32

_ROPE_FRAME_RESCALE = 5.0 / 3.0
_ROPE_FRAMES_PER_LATENT = (1, 4, 4, 4, 4)
_ROPE_SPATIAL_SCALE = 32


def tile_pad(n: int) -> int:
    return (n + TILE - 1) // TILE * TILE


@dataclass(frozen=True)
class Layout:
    position_ids: np.ndarray  # (S, 3) float64 (t, h, w)
    token_tags: np.ndarray  # (S,) int64
    text_indices: np.ndarray
    audio_indices: np.ndarray
    video_indices: np.ndarray

    @property
    def sequence_length(self) -> int:
        return int(self.position_ids.shape[0])


def patchify_video_latents(latents: np.ndarray, patch_size=PATCH_SIZE) -> np.ndarray:
    """(C, T, H, W) -> (rows, C * prod(patch)), frame-major then row-major."""
    pt, ph, pw = patch_size
    c, t, h, w = latents.shape
    x = latents.reshape(c, t // pt, pt, h // ph, ph, w // pw, pw).transpose(1, 3, 5, 0, 2, 4, 6)
    return np.ascontiguousarray(x.reshape(-1, c * pt * ph * pw))


def _spatial_position_grid(dim: int, patch: int, sqrt_area: float) -> np.ndarray:
    ratio = dim / sqrt_area
    left = (1.0 - ratio) / 2.0
    return np.linspace(left, left + ratio, dim // patch, endpoint=False) * _ROPE_SPATIAL_SCALE


def _temporal_position_grid(num_latent_frames: int, origin: float) -> np.ndarray:
    spans = np.array(
        [
            _ROPE_FRAME_RESCALE * _ROPE_FRAMES_PER_LATENT[i % len(_ROPE_FRAMES_PER_LATENT)]
            for i in range(num_latent_frames)
        ],
        dtype=np.float64,
    )
    return origin + np.concatenate([np.zeros(1, dtype=np.float64), np.cumsum(spans[:-1])])


def build_packed_sequence(
    num_text_tokens: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
    num_audio_latents: int = AUDIO_LATENTS,
    patch_size=PATCH_SIZE,
    audio_channels: int = AUDIO_CHANNELS,
) -> Layout:
    """`[text | audio | video]` rows with (t, h, w) rotary coordinates and modality tags."""
    _, ph, pw = patch_size
    rows_per_frame = (latent_height // ph) * (latent_width // pw)
    num_audio_rows = num_audio_latents * audio_channels
    audio_start = num_text_tokens
    video_start = audio_start + num_audio_rows
    seq_len = video_start + num_latent_frames * rows_per_frame

    position_ids = np.zeros((seq_len, 3), dtype=np.float64)
    position_ids[:num_text_tokens, 0] = np.arange(num_text_tokens, dtype=np.float64)
    sqrt_area = np.sqrt(latent_height * latent_width)
    height_grid = _spatial_position_grid(latent_height, ph, sqrt_area)
    width_grid = _spatial_position_grid(latent_width, pw, sqrt_area)
    hh, ww = np.meshgrid(height_grid, width_grid, indexing="ij")
    frame_grid = np.stack([hh.reshape(-1), ww.reshape(-1)], axis=-1)
    if num_audio_rows:
        audio_time = float(num_text_tokens) + np.arange(num_audio_latents, dtype=np.float64)
        position_ids[audio_start:video_start, 0] = np.tile(audio_time, audio_channels)
        position_ids[audio_start:video_start, 2] = np.concatenate(
            [
                np.full(num_audio_latents, float(width_grid[0])),
                np.full(num_audio_rows - num_audio_latents, float(width_grid[-1])),
            ]
        )
    frame_times = _temporal_position_grid(num_latent_frames, float(num_text_tokens))
    video = np.empty((num_latent_frames, rows_per_frame, 3), dtype=np.float64)
    video[:, :, 0] = frame_times[:, None]
    video[:, :, 1:] = frame_grid[None]
    position_ids[video_start:] = video.reshape(-1, 3)

    text_indices = np.arange(num_text_tokens, dtype=np.int64)
    audio_indices = np.arange(audio_start, video_start, dtype=np.int64)
    video_indices = np.arange(video_start, seq_len, dtype=np.int64)
    token_tags = np.empty(seq_len, dtype=np.int64)
    token_tags[text_indices] = TEXT_TAG
    token_tags[audio_indices] = AUDIO_TAG
    token_tags[video_indices] = VIDEO_TAG
    return Layout(position_ids, token_tags, text_indices, audio_indices, video_indices)


def build_row_timesteps(layout: Layout, video_t: float, audio_t: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Distinct sorted timesteps (float32) and the per-row index into them; text rows follow the video."""
    row_t = np.full(layout.sequence_length, video_t, dtype=np.float32)
    row_t[layout.audio_indices] = audio_t
    timesteps, indices = np.unique(row_t, return_inverse=True)
    return timesteps.astype(np.float32), indices.reshape(-1).astype(np.int64)


def shared_timesteps(per_sample: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Several `build_row_timesteps` results merged into one distinct-timestep table, with each sample's per-row
    indices into it, so the samples share one set of AdaLN tables."""
    shared = np.unique(np.concatenate([t for t, _ in per_sample]))
    return shared, [np.searchsorted(shared, t[i]).astype(np.int64) for t, i in per_sample]


def adaln_indices(layout: Layout, timestep_indices: np.ndarray) -> np.ndarray:
    return timestep_indices * MODALITY_NUM + layout.token_tags


def sample_timestep(shift: float, rng: random.Random) -> float:
    """t = 1 - sigma with sigma = shift * u / (1 + (shift - 1) * u); t = 1 clean, t = 0 pure noise."""
    u = rng.random()
    return 1.0 - shift * u / (1 + (shift - 1) * u)


# --------------------------------------------------------------------------- rotary


def qk_channel_permutation(head_dim: int = 128, rotary_dim: int = 96) -> np.ndarray:
    """`new[i] = old[perm[i]]`: rotate-half pairs (j, j + rotary/2) become adjacent (2j, 2j + 1); the rest stays.

    Applied to the output rows of to_q / to_k and to norm_q / norm_k, so the device rope rotates adjacent pairs.
    Attention scores are invariant to a shared q/k channel permutation, so nothing downstream changes."""
    half = rotary_dim // 2
    perm = np.empty(head_dim, dtype=np.int64)
    perm[0:rotary_dim:2] = np.arange(half)
    perm[1:rotary_dim:2] = np.arange(half) + half
    perm[rotary_dim:] = np.arange(rotary_dim, head_dim)
    return perm


def rope_tables(
    position_ids: np.ndarray, padded_len: int, head_dim: int = 128, rope_freq_dim: int = 16, theta: float = 10000.0
) -> tuple[np.ndarray, np.ndarray]:
    """(padded_len, head_dim) float32 cos / sin in the adjacent-pair layout; channels past 6 * rope_freq_dim pass
    through (cos 1, sin 0). Row j < 3 * rope_freq_dim uses axis j // rope_freq_dim and frequency j % rope_freq_dim."""
    rotary = 2 * 3 * rope_freq_dim
    inv_freq = 1.0 / (theta ** (np.arange(0, 2 * rope_freq_dim, 2, dtype=np.float32) / (2 * rope_freq_dim)))
    pos = position_ids.astype(np.float32)
    ang = (pos[:, :, None] * inv_freq[None, None, :]).reshape(pos.shape[0], -1)  # (S, 48): t | h | w
    cos = np.ones((padded_len, head_dim), dtype=np.float32)
    sin = np.zeros((padded_len, head_dim), dtype=np.float32)
    cos[: pos.shape[0], :rotary] = np.repeat(np.cos(ang), 2, axis=1)
    sin[: pos.shape[0], :rotary] = np.repeat(np.sin(ang), 2, axis=1)
    return cos, sin


def rope_trans_mat(tile: int = TILE) -> np.ndarray:
    """(1, 1, 32, 32): x @ mat = (-x_odd, x_even) on adjacent pairs."""
    mat = np.zeros((1, 1, tile, tile), dtype=np.float32)
    idx = np.arange(0, tile, 2)
    mat[..., idx, idx + 1] = 1.0
    mat[..., idx + 1, idx] = -1.0
    return mat


# --------------------------------------------------------------------------- timestep conditioning


def sinusoidal_timestep_embedding(timesteps: np.ndarray, dim: int = 256, max_period: float = 10000.0) -> np.ndarray:
    """diffusers `Timesteps(flip_sin_to_cos=True, downscale_freq_shift=0)`: [cos | sin], float32."""
    half = dim // 2
    exponent = -math.log(max_period) * np.arange(half, dtype=np.float32) / half
    args = timesteps.astype(np.float32)[:, None] * np.exp(exponent)[None, :]
    return np.concatenate([np.cos(args), np.sin(args)], axis=-1)


def _silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def time_embedding(timesteps: np.ndarray, weights: dict[str, np.ndarray]) -> np.ndarray:
    """fp32 `time_embedder` on host: (T, time_embed_dim). `weights` holds the four `time_embedder.*` arrays."""
    e = sinusoidal_timestep_embedding(timesteps, weights["time_embedder.linear_1.weight"].shape[1])
    h = _silu(e @ weights["time_embedder.linear_1.weight"].T + weights["time_embedder.linear_1.bias"])
    return (h @ weights["time_embedder.linear_2.weight"].T + weights["time_embedder.linear_2.bias"]).astype(np.float32)


def selection_matrix(indices: np.ndarray, padded_len: int, num_cols: int = TILE, fill: int | None = None) -> np.ndarray:
    """(padded_len, num_cols) one-hot rows: `sel @ table` gathers `table[indices]`. Padded rows select `fill`."""
    sel = np.zeros((padded_len, num_cols), dtype=np.float32)
    sel[np.arange(indices.shape[0]), indices] = 1.0
    if fill is not None and padded_len > indices.shape[0]:
        sel[indices.shape[0] :, fill] = 1.0
    return sel


def attention_mask(num_valid: int, padded_len: int) -> np.ndarray:
    """(1, 1, padded_len, padded_len) with 1 where the key is a real row; padded queries still see real keys."""
    mask = np.zeros((1, 1, padded_len, padded_len), dtype=np.float32)
    mask[..., :num_valid] = 1.0
    return mask
