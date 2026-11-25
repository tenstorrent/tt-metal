# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

# SPDX-License-Identifier: Apache-2.0

"""Torch (CPU) reference for the OpenVDN operations, copied from OpenVDN/vdn-minimax-h3 @e262cb57 (Apache-2.0) with
the training-path spelling and nothing else: `src/models/linear_attention/{scan,delta_rule}.py` and
`src/models/softmax_attention/window.py`. Used by the unit tests; the model's own repo stays the source of truth.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F


def window_bounds(num_frames: int, radius: int, chunk: int = 0) -> list[tuple[int, int]]:
    if chunk <= 0:
        return [(t - radius, t + radius) for t in range(num_frames)]
    return [(((t // chunk) - radius) * chunk, ((t // chunk) + radius + 1) * chunk - 1) for t in range(num_frames)]


def frame_statistics(kf: torch.Tensor, vf: torch.Tensor, beta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`_frame_statistics_pass(a_fp32=True)`, training path (plain fp32 matmul). kf, vf: [F,H,S,d] bf16; beta:
    [F,H,S] bf16."""
    kf32 = kf.float()
    scaled32 = kf32 * beta.unsqueeze(-1).float()
    vb = vf * beta.unsqueeze(-1).to(vf.dtype)
    a = torch.matmul(scaled32.transpose(-1, -2), kf32)
    a = 0.5 * (a + a.transpose(-1, -2))
    b = torch.matmul(vb.transpose(-1, -2).float(), kf32)
    return a, b


def vdn_inverse(a: torch.Tensor) -> torch.Tensor:
    """`(I + A)^-1` as `VdnDelta.factor_apply` forms it."""
    a32 = a.float()
    eye = torch.eye(a32.shape[-1], dtype=torch.float32).expand_as(a32)
    chol = torch.linalg.cholesky(a32 + eye)
    linv = torch.linalg.solve_triangular(chol, eye, upper=False, left=True)
    return linv.transpose(-1, -2) @ linv


def vdn_factor_apply(alpha: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    inv = vdn_inverse(a)
    return alpha.unsqueeze(-1) * inv, b.float() @ inv


def run_scans(
    transitions: torch.Tensor, injections: torch.Tensor, text_state: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """`_run_scans` after `factor_apply`."""
    num_frames = transitions.shape[0]
    start = torch.zeros_like(injections[0]) if text_state is None else text_state.to(injections.dtype)
    state, forward = start, []
    for f in range(num_frames):
        state = state @ transitions[f] + injections[f]
        forward.append(state)
    state, reverse = start, [None] * num_frames
    for f in range(num_frames - 1, -1, -1):
        state = state @ transitions[f] + injections[f]
        reverse[f] = state
    return torch.stack(forward), torch.stack(reverse)


def gather_linear_state(
    prefix: torch.Tensor,
    suffix: torch.Tensor,
    alpha: torch.Tensor,
    bounds: Sequence[tuple[int, int]],
    text_state: torch.Tensor | None = None,
) -> torch.Tensor:
    """`gather_linear_state(bridge="alpha")`."""
    num_frames = prefix.shape[0]
    last_before = torch.tensor([lo for lo, _ in bounds]) - 1
    first_after = torch.tensor([hi for _, hi in bounds]) + 1
    before_idx, after_idx = last_before.clamp(min=0), first_after.clamp(max=num_frames - 1)
    has_before, has_after = last_before >= 0, first_after < num_frames
    bridge_before, bridge_after = (last_before + 1).clamp(min=0), first_after.clamp(max=num_frames)
    frames = torch.arange(num_frames)

    state_before, state_after = prefix[before_idx], suffix[after_idx]
    if text_state is not None:
        text_state = text_state.to(state_before.dtype)
        state_before = torch.where(has_before.view(-1, 1, 1, 1), state_before, text_state)
        state_after = torch.where(has_after.view(-1, 1, 1, 1), state_after, text_state)
    log_alpha = torch.log(alpha.clamp_min(1e-12))
    log_prefix = torch.cat([torch.zeros_like(log_alpha[:1]), log_alpha.cumsum(0)])
    state_before = state_before * torch.exp(log_prefix[frames + 1] - log_prefix[bridge_before]).unsqueeze(2)
    state_after = state_after * torch.exp(log_prefix[bridge_after] - log_prefix[frames]).unsqueeze(2)
    if text_state is not None:
        return state_before + state_after
    return state_before * has_before.view(-1, 1, 1, 1) + state_after * has_after.view(-1, 1, 1, 1)


def window_softmax(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    video_start: int,
    num_frames: int,
    tokens_per_frame: int,
    bounds: Sequence[tuple[int, int]],
    scale: float,
    anchor_frames: str = "both",
) -> torch.Tensor:
    """`window_softmax_reference`. query/key/value: [total, H, d]."""
    total, heads, head_dim = query.shape
    video_end = video_start + num_frames * tokens_per_frame
    global_idx = torch.cat([torch.arange(0, video_start), torch.arange(video_end, total)])
    out = torch.empty_like(query)

    def sdpa(q_rows, k_rows, v_rows):
        attended = F.scaled_dot_product_attention(
            q_rows.permute(1, 0, 2).unsqueeze(0),
            k_rows.permute(1, 0, 2).unsqueeze(0),
            v_rows.permute(1, 0, 2).unsqueeze(0),
            scale=scale,
        )
        return attended.squeeze(0).permute(1, 0, 2)

    if global_idx.numel():
        out[global_idx] = sdpa(query[global_idx], key, value)
    global_key, global_value = key[global_idx], value[global_idx]
    frame_shape = (num_frames, tokens_per_frame, heads, head_dim)
    video_query = query[video_start:video_end].view(frame_shape)
    video_key = key[video_start:video_end].view(frame_shape)
    video_value = value[video_start:video_end].view(frame_shape)
    for frame in range(num_frames):
        lo, hi = max(bounds[frame][0], 0), min(bounds[frame][1], num_frames - 1)
        if anchor_frames in ("rows", "both") and frame in (0, num_frames - 1):
            lo, hi = 0, num_frames - 1
        extra = [
            f for f in ((0, num_frames - 1) if anchor_frames in ("columns", "both") else ()) if not lo <= f <= hi
        ]
        window_key = torch.cat(
            [global_key, video_key[lo : hi + 1].reshape(-1, heads, head_dim)]
            + [video_key[f].reshape(-1, heads, head_dim) for f in extra]
        )
        window_value = torch.cat(
            [global_value, video_value[lo : hi + 1].reshape(-1, heads, head_dim)]
            + [video_value[f].reshape(-1, heads, head_dim) for f in extra]
        )
        row = video_start + frame * tokens_per_frame
        out[row : row + tokens_per_frame] = sdpa(video_query[frame], window_key, window_value)
    return out
