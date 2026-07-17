# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canvas denoise attention-mask geometry (reference, #47462).

During denoise the decoder is **bidirectional** and cross-attends to the prompt
by concatenating encoder K/V **in front of** the canvas K/V (prefix-style). So
for canvas *queries* the key axis is ``[prompt (P) ; canvas (C)]`` of length
``P + C`` and the additive mask is ``[C, P + C]`` (0 = attend, -inf = masked).

CANONICAL VISIBILITY (resolves plan.md §10 against the source): the DiffusionGemma
decoder is **fully bidirectional for BOTH full-attention AND sliding layers** —
``modeling_diffusion_gemma.py:1399-1438`` builds the decoder mask with
``bidirectional_mask_function`` (``q_idx >= 0`` → always True) for every layer type
and uses ``sliding_window`` only to shape offsets / the SDPA skip hint, NEVER to
restrict visibility ("DiT module doesn't need a sliding mask and has to attend
fully to prev context and itself"). So the denoise mask is **all-attend** — the
local window does NOT gate which keys a canvas query sees.

:func:`build_canvas_denoise_mask` therefore returns an all-attend mask by default.
The symmetric-window bake is kept behind ``local_window=True`` ONLY to exercise the
ttnn SDPA windowed-mask path (``sliding_window_size`` and ``attn_mask`` are mutually
exclusive, ``sdpa_device_operation.cpp:67-68``) — it is **not** the canonical
denoise geometry and must not be used as the denoise oracle.
"""

from __future__ import annotations

import torch


def canvas_positions(prompt_len: int, canvas_len: int, *, device=None) -> torch.Tensor:
    """Absolute (RoPE) positions of the canvas tokens: ``[prompt_len + i]``."""
    return prompt_len + torch.arange(canvas_len, device=device)


def build_canvas_denoise_mask(
    prompt_len: int,
    canvas_len: int,
    *,
    local_window: bool = False,
    window_half: int | None = None,
    inclusive: bool = True,
    prompt_fully_visible: bool = False,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Additive ``[canvas_len, prompt_len + canvas_len]`` mask for canvas queries.

    Default (``local_window=False``) → **all-attend (zeros)** for every layer type,
    matching the canonical fully-bidirectional decoder (see module docstring). This
    is the denoise oracle.

    ``local_window=True`` (NON-canonical, op-test only) → symmetric window of
    half-width ``window_half`` over absolute positions; ``inclusive`` toggles
    ``|q-k| <= W`` vs ``< W``; ``prompt_fully_visible`` keeps all prompt keys visible
    and windows only canvas↔canvas. Use solely to drive the ttnn SDPA windowed-mask
    path, never as the denoise reference.
    """
    total_k = prompt_len + canvas_len
    if local_window:
        if window_half is None:
            raise ValueError("window_half is required when local_window=True")
        q_abs = canvas_positions(prompt_len, canvas_len, device=device).unsqueeze(1)  # [C, 1]
        k_abs = torch.arange(total_k, device=device).unsqueeze(0)  # [1, P+C]
        dist = (q_abs - k_abs).abs()
        allowed = dist <= window_half if inclusive else dist < window_half
        if prompt_fully_visible:
            allowed[:, :prompt_len] = True
    else:
        allowed = torch.ones(canvas_len, total_k, dtype=torch.bool, device=device)

    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )
