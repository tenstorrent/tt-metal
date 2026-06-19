# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canvas denoise attention-mask geometry (reference, #47462).

During denoise the decoder is **bidirectional** and cross-attends to the prompt
by concatenating encoder K/V **in front of** the canvas K/V (prefix-style). So
for canvas *queries* the key axis is ``[prompt (P) ; canvas (C)]`` of length
``P + C`` and the additive mask is ``[C, P + C]`` (0 = attend, -inf = masked).

Because ttnn SDPA makes ``sliding_window_size`` and ``attn_mask`` mutually
exclusive (`sdpa_device_operation.cpp:67-68`), the symmetric sliding window must
be **baked into an explicit mask** — this module is the reference for that bake,
and the oracle the device mask is PCC'd against.

Layer types (plan.md §2 / §6 #47462):
  * **full-attention (global):** canvas sees the whole prompt + whole canvas
    (fully bidirectional) -> all-attend.
  * **sliding (local):** symmetric window of half-width ``window_half`` (total
    ``2*window_half + 1``) over **absolute** positions; canvas query ``i`` has
    absolute position ``P + i``.

Open question (plan.md §10), parameterized rather than guessed: does the local
window extend over the prompt prefix, or is the prompt fully visible to the
canvas on local layers? ``prompt_fully_visible`` selects the variant; the
default (False) windows over the concatenated absolute positions, which
naturally covers the prompt tail for early canvas positions. Reconcile against
the HF reference once importable (#47468).
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
    is_sliding: bool,
    window_half: int | None = None,
    inclusive: bool = True,
    prompt_fully_visible: bool = False,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Additive ``[canvas_len, prompt_len + canvas_len]`` mask for canvas queries.

    ``is_sliding=False`` (full-attn layer) -> all-attend (zeros).
    ``is_sliding=True`` -> symmetric window of half-width ``window_half`` over
    absolute positions; ``inclusive`` toggles ``|q-k| <= W`` vs ``< W`` (the
    vLLM-``<W`` vs ttnn-centered reconciliation, plan.md §6 #47462);
    ``prompt_fully_visible`` keeps all prompt keys visible and windows only
    canvas<->canvas.
    """
    total_k = prompt_len + canvas_len
    if is_sliding:
        if window_half is None:
            raise ValueError("window_half is required for sliding (local) layers")
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
