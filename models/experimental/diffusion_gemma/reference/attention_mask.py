# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canvas denoise attention-mask geometry (reference, #47462).

During denoise the decoder is **bidirectional** and cross-attends to the prompt
by concatenating encoder K/V **in front of** the canvas K/V (prefix-style). So
for canvas *queries* the key axis is ``[prompt (P) ; canvas (C)]`` of length
``P + C`` and the additive mask is ``[C, P + C]`` (0 = attend, -inf = masked).

HF VISIBILITY: full-attention layers are fully bidirectional. Sliding layers use
HF's ``sliding_window_bidirectional_overlay`` when ``prompt_len`` grows beyond the
window, so they attend only keys with ``abs(q_idx - kv_idx) <= sliding_window``.

:func:`build_canvas_denoise_mask` returns an all-attend mask by default for
backwards compatibility and short-prompt tests. Pass ``layer_type="sliding_attention"``
plus ``sliding_window`` to reproduce HF sliding-layer visibility.
"""

from __future__ import annotations

import torch


def canvas_positions(prompt_len: int, canvas_len: int, *, device=None) -> torch.Tensor:
    """Absolute (RoPE) positions of the canvas tokens: ``[prompt_len + i]``."""
    return prompt_len + torch.arange(canvas_len, device=device)


def build_canvas_reveal_denoise_mask(
    prompt_len: int,
    canvas_len: int,
    p_max: int,
    *,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    enforce_sliding_window: bool = False,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Fixed-shape ``[canvas_len, p_max + canvas_len]`` reveal mask (paged-prefix Phase 1).

    Unlike :func:`build_canvas_denoise_mask`, the key axis prefix span is a CONSTANT
    ``p_max`` (not the growing ``prompt_len``), so the mask — and therefore the traced
    denoise attention graph — is shape-invariant across blocks (capture-once/replay-many,
    see ``doc/optimize_perf/paged_prefix_denoise_design.md``). The growing committed prefix
    is exposed purely through the mask CONTENT:

    - Canvas queries are anchored at the TRUE committed ``prompt_len`` (``q_abs = prompt_len + i``),
      NOT ``p_max`` — decoupling the read span from the reveal length is the whole point.
    - Prefix key slot ``j`` in ``[0, p_max)`` maps to absolute position ``j`` and is revealed
      iff ``j < prompt_len`` (committed). Uncommitted slots ``[prompt_len:p_max]`` are always
      ``neg_inf`` — this is the explicit reveal predicate (do NOT rely on a window to hide them).
    - Canvas key slot ``j'`` in ``[0, canvas_len)`` lives at key columns ``[p_max:p_max+canvas_len]``.

    ``enforce_sliding_window=False`` (Phase 1) → committed keys are all-attend (matches today's
    maskless production denoise path; the reveal is the ONLY masking). ``True`` (Phase 2) → also
    apply HF's bidirectional window ``abs(q_abs - k_abs) <= sliding_window`` (a decision change vs
    today → its own #48291 re-validation). ``layer_type='full_attention'`` ignores the window.
    """
    if p_max < prompt_len:
        raise ValueError(f"p_max ({p_max}) must be >= prompt_len ({prompt_len})")
    total_k = p_max + canvas_len
    i = torch.arange(canvas_len, device=device).unsqueeze(1)  # [C, 1] canvas row index
    q_abs = (prompt_len + i).to(torch.long)  # [C, 1] canvas absolute position
    # Key absolute position: prefix slot j -> abs j; canvas slot j' -> abs prompt_len + j'.
    prefix_abs = torch.arange(p_max, device=device)  # [p_max]
    canvas_abs = prompt_len + torch.arange(canvas_len, device=device)  # [C]
    k_abs = torch.cat([prefix_abs, canvas_abs]).unsqueeze(0).to(torch.long)  # [1, p_max+C]

    # Committed predicate: prefix columns require j < prompt_len; canvas columns always committed.
    committed = torch.zeros(total_k, dtype=torch.bool, device=device)
    committed[:p_max] = prefix_abs < prompt_len
    committed[p_max:] = True
    allowed = committed.unsqueeze(0).expand(canvas_len, total_k).clone()  # [C, p_max+C]

    if enforce_sliding_window and layer_type == "sliding_attention":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError("sliding_window must be positive for sliding_attention")
        allowed = allowed & ((q_abs - k_abs).abs() <= sliding_window)
    elif layer_type not in (None, "full_attention", "sliding_attention"):
        raise ValueError(f"unsupported layer_type {layer_type!r}")

    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )


def build_canvas_denoise_mask(
    prompt_len: int,
    canvas_len: int,
    *,
    layer_type: str | None = None,
    sliding_window: int | None = None,
    local_window: bool = False,
    window_half: int | None = None,
    inclusive: bool = True,
    prompt_fully_visible: bool = False,
    causal: bool = False,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Additive ``[canvas_len, prompt_len + canvas_len]`` mask for canvas queries.

    Default (``local_window=False`` and ``layer_type is None``) → all-attend
    (zeros), preserving the original short-prompt oracle behavior.

    ``layer_type="full_attention"`` → all-attend. ``layer_type="sliding_attention"``
    → HF-style bidirectional sliding visibility, requiring ``sliding_window``.

    ``causal=True`` (the COMMIT phase — #47557 commit batching) turns the mask into
    a *causal* prefix+canvas mask: a canvas query at absolute position
    ``prompt_len + i`` attends key position ``p`` iff ``p <= prompt_len + i`` (all of
    the frozen prefix ``0..prompt_len-1`` plus canvas positions ``0..i``). For a
    ``sliding_attention`` layer the causal window is additionally clipped to the last
    ``sliding_window`` positions (``prompt_len + i - p < sliding_window``). This is the
    per-token visibility that the sequential single-token decode-append produces
    (each committed token's decode SDPA attends causally over the frozen cache), so a
    single 256-query masked prefill reproduces the 256 sequential appends. ``causal``
    composes with ``layer_type`` (full vs sliding) and is mutually exclusive with the
    ``local_window`` op-test path.

    ``local_window=True`` (NON-canonical, op-test only) → symmetric window of
    half-width ``window_half`` over absolute positions; ``inclusive`` toggles
    ``|q-k| <= W`` vs ``< W``; ``prompt_fully_visible`` keeps all prompt keys visible
    and windows only canvas↔canvas. Use solely to drive the ttnn SDPA windowed-mask
    path, never as the denoise reference.
    """
    total_k = prompt_len + canvas_len
    if causal:
        if local_window:
            raise ValueError("causal=True is mutually exclusive with local_window=True")
        q_abs = canvas_positions(prompt_len, canvas_len, device=device).unsqueeze(1)  # [C, 1]
        k_abs = torch.arange(total_k, device=device).unsqueeze(0)  # [1, P+C]
        allowed = k_abs <= q_abs  # causal: key at or before the query's absolute position
        if layer_type == "sliding_attention":
            if sliding_window is None or sliding_window <= 0:
                raise ValueError("sliding_window must be positive for sliding_attention")
            # Last ``sliding_window`` positions inclusive of self (HF causal-sliding:
            # attend iff 0 <= q_abs - k_abs < sliding_window).
            allowed = allowed & ((q_abs - k_abs) < sliding_window)
        elif layer_type not in (None, "full_attention"):
            raise ValueError(f"unsupported layer_type {layer_type!r}")
        return torch.where(
            allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
        )
    if local_window:
        if window_half is None:
            raise ValueError("window_half is required when local_window=True")
        q_abs = canvas_positions(prompt_len, canvas_len, device=device).unsqueeze(1)  # [C, 1]
        k_abs = torch.arange(total_k, device=device).unsqueeze(0)  # [1, P+C]
        dist = (q_abs - k_abs).abs()
        allowed = dist <= window_half if inclusive else dist < window_half
        if prompt_fully_visible:
            allowed[:, :prompt_len] = True
    elif layer_type == "sliding_attention":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError("sliding_window must be positive for sliding_attention")
        q_abs = canvas_positions(prompt_len, canvas_len, device=device).unsqueeze(1)  # [C, 1]
        k_abs = torch.arange(total_k, device=device).unsqueeze(0)  # [1, P+C]
        allowed = (q_abs - k_abs).abs() <= sliding_window
    elif layer_type in (None, "full_attention"):
        allowed = torch.ones(canvas_len, total_k, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"unsupported layer_type {layer_type!r}")

    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )
