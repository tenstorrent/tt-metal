# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Canvas denoise attention-mask geometry (reference, #47462).

During denoise the decoder is **bidirectional** and cross-attends to the prompt
by concatenating encoder K/V **in front of** the canvas K/V (prefix-style). So
for canvas *queries* the key axis is ``[prompt (P) ; canvas (C)]`` of length
``P + C`` and the additive mask is ``[C, P + C]`` (0 = attend, -inf = masked).

HF VISIBILITY (#51080). Full-attention layers are fully bidirectional. Sliding
layers do **not** apply a ``abs(q_idx - kv_idx) <= sliding_window`` staircase:

* ``DiffusionGemmaDecoderModel.create_diffusion_decoder_attention_mask`` returns ``None`` for
  both masks on the ordinary unpadded ``DynamicCache`` path, so no sliding mask exists at all;
* the window is purely a CACHE-TRUNCATION effect — ``DynamicSlidingWindowLayer.update`` retains
  only ``full_key_states[:, :, -sliding_window + 1:, :]``, i.e. ``sliding_window - 1`` tokens;
* when a padding mask *is* materialized it is expanded from a 1-D per-key vector, so it has NO
  query-index dependence — every canvas row sees the same key set.

So sliding-layer denoise visibility = the ``sliding_window - 1`` most recent COMMITTED tokens,
all-attend, plus the full canvas. It is a per-KEY predicate on the committed prefix, not a
per-(query, key) distance predicate. Pinned by
``tests/test_reference.py``.

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
    hidden_prefix_span: tuple[int, int] | None = None,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Fixed-shape ``[canvas_len, p_max + canvas_len]`` reveal mask (paged-prefix Phase 1).

    Unlike :func:`build_canvas_denoise_mask`, the key axis prefix span is a CONSTANT
    ``p_max`` (not the growing ``prompt_len``), so the mask — and therefore the traced
    denoise attention graph — is shape-invariant across blocks (capture-once/replay-many).
    The growing committed prefix is exposed purely through the mask CONTENT:

    - Canvas queries are anchored at the TRUE committed ``prompt_len`` (``q_abs = prompt_len + i``),
      NOT ``p_max`` — decoupling the read span from the reveal length is the whole point.
    - Prefix key slot ``j`` in ``[0, p_max)`` maps to absolute position ``j`` and is revealed
      iff ``j < prompt_len`` (committed). Uncommitted slots ``[prompt_len:p_max]`` are always
      ``neg_inf`` — this is the explicit reveal predicate (do NOT rely on a window to hide them).
    - Canvas key slot ``j'`` in ``[0, canvas_len)`` lives at key columns ``[p_max:p_max+canvas_len]``.

    ``hidden_prefix_span=(lo, hi)`` additionally hides prefix slots ``[lo, hi)`` that are inside the
    committed span but hold no real token. The prefill pad slots are exactly that case: prefill
    right-pads the prompt to a tile multiple and writes K/V for the pad tokens, and the reveal
    predicate is evaluated with the PADDED length, so those keys are revealed even though they are
    garbage — and they sit immediately before the canvas, making the canvas's nearest context noise.

    ``enforce_sliding_window=False`` (Phase 1) → committed keys are all-attend (the reveal is the
    ONLY masking). ``True`` → additionally hide committed keys HF's sliding cache no longer
    retains, i.e. keep only absolute positions ``>= prompt_len - (sliding_window - 1)`` — a
    per-KEY predicate with no query dependence (see the module docstring).
    ``layer_type='full_attention'`` ignores the window.
    """
    if p_max < prompt_len:
        raise ValueError(f"p_max ({p_max}) must be >= prompt_len ({prompt_len})")
    total_k = p_max + canvas_len
    # Key absolute position: prefix slot j -> abs j; canvas slot j' -> abs prompt_len + j'.
    # Visibility here is entirely a per-KEY predicate, so no q_abs/k_abs grid is needed.
    prefix_abs = torch.arange(p_max, device=device)  # [p_max]

    # Committed predicate: prefix columns require j < prompt_len; canvas columns always committed.
    committed = torch.zeros(total_k, dtype=torch.bool, device=device)
    committed[:p_max] = prefix_abs < prompt_len
    committed[p_max:] = True
    if hidden_prefix_span is not None:
        lo, hi = (int(v) for v in hidden_prefix_span)
        if not 0 <= lo <= hi <= p_max:
            raise ValueError(f"hidden_prefix_span {(lo, hi)} must satisfy 0 <= lo <= hi <= p_max ({p_max})")
        committed[lo:hi] = False
    allowed = committed.unsqueeze(0).expand(canvas_len, total_k).clone()  # [C, p_max+C]

    if enforce_sliding_window and layer_type == "sliding_attention":
        if sliding_window is None or sliding_window <= 0:
            raise ValueError("sliding_window must be positive for sliding_attention")
        # HF retains only the last ``sliding_window - 1`` COMMITTED tokens in a sliding layer's
        # cache. Visibility is therefore a per-KEY property of the committed prefix with no
        # query dependence; the canvas is always fully visible.
        keep_from = prompt_len - (sliding_window - 1)
        retained = torch.zeros(total_k, dtype=torch.bool, device=device)
        retained[:p_max] = prefix_abs >= keep_from
        retained[p_max:] = True
        allowed = allowed & retained.unsqueeze(0)
    elif layer_type not in (None, "full_attention", "sliding_attention"):
        raise ValueError(f"unsupported layer_type {layer_type!r}")

    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )


def build_canvas_reveal_denoise_window_mask(
    prompt_len: int,
    canvas_len: int,
    span: int,
    lo: int,
    *,
    sliding_window: int,
    hidden_prefix_span: tuple[int, int] | None = None,
    neg_inf: float = float("-inf"),
    dtype: torch.dtype = torch.float32,
    device=None,
) -> torch.Tensor:
    """Fixed-shape ``[canvas_len, span + canvas_len]`` mask for a BOUNDED sliding-layer read.

    Companion to :func:`build_canvas_reveal_denoise_mask` for the per-layer bounded span: a
    sliding layer reads only ``span`` cache rows starting at absolute position ``lo`` instead of
    the whole ``p_max`` prefix, so prefix column ``r`` maps to absolute position ``lo + r``
    (whereas the full-span builder has ``lo == 0`` and column ``j`` maps to ``j``).

    A column is attended iff BOTH hold:

    * ``lo + r < prompt_len`` — the position is actually committed (the reveal predicate);
    * ``lo + r >= prompt_len - (sliding_window - 1)`` — HF's sliding cache still retains it.

    Canvas columns are always attended. With ``span`` tile-aligned and ``lo = max(0, prompt_len -
    span)``, the two regimes are:

    * ``prompt_len <= span``  -> ``lo == 0``, so this is the leading ``span`` columns of the
      full-span mask (identical content where they overlap);
    * ``prompt_len > span``   -> ``lo == prompt_len - span``, the commit predicate is vacuous and
      the retention predicate reduces to ``r >= span - (sliding_window - 1)``, which is
      ``prompt_len``-INDEPENDENT. So in steady state the mask stops changing between blocks.

    ``hidden_prefix_span=(lo_pad, hi_pad)`` hides the prefill pad slots, the same absolute-position
    span :func:`build_canvas_reveal_denoise_mask` takes. No column arithmetic is needed because
    ``prefix_abs`` already carries absolute positions; a pad slot outside ``[lo, lo + span)`` simply
    does not appear in this window and the predicate is a no-op there. That is the normal case once
    the window has scrolled: the pads sit at the START of the prompt, so as ``lo`` grows past
    ``hi_pad`` a sliding layer stops seeing them on its own and the mask returns to the
    ``prompt_len``-independent steady-state form above. Only the full-attention layers, which read
    the whole ``p_max`` prefix through the other builder, need the pads hidden forever.
    """
    if span <= 0:
        raise ValueError(f"span must be positive, got {span}")
    if lo < 0:
        raise ValueError(f"lo must be non-negative, got {lo}")
    if sliding_window <= 0:
        raise ValueError(f"sliding_window must be positive, got {sliding_window}")
    total_k = span + canvas_len
    prefix_abs = lo + torch.arange(span, device=device)
    keep_from = prompt_len - (sliding_window - 1)

    allowed = torch.zeros(canvas_len, total_k, dtype=torch.bool, device=device)
    allowed[:, :span] = ((prefix_abs < prompt_len) & (prefix_abs >= keep_from)).unsqueeze(0)
    if hidden_prefix_span is not None:
        lo_pad, hi_pad = (int(v) for v in hidden_prefix_span)
        # Only lo <= hi is required. Deliberately NOT bounded to this window: the pads routinely sit
        # outside [lo, lo + span) once it has scrolled, and that is a no-op, not an error.
        if not 0 <= lo_pad <= hi_pad:
            raise ValueError(f"hidden_prefix_span {(lo_pad, hi_pad)} must satisfy 0 <= lo <= hi")
        allowed[:, :span] &= ~((prefix_abs >= lo_pad) & (prefix_abs < hi_pad)).unsqueeze(0)
    allowed[:, span:] = True
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
        # Non-causal (denoise) sliding visibility is a per-KEY predicate: HF's sliding cache
        # retains only the last ``sliding_window - 1`` committed prompt positions, and the whole
        # canvas is always visible. It is NOT a ``abs(q_abs - k_abs) <= sliding_window``
        # staircase — see the module docstring and #51080.
        keep_from = prompt_len - (sliding_window - 1)
        prompt_abs = torch.arange(prompt_len, device=device)
        allowed = torch.zeros(canvas_len, total_k, dtype=torch.bool, device=device)
        allowed[:, :prompt_len] = (prompt_abs >= keep_from).unsqueeze(0)
        allowed[:, prompt_len:] = True
    elif layer_type in (None, "full_attention"):
        allowed = torch.ones(canvas_len, total_k, dtype=torch.bool, device=device)
    else:
        raise ValueError(f"unsupported layer_type {layer_type!r}")

    return torch.where(
        allowed, torch.zeros((), dtype=dtype, device=device), torch.full((), neg_inf, dtype=dtype, device=device)
    )
