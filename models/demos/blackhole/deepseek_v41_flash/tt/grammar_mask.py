# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Grammar masks for the speculative verify round (host side, torch only).

vLLM hands the plugin a PACKED bitmask per structured request: ``ceil(vocab / 32)`` int32 words, bit ``i`` of word ``w`` set = token ``32 * w + i`` may be
sampled. The plugin (``spec_grammar_bitmask_for_tt_batch``) lays it out as ``[W, width, words]`` over the logical rows of the step, all ones (-1) for a row
without a grammar and for the columns past the request's own rows. This module turns that block into what the verify round consumes and holds the
host reference of what the device does with it.

Device design (tt/spec_model.py, M1):

* ONE persistent int32 buffer ``[rows * T, words]`` (``T`` = ``U * n`` block rows of a mesh row, user-major ``u * n + j``) sharded over the mesh as
  ``dims=(0, 1)``: mesh row ``r`` holds its ``T`` block rows, mesh column ``c`` the ``words / cols`` words of its vocab shard. 129280 / 8 = 16160 = 505 words, so
  every shard holds whole words and there are no tail bits inside the vocabulary. The buffer is allocated before the first trace capture and updated in place
  (``ttnn.copy_host_to_device_tensor``) only on a step that carries a structured row; the rest of the time it stays all ones.
* the upload is the packed words (4040 int32 per block row), about ``rows * T * 16 KB`` per round across the mesh; the unpack (shift by arange(32), and 1, to a bias
  0 / -inf) runs on device inside the sampler trace S, on the logits of trace A, BEFORE the shared top-32 sampler sees them (the bias is applied once).
* upstream ``TTSampling.enable_device_grammar`` is not used: it refuses ``sampling_dp != 1`` and our adapter samples with ``sampling_dp = rows``.
"""

from __future__ import annotations

from typing import Sequence

import torch

VOCAB = 129280
ALL_ONES = -1  # an int32 word with every bit set: the row allows every token


def packed_words(vocab: int = VOCAB) -> int:
    return (int(vocab) + 31) // 32


def unpack_mask(packed: torch.Tensor, vocab: int = VOCAB) -> torch.Tensor:
    """``[..., words]`` int32 -> bool ``[..., vocab]``, True = allowed. Bits at or above ``vocab`` (the tail of the last word) are dropped."""
    packed = torch.as_tensor(packed)
    if packed.dtype != torch.int32:
        raise TypeError(f"a packed grammar mask is int32, got {packed.dtype}")
    words = int(packed.shape[-1])
    if words < packed_words(vocab):
        raise ValueError(f"packed mask holds {words} words, {vocab} tokens need {packed_words(vocab)}")
    shifts = torch.arange(32, dtype=torch.int32)
    bits = (packed.unsqueeze(-1) >> shifts) & 1  # arithmetic shift of a negative word: the & 1 keeps the bit asked for
    return bits.reshape(*packed.shape[:-1], words * 32)[..., :vocab].bool()


def mask_bias(packed: torch.Tensor, vocab: int = VOCAB, neg: float = float("-inf")) -> torch.Tensor:
    """The additive bias form (0 allowed, ``neg`` disallowed) the device adds to the logits."""
    allowed = unpack_mask(packed, vocab)
    return torch.where(allowed, torch.zeros((), dtype=torch.float32), torch.full((), neg, dtype=torch.float32))


def apply_mask_reference(logits: torch.Tensor, packed: torch.Tensor) -> torch.Tensor:
    """Host reference: ``logits [..., vocab]`` with the disallowed tokens at -inf (what the plugin's ``apply_grammar_bitmask`` does)."""
    vocab = int(logits.shape[-1])
    return logits.float() + mask_bias(packed, vocab)


def check_rows_allow_something(packed: torch.Tensor, vocab: int = VOCAB, what: str = "grammar mask") -> None:
    """Raise when a row allows no token inside the vocabulary: sampling from it would return an arbitrary token (a grammar that rejects everything is a bug upstream)."""
    packed = torch.as_tensor(packed)
    flat = packed.reshape(-1, packed.shape[-1])
    allowed = unpack_mask(flat, vocab)
    empty = (~allowed.any(dim=-1)).nonzero().flatten().tolist()
    if empty:
        raise ValueError(f"{what}: rows {empty} allow no token")


def permute_to_runner(
    block: torch.Tensor,
    rows: Sequence[tuple[int, int]],
    runner_rows: int,
    n: int,
    kv: int | None = None,
    *,
    column0_only: bool = True,
) -> torch.Tensor:
    """The plugin's ``[W, width, words]`` mask block -> the runner's ``[runner_rows * n, words]`` buffer layout (user-major ``r * n + j``).

    ``rows`` = ``(logical row i, runner row r)`` of every live row, exactly the ``(x[0], x[3])`` pairs ``_spec_verify`` builds for ``X[rr] = tokens[ii]``
    (``r`` is the user itself on the full-width runner, its index in ``runner.phys`` on a ``spec_buckets`` runner). Rows nobody maps to (idle users,
    padding) stay all ones. ``kv`` is the verify width in drafts (adaptive k): the block is sliced to its first ``kv + 1`` columns; ``n`` must equal that
    (``n = kv + 1``). ``column0_only`` (M1: structured rows decode with k = 0) keeps column 0 and leaves the others all ones; M2 passes False.
    Raises when a live row's kept columns hold an empty mask, or the block is narrower than ``n``.
    """
    block = torch.as_tensor(block)
    if block.dtype != torch.int32 or block.dim() != 3:
        raise TypeError(f"the mask block is int32 [W, width, words], got {block.dtype} {tuple(block.shape)}")
    W, width, words = (int(s) for s in block.shape)
    if kv is not None:
        if n != kv + 1:
            raise ValueError(f"verify width n={n} is not kv + 1 = {kv + 1}")
        block = block[:, : kv + 1]
        width = int(block.shape[1])
    if width < n:
        raise ValueError(f"the mask block has {width} columns, the verify round has {n}")
    out = torch.full((runner_rows, n, words), ALL_ONES, dtype=torch.int32)
    seen = set()
    for i, r in rows:
        if not 0 <= i < W:
            raise ValueError(f"logical row {i} outside the mask block of {W} rows")
        if not 0 <= r < runner_rows:
            raise ValueError(f"runner row {r} outside the {runner_rows} runner rows")
        if r in seen:
            raise ValueError(f"runner row {r} is mapped twice")
        seen.add(r)
        cols = 1 if column0_only else n
        out[r, :cols] = block[i, :cols]
        check_rows_allow_something(out[r, :cols], what=f"logical row {i} (runner row {r})")
    return out.reshape(runner_rows * n, words)


def has_structured_row(block: torch.Tensor | None, live_rows: Sequence[int]) -> bool:
    """True when any live logical row's column 0 differs from all ones (the step needs the masked round)."""
    if block is None:
        return False
    block = torch.as_tensor(block)
    return any(bool((block[i, 0] != ALL_ONES).any()) for i in live_rows)
