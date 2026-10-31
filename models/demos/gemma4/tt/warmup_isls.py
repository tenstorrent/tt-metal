# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Eager prefill warmup ladder (``GEMMA4_WARMUP_PREFILL_ISLS``).

The first request at a new input length pays a one-off cost on top of the
prefill itself: new chunk and attention shapes compile into the program cache
on first use. On the Blackhole Galaxy release run that was +2.9 s at 4K, +1.4 s
at 8K, +2.7 s at 16K and +7 s at 32K, each on the first request only, with
every later request at that length inside target. The trace warmup only
covers the trace buckets (<= one chunk), so nothing warmed the longer shapes.

``GEMMA4_WARMUP_PREFILL_ISLS="4096,8192,16384,32768"`` runs one eager batch-1
prefill per listed length at boot, once per process. Unset (default) warms
nothing extra, so serving behaviour is unchanged unless an entry opts in.
Lengths above the served context are skipped; nothing else is inferred.

``GEMMA4_PREFILL_PAD_BUCKETS`` (Gemma 4's own prefill padding policy, below) adds
its lengths to the ladder, so the finer buckets it names are compiled at boot too.

This module is host-only (no ttnn import) so the selection and the
orchestration can be unit-tested with a fake generator.
"""

import os
from functools import lru_cache
from typing import Callable, List, Optional

import torch
from loguru import logger

ENV_VAR = "GEMMA4_WARMUP_PREFILL_ISLS"

# ── Gemma 4's own prefill padding (GEMMA4_PREFILL_PAD_BUCKETS) ────────────────────────────────────────────────
#
# The shared ``get_padded_prefill_len`` (models/tt_transformers/tt/common.py) pads a prompt of 129..1024 tokens to
# 1024, so a chat-templated ~145-token benchmark prompt costs a 1024-token prefill: the served c1 TTFT of 150 /
# 189 / 260 ms on the QB2 12B / 26B / 31B entries and the Galaxy lane engine's ~275 ms 1x1024 frame (2026-10-31).
# The knob names finer buckets for that range ("256,512,1024": 129..256 -> 256, 257..512 -> 512, 513..1024 ->
# 1024; 1..128 and everything above 1024 pad as before) and is applied at Gemma 4's own call sites only: the
# shared generator loop still pads to its bucket, and ``prefill_trim_width`` cuts a single-user call back to
# Gemma 4's bucket before the device work (the pad past it is zeros the kernel never needs). Unset (default)
# keeps the shared buckets everywhere; a new length is a new program-cache signature and a new trace shape, so
# the knob's lengths join the warmup ladder (``warmup_prefill_isls``), the continuation tails, the trace ISLs
# (``generator_trace``) and the lane frames' admitted lengths (``lane_prefill``).
PREFILL_PAD_BUCKETS_ENV = "GEMMA4_PREFILL_PAD_BUCKETS"
STOCK_SHORT_PREFILL_BUCKET = 128
STOCK_LONG_PREFILL_BUCKET = 1024


@lru_cache(maxsize=8)
def _parsed_prefill_pad_buckets(raw: str) -> tuple:
    out = set()
    for item in str(raw or "").split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            logger.warning("Ignoring invalid {} entry {!r} (not an integer)", PREFILL_PAD_BUCKETS_ENV, item)
            continue
        if (
            value <= STOCK_SHORT_PREFILL_BUCKET
            or value > STOCK_LONG_PREFILL_BUCKET
            or value & (value - 1)
            or value % STOCK_SHORT_PREFILL_BUCKET
        ):
            logger.warning(
                "Ignoring invalid {} entry {!r} (a power of two in ({}, {}])",
                PREFILL_PAD_BUCKETS_ENV,
                item,
                STOCK_SHORT_PREFILL_BUCKET,
                STOCK_LONG_PREFILL_BUCKET,
            )
            continue
        out.add(value)
    return tuple(sorted(out))


def parse_prefill_pad_buckets(raw: Optional[str]) -> List[int]:
    """``"256, 512,1024,abc,96,2048"`` -> ``[256, 512, 1024]``: sorted unique powers of two in (128, 1024] (the
    range the shared policy folds into 1024; a power of two keeps every pow2 assumption of the trace keys, the
    RoPE chunk slice and the chunk grids). Other entries are dropped with a warning, once per distinct value."""
    return list(_parsed_prefill_pad_buckets(str(raw or "")))


def prefill_pad_buckets(env: Optional[dict] = None) -> List[int]:
    """The knob's buckets (empty = the shared policy)."""
    env = os.environ if env is None else env
    return parse_prefill_pad_buckets(env.get(PREFILL_PAD_BUCKETS_ENV))


def gemma4_padded_prefill_len(seq_len, buckets=None) -> int:
    """Gemma 4's padded prefill length: the shared ``get_padded_prefill_len`` unless it answers 1024 and the knob
    names a smaller bucket at or above ``seq_len`` (the smallest such; none -> 1024, as before)."""
    from models.tt_transformers.tt.common import get_padded_prefill_len  # lazy: this module stays ttnn-free

    seq_len = int(seq_len)
    stock = int(get_padded_prefill_len(seq_len))
    if stock != STOCK_LONG_PREFILL_BUCKET:
        return stock
    for bucket in prefill_pad_buckets() if buckets is None else buckets:
        if int(bucket) >= seq_len:
            return int(bucket)
    return stock


def new_prefill_pad_lengths(max_seq_len: Optional[int] = None, env: Optional[dict] = None) -> List[int]:
    """The knob's lengths the shared policy never pads to (a program set and a trace shape of their own), within
    the served context: what the boot warmup covers so no first request at them compiles."""
    return [
        bucket
        for bucket in prefill_pad_buckets(env)
        if bucket not in (STOCK_SHORT_PREFILL_BUCKET, STOCK_LONG_PREFILL_BUCKET)
        and (max_seq_len is None or bucket <= int(max_seq_len))
    ]


def prefill_trim_width(width, last_token_idx, num_cached_tokens=0, batch_size=1, buckets=None):
    """The Gemma 4 bucket a shared-loop prefill call of ``width`` padded tokens runs at, or None when the call keeps
    its width: the knob off, a batched call (``batch_size`` > 1 or a per-slot index list: the shared batched
    consumption reshapes by the width it padded to), no last-token index, or no bucket below the width. The call's
    rows are ``last_token_idx - num_cached_tokens + 1`` (the index is absolute, cached tokens included, as the
    shared loop hands it over; a multi-chunk walk's chunks are whole buckets already, so they never trim)."""
    buckets = prefill_pad_buckets() if buckets is None else list(buckets)
    if not buckets or last_token_idx is None or isinstance(last_token_idx, (list, tuple)) or int(batch_size or 1) != 1:
        return None
    rows = int(last_token_idx) - int(num_cached_tokens or 0) + 1
    if rows <= 0:
        return None
    bucket = gemma4_padded_prefill_len(rows, buckets)
    return bucket if bucket < int(width) else None


def parse_warmup_isls(raw: Optional[str]) -> List[int]:
    """``"4096, 8192,abc,,16384"`` -> ``[4096, 8192, 16384]`` (sorted, unique, > 0).
    Invalid or non-positive entries are dropped with a warning, so a mis-set
    env var is visible in the boot log instead of silently warming nothing."""
    if not raw:
        return []
    out = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = int(item)
        except ValueError:
            logger.warning("Ignoring invalid {} entry {!r} (not an integer)", ENV_VAR, item)
            continue
        if value <= 0:
            logger.warning("Ignoring invalid {} entry {!r} (must be > 0)", ENV_VAR, item)
            continue
        out.add(value)
    return sorted(out)


def warmup_prefill_isls(max_seq_len: Optional[int], env: Optional[dict] = None) -> List[int]:
    """The ladder to run: every env length within the served context, plus the lengths
    ``GEMMA4_PREFILL_PAD_BUCKETS`` adds to the padding policy (``new_prefill_pad_lengths``:
    each a program set of its own that nothing else warms under bounded sliding, where
    the trace sweep stops at the short cap). Nothing is inferred as "already covered":
    the trace buckets only warm a shape when prefill tracing is on, and the first 4K
    request on an entry with tracing off paid 1.9 s vs 0.7 s because 4096 had been
    skipped as a trace bucket."""
    env = os.environ if env is None else env
    ladder = set(parse_warmup_isls(env.get(ENV_VAR)))
    ladder.update(new_prefill_pad_lengths(env=env))
    return sorted(isl for isl in ladder if max_seq_len is None or isl <= int(max_seq_len))


def run_prefill_ladder(
    generator,
    kv_cache,
    prefill_forward: Callable,
    ladder: List[int],
    *,
    chunk: int,
) -> List[int]:
    """Run one eager batch-1 prefill per ``ladder`` length on every data-parallel
    model of ``generator``, once per process. Returns the lengths warmed.

    The once-per-process flag is set before the prefills run, deliberately: the
    plugin calls the prefill warmup twice (compile pass, then capture pass) and
    the ladder must not run a second time; a prefill that raises propagates and
    fails the boot, so there is nothing to retry on the second pass.

    With paged attention off (no page table) a length beyond one chunk cannot
    be prefilled, so the ladder stops at the first such length. The mock page
    table carries real block ids (``warmup_page_table_with_real_blocks``): the
    chunks past the first of a length above the chunk are eager continuations,
    which then run the served programs over real blocks instead of logging the
    generator's "chunk_page_table has no real block ids" warning on every one.
    """
    if getattr(generator, "_warmed_prefill_isl_ladder", False):
        return []
    generator._warmed_prefill_isl_ladder = True
    warmed: List[int] = []
    for model_id in range(generator.data_parallel):
        for isl in ladder:
            warmup_args = generator._mock_tokens(1, isl, kv_cache, model_id)
            if warmup_args.get("page_table") is None and isl > chunk:
                logger.warning(
                    "Skipping prefill warmup at ISL {}: longer than the {}-token chunk with paged attention off",
                    isl,
                    chunk,
                )
                break
            warmup_args["page_table"] = warmup_page_table_with_real_blocks(
                warmup_args["page_table"], pool_blocks=_kv_pool_blocks(kv_cache, model_id)
            )
            logger.info("Warming up eager prefill at ISL {} ({})", isl, ENV_VAR)
            prefill_forward(
                **warmup_args,
                kv_cache=kv_cache,
                enable_trace=False,
                model_id_warmup=model_id,
                sampling_params=None,
                warmup_prefill=False,
            )
            if model_id == 0:
                warmed.append(isl)
    return warmed


#: The tail a prefix-cache continuation warmup prefills past one chunk of cached tokens (the shape a served APC hit
#: takes: the uncached remainder of a prompt, padded to the SDPA chunk alignment).
CONTINUATION_TAIL = 128


def continuation_tails(chunk: int, padded_len: Callable[[int], int], *, skip_chunk: bool = False) -> List[int]:
    """Every length an eager continuation past a cached prefix can take: the padded prefill buckets ``padded_len``
    maps a remnant of 1..``chunk`` tokens to (128, 1024, 2048, ... up to the chunk; ``get_padded_prefill_len``, or
    128, 256, 512, 1024, ... with ``GEMMA4_PREFILL_PAD_BUCKETS`` through ``gemma4_padded_prefill_len``), each
    its own program set (the chunk length is a compile-time argument of the attention kernels). The served
    12B QB2 paid 4.0 s on the first 1026-token remnant past a 4096-token first chunk (padded to 2048, a length the
    ISL ladder never runs), 2026-10-30. ``skip_chunk`` drops the chunk-long tail (the ladder's second chunk of an
    ISL above the chunk already ran it)."""
    chunk = int(chunk)
    tails = sorted({int(padded_len(n)) for n in range(1, chunk + 1)})
    return [t for t in tails if t <= chunk and not (skip_chunk and t == chunk)]


def warmup_page_table_with_real_blocks(page_table, pool_blocks: Optional[int] = None):
    """A mock page table with real block ids: the stock warmup table is all zeros, and block 0 is vLLM's null block,
    so the continuation's prior-window reads (``_paged_tail_block_range`` drops leading null blocks and finds no
    tail; the window frame reads block 0) never ran at boot and compiled on the first served hit. Each column gets
    its own id ``1, 2, ...`` (wrapping inside ``pool_blocks`` when the pool is known); a table that already holds
    real ids is returned as is. Non-tensor tables (fakes) pass through."""
    if not torch.is_tensor(page_table) or page_table.numel() == 0 or int(page_table.max()) > 0:
        return page_table
    table = page_table if page_table.dim() > 1 else page_table.unsqueeze(0)
    ids = torch.arange(1, table.numel() + 1, dtype=torch.int64)
    if pool_blocks is not None and int(pool_blocks) > 2:
        ids = (ids - 1) % (int(pool_blocks) - 1) + 1  # real ids stay in [1, pool_blocks)
    out = ids.reshape(table.shape).to(page_table.dtype)
    return out if page_table.dim() > 1 else out.reshape(page_table.shape)


def _kv_pool_blocks(kv_cache, model_id) -> Optional[int]:
    """Blocks of the model's layer-0 K pool (``[blocks, kv_heads, block, head_dim]``), or None when unknown."""
    try:
        return int(kv_cache[model_id][0][0].shape[0])
    except (TypeError, IndexError, AttributeError, KeyError):
        return None


def run_prefix_continuation_warmup(
    generator, kv_cache, prefill_forward: Callable, *, chunk: int, tail: int = CONTINUATION_TAIL, tails=None
) -> bool:
    """Run one eager batch-1 prefill of each tail length past ``chunk`` cached tokens (``start_pos=[chunk]``) on every
    data-parallel model of ``generator``, once per process: the shapes a prefix-cache hit (or a scheduler chunk past
    the first) prefills in serving. ``tails`` defaults to ``[tail]``; the warmup sweep passes every reachable padded
    length (``continuation_tails``). The continuation is eager by design and its first use compiled on the request
    that hit the cache (the 12B QB2's first cached request took 2.7 s against 0.6 s for the next, 2026-10-30). The
    mock page table carries real block ids (``warmup_page_table_with_real_blocks``) so the continuation runs the same
    programs a served hit runs (the prior window read from the paged cache). The global layers' SDPA table is sized
    from the served context, not the prompt (``attention/continuation.py`` ``continuation_sdpa_table_cols``), so one
    tail here compiles the program a hit of that tail runs at any position and prompt length. Returns whether it
    ran: with paged attention off (no page table) there is no cached prefix to continue from."""
    if getattr(generator, "_warmed_prefix_continuation", False):
        return False
    generator._warmed_prefix_continuation = True
    tails = [int(tail)] if tails is None else [int(t) for t in tails]
    for model_id in range(generator.data_parallel):
        for t in tails:
            warmup_args = generator._mock_tokens(1, int(chunk) + t, kv_cache, model_id)
            if warmup_args.get("page_table") is None:
                logger.warning("Skipping the prefix-cache continuation warmup: paged attention off (no page table)")
                return False
            warmup_args["page_table"] = warmup_page_table_with_real_blocks(
                warmup_args["page_table"], pool_blocks=_kv_pool_blocks(kv_cache, model_id)
            )
            logger.info("Warming up the eager prefix-cache continuation: {} tokens past {} cached", t, chunk)
            prefill_forward(
                **warmup_args,
                kv_cache=kv_cache,
                start_pos=[int(chunk)],
                enable_trace=False,
                model_id_warmup=model_id,
                sampling_params=None,
                warmup_prefill=False,
            )
    return True
