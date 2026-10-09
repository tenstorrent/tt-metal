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

This module is host-only (no ttnn import) so the selection and the
orchestration can be unit-tested with a fake generator.
"""

import os
from typing import Callable, List, Optional

from loguru import logger

ENV_VAR = "GEMMA4_WARMUP_PREFILL_ISLS"


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
    """The ladder to run: every env length within the served context. Nothing is
    inferred as "already covered": the trace buckets only warm a shape when prefill
    tracing is on, and the first 4K request on an entry with tracing off paid
    1.9 s vs 0.7 s because 4096 had been skipped as a trace bucket."""
    env = os.environ if env is None else env
    return [isl for isl in parse_warmup_isls(env.get(ENV_VAR)) if max_seq_len is None or isl <= int(max_seq_len)]


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
    be prefilled, so the ladder stops at the first such length.
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
