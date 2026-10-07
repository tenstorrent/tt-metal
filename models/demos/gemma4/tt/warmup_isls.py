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
Lengths above the served context, or at or below the longest length the
existing warmup already covers, are skipped.
"""

import os
from typing import Iterable, List, Optional

ENV_VAR = "GEMMA4_WARMUP_PREFILL_ISLS"


def parse_warmup_isls(raw: Optional[str]) -> List[int]:
    """``"4096, 8192,abc,,16384"`` -> ``[4096, 8192, 16384]`` (sorted, unique, > 0)."""
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
            continue
        if value > 0:
            out.add(value)
    return sorted(out)


def warmup_prefill_isls(
    max_seq_len: Optional[int],
    already_warmed: Iterable[int] = (),
    env: Optional[dict] = None,
) -> List[int]:
    """The ladder to run: env lengths within the served context and longer
    than anything ``already_warmed`` (the base/trace warmup lengths)."""
    env = os.environ if env is None else env
    floor = max([int(x) for x in already_warmed] or [0])
    ladder = []
    for isl in parse_warmup_isls(env.get(ENV_VAR)):
        if max_seq_len is not None and isl > int(max_seq_len):
            continue
        if isl <= floor:
            continue
        ladder.append(isl)
    return ladder
