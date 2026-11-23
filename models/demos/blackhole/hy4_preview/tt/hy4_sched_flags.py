# SPDX-License-Identifier: Apache-2.0
"""Flag + live-switch-file helper of the Hy4 scheduler / serving features (program hy4sched, 2026-11-23). Pure python, no ttnn / torch / vllm.

Every feature of this family is OFF by default and has two switches so that a booted server can be bisected without a restart:

    env      HY4_<NAME>=1                 enables the machinery at boot (and is the value used when no file exists)
    file     HY4_<NAME>_FILE=<path>       or, when that is not set, ``$HY4_SCHED_DIR/<name>`` (default dir /out17): a file holding 0 / 1, read at most every 0.2 s; it overrides the env value in BOTH
                                          directions, so a server booted with nothing set but the install hooks can still switch a feature on, and one booted with =1 can switch it off.

``enabled(name)`` is cheap (one cached stat/read per 0.2 s) and never raises. The names are the ``FEATURES`` keys.
"""
from __future__ import annotations

import os
import time

# feature -> (env var, file base name, what it does)
FEATURES = {
    "cached_tokens": (
        "HY4_CACHED_TOKENS",
        "cached_tokens",
        "usage.prompt_tokens_details.cached_tokens from the row-local cache (scheduler-time peek_home + engine actual by prompt fingerprint)",
    ),
    "ridmap": (
        "HY4_RIDMAP",
        "ridmap",
        "request id -> mesh row map by prompt fingerprint (the image plugin does not pass request_ids), so the hook's finished_ids frees the right row",
    ),
    "evict_cost": (
        "HY4_PFX_EVICT_COST",
        "evict_cost",
        "victim of a free-row overwrite = the cheapest chain to recompute x probability of reuse (HY4_PFX_EVICT=cost semantics, live switchable)",
    ),
    "tier_log": (
        "HY4_PFX_TIER_LOG",
        "tier_log",
        "one `pfx_tiers` line per prefill call: hit tokens / copies / stash / blocks counters per tier",
    ),
    "pos_buckets": (
        "HY4_POS_BUCKETS",
        "pos_buckets",
        "decode position buckets (reserved: only wired when ITL vs context shows a gap)",
    ),
}

_cache: dict = {}
_TTL = 0.2


def _file_for(name):
    env, base, _ = FEATURES[name]
    f = os.environ.get(env + "_FILE")
    if f:
        return f
    return os.path.join(os.environ.get("HY4_SCHED_DIR", "/out17"), base)


def enabled(name, default=False):
    env = FEATURES[name][0]
    now = time.monotonic()
    ent = _cache.get(name)
    if ent is not None and now - ent[0] < _TTL:
        return ent[1]
    v = os.environ.get(env)
    val = default if v is None else (v.strip() == "1")
    try:
        with open(_file_for(name)) as fh:
            s = fh.read().strip()
        if s in ("0", "1"):
            val = s == "1"
    except (OSError, ValueError):
        pass
    _cache[name] = (now, val)
    return val


def reset_cache():
    _cache.clear()
