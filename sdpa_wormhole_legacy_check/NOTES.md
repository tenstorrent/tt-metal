# SDPA recipe stack: Wormhole legacy-kernel check (wh-06, IRD 130308)

BASE = dfaf6dc802f (main at merge-base), HEAD = d640c711 (cglagovich/sdpa-dit-explicit-recipes).
Both built in the same directory /localdev/cglagovich/whcheck/tt-metal (serial checkout), so JIT
kernel ELFs are comparable byte-for-byte. Per-build JIT caches: cache_base / cache_head.

Status: in progress (building BASE).
