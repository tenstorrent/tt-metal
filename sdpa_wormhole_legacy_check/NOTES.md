# SDPA recipe stack: Wormhole legacy-kernel check (wh-06, IRD 130308)

BASE = dfaf6dc802f (main at merge-base), HEAD = d640c711 (cglagovich/sdpa-dit-explicit-recipes).
Both built in the same directory /localdev/cglagovich/whcheck/tt-metal (serial checkout), so JIT
kernel ELFs are comparable byte-for-byte. Per-build JIT caches: cache_base / cache_head.

Status: in progress (building BASE).

## Host topology finding
wh-06 is 4 *unconnected* n300 cards (fabric auto-discovery: every chip has degree 1). Opening a
device with all 8 chips visible fails (`Physical chip id 0 not found in control plane chip mapping`),
so everything runs with TT_VISIBLE_DEVICES=<card> (one n300 = 1x2 mesh), 4 cards in parallel with
per-card locks/dirty flags (scripts/run_safe_pytest_card2.sh). No T3K: T3K 2x4 tests replaced by
1x2 ring joint drivers (tools/test_whcheck_ring_joint.py).

## BASE (dfaf6dc802f) done 19:04 UTC: see results/base_summary.txt
- exp ring on 1x2 FABRIC_1D_RING hangs on BASE (fabric eth cores time out) -> ELF compare only.
- watcher + fabric (1x2) fails on BASE: cq_dispatch idle_erisc.elf overflows under watcher -> watcher
  compare limited to single-chip dense/joint matrix configs.
