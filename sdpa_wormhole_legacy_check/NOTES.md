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

## HEAD (d640c711) results, 19:10-19:45 UTC
- Test counts identical to BASE: unit 88p/4s, joint 48p/8s, chunked 36p, nightly-prefill subset 156p/6s,
  ring joint 1x2 12p, dit legacy 6p, matrix 20/20 OK, watcher matrix 9/9 OK.
- Bitwise: all 20 matrix outputs identical BASE vs HEAD.
- ELF compare (all SHF_ALLOC sections, exact JIT key match): 1015 kernels SAME, 0 DIFF, no HEAD-only key
  from a shared test (HEAD-only keys come from HEAD-only tests run later). Watcher build: all SAME.
- Host checks: precision raises "qualified only on Blackhole"; inputs_prepared w/o precision rejected;
  chunk 0 rejected for sdpa/joint; BUT chunked_scaled_dot_product_attention with chunk 0 -> SIGFPE
  (BASE: TypeError). Fix on cglagovich/sdpa-wormhole-fixes.
- exp ring on 1x2 ring: BASE hangs; HEAD fails fast "Requested link index 2 is out of bounds" (new
  ring_size==2 link-offset routing in the legacy exp ring factory needs 2*num_links channels).
- test_sdpa_numerics_compatibility::test_sdpa_legacy_defaults[explicit-lofi] HANGS on WH (HEAD).
  Probe: legacy dense SDPA LoFi + q_chunk 256 hangs for any grid/k chunk; LoFi q128 fine; HiFi2/3 fine.
  Checking BASE now.
