# I2: TILE-native input/output (+ UINT16 index emission) for topk_large_indices

Charter I2, root-fixing I1 (the k1024@2048 routed anomaly = single-core TilizeWithValPadding),
bundling the opt-in index-dtype emission. Branch nkapre/sorting, Blackhole p150a, 2026-08-17.
Tree left UNCOMMITTED with only the kept wins; nothing pushed.

## What landed (all opt-in; defaults byte-identical)

Op (`ttnn/cpp/ttnn/operations/experimental/topk_large_indices/`):
- `tile_output: bool = False` — the writer scatters the 16-element result slices directly into
  their TILE positions (BH DRAM WRITES need only 16 B alignment; every slice run is 32 B-aligned)
  and zero-fills tile padding rows. Pad writes are PRE-ISSUED at kernel start (pad rows are
  disjoint from data rows) so their NoC injection hides under the compute pipeline instead of
  tailing it. Requires k % 32 == 0. New kernels `writer_flex.cpp`, `writer_tree_flex.cpp`,
  shared header `topk_large_indices_writer_flex_common.hpp` — selected only when an opt-in is
  active; the default writer sources/binaries stay byte-identical.
- `index_dtype: Optional[DataType] = None` — UINT16 opt-in, writer-side narrowing (packs two
  uint16 per 32-bit RISC store, halving store count). Sentinel 0xFFFFFFFF -> 0xFFFF, bit-identical
  to the former typecast. Runtime-validated: searched width (valid_length or last dim) <= 65535
  so winners provably fit 16 bits.
- TILE-layout input accepted — NO new parameter; the op dispatches on `input.layout()` (already
  in the program hash). New `reader_tile.cpp` (both factories): per 16-element slice one 64 B DRAM
  read at the 64-aligned offset into a 64 B-stride staging slot (BH DRAM READS require 64 B
  CONGRUENCE: (l1 & 63) == (dram & 63) — sanitize.h:602; the CB's 32 B-stride slice destinations
  alternate mod 64, so staging is mandatory), then a 32 B local L1 copy into the chunk CB. Reads
  only the row's own face runs — no 32x tile-padding read amplification.
- `tile_output` and `index_dtype` added to the program hash; compute kernels UNTOUCHED
  (ckernel_sfpu_topk.h blast radius untouched; the op's kernel tree stays disjoint).
- nanobind: `tile_output=`, `index_dtype=` kwargs (defaults preserve today's behavior exactly;
  direct callers like the DSA indexer are bit-for-bit unchanged).

Routed composite (`ttnn/cpp/ttnn/operations/reduction/topk/topk.cpp`) — measured policy:
- `large_k_route_k_multiple` 16 -> 32 (TILE output needs whole output tile columns).
- TILE input: single flattened row AND padded W >= 32768 (`large_k_route_tile_input_min_width`).
  Measured: untilize ~18 us at W=65536 vs ~7 us reader overhead (win); untilize 1.6-1.9 us at
  W=2048 vs ~5.5 us reader overhead (loss).
- TILE output + native UINT16: k_rounded <= 1024 (`large_k_route_tile_output_max_k`). Tilize of a
  <= 32-tile output hits TilizeWithValPadding's pathological single-core factory (20-78 us — the
  I1 anomaly); the native scatter writer replaces it for ~2-5 us. At k_rounded == 2048 tilize is
  multi-core and cheap (3-4 us) while the scatter tail is ~10 us/stream, so the RM-output +
  tilize (+ typecast at padded <= 65535) chain is kept there.

## Stage profile (single-row routed cells, tracy DEVICE KERNEL sums, 5-iter means, us)

BEFORE (dir stage_before/):
| cell        | untilize | op   | tilize_v | tilize_i | typecast | total |
|-------------|----------|------|----------|----------|----------|-------|
| k512@2048   | 1.9      | 6.3  | 20.3     | 39.3     | 2.1      | 69.7  |
| k1024@2048  | 1.9      | 10.9 | 39.8     | 77.9     | 2.1      | 132.6 |
| k2048@2048  | 1.6      | 18.2 | 2.8      | 4.3      | 2.0      | 28.8  |
| k512@65536  | 17.9     | 13.7 | 20.3     | 39.3     | —        | 91.2  |
| k2048@65536 | 18.0     | 37.1 | 3.1      | 4.3      | —        | 62.5  |

AFTER, final policy (dir stage_after3/):
| cell        | chain                                   | total | delta |
|-------------|-----------------------------------------|-------|-------|
| k512@2048   | untilize 1.7 + op 11.9 (tile-out+u16)   | 13.5  | -81%  |
| k1024@2048  | untilize 1.6 + op 19.8 (tile-out+u16)   | 21.4  | -84%  |
| k2048@2048  | unchanged chain (policy)                | 28.9  |  0%   |
| k512@65536  | op 20.9 alone (tile-in + tile-out + u16)| 20.9  | -77%  |
| k2048@65536 | op 44.1 (tile-in) + 2x tilize           | 51.5  | -18%  |

Intermediate rounds that SET the policy (kept in stage_after/, stage_after2/):
- Round 1 (tile in+out everywhere, per-row pads, per-element narrow): k2048@2048 regressed
  28.8 -> 67.7 (single-core writer: ~38 us of pad+scatter posting + ~12 us u16 narrowing).
- Fixes: pad pre-issue at kernel start; paired-word narrow; then route policy split by k and W.
- Round 2 (fixes, k2048 -> tilize arm, tile-in still unconditional): k2048@2048 32.7 (tile-in
  reader +5.5 us where untilize was 1.6) -> round 3 adds the W >= 32768 tile-input floor.

## Guards (charter rule 3 minimum, run after EVERY build round — 3 rounds, all identical)

- test_topk_contract.py default: 62 passed, 1 skipped. FULL (TOPK_CONTRACT_FULL=1): 72 passed.
- tli nightly test_topk_large_indices.py: 181 passed (incl. ~20 NEW flex tests) + 2 failed =
  test_topk_large_indices_production_perf_check[prefill, bounded_cache], which fail at the
  "Real-time profiler must be active (needs IOMMU)" gate BEFORE any op call — environment-gated
  (requires_host_iommu perf-runner marker), pre-existing on this box, provably not code.
- reduce/test_topk.py: 220 passed, 8 skipped, 80 xfailed.
- Logs: guard_*.log, flex_tests_*.log in this directory.

New op-level coverage (appended flex section in test_topk_large_indices.py): BIT-IDENTITY vs the
default program (same compute => same result, writers only reformat) for tile_output (row-parallel,
tree, rank-3 slab padding, 33-row tile straddle, exact-32-row fill), tile_input (odd face rows =
the staged 64 B congruence path, non-tile-multiple widths, last-chunk winners masked against tile
padding, all-equal ties, valid_length stale-tail), uint16 (row/tree, -inf sentinel 0xFFFF, RM and
TILE), the exact route combos (single-row tile-in+tile-out+u16; 32-user sampling shape), error
cells (u16 width > 65535; tile_output k % 32 != 0), and program-cache growth/reuse across flag flips.

## A/B — canonical harness, competition routed layer (5 iters, ns_median; verified, 0 WRONG)

BEFORE dir: generated/canonical_sweep/comp3 (pinned campaign baseline).
AFTER dir:  generated/canonical_sweep/i2_tile_native_routed.

| k    | W      | before us | after us | delta  |
|------|--------|-----------|----------|--------|
| 512  | 2048   | 70.4      | 13.4     | -81.0% |
| 512  | 4096   | 71.6      | 14.3     | -80.0% |
| 512  | 8192   | 74.0      | 16.6     | -77.6% |
| 512  | 16384  | 79.1      | 22.2     | -71.9% |
| 512  | 32768  | 86.4      | 19.9     | -77.0% |
| 512  | 65536  | 91.7      | 20.9     | -77.2% |
| 512  | 131072 | 107.4     | 29.7     | -72.3% |
| 512  | 262144 | 136.8     | 45.3     | -66.9% |
| 1024 | 2048   | 134.2     | 21.3     | -84.1% |  <- the I1 anomaly cell (target ~35: beaten)
| 1024 | 4096   | 135.9     | 22.9     | -83.1% |
| 1024 | 8192   | 138.9     | 25.5     | -81.6% |
| 1024 | 16384  | 144.9     | 31.7     | -78.1% |
| 1024 | 32768  | 152.7     | 30.6     | -80.0% |
| 1024 | 65536  | 158.1     | 30.2     | -80.9% |
| 1024 | 131072 | 173.0     | 38.5     | -77.7% |
| 1024 | 262144 | 205.6     | 55.5     | -73.0% |
| 2048 | 2048   | 29.5      | 28.8     | -2.4%  |  <- kept-chain parity (policy)
| 2048 | 4096   | 33.1      | 32.4     | -2.1%  |
| 2048 | 8192   | 38.4      | 37.5     | -2.4%  |
| 2048 | 16384  | 46.3      | 45.3     | -2.2%  |
| 2048 | 32768  | 55.5      | 47.6     | -14.2% |
| 2048 | 65536  | 63.4      | 51.5     | -18.9% |  <- anchor (target ~40: partial, see below)
| 2048 | 131072 | 79.4      | 61.7     | -22.3% |
| 2048 | 262144 | 113.8     | 80.1     | -29.6% |

Sampling scenarios (BEFORE dir: i5_landing; AFTER dir: i2_tile_native_scenarios):
| scenario             | before routed | after routed | note |
|----------------------|---------------|--------------|------|
| sampling_qwen36_tp4  | 216.9 us      | 213.3 us     | -1.7%; target "below 217" met |
| sampling_1chip_split | 214.9 us      | 210.7 us     | -2.0%; target "below 215" met |
| sampling_tp8_pow2    | 171.2 us      | 171.3 us     | +0.1% = noise; this cell does NOT route (pow2 W < 65535 keeps the stock multi-core bitonic) — unchanged code path |

No cell regressed; everything kept. The k2048 narrow-W -2% band is the unchanged chain (day-to-day
drift within noise); the sampling wins are small because their conversion overhead was already
small ([32, 32] outputs -> tiny tilizes) — the dominant sampling cost is the op itself (multi-row
untilize + 75-chunk rows), out of I2 scope.

## Anchor honesty (k2048@65536: 63.4 -> 51.5, not ~40)

The ~40 us aspiration assumed the conversion envelope was the whole gap. Post-change decomposition:
op-with-TILE-input 44.1 us (RM-input op was 37.1; the +7 us is the tile reader's staged scatter,
already ~2.5x cheaper than the 18 us untilize it replaced) + 7.4 us of cheap multi-core tilizes.
Getting to ~40 needs either a faster tile-input read path (halving the staged read count via
congruent-direct reads for the even-delta half) or a k=2048-capable native tile writer with a
sub-4 us tail — both op-internal, neither a conversion-chain deletion. Logged as follow-up.

## u16 call-site patch (mission step 5) — evaluated, NOT applicable

- models/common/sampling/tt_sampling.py:841-850 — the routed-arm typecast is a CONDITIONAL dtype
  normalization (fires only when the route's index dtype mismatches the stock contract). The
  route's dtype contract is unchanged (u16 iff padded <= 65535, now emitted natively), so the
  guard fires in exactly the same rare cases (expected-u32 upcast for misaligned multi-step
  halves) and must stay.
- models/common/modules/sampling/sampling_1d.py:384 — unconditional typecast to int32 for the
  device-offset add; required regardless of u16 vs u32 input.
No models/ diff prepared because none is droppable. The u16 win realized is the deletion of the
route-INTERNAL typecast op on every u16-eligible routed call (k_rounded <= 1024 arm).

## Artifacts in this directory

- stage_before/, stage_after/, stage_after2/, stage_after3/ — tracy per-stage CSVs + logs per round
- flex_tests_1.log, flex_tests_2.log — new-test runs (round 1: 42/43 + one test-bug; round 2+: 43/43)
- guard_{tli,contract_default,contract_full,reduce_topk}.log — final-round guard logs
- ab_routed.log, ab_scenarios.log — canonical harness runs
- build1.log, build2.log, build3.log — the three host rebuilds (each BEFORE any measurement grid)
- compare_ab.py — the table generator above

## Tree state (uncommitted, ready for orchestrator review)

Modified: topk_large_indices device_operation{.cpp,.hpp}, device_operation_types.hpp,
program_factory.cpp, nanobind.cpp; reduction/topk/topk.cpp; tests/.../test_topk_large_indices.py.
New: device/kernels/{reader_tile.cpp, writer_flex.cpp, writer_tree_flex.cpp,
topk_large_indices_writer_flex_common.hpp}.
Pre-existing unrelated modifications left untouched: .github/workflows/package-and-release.yaml,
paper-topk/evidence/paper/evidence.md, untracked lx-reset / n150 yaml / eltwise_poly / _ttnn.so.release.

## Follow-ups (not landed)

- Anchor residual: op-internal tile-reader and k2048 writer-tail optimizations (above).
- k512@2048-class cells idle ~1 us on the table: tile-input actually measured slightly BETTER at
  k<=512 narrow widths (12.6 vs 13.5 at k512@2048) because the K512 window's 32-slice chunks are
  cheap and the tree overlaps them; the W >= 32768 floor trades that ~1 us for a simple, safe gate.
- Sampling dominant cost is now the op itself (multi-row wide rows), not conversions.
