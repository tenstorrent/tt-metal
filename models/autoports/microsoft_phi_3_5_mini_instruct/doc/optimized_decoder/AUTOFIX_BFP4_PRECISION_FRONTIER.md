# BFP4 precision-frontier evidence

Date: 2026-07-30
Device: single-chip Blackhole

## Decision

Use BFP4_B for attention QKV/output, MLP gate/up, and MLP down weights.
Decode down retains the selected 16-core DRAM-width-sharded LoFi topology.
The supplied `OptimizationPolicy` is now explicitly preserved by
`OptimizedDecoder.from_state_dict`; previously non-default policies were
silently replaced by the class default during base construction.

## Isolated real-weight diagnostic

These projection-local PCC values are diagnostic, not the functional gate:

| Batch | QKV PCC | Output PCC | Down PCC |
|---:|---:|---:|---:|
| 1 | 0.994915 | 0.992651 | 0.993197 |
| 32 | 0.994211 | 0.993213 | 0.992963 |

## Cumulative real-weight decoder frontier

All candidates passed the whole-layer PCC threshold of 0.995. Relative timing
was measured with 100 trace replays per candidate in the same process.

| Batch | Candidate | PCC | Mean ms |
|---:|---|---:|---:|
| 1 | shipped BFP8 attention/down | 0.999238 | 0.667450 |
| 1 | BFP4 attention | 0.999243 | 0.652481 |
| 1 | BFP4 down | 0.998899 | 0.662148 |
| 1 | BFP4 attention + down | 0.998906 | 0.646263 |
| 32 | shipped BFP8 attention/down | 0.999206 | 0.873266 |
| 32 | BFP4 attention | 0.999160 | 0.857863 |
| 32 | BFP4 down | 0.998849 | 0.868357 |
| 32 | BFP4 attention + down | 0.998792 | 0.855436 |

The combined candidate improved B1 by 3.18% and B32 by 2.04%.
A second run through the fixed public policy path reproduced B1 at 0.648088 ms
and B32 at 0.853377 ms.

## Final correctness

The complete optimized suite passed all ten checks after a source-contract-only
comment fix. Real-weight PCC:

- Prefill S31/S33/S65: 0.998782 / 0.998704 / 0.998763
- Decode B1/B32: 0.998787 / 0.998885
- Traced decode B1/B32: 0.999026 / 0.998831
- Paged transition prefill/decode: 0.998744 / 0.998771
- Advertised-context decode: 0.998697

## Final performance

The exact candidate harness passed four tests:

| Path | Batch | Optimized mean ms | Fused mean ms |
|---|---:|---:|---:|
| Traced decode | 1 | 0.646986 | 1.047414 |
| Traced decode | 32 | 0.810999 | 1.210580 |
| Warmed prefill S128 | 1 | 1.395797 | 1.572368 |
| Warmed prefill S128 | 32 | 30.266554 | 37.320119 |

Raw logs:

- `/tmp/phi_bfp4_frontier.txt`
- `/tmp/phi_bfp4_cumulative_fixed.txt`
- `/tmp/phi_bfp4_full_correctness.txt`
- `/tmp/phi_bfp4_final_perf.txt`

## Stage-review closure rerun

The reproducer now makes every candidate's attention, gate/up, and down dtype
explicit. The tracked runner artifact
`bfp4_precision_frontier_runner.txt` reproduced the decision:

- B1: explicit BFP8 attention/down 0.667545 ms at PCC 0.999238; combined
  BFP4 0.647355 ms at PCC 0.998920.
- B32: explicit BFP8 attention/down 0.873244 ms at PCC 0.999206; combined
  BFP4 0.852540 ms at PCC 0.998803.

The BFP4/LoFi decode geometry matrix is preserved in
`bfp4_lofi_decode_geometry_runner.txt`. It covered 16/32 cores and all legal
K-blocks through 6 for hidden-size projections and through 16 for down.
The stack-compatible down winner was 16 cores, K-block 16:
0.061074/0.061515 ms at B1/B32 versus K-block 8 at
0.065233/0.065151 ms. Full-layer promotion passed 10/10 correctness checks
and improved traced decode to 0.643175 ms (B1) and 0.806157 ms (B32).

Explicit prefill K-block 4/8 evidence is preserved in
`prefill_explicit_config_runner.txt`. At B32, QKV rejected with exact L1
allocations 1,719,040 and 2,147,072 bytes versus the 1,572,864-byte limit;
gate/up also rejected both candidates (K-block 8 required 3,322,624 bytes).
Output and down were legal; down was the locally integrable material winner.
The adaptive 64-core, K-block-8 prefill down passed non-aligned S31/S33/S65
and the complete 10-test correctness suite. Final warmed S128 prefill improved
from 1.395797 to 1.351156 ms at B1 and from 30.266554 to 24.148464 ms at B32.
Final traced decode remained 0.642772/0.807652 ms.

Tracked final proof:

- `correctness_prefill_down_explicit_runner.txt` — 10 passed.
- `perf_bfp4_block16_prefill_down_runner.txt` — 4 passed.
