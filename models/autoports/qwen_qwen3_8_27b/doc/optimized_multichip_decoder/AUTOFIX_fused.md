# Native DRAM-sharded fused candidate repair

Source-backed diagnosis: `AUTODEBUG_fused.md`. Both fused factories accept DRAM-sharded B, but the selected native paths violate kernel contracts. The AG host serializes nine fusion words; the nonwaiting receiver consumes seven before DRAM descriptors. Its weight reader also ignores ring-reordered K tile IDs. Native RS describes20 tiles of N per core but reserves/pushes10-wide blocks, overlapping rows and exceeding reserved CB storage.

Verified the receiver constructor, trailing DRAM descriptor parsing and DRAM reader loop against the report. These are concrete source contradictions, not a claim that DRAM-sharded fusion is categorically unsupported. All four AG runs timed out with preserved triage and successful bounded recovery. All four RS runs completed with NaN PCC; correlation alone does not identify literal tensor NaNs.

Adaptations:

1. `fused_interleaved_control` keeps the exact2D grid, K blocks, subblocks and residual contract, changing only B to interleaved DRAM at setup. This isolates the native weight reader without paying a per-token weight conversion. Earlier `fused_refine` passed but also changed AG to1D, so it was not a one-variable AG control.
2. `fused_rs_native_repair` keeps native DRAM-sharded B and sets explicit output block width20, matching per-core N. K multipliers1/2 both remain tested. The prediction is correct full-layer outputs if the width contradiction was the RS failure cause; its traced latency then competes with the current best separate DRAM-sharded matmul/direct-AR family.

The production path was not switched to a failing candidate. The following controls and repairs completed.

Same-program weight-layout controls completed: all eight pass. Median warmed traced host decode (us):

| Layer | Candidate | us | Minimum PCC |
|---|---|---:|---:|
| 0 | both_agk10 | 718.117 | 0.999958487 |
| 0 | both_agk20 | 703.859 | 0.999958487 |
| 0 | rs_k1 | 597.495 | 0.999958487 |
| 0 | rs_k2 | 596.203 | 0.999958487 |
| 3 | both_agk10 | 621.211 | 0.999976088 |
| 3 | both_agk20 | 605.911 | 0.999928676 |
| 3 | rs_k1 | 527.190 | 0.999949333 |
| 3 | rs_k2 | 527.050 | 0.999943123 |

The only policy change from each native failure is fused_dram_sharded_weights=false. Weight conversion is once at setup. Compatible hidden-sharded residuals remain through layer output; the harness gathers only outside timing. These passing controls are slower than the current separate projection/direct-AR family. Native RS width repair also passes: fused_rs_native_repair records approximately525.84/457.05us at B1, slower than the separate direct-AR family. At B32 it becomes competitive under the hidden-sharded residual contract; row_family_final32 and compact_boundaries32 measure it with compact column boundaries. Native output blocks remain20, matching the reader descriptor. No generic kernel source is modified. Final selection is recorded separately after final-default validation.
