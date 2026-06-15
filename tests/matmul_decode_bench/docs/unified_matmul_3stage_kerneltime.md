# Unified matmul_decode vs best-explicit native — DEVICE KERNEL DURATION (deep-plan_14 post-build)

> **GOAL (aggressive, honest):** drive EVERY pi05 matmul projection to DEVICE KERNEL DURATION strict-beat
> the best-explicit native `ttnn.linear` (`ratio = mmd_KERNEL ÷ native_best_explicit_KERNEL < 1−ε`).
> **THE CEILING (stated up front):** compute-bound prefill rows (SigLIP/VLM, M=256/288) run on the SAME
> systolic FPU and the SAME `SUBBLOCK_HW_CHOICES` subblock table as native. Once M-split dispatch is
> collapsed and the subblock/k-config is tuned, several such rows CANNOT strict-beat single-chip. A row
> that after all levers cannot drop below native is documented IMPROVED-but-LOSS / TIE-CEILING with
> per-op tracy evidence — the honest terminal, NOT a faked beat.

**Metric: `DEVICE KERNEL DURATION [ns]`** (CSV col 20, the compute-kernel time), per single block-forward, µs.
Host-only re-derivation via `extract_perf.py METRIC=KERNEL` (deep-plan_14 §9.1; back-tested: mmd SigLIP qkv
reproduces 45.97 exactly; the full frozen table reproduces within rounding). Fork HEAD `e4500c1f`; N_ITERS=5.
**KERNEL ε = 0.012 (1.196% 3-repeat spread, re-derived on the KERNEL column from `eps_SigLIP_qkv.csv`).**

- **mmd** = unified op (`MatmulDecodeDeviceOperation`), sum over all device calls in the measured
  `MMD_<stage>_<proj>` region ÷ 5. **state** column gives the adopted execution path.
- **native** = best-explicit `ttnn.linear`: min per-call KERNEL across the frozen ~10-config NATEXP sweep
  (`deep-work/mmd13_csvs/nat_*.csv`, KERNEL-mode min; NO re-sweep). NOTE: prior table's 55.25 (qkv) /
  238.27 (down) are the FW-era per-call solve; consistent KERNEL-mode min is 55.441 / 236.716 (the
  per-call divisor differs ~0.3-0.7%; the mmd column reproduces exactly).
- **ratio = mmd ÷ native**. Verdict: `<1−ε` STRICT-BEAT; `|r−1|≤ε` TIE-CEILING; `>1+ε` IMPROVED-but-LOSS.

| stage | proj | [M,K,N] | dtype | state (scheme) | mmd_KERNEL µs | native µs | ratio | subblock | VERDICT |
|---|---|---|---|---|--:|--:|--:|---|---|
| SigLIP | qkv | 256,1152,4608 | bf16 | chunked-M=32 (8 calls) | 46.18 | 55.44 | **0.833** | auto out_w | **STRICT-BEAT** |
| SigLIP | o | 256,1536,1152 | bf16 | whole-M out_h=8 (1 call) | 33.89 | 20.43 | 1.659 | M-fill 8×1 | IMPROVED-but-LOSS |
| SigLIP | fc1 | 256,1152,4320 | bf8_b | whole-M out_h=8 (1 call) | 54.57 | 41.37 | 1.319 | M-fill 8×1 | IMPROVED-but-LOSS |
| SigLIP | fc2 | 256,**4320**,1152 | bf8_b | **WIDTH-temporal k_stream (1 call, ks=15, G=9)** | **123.95** | 42.09 | **2.945** | out_w (npc=1) | IMPROVED-but-LOSS (Lever-2; was chunked 131.73/3.130) |
| VLM | qkv | 288,2048,2560 | bf8_b | chunked-M=32 (9 calls) | 56.24 | 38.38 | 1.466 | auto out_w | IMPROVED-but-LOSS |
| VLM | o | 288,2048,2048 | bf8_b | chunked-M=32 (9 calls) | 56.83 | 37.35 | 1.521 | auto out_w | IMPROVED-but-LOSS |
| VLM | gate | 288,2048,16384 | bf8_b | chunked-M=32 (9 calls) | 263.96 | 254.61 | 1.037 | auto out_w | IMPROVED-but-LOSS |
| VLM | up | 288,2048,16384 | bf8_b | chunked-M=32 (9 calls) | 264.00 | 254.69 | 1.037 | auto out_w | IMPROVED-but-LOSS |
| VLM | down | 288,16384,2048 | bf8_b | host-G G=2 (18 calls) [temporal BUILT, NOT adopted] | 404.71 | 236.72 | 1.710 | auto out_w | IMPROVED-but-LOSS |
| DENOISE | gate | 64,1024,4096 | bf8_b | chunked (2 calls) | 9.95 | 34.64 | **0.287** | auto out_w | **STRICT-BEAT** |
| DENOISE | up | 64,1024,4096 | bf8_b | frozen (2 calls) | 9.83 | 34.63 | **0.284** | auto out_w | **STRICT-BEAT** |
| DENOISE | down | 64,4096,1024 | bf8_b | frozen (2 calls) | 22.76 | 35.19 | **0.647** | auto out_w | **STRICT-BEAT** |

**STRICT-BEAT (4/12):** SigLIP qkv (0.833), DENOISE gate/up/down (0.287/0.284/0.647). UNCHANGED winners,
no regression through the Lever-0/Route-B shared edits.

**IMPROVED-but-LOSS (8/12):**
- SigLIP o, fc1 — fat-M-fill (whole-M, out_h>1) collapsed 8 thin calls → 1 fat call: o 38.58→33.89
  (−12%), fc1 65.15→54.57 (−16%). Still above native because native shards N=1152/4320 to get out_w
  fill that the WIDTH op's output-spec (cap-divisor out_cores) cannot reproduce for these N (npc=1).
- VLM qkv, o — whole-M doesn't fit L1 at low a_cores (K=2048, M_tiles=9); forcing a_cores=32 to fit
  FRAGMENTS A and made KERNEL WORSE (87/92µs vs chunked 56µs). Adopted state = chunked (56µs). LOSS.
- VLM gate, up — whole-M doesn't fit L1 at any a_cores (npc=8 in1/out too large); stay chunked.
  Near-tie at 1.037 but > 1+ε (ε=0.012). LOSS by a thin margin.
- SigLIP fc2 (2.945) — **Lever-2 WIDTH-temporal k_stream BUILT + PCC-validated** (PCC_torch 0.99997 /
  PCC_native 0.99990; ks=15, G=9, M_tiles=8 single output-group fp32 DST). Temporal single-call 123.95µs
  modestly beats the chunked-M 131.73µs (−6%) but stays far above native 42.09 (the 9 serial sender
  broadcasts + gather/done handshakes dominate; the GEMM is FPU-bound = native). Adopted (better than
  chunked). Still IMPROVED-but-LOSS.
- VLM down (1.710) — **Lever-2 temporal BUILT but NOT adopted.** Temporal single-call (ks=8, G=64,
  M_tiles=9, bf16 DST single-group) PCC_torch 0.9931 / PCC_native 0.9861, KERNEL 556.06µs = 2.349× —
  WORSE than the host-G fallback (404.71µs/1.710×). The 64 serial sender-broadcast rounds make temporal
  slower than the 18-call host-G split. fp32 DST (needed for full PCC, native is itself only 0.9986 vs
  torch on this lossy K=16384 bf8_b shape) requires M_tiles=9>8 → a 2-output-group A re-stream whose
  cross-group path did not converge (group≥1 garbage); the 2-call fp32 form gives PCC_torch 0.99997 but
  is multi-call. Host-G fallback STAYS (PCC-green, faster). LOSS.

**fat-M-fill landed + PCC-validated** (P0-A out_h>1 proven via Route-B; fc1 {2×1} PCC 1.000008,
{2×3}/{4×1} bf8_b 0.999842). The KERNEL win is real on o/fc1 but does not strict-beat native (same FPU,
worse N-shard). All 22 inventory shapes PCC ≥0.99 vs torch AND native (M0 + M1 gates, bit-identical at
defaults). KERNEL ε=0.012. **Lever-2 WIDTH-temporal k_stream BUILT (deep-plan_14_lever2_execution.md):
fc2 adopted (123.95µs, 2.945×, PCC 0.99997/0.99990); VLM-down built but host-G fallback faster (temporal
556µs > 405µs) so NOT adopted.** Source CSVs in `/tmp/deep-plan-20260611-205629-kernelbeat/` +
`deep-work/mmd13_csvs/`; temporal CSVs `/tmp/fc2_temporal.csv` + `/tmp/down_temporal.csv`. Tracy-only
device time throughout.

**(BEFORE — deep-plan_13 baseline, for reference)**: SigLIP o 38.58/1.89, fc1 65.15/1.58, fc2 131.44/3.12,
VLM qkv 56.04/1.45, o 56.40/1.50, gate/up 1.04, down 405.76/1.70 — same 4 winners. deep-plan_14 delta:
o 1.89→1.66, fc1 1.58→1.32 (fat-M-fill); others ≈ unchanged (whole-M not beneficial / not gated to Lever-2).

---

## deep-plan_15 — gather_in0 fat-fill (weight-stationary 2D, native FPU + L1-resident in1) — KERNEL companion

**WHAT THIS ADDS.** A NEW path inside `MatmulDecodeLinear` (env gate `WS2D_GATHER=1`,
`_build` byte-untouched): `Matmul1D` with `gather_in0=True`
(`MatmulMultiCoreReuseMultiCast1DProgramConfig`) — op-code **`MatmulDeviceOperation`** (NATIVE,
NOT `MatmulDecodeDeviceOperation`), in1 (weight) kept **L1-RESIDENT WIDTH_SHARDED** across calls,
GATE-LEGAL fat subblock (`out_subblock_w == per_core_N`, `out_subblock_h > 1` honoring the
`:1119` validator — the gather path has NO re-stride writer so the gate is real dataflow
correctness). Measured for the 3 `per_core_N ≤ 2` allowed-fat rows (SCOPE-A, zero C++).

**Parse:** `EXTRACT_MODE=mmsweep METRIC=KERNEL MMSWEEP_OP=native` (the additive switch sums
`MatmulDeviceOperation` col-20 ÷5; default `MMSWEEP_OP` still sums `MatmulDecodeDeviceOperation`).
**Back-test (HARD pre-gate):** `MMSWEEP_OP=mmd` over the frozen `mmd_*.csv` reproduced SigLIP qkv
**45.969 EXACT** + the full frozen table BOTH pre- and post-edit; `MMSWEEP_OP=native` over the same
frozen MMD CSVs yields 0.000/calls=0 (the STOP sentinel, as designed). Fork HEAD `e4500c1f`;
N_ITERS=5; ε=0.012. **Δ = new_ratio − prior_ratio (negative = improvement toward native).**

| stage | proj | [M,K,N] | dtype | state (scheme) | ws2d_KERNEL µs | native µs | ratio | Δ vs prior (prior→new) | subblock (calls/fwd) | VERDICT |
|---|---|---|---|---|--:|--:|--:|---|---|---|
| SigLIP | o | 256,1536,1152 | bf16 | gather_in0 fat-fill (resident in1, fp32) | 47.55 | 20.43 | 2.327 | 1.659 → 2.327 (+0.668) | oh=4 ow=1, cores=48 (1 call) | LOSS-UNMOVED |
| VLM | o | 288,2048,2048 | bf8_b | gather_in0 fat-fill (resident in1, fp32) | 74.53 | 37.35 | 1.995 | 1.521 → 1.995 (+0.474) | oh=3 ow=1, cores=64 (1 call) | LOSS-UNMOVED |
| VLM | qkv | 288,2048,2560 | bf8_b | gather_in0 fat-fill (resident in1, bf16, N-pad 2560→4096) | 121.29 | 38.38 | 3.161 | 1.466 → 3.161 (+1.695) | oh=3 ow=2, cores=64 (1 call) | LOSS-UNMOVED |
| SigLIP | qkv | 256,1152,4608 | bf16 | chunked (winner protected, Δ=0) | 46.18 | 55.44 | 0.833 | 0.833 → 0.833 (0.000) | auto out_w | WINNER-PROTECTED |
| SigLIP | fc1 | 256,1152,4320 | bf8_b | per_core_N>2 needs re-stride writer → chunked (unchanged) | 54.57 | 41.37 | 1.319 | 0.000 | (NOFIT-NULL) | NOFIT-NULL |
| SigLIP | fc2 | 256,4320,1152 | bf8_b | per_core_N>2 needs re-stride writer / gather OOM → WIDTH-temporal (unchanged) | 123.95 | 42.09 | 2.945 | 0.000 | (NOFIT-NULL) | NOFIT-NULL |
| VLM | gate | 288,2048,16384 | bf8_b | gather_in0 NOFIT (N=16384) → chunked (unchanged) | 263.96 | 254.61 | 1.037 | 0.000 | (NOFIT-NULL) | NOFIT-NULL |
| VLM | up | 288,2048,16384 | bf8_b | gather_in0 NOFIT (N=16384) → chunked (unchanged) | 264.00 | 254.69 | 1.037 | 0.000 | (NOFIT-NULL) | NOFIT-NULL |
| VLM | down | 288,16384,2048 | bf8_b | gather_in0 NOFIT (K=16384) → host-G (unchanged) | 404.71 | 236.72 | 1.710 | 0.000 | (NOFIT-NULL) | NOFIT-NULL |
| DENOISE | gate | 64,1024,4096 | bf8_b | chunked (winner protected, Δ=0) | 9.95 | 34.64 | 0.287 | 0.000 | auto out_w | WINNER-PROTECTED |
| DENOISE | up | 64,1024,4096 | bf8_b | frozen (winner protected, Δ=0) | 9.83 | 34.63 | 0.284 | 0.000 | auto out_w | WINNER-PROTECTED |
| DENOISE | down | 64,4096,1024 | bf8_b | frozen (winner protected, Δ=0) | 22.76 | 35.19 | 0.647 | 0.000 | auto out_w | WINNER-PROTECTED |

**HEADLINE (the NULL the data shows):** native-fat-fill via gather_in0 does **NOT** close the prefill
KERNEL gap on any of the 3 measurable allowed-fat rows — every one REGRESSED vs the prior WIDTH path
(SigLIP o 1.659→2.327, VLM o 1.521→1.995, VLM qkv 1.466→3.161; all Δ>0). The dispatch collapsed to
**1 call/fwd** (vs the WIDTH path's 8–9 thin M-tile calls), confirming the op-code is `MM_NATIVE`
(calls/fwd≠0 = not the STOP sentinel) — but the per-call KERNEL is WORSE because the gather substrate
forces a **K-tile-divisible core grid** (`num_cores | K_tiles`) that yields wasteful N geometry:
`per_core_N=1` starves the FPU's N-fill (each core does a thin 1-tile column; the fat `oh` fills M but
N is underfilled), and VLM qkv pays a 2560→4096 (+60%) N-pad on top. **Fat-M alone does not beat native
when the N-shard is worse than native's.** This is the predicted TIE-CEILING/LOSS terminal: gather_in0
fat-fill runs the same systolic FPU but on a less-favorable grid → measured NULL, an honest SUCCESS.

**Footnotes (anti-overclaim):** (1) KERNEL excludes glue/staging; residency is INVISIBLE on this metric
by construction (deep-plan_8) — the KERNEL story is FAT-FILL only. (2) native = FROZEN best-explicit
KERNEL min, NO re-sweep. (3) ε=0.012. (4) back-test reproduced 45.969 exact + full frozen table on the
mmsweep KERNEL default-op-code path, BOTH pre- and post-`MMSWEEP_OP`-edit. (5) ws2d op-code =
`MatmulDeviceOperation` (`MM_NATIVE`), summed via `MMSWEEP_OP=native`. (6) fork HEAD e4500c1f; N_ITERS=5;
tracy-only. (7) per_core_N>2 rows (fc1/fc2/gate/up/down) STAY on the deep-plan_14 path — fat-M is
unreachable on gather_in0 without a SCOPE-B re-stride output writer (NOT built; the :1119 gate is real
dataflow correctness, NOT relaxable without the writer). (8) The SECONDARY residency/TOTAL number lives
in deep-plan_15_execution.md §7, NEVER in this ratio column.

---

## deep-plan_16 — GATED resident-in1 2D-mcast (`MatmulMultiCoreReuseMultiCast`, native FPU, L1-resident in1)

**WHAT THIS ADDS.** A GATED branch on native's 2D-mcast factory
(`device/factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp`,
`create_program_mcast_in0_in1_descriptor`), env `TT_MM_RESIDENT_IN1_NO_MCAST=1`, default-OFF and
BYTE-IDENTICAL when unset. The fix RESIZES the in1 CB FIFO `total_size` to the full resident shard
so the `IN1_SHARDED` sender's `reserve_back(in1_block_num_tiles * num_blocks)` fits. ROOT CAUSE
(MEASURED by the S1 bisect, candidate #1 = CB over-reserve, NOT the stale-mcast): the in1 CB
`total_size` was sized for ONE inner-dim block × MCAST_INPUT_BUFFERING_DEPTH(=2), and
`set_globally_allocated_address` / `cb_desc.tensor` keeps `total_size_` unchanged
(circular_buffer_config.cpp L207) → the resident reserve of `num_blocks(=6) > 2` blocks deadlocks.
op-code = **`MatmulDeviceOperation`** (native; `MMSWEEP_OP=native`, calls/fwd=1.0 confirmed pure).
Fork HEAD `e4500c1f`; N_ITERS=5; ε=0.012. Back-test reproduced 45.969 EXACT pre+post-edit.

LAYOUT-A single-row geometry (full-K per core, N split across columns, mcast auto-OFF since
0 receivers). Default DRAM path byte-identical (gate-off still hangs the resident cell exactly as
stock — the gate is load-bearing; the default DRAM matmul PCC 0.999998 unchanged; no kernel
compile-time arg added → TensorAccessorArgs offsets unchanged).

### Single-op KERNEL (resident, LAYOUT-A single-row) vs frozen native min
| stage | proj | [M,K,N] | dtype | layout | EXPECTED | variant µs | native µs | ratio | Δ vs gather-resident-prior | calls/fwd | PCC_torch/native | VERDICT |
|---|---|---|---|---|---|--:|--:|--:|---|--:|---|---|
| SigLIP | o | 256,1536,1152 | bf16 | A (single-row) | TIE-optimistic / base-rate LOSS | 83.936 | 20.433 | **4.108** | 2.327 → 4.108 (+1.781) | 1.0 | 0.999998 / 1.000000 | IMPROVED-but-LOSS |
| DENOISE | gate | 64,1024,4096 | bf8_b | A (single-row) | reuse-target | 55.461 | 34.637 | 1.601 | (no resident prior) | 1.0 | 0.999973 / 1.000000 | IMPROVED-but-LOSS |
| DENOISE | down | 64,4096,1024 | bf8_b | A (single-row) | reuse-target | 57.959 | 35.190 | 1.647 | (no resident prior) | 1.0 | 0.999971 / 1.000000 | IMPROVED-but-LOSS |
| VLM | o | 288,2048,2048 | bf8_b | A (single-row) | — | — | 37.354 | — | — | — | — | NOFIT-NULL (L1 clash, program.cpp:1476) |

**HEADLINE (the honest single-op terminal):** every resident-in1 single-op row LOSES, consistent
with the §0.2 base rate (100% of resident-in1 single-op measurements lost 2-3×; here 1.6-4.1×).
The single-row LAYOUT-A geometry is FPU-starved (only `nc` cores in ONE row, e.g. SigLIP o = 9
cores × `per_core_N=4`) vs native's full 2D grid (72 cores). KERNEL is invisible to residency by
construction — the resident weight DELIVERY changes but the single-row OUTPUT parallelism is far
worse than native's 2D fat fill. SigLIP single-op TIE (the optimistic target) was NOT reached.

### reuse-TOTAL (denoise 10-step loop, matmul-op DEVICE KERNEL; LAYOUT-A → cross-core-reduce term = 0)
3-leg (N native DRAM-stream / R resident-once / X per-step re-stage), SAME single-row program
config for N and R (apples-to-apples), SEED=1234, M=64, bf8, N_STEPS=10, N_ITERS=5. Residency
EVIDENCED: all 3 resident buffer-addresses STABLE across the loop; leg R has ZERO I2S/S2I rows
(weight staged once in warm-up), leg X adds 50 device-captured `InterleavedToShardedDeviceOperation`
re-stage rows R does not (R-vs-X isolation OK). Leg N has ZERO I2S (native folds the in1 DRAM read
inside the matmul op).

| leg | matmul-op µs/loop | I2S rows | note |
|---|--:|--:|---|
| N native (same config, in1 DRAM) | **1755.63** | 0 | native re-reads in1 DRAM inside each matmul |
| R resident (in1 L1-resident) | **1689.30** | 0 | weight staged ONCE; reused 10 steps |
| X re-read (per-step re-stage) | matmul == R (554.8 vs 554.7 us gate); +50 I2S | 50 | residency-OFF control |

- **vs native-SAME-config: resident BEAT, ratio 0.962** (3.8%, beyond ±2%). Residency saves the
  in1 DRAM read native folds into its matmul-op (R-vs-X confirms the matmul-op time is config-bound,
  not residency-bound; the win is N-vs-R: native pays the folded DRAM read, resident reads L1).
- **vs native-BEST-explicit config (frozen min, full 2D grid): resident LOSES, ratio 1.626**
  (1689.30 vs 30×34.637 = 1039.1 us/loop). The single-row geometry tax exceeds the DRAM-read saving.

**HONEST verdict:** a residency-attributable reuse BEAT exists ONLY when native is constrained to the
same single-row geometry; against native's best 2D config the single-row resident loses. Terminal (b/c).
