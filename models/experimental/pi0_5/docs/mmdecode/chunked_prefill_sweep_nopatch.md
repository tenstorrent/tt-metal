# Chunked-prefill sweep — no-patch committed-base `matmul_decode` (Blackhole P150)

> **DEVICE KERNEL DURATION** (min-of-5, tracy col-20) via `extract_perf METRIC=KERNEL`. Tree clean @ `e4500c1f`, **no patches**. PCC≥0.99 vs torch on every FIT cell.
>
> **NATIVE BASELINE — read this.** The reference for the verdict is the **GOLDEN best-explicit native** = the frozen min-per-call KERNEL over the explicit native config sweep (`deep-work/mmd13_csvs/nat_*.csv`, `EXTRACT_MODE=natexp`), the SAME baseline as `unified_matmul_3stage_kerneltime.md`. The `Nat unch` column inside the µs table below is this sweep's **plain `ttnn.linear` (auto/default program config)** — it is NOT the golden baseline and runs 1.3–4× slower than golden on the shapes where TTNN's default geometry is poor (SigLIP o, fc2; VLM qkv/o/down). The in-table `(×f)` factors are therefore relative to *plain-default native* and OVERSTATE mmd's win; the **verdict table uses golden native** and is authoritative.

## STATUS: COMPLETE — 69 TIMED / 24 INVALID / 6 NOFIT (99 cells)

- Back-test gate `45.969` reproduced exactly. tt-metal `e4500c1f`, clean tree.
- **Profiler-segfault root cause (resolved):** the sweep driver passed `--op-support-count 200000`, which faults the device-profiler buffer dump at `close_device`; **20000** (canonical) dumps cleanly. Not a board/firmware/cache issue — reboot/driver-reload/cold-cycle were unrelated. Fixed in `run_sweep.sh`.
- **Native framing (the regression flagged):** this sweep's `Nat unch` cells are un-tuned default `ttnn.linear`. Against the GOLDEN best-explicit native (frozen `nat_*.csv`, min over the explicit-config sweep) the picture matches the established `unified_matmul_3stage_kerneltime.md` terminal — mmd STRICT-BEATs only DENOISE; SigLIP/VLM prefill is IMPROVED-but-LOSS. See the golden verdict table.


## TABLE — KERNEL µs (×factor vs THIS-SWEEP plain-default `ttnn.linear`, NOT golden — see verdict table for golden)

### SigLIP (M=256)
| proj [M·K·N dtype] | Nat unch | mmdFULL unch | mmdPART unch | Nat T32 | mmdF T32 | mmdP T32 | Nat T64 | mmdF T64 | mmdP T64 |
|---|---|---|---|---|---|---|---|---|---|
| qkv [256·1152·4608 bf16] | 55.0 (×1.00) | INVALID[out-core>104] | 66.9 (×1.22) | 226.8 (×4.13) | INVALID[out-core>104] | 105.4 (×1.92) | 121.3 (×2.21) | INVALID[out-core>104] | 83.7 (×1.52) |
| o [256·1536·1152 bf16] | 47.7 (×1.00) | 33.1 (×0.69) | 27.4 (×0.57) | 191.3 (×4.01) | 38.5 (×0.81) | 38.9 (×0.82) | 107.7 (×2.26) | 35.5 (×0.74) | 32.5 (×0.68) |
| fc1 [256·1152·4320 bf8] | 54.5 (×1.00) | INVALID[out-core>104] | 67.1 (×1.23) | 187.7 (×3.45) | INVALID[out-core>104] | 109.5 (×2.01) | 106.9 (×1.96) | INVALID[out-core>104] | 86.2 (×1.58) |
| fc2 [256·4320·1152 bf8] | 173.0 (×1.00) | NOFIT[Kc not tile-aln] | NOFIT[cannot N-chunk] | 961.2 (×5.56) | 131.0 (×0.76) | NOFIT[cannot N-chunk] | 508.1 (×2.94) | 129.4 (×0.75) | NOFIT[cannot N-chunk] |

### VLM (M=288)
| proj [M·K·N dtype] | Nat unch | mmdFULL unch | mmdPART unch | Nat T32 | mmdF T32 | mmdP T32 | Nat T96 | mmdF T96 | mmdP T96 |
|---|---|---|---|---|---|---|---|---|---|
| qkv [288·2048·2560 bf16] | 73.0 (×1.00) | 50.4 (×0.69) | INVALID[L1-CB clash] | 379.7 (×5.20) | 56.2 (×0.77) | 103.8 (×1.42) | 151.1 (×2.07) | 51.3 (×0.70) | 84.6 (×1.16) |
| o [288·2048·2048 bf16] | 63.9 (×1.00) | 50.5 (×0.79) | INVALID[L1-CB clash] | 339.8 (×5.32) | 56.6 (×0.89) | 102.6 (×1.61) | 138.5 (×2.17) | 51.5 (×0.81) | 81.6 (×1.28) |
| gate [288·2048·16384 bf16] | 259.4 (×1.00) | NOFIT[no a_cores fit bank] | INVALID[L1-CB clash] | 1550.8 (×5.98) | INVALID[out-core>104] | 818.7 (×3.16) | 560.5 (×2.16) | INVALID[out-core>104] | INVALID[L1-CB clash] |
| up [288·2048·16384 bf16] | 259.4 (×1.00) | NOFIT[no a_cores fit bank] | INVALID[L1-CB clash] | 1550.9 (×5.98) | INVALID[out-core>104] | 819.0 (×3.16) | 560.4 (×2.16) | INVALID[out-core>104] | INVALID[L1-CB clash] |
| down [288·16384·2048 bf8] | 415.7 (×1.00) | INVALID[L1-CB clash] | INVALID[rejected] | 2586.4 (×6.22) | 421.1 (×1.01) | INVALID[L1-CB clash] | 1025.8 (×2.47) | 415.5 (×1.00) | INVALID[rejected] |

### DENOISE (M=64)
| proj [M·K·N dtype] | Nat unch | mmdFULL unch | mmdPART unch | Nat T32 | mmdF T32 | mmdP T32 |
|---|---|---|---|---|---|---|
| gate [64·1024·4096 bf16] | 24.7 (×1.00) | INVALID[out-core>104] | 23.2 (×0.94) | 44.0 (×1.78) | INVALID[out-core>104] | 29.7 (×1.20) |
| up [64·1024·4096 bf16] | 24.7 (×1.00) | INVALID[out-core>104] | 23.2 (×0.94) | 44.0 (×1.78) | INVALID[out-core>104] | 29.7 (×1.20) |
| down [64·4096·1024 bf8] | 66.1 (×1.00) | 22.1 (×0.33) | 17.0 (×0.26) | 117.9 (×1.78) | 22.9 (×0.35) | 18.8 (×0.28) |

## VERDICT (authoritative) — best mmd unchunked-FIT vs GOLDEN best-explicit native

Golden native = frozen min-per-call KERNEL over the explicit native config sweep
(`deep-work/mmd13_csvs/nat_*.csv`, `EXTRACT_MODE=natexp METRIC=KERNEL`) — same baseline as
`unified_matmul_3stage_kerneltime.md`. ratio = best-mmd ÷ golden-native. Tie band ε=0.012:
`<1−ε` STRICT-BEAT · `|r−1|≤ε` TIE · `>1+ε` LOSS. (plain-nat = this sweep's default `ttnn.linear`,
shown only to expose the framing gap.)

| proj | plain-nat µs | **golden-nat µs** | best mmd unchunked | ratio vs golden | VERDICT |
|---|--:|--:|---|--:|---|
| SigLIP.qkv | 55.0 | **55.44** | 66.9 (partial) | 1.206 | LOSS |
| SigLIP.o | 47.7 | **20.43** | 27.4 (partial) | 1.340 | LOSS |
| SigLIP.fc1 | 54.5 | **41.37** | 67.1 (partial) | 1.621 | LOSS |
| SigLIP.fc2 | 173.0 | **42.09** | — (unchunked NOFIT; chunked mmd_full_T32 131.0 → 3.11) | 3.112 | LOSS |
| VLM.qkv | 73.0 | **38.38** | 50.4 (full) | 1.314 | LOSS |
| VLM.o | 63.9 | **37.35** | 50.5 (full) | 1.352 | LOSS |
| VLM.gate | 259.4 | **254.61** | — (unchunked NOFIT/INVALID; partial_T32 818.7 → 3.22) | 3.216 | LOSS |
| VLM.up | 259.4 | **254.69** | — (unchunked NOFIT/INVALID; partial_T32 819.0 → 3.22) | 3.216 | LOSS |
| VLM.down | 415.7 | **236.72** | — (unchunked INVALID; chunked mmd_full_T96 415.5 → 1.76) | 1.755 | LOSS |
| DENOISE.gate | 24.7 | **34.64** | 23.2 (partial) | 0.669 | **STRICT-BEAT** |
| DENOISE.up | 24.7 | **34.63** | 23.2 (partial) | 0.671 | **STRICT-BEAT** |
| DENOISE.down | 66.1 | **35.19** | 17.0 (partial) | 0.483 | **STRICT-BEAT** |

**STRICT-BEAT (3/12):** DENOISE gate/up/down only. **LOSS (9/12):** every SigLIP + VLM prefill row.

## PER-OPTIMIZATION KERNEL µs (one column per optimization)

Every distinct optimization path as its own column, KERNEL µs (min-of-5). The two `golden*` columns are
the patched-binary reference (`unified_matmul_3stage_kerneltime.md`); all other columns are this clean
no-patch `e4500c1f` sweep. `Tbest` = best of the valid chunked-T forms (T noted). `—` = NOFIT/INVALID
on the committed op. **Ratio rows below each use golden best-explicit native as the denominator.**

| proj [M·K·N dtype] | plain native (default linear) | **golden native** (best-explicit sweep) | mmd FULL (resident, T=M) | mmd PARTIAL (K-split, T=M) | mmd FULL chunked (Tbest) | mmd PARTIAL chunked (Tbest) | **golden mmd** (adopted lever) |
|---|--:|--:|--:|--:|--:|--:|--:|
| SigLIP.qkv [256·1152·4608 bf16] | 55.0 | **55.4** | — INVALID | 66.9 | — INVALID | 83.7 (T64) | **46.2** (chunked-M=32) |
| SigLIP.o [256·1536·1152 bf16] | 47.7 | **20.4** | 33.1 | 27.4 | 35.5 (T64) | 32.5 (T64) | **38.6** (whole-M out_h=8) |
| SigLIP.fc1 [256·1152·4320 bf8] | 54.5 | **41.4** | — INVALID | 67.1 | — INVALID | 86.2 (T64) | **65.2** (whole-M out_h=8) |
| SigLIP.fc2 [256·4320·1152 bf8] | 173.0 | **42.1** | — NOFIT | — NOFIT | 129.4 (T64) | — NOFIT | **131.4** (WIDTH-temporal k_stream) |
| VLM.qkv [288·2048·2560 bf16] | 73.0 | **38.4** | 50.4 | — INVALID | 51.3 (T96) | 84.6 (T96) | **56.0** (chunked-M=32) |
| VLM.o [288·2048·2048 bf16] | 63.9 | **37.4** | 50.5 | — INVALID | 51.5 (T96) | 81.6 (T96) | **56.4** (chunked-M=32) |
| VLM.gate [288·2048·16384 bf16] | 259.4 | **254.6** | — NOFIT | — INVALID | — INVALID | 818.7 (T32) | **264.0** (chunked-M=32) |
| VLM.up [288·2048·16384 bf16] | 259.4 | **254.7** | — NOFIT | — INVALID | — INVALID | 819.0 (T32) | **264.0** (chunked-M=32) |
| VLM.down [288·16384·2048 bf8] | 415.7 | **236.7** | — INVALID | — INVALID | 415.5 (T96) | — INVALID | **405.8** (host-G G=2) |
| DENOISE.gate [64·1024·4096 bf16] | 24.7 | **34.6** | — INVALID | 23.2 | — INVALID | 29.7 (T32) | **9.9** (chunked, 2 calls) |
| DENOISE.up [64·1024·4096 bf16] | 24.7 | **34.6** | — INVALID | 23.2 | — INVALID | 29.7 (T32) | **9.8** (frozen, 2 calls) |
| DENOISE.down [64·4096·1024 bf8] | 66.1 | **35.2** | 22.1 | 17.0 | 22.9 (T32) | 18.8 (T32) | **22.8** (frozen, 2 calls) |

### Same matrix as ratio ÷ golden best-explicit native (ε=0.012; **bold** = STRICT-BEAT)

| proj | plain native | golden native | mmd FULL | mmd PART | mmd FULL Tbest | mmd PART Tbest | golden mmd |
|---|--:|--:|--:|--:|--:|--:|--:|
| SigLIP.qkv | 0.99 | 1.00 | — | 1.21 | — | 1.51 | **0.83** |
| SigLIP.o | 2.34 | 1.00 | 1.62 | 1.34 | 1.74 | 1.59 | 1.89 |
| SigLIP.fc1 | 1.32 | 1.00 | — | 1.62 | — | 2.08 | 1.57 |
| SigLIP.fc2 | 4.11 | 1.00 | — | — | 3.07 | — | 3.12 |
| VLM.qkv | 1.90 | 1.00 | 1.31 | — | 1.34 | 2.21 | 1.46 |
| VLM.o | 1.71 | 1.00 | 1.35 | — | 1.38 | 2.19 | 1.51 |
| VLM.gate | 1.02 | 1.00 | — | — | — | 3.22 | 1.04 |
| VLM.up | 1.02 | 1.00 | — | — | — | 3.22 | 1.04 |
| VLM.down | 1.76 | 1.00 | — | — | 1.76 | — | 1.71 |
| DENOISE.gate | 0.71 | 1.00 | — | **0.67** | — | 0.86 | **0.29** |
| DENOISE.up | 0.71 | 1.00 | — | **0.67** | — | 0.86 | **0.28** |
| DENOISE.down | 1.88 | 1.00 | **0.63** | **0.48** | **0.65** | **0.53** | **0.65** |

Reading the columns: **golden mmd** strict-beats golden native only on SigLIP.qkv + DENOISE×3 (the
established golden winners). On the clean no-patch base the only mmd column that still strict-beats is on
**DENOISE** (FULL/PARTIAL ×0.48–0.67); every SigLIP/VLM mmd column loses, and the no-patch mmd is itself
2.3–3.1× off the golden mmd on DENOISE gate/up and VLM gate/up (the rows whose golden numbers needed the
tuned levers). Plain native also loses to golden native by 1.3–4× on the poorly-defaulted shapes.

## mmd ALSO regressed from the golden table (different cause than native)

The golden `unified_matmul_3stage_kerneltime.md` mmd column was measured on the **patched binary** with
the adopted tuning levers (chunked-M=32 fat-fill, Lever-2 WIDTH-temporal k_stream, host-G splits,
frozen-tuned subblocks). **This sweep is the CLEAN committed base `e4500c1f` with NONE of those levers**
(the fat-fill `out_subblock` knobs / k_stream toggle / §6.4 reduce fix are not in the tree) AND it runs
the structural unchunked/fixed-T cells, not the golden adopted scheme. So mmd here is slower than the
golden mmd on most rows:

| proj | golden mmd µs (scheme) | best mmd this-sweep µs (form) | sweep/golden |
|---|--:|--:|--:|
| SigLIP.qkv | 46.18 (chunked-M=32, 8 calls) | 66.9 (mmd_partial unch) | 1.45 |
| SigLIP.o | 38.58 (whole-M out_h=8) | 27.4 (mmd_partial unch) | 0.71 |
| SigLIP.fc1 | 65.15 (whole-M out_h=8) | 67.1 (mmd_partial unch) | 1.03 |
| SigLIP.fc2 | 131.44 (WIDTH-temporal k_stream) | 129.4 (mmd_full_T64) | 0.98 |
| VLM.qkv | 56.04 (chunked-M=32) | 50.4 (mmd_full unch) | 0.90 |
| VLM.o | 56.40 (chunked-M=32) | 50.5 (mmd_full unch) | 0.90 |
| VLM.gate | 263.96 (chunked-M=32) | 818.7 (mmd_partial_T32) | **3.10** |
| VLM.up | 264.00 (chunked-M=32) | 819.0 (mmd_partial_T32) | **3.10** |
| VLM.down | 405.75 (host-G G=2, 18 calls) | 415.5 (mmd_full_T96) | 1.02 |
| DENOISE.gate | 9.90 (chunked, 2 calls) | 23.2 (mmd_partial unch) | **2.34** |
| DENOISE.up | 9.83 (frozen, 2 calls) | 23.2 (mmd_partial unch) | **2.36** |
| DENOISE.down | 22.76 (frozen, 2 calls) | 17.0 (mmd_partial unch) | 0.75 |

**The big mmd regressions are DENOISE gate/up (×2.3, golden 9.9µs vs sweep 23.2µs) and VLM gate/up
(×3.1, golden 264µs vs sweep 819µs)** — precisely the rows whose golden numbers depended on the
tuned chunked-M=32 / frozen 2-call schemes that the no-patch base does not have. (A few rows are
*faster* than golden — SigLIP o/fc2, VLM qkv/o — because the no-patch unchunked resident-FULL form
beats the golden adopted chunked form on those shapes; that does not change their LOSS vs golden native.)

**Net:** this no-patch sweep is NOT a re-measurement of the golden table — it is the clean committed
base, and BOTH its native (plain default vs golden best-explicit) and its mmd (untuned structural cells
vs golden adopted levers) sit below the golden numbers. The golden table remains the patched-binary,
best-scheme reference; reproducing its mmd numbers requires the deep-plan tuning levers, which are out
of scope for this no-patch run.

**Note on SigLIP qkv:** the golden qkv win (0.833 vs golden native) used the CHUNKED-M=32 mmd scheme
(46.18µs). This no-patch sweep's qkv mmd is unchunked-partial 66.9µs (×1.21 vs golden native) and its
chunked-T32 form is 105.4µs (worse) — so that beat is scheme-specific and is not reproduced by the
structural cells here.

## Honest read-out (against GOLDEN best-explicit native)

**0. Framing correction (the flagged regression).** The earlier read-out compared mmd to this
sweep's **plain-default `ttnn.linear`**, which is 1.3–4× slower than the golden best-explicit native
on poorly-default-tiled shapes (SigLIP o 47.7 vs golden 20.4; VLM qkv 73.0 vs 38.4; fc2 173.0 vs 42.1).
That made several mmd cells look like wins. Against the **golden** baseline the result flips and matches
the established terminal.

**1. mmd STRICT-BEATs golden native on DENOISE only (3/12).** DENOISE down ×0.483 (17.0 vs golden 35.19),
gate ×0.669, up ×0.671. These are the small-M (M=64), low-K reuse-friendly shapes where the
width/partial-sharded resident weight genuinely out-utilizes native's 2D-mcast. Same 3 winners as the
golden `unified_matmul_3stage_kerneltime.md` study.

**2. mmd LOSES golden native on every SigLIP + VLM prefill row (9/12).** SigLIP o ×1.34, fc1 ×1.62,
qkv ×1.21; VLM qkv ×1.31, o ×1.35. These are compute-bound (M=256/288), run the SAME systolic FPU as
native, and native's best-explicit 2D grid out-fills the WIDTH/partial op's output geometry — the
documented TIE-CEILING/LOSS. The qkv beat the golden study recorded (0.833) was the **chunked-M=32** mmd
scheme (46.18µs); this no-patch sweep's unchunked qkv mmd is 66.9µs (×1.21) and its chunked-T32 form is
105.4µs (far worse), so that beat is scheme-specific and not reproduced here.

**3. wide-N: PARTIAL fits where FULL can't, but never wins.** SigLIP qkv/fc1 and VLM gate/up exceed the
104-core output cap on FULL (NOFIT/INVALID); PARTIAL K-split FITs (the structural win) but the cross-core
reduce makes it ×1.2 (SigLIP) to ×3.2 (VLM gate/up T32) vs golden native. Fitting ≠ faster.

**4. chunking is never a speed win, same-scheme.** Native chunked ×1.8–6.2 vs its own one-shot
(weight re-streamed per chunk); mmd chunked ≥ mmd unchunked wherever unchunked fits. Chunking's only
role is fit-enabling for weights too large for one resident call (SigLIP fc2 mmd_full_T32, VLM
gate/up/down) — and even those lose to golden native (fc2 ×3.11, VLM down chunked-FULL ×1.76).

**Bottom line (golden framing):** the committed-base `matmul_decode` **strict-beats golden best-explicit
native only on DENOISE (gate/up/down, ×0.48–0.67)**; on all SigLIP/VLM prefill projections it is
IMPROVED-but-LOSS (×1.2–1.6 where it fits one-shot, ×1.8–3.2 where it must chunk/partial). One-shot beats
chunking for the same scheme. This is consistent with the golden `unified_matmul_3stage_kerneltime.md`
terminal; the apparent broader wins in the prior draft were an artifact of comparing against un-tuned
default native rather than the golden best-explicit baseline.


## Artifacts
- Per-cell CSVs + logs: `deep-work/chunked_sweep_nopatch_csvs/{<tag>.csv,raw/<tag>.log}`
- Per-cell extract logs: `deep-work/chunked_sweep_nopatch_csvs/extract_logs/<tag>.extract.log`
- **Authoritative min-of-5 data (table source):** `deep-work/chunked_sweep_nopatch_csvs/kernel_min_of5.tsv`
  (tag, op, min_us, calls/fwd, reps=5, spread%). Built by `extract_minof5.sh` → `min_of5.py`,
  which reuses the back-tested `extract_perf.parse_mmsweep` (METRIC=KERNEL, col-20) and takes the
  MIN per-forward summed-KERNEL across the 5 `MMD_<stage>_<proj>_r0..4` rep regions.
- Avg-of-5 cross-check: `kernel_table_data.tsv` (≤1.55% from min in the worst cell; verdicts identical).
- Sweep driver (fixed): `deep-work/chunked_sweep_nopatch_csvs/run_sweep.sh` (op-support-count=20000)
- Sweep log: `deep-work/profiling_logs/full_sweep.log`
- Canonical KERNEL extractor: `tests/models/pi05_chunked_mmdecode/extract_perf.py`
  (`EXTRACT_MODE=mmsweep METRIC=KERNEL MMSWEEP_OP=native|mmd`).
