# C/D FP32 recurrence fusion, sprint v3

## Outcome and scope

The frozen `l1_early` candidate removes the separate numerator/denominator state-update pass when **all 256 query rows have an unchanged, finite running maximum**. It accumulates the new FP32 PV numerator and denominator directly into the old FP32 L1 banks. The guard is scanned incrementally during QK work, rather than as a blocking whole-Q scan before PV.

On the sustained resident repeated-KV compute test, this reduces time **6.26% for D and 8.81% for C**, relative to the best pre-v3 implementations. These are conditional compute-throughput gains, not full-chip/model speedups. With distinct normal inputs at 256K on two cores, reductions are **1.58% D / 2.55% C**; some D short/changing-max cases regress about 0.6%.

Everything is private experimental code; no production source or canonical recipe was changed. Tested geometry is noncausal, unmasked Q256/K512/D128, one original KV input slot, BF16 Q/K/V/output. Reader/writer sources, CB capacities/formats, input preparation, exp coefficients, fidelity, and Q/K block sizes remain unchanged. No claim covers arbitrary shapes, masks, ring mode, attention sinks, latent-V, or other hardware.

| Variant | QK / PV | State and DST | Other preserved choices | Fresh comparison baseline |
|---|---|---|---|---|
| D | HiFi4 / HiFi4 | FP32 numerator, denominator, score CBs and DST | Full-FP32 subtraction, accurate exp path, denominator phases 0+2, no QKV preprocessing | v2 scoped function-O2 winner |
| C | HiFi4 / HiFi2 | FP32 numerator, denominator, score CBs and DST | Existing cheaper subtraction and matched cubic exp, original denominator arithmetic, no QKV preprocessing | v1 scheduling winner, O3 |

D retains its scoped `#pragma GCC optimize("O2")` wrapper; this is not a global compiler/linker optimization-level change. Both variants retain v1 identity4/state-unpack-batch/lazy-init/max-scan changes. C retains its C-refine and C-refine-hoist flags.

## Sustained resident compute result

Source: `integrity-early-sustained-v2.json`, run by the parent on reservation 224379 after fresh device/control smoke tests. There are nine alternating-order rounds per variant and three implementations per round (54 records). Each record has two warmup and three measured trace replays. Q repeats eight times, each with 512 K chunks. Useful work is 274,877,906,944 FLOPs per invocation on one core.

| Variant | Best pre-v3 baseline ms | Late whole-Q guard ms | Early guard ms | Early time reduction | Early TFLOP/s/core |
|---|---:|---:|---:|---:|---:|
| D | 327.670747 | 316.005588 | 307.173438 | 6.2555% | 0.894862 |
| C | 244.263508 | 231.457498 | 222.732721 | 8.8146% | 1.234116 |

Values are medians of per-record median unprofiled blocking trace times, not profiler durations. TFLOP/s counts useful QK+PV work only. Repeated identical KV makes the guard true after the first chunk; this is a favorable compute upper-bound workload, not a representative guard frequency for normal distinct inputs. We do not extrapolate these per-core figures to chip throughput.

All 54 records passed numerical gates, raw eager/replay equality, actual device-input before/after hashes, and selected-source before/after hashes. The original D control is the v2 O2 baseline, not the older/slower v1 O3 implementation.

The fusion does change output bits in this long repeated-KV test: D L2 is 0.177935→0.177947%, C 0.380958→0.380972%; candidate-to-baseline output distances are 0.002132% and 0.006623%, respectively. Both comfortably satisfy the relative-regression gate.

## Distinct-Q/K/V, unchanged data-movement check

These are two-core executions with Q=2048 (four different Q blocks per core), the same Q256/K512 geometry and unchanged reader/writer/input buffering. Seed 1244, normal and changing-max inputs, three alternating-order paired rounds, five warmup and nine measured trace replays per record. Source artifacts are `integrity-early-distinct-{D,C}-{32768,262144}-v2.json`. Values below are median per-record trace times; negative time change is better. This is not the resident/no-DM test and not a full-chip measurement.

| Variant | K length | Distribution | Best baseline ms | Early ms | Time change |
|---|---:|---|---:|---:|---:|
| D | 32768 | Normal | 21.795763 | 21.928185 | +0.608% |
| D | 32768 | Changing max | 22.125269 | 22.264042 | +0.627% |
| D | 262144 | Normal | 167.519197 | 164.874338 | -1.579% |
| D | 262144 | Changing max | 176.393981 | 177.456971 | +0.603% |
| C | 32768 | Normal | 16.628947 | 16.593357 | -0.214% |
| C | 32768 | Changing max | 16.935053 | 16.857892 | -0.456% |
| C | 262144 | Normal | 126.133132 | 122.914583 | -2.552% |
| C | 262144 | Changing max | 134.909364 | 134.298773 | -0.453% |

All 48 records above passed numerical, eager/replay, actual device-input, and selected-source integrity checks, independently audited locally. D normal 256K changes 18 BF16 output elements out of 262144; L2 changes 0.18024531→0.18024861%, candidate-to-baseline distance 0.00299333%. C normal 256K changes 17 elements; L2 changes 0.40210145→0.40210026%, distance 0.00120389%. Other cases above are bit-identical to their original baseline. The synthetic monotonically changing-max 256K case has a very large inherited error (D L2 63.26%, C 63.52%); preservation of that error is not an absolute accuracy endorsement.

Thus the resident improvement is real, but not a blanket model-performance improvement: the guard rarely/never wins on some distributions, and its overhead can slightly regress short or changing-max D workloads. An opt-in, explicitly scoped long-context implementation is more defensible than an unconditional generic replacement; no unmeasured crossover is asserted.

## Numerical qualification and integrity

The per-case gate is `candidate L2% <= 1.05 * baseline L2% + 0.0001` percentage points, against the same blocked FP64 reference made from the original BF16 inputs. Exactly zero references use `max_abs <= max(1e-6, 1.05 * baseline_max_abs)`. Passing this relative-regression gate does **not** mean every stress input meets an absolute 0.5% accuracy target.

The successor integrity harnesses require at least two actual trace replays even for zero timed iterations; raw BF16 uint16 output equality with eager execution; host-input equality with device readback; device Q/K/V unchanged after execution; and immutable hashes of a selected source set. That set includes recipes, private kernels, relevant baseline/numeric helpers, dataflows and metrics, but is **not the entire transitive SDK/compiler closure**. Remote AppleDouble `._` metadata files appear in some pins; independent local auditing skips those non-code sidecars and rehashes every real source.

| Completed authoritative artifact | Coverage | Records | Result |
|---|---|---:|---|
| `integrity-k1.json` | C/D; K=512 first=last chunk; two Q repetitions; normal, V=3.25, V=0; original/late/early | 18 | All numerical and integrity checks pass |
| `integrity-stress32.json` | C/D; K=32768 distinct KV; seed 1243; 15 distributions; original/late/early | 90 | All checks pass; both candidates bit-identical to original on all 30 variant/case combinations |
| `integrity-early-sustained-v2.json` | C/D resident Q-repeat=8, K-chunks=512; nine rounds; original/late/early | 54 | All checks pass |
| `integrity-early-boundary-k2-v2.json` | C/D; two K chunks, two Q repetitions; seed 1245; six distributions; original/early | 24 | All checks pass; only D transitions changes output bits |
| `integrity-early-boundary-k3-v2.json` | C/D; three K chunks, two Q repetitions; seed 1245; six distributions; original/early | 24 | All checks pass; only D coherent changes output bits |
| `integrity-early-heldout256-v2.json` | C/D; K262144 distinct KV; held-out seed 1245; seven distributions; original/early | 28 | All checks pass |
| Four `integrity-early-distinct-*-v2.json` files | C/D; Q2048 on two cores; distinct K32768/262144; normal/changing max; three paired rounds | 48 | All checks pass; timing and numerical details above |

Across these final integrity artifacts there are **286 execution records: 116 early-candidate executions, 116 original controls, and 54 late-candidate intermediate controls**. Do not call these 286 independent candidate test cases. The dedicated accuracy/boundary suites contain 74 early-candidate cases plus their 74 original controls (and 36 additional late controls). Timed suites add 42 early executions of ten distinct configurations: two sustained configurations repeated nine times and eight distinct-DM configurations repeated three times. All records were independently audited against local real-source SHA pins and the recorded numerical/integrity checks.

The worst relative L2 increase among **early-candidate records only**, including the timed qualification, is **0.018911% relative** (not 0.018911 percentage points), far below the allowed 5% relative increase: held-out C/coherent/256K rises from 0.366189975% to 0.366259225% L2, a **0.0000692503 percentage-point** increase. Its output distance from baseline is 0.00742431%. The zero-reference V=0 cases pass the explicit absolute-error gate rather than a ratio.

The 15 stress distributions are normal, scaled QK, outliers, common Q, common K, common V, uniform, V=1, V=3.25, V=0, repeated coherent KV, cancellation, tiny updates, identity/max-change transitions, and continually changing maxima. JSON records retain L2/PCC, row p95/p99/worst L2, reference-near-zero absolute diagnostics, and candidate-to-baseline distance. PCC is undefined for constant outputs and is recorded as null, not claimed to be perfect.

The two-/three-chunk boundary suites use normal, transitions, coherent, cancellation, V=3.25 and V=0. Their only changed-output cases have tiny distances: D K2 transitions 0.0000001275% and D K3 coherent 0.0002302%. Q repetitions exercise state-bank reset across jobs; the separate Q2048/two-core checks use genuinely different query blocks.

The held-out 256K suite uses normal, outliers, common V, V=3.25, cancellation, coherent KV and tiny updates. Normal-input L2 is D 0.180499874→0.180506911% and C 0.393721294→0.393721294%; cancellation remains D 0.773052% / C 0.903700%. All row/tail diagnostics and near-zero errors remain available in the raw records; the acceptance gate itself is the per-case global-L2/zero-reference rule above, not an undisclosed row-error threshold.

Representative 32K seed-1243 errors (early and original are bit-identical here):

| Distribution | D L2% | C L2% |
|---|---:|---:|
| Normal | 0.178024 | 0.376902 |
| Outliers | 0.351295 | 0.419111 |
| Common K | 0.711806 | 1.017390 |
| Cancellation | 1.188088 | 1.249439 |
| Changing maximum stress | 6.038275 | 8.152069 |

These inherited stress errors are preserved, not solved by scheduling/fusion. In the common-V stress, low global L2 conceals loss of small query-varying output components after BF16 output rounding; the centered-query-variation metric records that loss (100% for the evaluated offset case). Constant V alone is an insufficient regression test: an earlier broken pack-mode transition corrupted both numerator and denominator in a way that canceled on constant V.

Earlier `both-*`, `early-smoke`, and `early-first-screen` experiments have numerical/replay checks but predate device-input immutability assertions. They are supporting exploration, not substituted for the integrity-qualified final evidence.

## Source/lifetime review

The frozen early candidate is `early_guard.hpp`, used by `resident_early.cpp` and `compute_early.cpp`. Relevant locations:

- Lines 2185–2227: reset per-call sum cleanup state; finite, low-16-bit BF16 maximum equality scan. NaN/Inf maxima reject the shortcut; signed-zero bit differences reject it conservatively.
- Lines 2404–2409: scan each preceding max row after issuing the next QK block. Rows 0–6 are scanned here; the final row is scanned at lines 2575–2586. Only UNPACK owns the intermediate flag. One uniform mailbox exchange supplies the final decision to MATH/PACK.
- Lines 2572–2604: explicit fixed geometry/noncausal/non-ring/no-sink/no-latent-V assertions; wait/pop the whole old numerator bank, swap CB aliases, and similarly reuse the old sum bank. Exact capacities of 32 numerator tiles and eight sum tiles ensure a full pop/reserve wraps to the same physical L1 addresses. A larger-capacity/differently laid-out CB would invalidate this argument.
- Lines 2684–2747: original split-drain PV arithmetic is retained, but L1 accumulation starts at the first partial when reusing old state. Critically, accumulation is reset to zero even after partial zero before the next softmax overwrite.
- Lines 2885–2889: denominator pack accumulation targets the reused old FP32 sum bank. Phase 1's FP32 path has no writes to `cur.sum`; its reserve does not advance/publish the unused alternate bank.
- Lines 2971–2985: preserve the correction CB's publication/wait/pop handshake despite omitting its state arithmetic. Keep pack-mode transitions; no fence removal is part of this candidate.
- Lines 3556–3564: suppress the old sum pop only when the whole bank was already consumed; the usual outer accumulator swap still propagates updated aliases. Final normalization drains current banks; first-Q initialization and next-Q reuse remain intact.

`SDPA_FP32_PIPELINE` is explicitly rejected near the header top because it bypasses the instrumented QK loop. `qkt_subblock_h == 1` is asserted. This private header is not a generic drop-in streaming replacement: geometry assertions alone do not establish mask/tail/ring support. Production integration must preserve the exact capacity/lifetime assumptions or provide an ordinary-path fallback.

Numerically this is not proven bit-equivalent for all inputs. FP32 L1 accumulation consumes old full-FP32 state without SrcA/B TF32 ingress, but row zero changes the addition association from `PV_partials + old_state` to `old_state + PV_partial_0 + ...`. The original Blackhole post-unpack/zero-valid workaround remains in place. Finite identity of the maximum means alpha is one; changed-max chunks take the ordinary rescaled-state path. Observed output identity on many cases is empirical, not a claim that floating-point reassociation is exact.

## Explored and rejected alternatives

- **Full row-zero PV reduction before one pack:** numerically acceptable short tests, but slower (D roughly 82.11→83.56 ms; C 61.20→61.63 ms at q8/k128). It loses overlap between split PV and exp drain.
- **Preload old FP32 numerator into DST then PV:** even batched four-tile unpack was slower (D 82.10→83.39 ms; C 61.21→63.16 ms). The added unpack/reconfiguration cost outweighed the removed state pack. This explored FPU accumulation ordering but is not retained.
- **Numerator-only in-place L1 reuse:** essentially neutral/slower; retaining the sum state pass left too much overhead. Both numerator and denominator must be reused to obtain the measured gain.
- **Late whole-Q guard:** a real resident gain, but inferior to the frozen early guard. At distinct 32K its normal-input timing regressed 0.68% D / 0.34% C; at 256K its gains were only 0.15% D / 0.64% C. These are historical late-guard results, not early-candidate timings.
- **Pack-mode leak negative control:** initial numerator-only K2 normal tests produced huge errors because L1 accumulate stayed enabled into the next exp overwrite. `l1-k2-smoke.json` preserves the failure. The corrected first-partial reset and normal-K2 regression coverage are mandatory. Constant V masked this failure.

The late candidate's measured guard frequency on distinct D Q2048/K262144/two-core data was 35.25% over eligible nonfirst chunks for normal inputs, zero for changing-max stress, and 100% for coherent repeated KV (`guard256-D-fixed/.logs/profile_log_device.csv`). These were actual device counters, not a host independence model. Guard instrumentation is disabled in decisive timings. This conditional frequency is why resident gains must not be advertised as generic model speedups.

## Frozen pins and reproduction

| Source | SHA256 |
|---|---|
| `early_guard.hpp` | `e689d9039fc56347e1e313c08bc093074e7fb0c130ce7980e7a553f3f0e43bed` |
| `resident_early.cpp` | `60a8601dd0a3e17ad5cb20e9bec9e24b038b10f59e0a436395f3205c94bde3aa` |
| `compute_early.cpp` | `a32a2708a51336b64162ef5ee74d484807ad8a903914c5607d04f8e663457ddc` |
| Late `compute_streaming.hpp` | `e1fc7330af846d606a2460adca6ade49443be16c38a01f2a6d15e7986cc84acf` |
| `integrity_core.py` | `4c50ab2623fa52dbd6e2b2f6a21e2b2d4ce5689cd243afc708378db78b0d4ca5` |
| `integrity_bench.py` | `1d41b84fbcfd8f9710acd405e766e4666f695dd6d5af54fe939fd219ffe39963` |
| `integrity_fullchip.py` | `92105fb5b4e5480544db73bfef6725840b9d5c5ac7a8d22c039624f98682947a` |

Retained candidate defines are `SDPA_V3_L1_INPLACE=1`, `SDPA_V3_L1_BOTH=1`; D additionally `SDPA_V3_O2=1`. Keep all baseline recipe/v1 winner flags. Do not enable inactive rejected prototype flags (`SDPA_V3_FULL_PV`, `SDPA_V3_IDENTITY_FUSION`) or profiling guard counters for performance. The frozen header includes their inactive scaffolding as experiment provenance; later production extraction should retain only the reviewed early-scan, bank-reuse, pack-mode, SALAD-handshake and cleanup changes, then requalify that extracted source.

Run under the shared serialized `compute-sprint-v1/run_locked.sh` on logical device zero; no nested locks or automatic reset. Example arguments (paths relative to repo root):

```sh
bash experiments/sdpa-l2/compute-sprint-v1/run_locked.sh \
  experiments/sdpa-l2/compute-sprint-v3/fp32/integrity_bench.py \
  --label UNIQUE_LABEL --k-chunks 512 --q-repeats 8 \
  --algorithms baseline,l1_both,l1_early --warmup 2 --iters 3 --rounds 9
```

The device kernels were JIT compiled and exercised on Blackhole; this report-only update needs no host rebuild. The existing prebuilt host stack is unchanged. Full provenance and raw per-record outputs are retained alongside JSON results.
