# Streaming SDPA recipe qualification

Initial qualification: 2026-09-21; single Blackhole P100. Main base
`dfaf6dc802f0a1321bbb2578ba7c4a0fb9b71ab8`, Clang 20, SFPI 7.80.0,
Python 3.10, Torch 2.11 CPU. Host libraries and bindings were rebuilt and
installed from this branch; device kernels were compiled from its sources.
No research library, include override or Python operation monkeypatch was used.

The numerical reference is frozen snapshot
`e13f445161ad598de9700edf915e5ed7faa6dc34`
([published tag `sdpa-recipes-20260921-v1`](https://github.com/tenstorrent/tt-metal/tree/sdpa-recipes-20260921-v1)).
The production test fixture pins the source evidence hashes, original and
prepared input digests, output digests and per-case metrics. Experimental sources
are not imported. The [frontier plot](images/sdpa_precision_frontier.png) is copied
unchanged from the frozen research evidence; its resident timings are historical,
not new measurements of this production branch.

## Numerical preservation

All **147/147** production outputs were bit-for-bit identical to the frozen
outputs: seven variants times 21 cases. Original/prepared input digests also
matched. The independent bounded-memory FP64 oracle agrees with dense FP64
attention to `rtol=atol=1e-12` on its cross-check.

The core suite has normal, clipped, scaled-down Q/K, scaled-up Q/K, outlier,
and uniform-attention inputs at KV4K/32K/256K, Q256, one head, D128.
Common-Q/K/V stress uses KV32K. The L2 regression gate is per-case
`L2_percent <= 1.05 * frozen_L2_percent + 0.0001`; it is not a blanket accuracy
promise. PCC, maximum absolute error and row-wise L2 tails are recorded too.

| Variant | Core-suite L2 range (%) | Normal, KV256K L2 (%) | Normal, KV256K PCC | Common-K stress L2 (%) |
| --- | ---: | ---: | ---: | ---: |
| A | 1.461–96.543 | 18.7367 | 0.99746822 | 16.8017 |
| B | 1.014–5.718 | 3.1972 | 0.99972119 | 16.7796 |
| C | 0.164–0.413 | 0.3778 | 0.99999286 | 1.0563 |
| D | 0.152–0.290 | 0.1782 | 0.99999841 | 0.7302 |
| E_bf16 | 1.660–7.845 | 3.5853 | 0.99958282 | 46.9110 |
| E_bfp8 | 1.634–7.975 | 3.6462 | 0.99956362 | 46.9229 |
| E_bfp4 | 11.190–41.134 | 16.4053 | 0.98682611 | 79.8916 |

Uniform attention at long context exposes A's severe BF16 recurrent-state
loss. Common-K stress remains challenging even for C/D. The port preserves
these limitations; passing a regression gate is not passing a universal 0.5%
accuracy threshold. The other stress errors remain in the fixture and reports.

## Execution and compatibility checks

The final combined release run passed **275 tests**, with two pre-existing
prefill skips. This includes the 147 frozen comparisons, independent reference
check, preparation/recipe contracts, legacy/component tests and the 14 real
activation replays below. The thirteen host GoogleTests also passed.

- A/B/C/D at one, two and three KV chunks, plus all seven variants at Q768,
  two heads, K1536, four cores: normal/uniform/changed-max/constant-V/zero-V.
  These exercise unequal forwarding-chain lengths, an odd final compensation
  group, CB reuse and per-Q state reset.
- Fresh live addresses on cache hits, unchanged cache-entry count, two actual
  trace replays, and original/prepared input immutability.
- E's four preparation contracts against independent rounding oracles:
  Q RNE7, KV RNE5 BF16, RNE5+BFP8, BFP4-grid RNE+saturation. Normal values,
  explicit ties/exponent changes and zeros; exact decoded values; fresh-address
  cache reuse and actual traces. Six unsupported preparation cases reject.
- Fifteen incompatible attention configurations reject before program dispatch.
- Named A equals legacy streaming bit-for-bit at five heads, Q768,
  K512/1536/32768, Q256/K512 blocks and a 5x2 grid.
- Thirteen policy/resolver GoogleTests; eleven legacy-config tests and eleven
  FP32-state component tests. The latter independently verify separate FP32
  rescale/L1-add rounding and reciprocal/normalization, including cancellation.
- Existing prefill coverage includes GQA, dense bias, sliding windows,
  attention sinks and packed inputs. Existing callers retain legacy dispatch.

The focused Watcher/lightweight-assert/LLK-assert suite passed **105 tests**.
An additional mixed-recipe cache test passed in release and Watcher builds:
all seven recipes on the same live tensors, reverse-order cache hits, and two
replays of a trace containing every recipe matched the frozen output digests.
The broader Watcher run encountered one legacy BFP8 noncausal sliding-window
program-size limit: **74,224 bytes versus 70,656 available**. Repeating that
exact case with the original main `compute_streaming.hpp`, before helper
extraction, produced the same sizes and failure. This is a main-header JIT
control using the rebuilt host library, not a separate full build of main.
The case passes in release mode. No new skip was added to conceal it.

Watcher-only size optimization makes all explicit recipes fit; release
optimization settings remain unchanged. Watcher timings are not performance
results.

## Real model activation replay

Two saved FLUX.2 captures, `dual.0` and `single.47`, from the first denoising
step of the prior block-sweep-source qualification were replayed through the
production API. SHA256s are pinned in `test_sdpa_recipe_model_capture.py`.
Each contains four sampled global heads and all 4,608 tokens, BF16 D128.
The default host grid uses the production per-head forwarding chains.

All **14 replays passed**, with finite outputs, immutable original inputs and
two bitwise-exact trace replays each. Error is against independent FP64 SDPA
on the original captured BF16 inputs, including E's preparation error.

| Variant | Dual 0 L2 (%) | Single 47 L2 (%) |
| --- | ---: | ---: |
| A | 1.83262 | 1.02270 |
| B | 1.80987 | 0.99319 |
| C | 0.20602 | 0.17269 |
| D | 0.17150 | 0.16750 |
| E_bf16 | 2.43236 | 1.08504 |
| E_bfp8 | 2.44016 | 1.09074 |
| E_bfp4 | 10.61233 | 4.67989 |

This is a new attention-level test of actual activations, **not** a new
pretrained transformer-block, image/video, CLIP or multi-device model test.
Raw captures remain external artifacts, not files distributed in the PR.

## Performance

Fixed Q256/K512/D128; one active core; original family CB depths. Resident
measurement reuses one physical Q block and one physical KV block for 16 Q
jobs and 512 KV chunks, with no steady-state input DRAM reads. It uses the
same production compute source, numerical defines and CB formats. Its repeated
inputs are a compute diagnostic, not a representative accuracy distribution.

Resident TFLOP/s/core is useful QK+PV work divided by blocking trace wall time
(three warmups, nine timed replays, median). It includes dispatch and startup;
it is **not** a hardware-counter FPU utilization metric. Distinct-input device
time is separately measured with Tracy, Q256/K32768, one head; nine steady
replays. The final column is unprofiled trace wall time including explicit
on-device preparation from already-device-resident BF16 inputs, not host
conversion, model execution or communication.

| Variant | Resident TFLOP/s/core | Distinct-input device time (ms) | With preparation, trace wall time (ms) |
| --- | ---: | ---: | ---: |
| A | 1.9952 | 2.1778 | 2.2024 |
| B | 1.7506 | 2.6864 | 2.7047 |
| C | 1.2343 | 4.1250 | 4.1401 |
| D | 0.8951 | 5.4632 | 5.4817 |
| E_bf16 | 2.1235 | 2.2835 | 2.4375 |
| E_bfp8 | 2.0917 | 2.2888 | 2.4328 |
| E_bfp4 | 2.0906 | 2.2905 | 2.4328 |

All 28 unprofiled benchmark cases passed (resident, distinct normal,
distinct changing-max, and with preparation), plus seven profiled cases.
Changing-max distinct-input trace times were 2.20/2.73/4.24/5.60 ms for
A/B/C/D and 2.26–2.32 ms for E. These timings are not portable CI thresholds.
This P100 is a different hardware cohort from the historical P150 research
runs: do not interpret cross-cohort absolute timing differences as a port
regression or improvement. No new full-chip or model speedup is claimed.

## Cleanup requalification (2026-09-22)

Compared pre-cleanup `1a7825207e9` with compute revision `96de2e927c5` on the
same reserved Blackhole (`yyzo-bh-26`), with the same build/toolchain and separate
JIT caches. The final source differs from the release-tested source only in
comments; the final revision was also compiled and run under Watcher.

- Both release runs: **276 passed, 2 existing skips**, including 14 real-activation
  replays. All **147 frozen output hashes matched exactly** in both runs; all
  recorded numerical metrics and hashes were unchanged. Exact equality was
  required for this cleanup comparison. Ongoing accuracy CI gates on the per-case
  L2 limit and records historical digest equality without requiring it; cache and
  trace checks still require exact equality to fresh outputs from the same build.
- The normalized fixture expands to exactly the original metadata, hashes and
  metrics. No test cases or input distributions were removed.
- Final Watcher/assert run: **106 passed**. Host-policy tests: **13 passed**.
  Reader/writer code, preparation, host dispatch, CB capacities and Q/K blocking
  were unchanged.
- Both matched performance runs passed all **28 cases**. Resident trace time
  changed by -0.67% to +0.20%; distinct, changing-max and preparation-inclusive
  modes changed by -1.05% to +0.30%. These small differences do not establish a
  speedup or a material regression.

| Variant | Resident TFLOP/s/core before | After |
| --- | ---: | ---: |
| A | 1.9951 | 1.9952 |
| B | 1.7507 | 1.7568 |
| C | 1.2343 | 1.2337 |
| D | 0.8951 | 0.8933 |
| E_bf16 | 2.1236 | 2.1379 |
| E_bfp8 | 2.0918 | 2.0896 |
| E_bfp4 | 2.0908 | 2.0910 |

Artifacts: external `sdpa-pr1-cleanup-validation-20260922/`, with
`sdpa-cleanup-{before,after,watcher,host}.xml`, matching logs, and
`sdpa-cleanup-{before,after}-perf.xml`. `compare.py` checks coverage, numerical
equality and timing deltas; `run.sh` and `watcher.sh` capture the run environment.

## Review follow-up (2026-09-22)

Rebuilt revision `c0295ea207d` on the same Blackhole after sharing production
compute/CB descriptors with the resident benchmark and unifying host exceptions.

- Release: **281 passed, 2 existing skips**, including all 14 FLUX.2 replays.
- Watcher/asserts: **111 passed**; host policy/resolver: **13 passed**.
- All 147 frozen output hashes and recorded numerical metrics remained exact.
  Historical output hashes are diagnostic properties; L2 is the accuracy gate.
  Cache/trace equality within a build remains mandatory.
- Performance: **28 passed**. Resident timing changed by less than 0.02%;
  other modes remained within 1% of the preceding cleanup run.
- Five added cases cover accepted default scales and rejection of BF16-rounded
  and nonfinite scales. Recipes, arithmetic and data movement are unchanged.

Artifacts: external `sdpa-pr1-cleanup-validation-20260922/sdpa-review-*`;
`review-compare.py` checks coverage, numerical equality and timing deltas.

## PR2 joint-adapter qualification (2026-09-22)

Revision `36d92f2e167`, same Blackhole/toolchain as PR1. The initial adapter
reuses the compute kernel, numerical policies, preparation kernels and CB depths.
Only segment addressing and shared host dispatch change in that revision. The
subsequent whole-tile tail qualification is recorded below; ring remains disabled.

- Joint release: **70 passed**, including 42 synthetic numerical cases and
  14 pinned FLUX.2 capture replays. All **56 numerical cases match dense output
  bit-for-bit** within this build; FP64 L2/PCC and row-error metrics are recorded.
- PR1 regression: **281 passed, 2 existing skips**. All 147 frozen hashes and
  recorded numerical metrics remain exact.
- Combined Watcher/asserts: **181 passed**. Seven opt-in performance cases skip
  in correctness runs; they are exercised separately.
- Performance: **35 passed** (28 existing modes plus 7 joint-versus-dense).
  Resident time changed by less than 0.1%; other existing modes stayed within 1%.
- Legacy joint smoke: **4 passed**. Host policy/resolver: **13 passed**.
- Fresh-address cache reuse, changed segment boundaries, both output buffers,
  trace replay, input immutability and unsupported-input rejection are covered.

Matched trace wall times below use Q=K=4608, four heads, D128, 4x4 grid,
Q256/K512, and a 4096+512 joint split. Preparation is outside timing.
These are not device-profiler times or full-model speedups.

| Variant | Dense ms | Joint ms | Joint overhead |
| --- | ---: | ---: | ---: |
| A | 1.5994 | 1.6295 | +1.88% |
| B | 1.9180 | 1.9190 | +0.05% |
| C | 3.0223 | 3.0236 | +0.04% |
| D | 3.9581 | 3.9603 | +0.06% |
| E_bf16 | 1.6560 | 1.6554 | -0.04% |
| E_bfp8 | 1.6519 | 1.6552 | +0.20% |
| E_bfp4 | 1.6280 | 1.6277 | -0.02% |

Artifacts: external `sdpa-pr2-validation-20260922/`, including logs/XML,
`run.sh`, `qualify.sh` and `compare.py`.

## PR2 joint chunk-tail qualification (2026-09-22)

Revision `77cb2f5f136`, same Blackhole/toolchain. Joint sequences may now end
inside a Q256/K512 chunk, while individual segments remain tile-aligned. Dense
eligibility, recipe arithmetic, preparation and CB depths are unchanged.

- Joint release: **125 passed**, seven opt-in performance skips. Added 49 tail
  numerical cases and seven tail cache/trace cases; removed the obsolete
  whole-chunk-length rejection. The sub-tile-padding rejection remains.
- All **56 aligned synthetic/captured comparisons remain bitwise equal** to dense.
- PR1 regression: **281 passed**, two existing skips. All **147 frozen hashes**
  and previously recorded numerical metrics remain exact.
- Watcher/asserts: **236 passed**, seven opt-in performance skips.
- Performance: **35 passed**. Resident timing changed by less than 0.1%; other
  existing and aligned-joint measurements stayed within 1.3% of the prior run.
  These trace-wall comparisons do not measure the new tail-mask overhead.
- Legacy joint: **4 passed**. Host policy/resolver: **13 passed**.

Tail cases cover short sequences, unequal Q/K lengths, multiple heads, unequal
chain job counts, normal/uniform/changed-max inputs, constant V and zero V.
They check independent FP64 error, input immutability, both output shapes,
fresh-address cache hits, segment repartitioning and trace replay.

| Variant | Maximum tail-case L2 % | Minimum nonconstant-case PCC |
| --- | ---: | ---: |
| A | 4.4375 | 0.999010 |
| B | 4.4366 | 0.999011 |
| C | 0.3946 | 0.999992 |
| D | 0.1798 | 0.999998 |
| E_bf16 | 3.6670 | 0.999335 |
| E_bfp8 | 3.7373 | 0.999306 |
| E_bfp4 | 16.7605 | 0.985904 |

Two bring-up findings are retained in the tests/evidence:

- Masking must execute on **PACK after its DST wait**, not MATH: streaming
  exponential/compensation already owns the SFPU on PACK. A math-thread fill
  raced that work and caused nondeterministic E results. The shared mask hook
  now follows existing ownership and waits for SFPU completion before packing.
- Duplicating K/V preserves exact attention but not finite-precision reduction
  error. On Q512/K256 uniform attention, A/B yield **1.494%**, exactly matching
  the existing masked implementation; duplicated K512 gives **0.819%**. The
  duplicate comparison is diagnostic, not an equal-error gate. A/B additionally
  gate against the legacy tail result. Constant-V tail cases gate below 1%
  (0.4% for C/D), preventing a missing key mask from hiding behind PCC.

Artifacts: external `sdpa-pr2-validation-20260922/sdpa-pr2-tail-*`, including
release, regression, Watcher, performance, legacy-joint and host logs/XML.
`qualify-tails.sh` reproduces the runs; `compare-tails.py` checks coverage,
numerical preservation and timing deltas. Sub-tile tails and ring integration
remain open PR2 gates; this is not a complete PR2 merge qualification.

## PR2 sub-tile tails, batch/GQA and SPMD meshes (2026-09-23)

Qualified source: `2a046f66bfe` (sub-tile tails), followed by `d3025e93e82`
(batch/GQA and uniform meshes). Tests ran on bh-32, IRD reservation 129246,
two P150b Blackholes, firmware 19.12.0 and SFPI 7.80.0[956]. The build used
the source overlay corresponding to these commits; the remote clone's Git HEAD
alone does not identify that overlay.

- Sub-tile release suite: **142 passed**. Dense/joint segment lengths include
  1, 15, 17, 31, 33, 255, 257, 511, 513 and 767 rows, unequal Q/K lengths,
  NaN-poisoned physical padding, constant/zero V and changed maxima.
- Batch/GQA and two-device mesh suite: **52 passed in release and 52 in
  Watcher**, no skips. GQA equals repeated-KV attention bit-for-bit. Mesh tests
  cover replicated, head-sharded and query-sharded inputs, plus trace replay.
- Recipe, frozen accuracy, legacy numerics/prefill, component, FLUX.2 capture
  and joint regression: **386 passed**, nine existing/opt-in skips. All **147
  frozen outputs remain bit-identical**.
- Tail/preparation/joint/recipe Watcher regression: **336 passed**, seven
  opt-in performance skips. Host policy/addressing tests: **14 passed**.
- Matched performance: **35 passed**. Compared with the committed pre-tail
  baseline on this same P150b, resident trace-wall medians change by at most
  **0.03%**. No measured aligned case regressed by more than 0.04%; apparent
  improvements of up to 2% in short tests are not claimed as kernel speedups.
  These comparisons do not quantify tail-mask overhead.

| Variant | Maximum sub-tile-case L2 % | Resident TFLOP/s/core |
| --- | ---: | ---: |
| A | 2.9147 | 1.995 |
| B | 2.9137 | 1.757 |
| C | 0.3906 | 1.234 |
| D | 0.1778 | 0.893 |
| E_bf16 | 2.8932 | 2.137 |
| E_bfp8 | 2.9368 | 2.089 |
| E_bfp4 | 16.1242 | 2.090 |

Resident throughput uses the existing Q256/K512 repeat-input test and trace-wall
timing, not device-profiler duration. The error column is the maximum over the
new sub-tile suite, not a replacement for the frozen stress qualification.

Artifacts: external `sdpa-pr2-validation-20260923/`, including
`tails-release`, `geometry-{release,watcher,perf,host}`, `recipe-regression`
and `tails-watcher` logs/XML. `qualify-geometry.sh` records the command sequence.
SPMD execution does not exchange KV between devices: **ring continuation and
communication are not qualified by these results**. No model defaults change.

## PR2 continuation, ring and pretrained Wan block (2026-09-23)

Source: `c0ffe6098d0`, on the same two P150b devices described above.
Continuation and ring integration change state lifetime, not arithmetic,
preparation, blocking or input buffer depths.

- Continuation: **162 release + 162 Watcher passed**, covering retained/reloaded
  Q, odd/even splits and raw DRAM checkpoints. Another complete Q block
  overwrites the live banks between save and restore.
- Two-device ring: **168 release + 168 Watcher passed**, covering single/multi-Q,
  uneven chains, GQA, batch 2, replicated/sharded joint KV, skipped shards and
  a 777-key logical tail. Inputs are normal, uniform and increasing-max.
- B/C/D/E match dense attention **bit-for-bit in the same KV order**, on both
  chips (288 comparisons). This compares current implementations, not frozen
  digests. Ring KV order can differ by rank.
- A retains the legacy ring wrapper. All variants also gate independent FP64
  L2 against the same-order dense result (at most 1.05×), and record PCC.
  C/D retain absolute bounds. The frozen stress qualification is unchanged.
- Trace replay and prepared-source immutability are checked. Unsupported
  precision/feature combinations reject before launching compute.

| Variant | Worst ring L2 % | Minimum PCC | Ring ms, 64 cores/chip | 32 cores/chip | 8 cores/chip |
| --- | ---: | ---: | ---: | ---: | ---: |
| A | 10.981 | 0.994095 | 0.857 | 1.134 | 4.420 |
| B | 10.981 | 0.994095 | 1.038 | 1.989 | 7.624 |
| C | 0.750 | 0.999972 | 1.140 | 2.227 | 8.580 |
| D | 0.186 | 0.999998 | 1.469 | 2.886 | 11.222 |
| E_bf16 | 8.500 | 0.996433 | 1.027 | 1.905 | 7.288 |
| E_bfp8 | 8.520 | 0.996416 | 0.995 | 2.007 | 7.700 |
| E_bfp4 | 21.063 | 0.978719 | 0.987 | 2.005 | 7.679 |

Error extrema include increasing-max stress inputs, where HiFi2/LoFi error
remains substantial. Ring does not fix that recipe limitation. Timing uses
Q/K=4096 **per device**, B1/H4/D128, Q256/K512, nine trace-wall samples after
warmup, excluding preparation. The 64-core case has one resident Q block/worker;
the 32/8-core cases assign two/eight and stage state. These columns change compute
parallelism and **do not isolate checkpoint overhead**. B/E ring code uses
`-Os` to fit the kernel configuration buffer; C/D use `-O2`; Watcher uses `-Os`.
Dense codegen is unchanged. After continuation extraction, dense/resident
comparisons showed at most 0.26% median regression (0.21% resident).

### Fresh pretrained attention module

`test_sdpa_recipe_wan.py` loads Wan2.2-T2V-A14B's first self-attention module
from pinned HF revision `5be7df9619b54f4e2667b2755bc6a756675b5cd7`, downloading
only the checkpoint shard containing it. Converted TT weights are cached once
and reused. All **16 cases passed**, including trace replay.

This covers QKV projections, learned Q/K norm, SDPA and output projection,
**not a full transformer block or generated-video evaluation**. Inputs are
seeded BF16 synthetic hidden states; this smoke test does not apply RoPE. The
reference runs the pretrained module in CPU FP32. Existing PCC ≥0.988 is
supplemented with an L2 gate: 10% for A–D/legacy, 20% for E. These module gates
are distinct from pure-SDPA qualification.

| Variant | L2 %, 1024 / 2048 tokens | Minimum PCC | Module ms, 1024 / 2048 tokens |
| --- | ---: | ---: | ---: |
| Legacy Wan | 2.598 / 2.212 | 0.999715 | 1.869 / 2.988 |
| A | 2.688 / 2.271 | 0.999704 | 1.897 / 3.004 |
| B | 2.656 / 2.243 | 0.999702 | 1.915 / 3.254 |
| C | 1.053 / 0.982 | 0.999971 | 1.933 / 3.205 |
| D | 1.000 / 0.940 | 0.999973 | 1.955 / 3.288 |
| E_bf16 | 2.760 / 2.351 | 0.999663 | 2.013 / 3.413 |
| E_bfp8 | 2.771 / 2.343 | 0.999654 | 1.764 / 2.934 |
| E_bfp4 | 11.007 / 9.748 | 0.993924 | 1.734 / 2.795 |

Module timings include E preparation, projections, normalization and
communication, excluding compilation/weight conversion. They are trace-wall
medians, not pure SDPA timings. No model defaults change. Full-model image/video
quality and longer contexts remain gates for default migration.

Artifacts: `sdpa-pr2-validation-20260923/continuation-final-{release,watcher}`,
`continuation-{frozen,perf}`, `ring-wide-{release,watcher}`, `ring-perf`,
`wan-pretrained-v2` and final regression files. Watcher completed all ring
tests but reported an Ethernet teardown timeout; a board reset recovered
both devices before release timing and model testing.

### Final regression sweep

The combined sweep passed **910 functional/numerical cases**, with 51 expected
skips (49 opt-in timings and two existing OOM exclusions). All **147 frozen
outputs remain bit-identical**. It then exposed a new negative-test setup error:
the test referenced a Python config class that is not exported. Switching the
test to `init_device_compute_kernel_config` fixed it; **all 10 rejection cases
passed** on rerun, without a production-code change.

Four existing legacy ring regressions passed: BF16/FP32 causal GQA, causal
chunked prefill and attention-sink accuracy/determinism/cache reuse. The final
ring benchmark passed all **21 cases**, and the fresh Wan module repeated
**16/16** successfully. Files: `final-release`, `final-rejection`,
`legacy-ring-regression`, `final-ring-perf` and `final-wan` XML/logs.

Final dense/resident throughput: **28/28 passed**. Relative to the matched
pre-continuation geometry build, worst median increase is **0.34%** including
preparation, **0.26%** excluding it, and **0.21%** for resident compute. These
small trace-wall differences are not claimed as a speedup. Host policy/addressing:
**14/14 passed** (`final-perf`, `final-host`). The repository-wide time-budget
validator passes all 271 CI budget buckets after the job split.

## DiT adoption geometry and callers (2026-09-24)

Branch `cglagovich/sdpa-dit-adoption`, stacked on PR2, on bh-32 (two P150b).
Scope: remove research-phase geometry constraints for DiT callers without
changing the frozen Q256/K512/D128 outputs.

| Check | Result |
| --- | --- |
| Frozen outputs, recipe suite, accuracy baselines | 218 passed; all 147 frozen outputs bit-identical |
| Q chunk 128-320 (dense/joint, all recipes incl. odd for B/E) | 119 passed + 83 odd-chunk passed; L1 skips only |
| K chunk 256/384 (dense/joint) | 176 + 3 B@K256 cases re-gated (see below) |
| D64 / D256 (dense/joint, all recipes) | 196 passed, 8 L1 skips |
| Ring: Q/K blocking, D64, D256 | 564 passed (incl. D64); D256 70 passed |
| Exp ring single- and multi-pass (1x2) | 124 passed |
| Continuation | 162 passed |
| Watcher subset: odd Q (B/E), D64/D256, K256/K384 | 12 + 31 + 35 passed; L1 skips only |
| After barrier change: regression / exp ring / geometry | 565 / 131 / 418 + 105 passed |
| Final consolidated run @63de9dff: dense+joint / ring+continuation+mesh / exp ring / model smoke / model host | 960 (+1 stale prep expectation, fixed) / 817 / 131 / 68 / 103 passed |
| Joint, tails, GQA, preparation, FP32 state | 595 passed (combined regression) |
| Model device smoke (random weights; FLUX.1/2, Qwen path, Mochi, Wan, LTX-2 video/audio, MiniMax H3, SD3.5, Motif, Ideogram4) | all passed; 1x1 and 1x2 |

Numerical notes:

- Non-Q256 blocking is **not bit-identical** to Q256 (about one BF16 ulp on the
  first PV row group of each Q chunk); gated on determinism and FP64 accuracy.
- COMPENSATED at K256 keeps less of its long-context advantage (8192-key uniform:
  1.48% vs 1.04% at K512, 0.93% at K384), still below FAST (2.0%).
- D64/D256 are gated against the D128 recipe on the same generator. The
  concatenated D256 changed-max stress is harsher than its D128 counterpart and is
  recorded rather than gated relative to D128.
- Odd-chunk COMPENSATED/LOW_PRECISION dense kernels compile with -Os to fit the
  kernel config buffer. On exp ring those recipes hang at Q224 and are rejected on
  the host until fixed; FAST/BALANCED/ACCURATE exp ring Q224 passes.

Matched-chunk trace-wall timing (single P150b, 10 heads, 8192 x 8192, D128, full
grid, preparation excluded; legacy = HiFi2, BF16 destination, exact exponential,
as DiT callers configure it). Milliseconds:

| Blocking | Legacy | A | B | C | D | E_bf16 | E_bfp8 | E_bfp4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Q256/K512 | 1.889 | 1.785 | 2.171 | 3.352 | 4.349 | 1.857 | 1.849 | 1.835 |
| Q128/K512 | 3.225 | 2.443 | 2.466 | 3.449 | 4.384 | 2.387 | 1.872 | 1.834 |
| Q256/K256 | 1.911 | 1.937 | 2.710 | 3.966 | 5.062 | 2.432 | 2.413 | 2.444 |
| Q224/K384 | 3.040 | 2.351 | 4.064 | 4.276 | 5.528 | 3.919 | 4.228 | 4.225 |
| Q320/K256 | 2.384 | 2.419 | 3.338 | 4.892 | 6.322 | 2.998 | 3.019 | 3.008 |

These are single-op trace timings, not model throughput, measured after raising the
chain-head reader's DRAM read-barrier interval from 2 to 16 tiles (before it,
FAST was 12% slower than legacy at Q256 and 30% slower at Q128; 1- and 4-core
timings are unchanged). FAST and LOW_PRECISION now match or beat legacy at matched
blocking; BALANCED and ACCURATE cost 1.8-2.6x legacy, which is their HiFi4/FP32
arithmetic, not dataflow.

### Ring and exp-ring timing (1x2, 2026-09-24)

FAST matches legacy on ring and exp ring within about 1-3%; no dataflow
bottleneck remained after the reader barrier change. BF16 recipe compute kernels
had been built at -Os on every TRISC to fit the kernel config buffer. They now
size-optimize only the pack thread (unpack/math at O2), which fits every ring,
exp-ring and odd-chunk case and recovers most of the O2 speed. Trace-wall
milliseconds (10 heads, 4096 rows/device, D128; exp ring 20 Q chunks/head):

| Case | Legacy | A | B | E_bf16 | E_bfp8 | E_bfp4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ring Q256/K512 (before -> after) | 1.53 | 1.55 | 2.11 -> 1.87 | 2.01 -> 1.81 | 2.08 -> 1.53 | 2.08 -> 1.52 |
| ring Q256/K256 | 1.56 | 1.56 | 2.67 -> 2.12 | 2.59 -> 1.96 | 2.77 -> 1.93 | 2.76 -> 1.95 |
| exp ring Q256 two-pass | 1.48 | 1.48 | 2.42 -> 1.83 | 2.32 -> 1.55 | 2.45 -> 1.55 | 2.44 -> 1.54 |
| exp ring Q224 two-pass | 1.15 | 1.15 | 2.09 -> 1.64 | 2.03 -> 1.44 | 2.11 -> 1.44 | 2.10 -> 1.43 |
| dense Q224/K512 | 1.14 | 1.30 | 2.08 -> 1.71 | 2.05 -> 1.53 | 2.11 -> 1.50 | 2.14 -> 1.47 |

C/D keep O2 and cost 1.6-2.4x legacy (HiFi4/FP32 arithmetic). At the H3
worker L1 budget, C/D ring at Q256/K512 and Q320/K384 exceed L1 by about 1 KB.

## Continuous coverage

The [SDPA sanity matrix](../tests/pipeline_reorg/ttnn_sanity_tests.yaml) separates
legacy operations, recipe numerics and recipe adapters. The two recipe jobs
run on Blackhole P100/P150 with 15/20-minute budgets, instead of adding this
coverage to the legacy job's 15-minute limit. The combined timeouts remain
within existing TTNN sanity budgets.
[Sanity CI](../.github/workflows/sanity-tests.yaml) runs on pushes to main and
scheduled Blackhole runs. Model-capture replay requires `SDPA_MODEL_CAPTURE_DIR`;
throughput requires `TEST_SDPA_RECIPE_PERF=1`. Those two suites are opt-in, not
automatic post-commit coverage.

The Blackhole e2e matrix additionally selects SPMD mesh and ring suites on P300 (two
chips); single-device sanity cannot qualify ring communication. This CI entry
is wired but has not been run by hosted CI as part of the local qualification.
Fresh pretrained Wan testing is opt-in with `TEST_SDPA_RECIPE_WAN=1`.

## Reproduction and evidence

Build and install this branch with TTNN tests enabled, configure `TT_METAL_HOME`,
`PYTHONPATH` and `LD_LIBRARY_PATH` for that build, and use a separate writable
`TT_METAL_CACHE`. Run under the repository's cooperative device lock:

```bash
scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipes.py \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_input_preparation.py \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipe_accuracy.py \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_numerics_compatibility.py \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_fp32_state.py \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_prefill.py

# Add --dev for Watcher/assert checks; see the legacy size limitation above.
TEST_SDPA_RECIPE_PERF=1 scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipe_performance.py
TEST_SDPA_RECIPE_PERF=1 scripts/run_safe_pytest.sh --profile \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipe_performance.py -k distinct

# Optional external real-activation captures, validated against pinned hashes:
SDPA_MODEL_CAPTURE_DIR=/path/to/block-sweep-source scripts/run_safe_pytest.sh \
  tests/ttnn/unit_tests/operations/sdpa/test_sdpa_recipe_model_capture.py
```

XML properties retain metrics, digests and timings. Raw logs/XML are retained
in the external `sdpa-pr1-validation-20260921` artifact directory:
`sdpa-pr1-final-release`, `sdpa-pr1-qualified-watcher`, `sdpa-pr1-final-host`,
`sdpa-pr1-model-captures`, `sdpa-pr1-performance`, `sdpa-pr1-device-profile`,
and `sdpa-pr1-main-header-control`.
The additional cache check uses `sdpa-pr1-cache-policy-{release,watcher}`.
Device timing comes from
`ops_perf_results_2026_09_21_22_31_02.csv`, trace IDs 0–6 mapping to the table
order, replay sessions 4–12.

Remaining rollout work: other dimensions/platforms, causal/masked/sink/windowed
attention, paged/indexed/chunked caches, MLA, larger ring topologies and full-model
default migration. Unsupported explicit recipes never silently fall back.
The legacy non-streaming implementation remains until its coverage is replaced.
