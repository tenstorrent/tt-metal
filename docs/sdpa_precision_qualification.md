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
Only segment addressing and shared host dispatch change; joint tails/ring are
not enabled yet.

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

## Continuous coverage

The [SDPA sanity group](../tests/pipeline_reorg/ttnn_sanity_tests.yaml) runs the
entire `tests/ttnn/unit_tests/operations/sdpa` directory on Blackhole P100/P150.
[Sanity CI](../.github/workflows/sanity-tests.yaml) runs on pushes to main and
scheduled Blackhole runs. Model-capture replay requires `SDPA_MODEL_CAPTURE_DIR`;
throughput requires `TEST_SDPA_RECIPE_PERF=1`. Those two suites are opt-in, not
automatic post-commit coverage.

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

Remaining rollout work: feature/platform eligibility, masked tails (including
Wan's valid-key tail), multi-device/model caller migration and fresh pretrained
model qualification. These are not supported by silently selecting another
recipe. The legacy non-streaming implementation remains until that coverage
is replaced; this draft does not claim it can be deleted yet.
