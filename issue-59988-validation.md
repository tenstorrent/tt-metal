# Issue #59988 — LoudBox validation record

Completed on 2026-10-09, based on `48468ff598a`. This is local KDA-layer validation, not Galaxy/SC4 or full-checkpoint qualification. See [the plan](issue-59988-plan.md) and [independent plan reviews](issue-59988-reviews.md).

Subsequent [implementation code reviews by Codex Astra and Claude Code / Opus 5.5](issue-59988-code-reviews.md) identified a multi-rank migration failure-path defect and performance-test/CI scope concerns. Draft-PR cleanup adds coordinated completion status and early eligibility checks, isolates the LB policy benchmark, and restores the shared CI reference. A focused Codex Astra follow-up found no new correctness issues in this cleanup; real multi-host qualification remains deferred.

Hardware: eight Blackhole P150b devices. Test commands use `scripts/run_safe_pytest.sh`; local logs are under `/tmp/kda-59988/`.

| Evidence | Current result |
| --- | --- |
| Baseline `./build_metal.sh --release` | Passed; `build-baseline.log`. |
| Modified `./build_metal.sh --release` | Passed before device validation (`build-policy.log`) and again after final API documentation (`build-final.log`). |
| New native reader policy tests, K/V=32 and 128 | Seven passed after reset removal: both policies, zero/positive/wrapped positions, cache reuse, positive-offset capture, two requests, NaN seed poison, and rejection of policy-enabled affine scan on computed SP entries; `post-reset-contracts.log`. |
| Small KDA/K3 adapter fixtures | SP1xTP8, SP2xTP4, SP4xTP2 eager/trace request restart and padding passed; `policy-adapter.log`. |
| Production state geometry | Three passed (96 heads, K=V=128, reduced hidden width): SP1xTP8, SP2xTP4, SP4xTP2. Bit-exact generic explicit-zero control, eager/trace equality, CPU references, stable carries, real slab export/import; `policy-adapter-production.log`. |
| Checks after reset removal | 65 passed: seven native policy/rejection cases, host construction/runtime/ack contracts, four allocation cases, and eager slot isolation; `post-reset-contracts.log`. All six adapter matrix cases also passed after removal, including poisoning live carries and importing completed slabs before continuation; `post-reset-adapter.log`. |
| Short first requests | Six passed, seven other cases deselected: 32- and 96-token first chunks followed by continuation, on all three meshes at K/V=32 and 128; `fresh-short-matrix.log`. Inactive ranks/groups are included. |
| Bonus synthetic transformer | One passed on LB SP2xTP4: one-shot/chunked CPU references, second request from dirty state, stable carry addresses and slab equality. Real embedding/norms/KDA/dense FFN with explicit plain residual arm; `bonus-transformer.log`. |
| CI selection for the bonus | Added to `bh-lb-disaggregated-prefill-accuracy` in `tests/pipeline_reorg/blackhole_e2e_tests.yaml`, using the safe runner. BH e2e schedules this LB job every eight hours. YAML command lint, generated LB matrix and shell syntax passed; safe-runner collection selects its one test (`lb-transformer-ci-collection.log`). The prior device execution took 42.33 seconds. Remote CI execution is pending. |
| Native compatibility | 144 passed, nine native performance cases deselected. Covers the three changed operations, downstream recurrent scan, padding, defaults, invalid inputs and program caching; `native-compatibility.log`. |
| Recurrence and convolution compatibility | Local recurrence: 10 passed, six distributed cases deselected (`recurrence-local.log`). Distributed recurrence: six passed, ten local cases deselected (`recurrence-distributed.log`). Convolution: eight passed (`convolution-compatibility.log`). |
| Generic state and slabs | Stateful default-off layer: two passed (`stateful-compatibility.log`). Local TP4 slab round trips: two passed, two larger cases deselected (`slab-compatibility.log`). |
| Existing dynamic trace, chronology and padding | 24 passed, including production SP1 trace geometry in both mesh orientations and the exhaustive aligned-start sweep; `layer-compatibility.log`. |
| Distributed chain on LB | Six passed, six Galaxy cases deselected, using the supported 2D fabric profile; `chain-distributed-fabric2d.log`. The original six skipped cases do not count. |
| Final host contracts | 32 passed: runtime/construction contracts, positional config compatibility, and two-sided performance-band checks; `final-host-contracts.log`. |
| Original local LB latency experiment | One passed with the temporary local reference; `perf-policy-bracketed.log`. Shared CI reference subsequently restored; see the separate LB-only rerun below. |
| Program profile | Normal correctness and profiled runs each passed one test; `profile-correctness.log` and `request-profile.log`. The underlying profiled pytest result was checked independently of the wrapper status. |

## Allocation baseline

Measured persistent allocation per device at TP4, including DRAM bank padding, after warming program-owned allocation buffers:

| Layers | Slots | Cache before removal (bytes/device) | Cache after removal (bytes/device) | Retained slabs (bytes/device) |
| --- | --- | --- | --- | --- |
| 1 | 1 | 3,440,640 | 1,720,320 | 1,628,160 |
| 3 | 1 | 10,321,920 | 5,160,960 | 4,884,480 |
| 1 | 2 | 5,160,960 | 3,440,640 | 3,256,320 |
| 3 | 2 | 15,482,880 | 10,321,920 | 9,768,960 |

One native recurrent/convolution pair allocates 1,720,320 bytes/device. The measured saving is exactly that amount (**1.640625 MiB per layer per device**), independent of slot count. Slab allocations are unchanged. These are measured LB per-device values; there is no measured SC4 result.

## Performance baseline and test development findings

- Before native/Python policy changes, synthetic SP2xTP4, 5,120 tokens measured a median 8.165480 ms over five samples of ten trace replays. The existing two-sided 8.758 ms ±3% gate failed because this machine was faster than its lower bound. No performance pass is claimed from that run. See `perf-baseline.log`.
- The first baseline command omitted the architecture parametrization prefix and selected no tests. The correct node is `test_synthetic_kimi_k3_perf[blackhole-SP2xTP4-fabric-1d]`.
- A tiny one-head slab normalized to unsupported WIDTH_SHARDED storage. Slab-bound tests now use production head counts and K/V dimensions; the small 32-wide fixture remains separate.
- The production-width state has larger peak error against the CPU oracle than the small fixture's calibrated 0.6 relative-L∞ bound. Production cases retain PCC≥0.999, relative RMSE≤0.05, output/convolution peak gates, and bit-exact comparison against the generic grouped path supplied with an explicit zero seed. Existing small-fixture thresholds are unchanged.
- Allocation measurements warm the matching slab geometry before taking deltas, separating a 16-KiB program-owned DRAMZeroFill buffer from persistent model storage.
- The unchanged generic control measured 8.174058 ms and 8.166991 ms in subsequent runs, agreeing with the before-change 8.165480 ms result. The initial local run used an 8.17 ms reference. Review cleanup restores the original shared 8.758 ms reference; local ratio comparisons now run in a separate LB-only test. No CI rebaseline is included.
- Sequential policy measurements showed increasing latency across successive sample groups. The comparison now brackets each enabled sample with controls at the same absolute position and uses their average, rather than comparing an early control with a later policy run. The final bracketed run passed; raw samples are in `perf-policy-bracketed.log`.
- Combining single-device and mesh model suites in one pytest process failed at fixture setup (`SetFabricConfig` while a device remained open), after eight local recurrence passes. Those groups now run in separate safe-runner processes; no test assertion was weakened to bypass the fixture error.
- The distributed chain suite collected Galaxy TorusXY cases alongside LB cases, activating the topology guard that skips LB 2x4 with FABRIC_1D. Its LB parametrization now uses the existing FABRIC_2D profile; Galaxy remains unchanged. The first six skips in `chain-distributed.log` are diagnostic only.

## Final local latency measurement

Synthetic production geometry, SP2xTP4, 5,120 tokens, five samples of ten replays:

| Measurement | Result |
| --- | --- |
| Generic standalone control | 8.166937 ms median; historical local measurement. It does not meet the original two-sided 8.758 ms CI band (faster than the lower bound). |
| Enabled policy, absolute start zero | Median enabled/matching-control ratio 1.000156 (+0.016%); passes ≤3% overhead gate. |
| Enabled policy, positive start with nonzero carry | Median enabled/matching-control ratio 1.000011 (+0.001%); passes ≤3% overhead gate. |
| Request head including carry commit and slab export | 10.055817 ms median enabled versus 10.228228 ms control including the historical reset; ratio 0.983144 (1.69% faster). |

The longer run's controls rose from roughly 8.2 to 10.1 ms, so absolute times from different sample groups are not directly comparable. Each policy sample is bracketed by controls. These are local layer measurements, not full-model throughput.

## Device program profile and seed-read evidence

The synthetic TP4 profile uses production state geometry (96 heads, K/V=128), hidden width 256, and 256 tokens. It labels a test-only reconstruction of the historical reset separately from the generic forward and the enabled-policy forward. Outputs and both carries are bit-identical between the old reset plus forward and the new dirty-state request start.

Local artifact (not committed): `generated/profiler/reports/2026_10_09_18_47_54/ops_perf_results_2026_10_09_18_47_54.csv`. Parsed counts are also saved locally in `/tmp/kda-59988/profile-summary.json`.

| Marked region | Timed device programs per chip, identical on all eight chips |
| --- | --- |
| Historical reset | 7: two carry copies plus five slab-export programs (one cache fill, two reshapes, one permutation, one slice write). |
| Historical forward and commit/export | 49. |
| Enabled initialization, forward and commit/export | 49; the separate seven-program reset is absent. |

The CSV includes program hashes and nonempty device kernel timings for all counted rows, including the reshape operations. It records the policy enabled on the distributed chain and convolution, while the downstream affine scan retains its default-off policy. Ordinary commit/export is preserved.

This operation-level profile does **not** measure individual DRAM transactions inside a kernel. Absence of external seed reads is established by the native reader branches, their local-zero barriers, and the dirty/NaN seed tests: the chain and SP1 affine readers select a zero fill instead of issuing the initial-state read at absolute zero; convolution skips only the external request-history read and preserves predecessor history. No hardware bus-counter measurement is claimed.

## Reproduction commands

All commands run from the project root. The two recurrence selections intentionally use separate processes to avoid overlapping single-device and mesh fixtures. The interval matrix ran before the `fresh_short` parameter was added; its original six cases are now selected with `-k intervals`.

```bash
./build_metal.sh --release

scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_request_start_policy.py models/demos/deepseek_v3_d_p/tests/kimi_k3/test_runtime_contract.py models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_migration_stages.py models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_cache_allocation.py models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_padding.py::test_k3_eager_request_restart_preserves_other_slot -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_padding.py -k intervals -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_kda_padding.py -k fresh_short -q

scripts/run_safe_pytest.sh tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_chain_affine_transforms.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_affine_exclusive_scan.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_qkv_causal_conv1d_silu.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_prefix.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_recurrent_chunk_scan.py tests/ttnn/nightly/unit_tests/operations/experimental/kda/test_padding_recurrence.py -k 'not performance' -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_chain_affine_transforms.py -k SP2xTP4 -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_recurrence.py -k 'not distributed' -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_recurrence.py -k distributed -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/components/test_convolution.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_stateful.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/layer/test_dynamic_trace.py models/demos/deepseek_v3_d_p/tests/kda/layer/test_actual_start.py models/demos/deepseek_v3_d_p/tests/kda/layer/test_padding_early_exit.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/test_state_adapter_device.py -k 2x4 -q

scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_transformer_kda_loudbox.py -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kimi_k3/test_runtime_contract.py models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_synthetic_performance_uses_two_sided_margin -q
KDA_PERF_SKU=bh_loudbox scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_layer_perf.py::test_request_initialization_perf_loudbox -q
scripts/run_safe_pytest.sh models/demos/deepseek_v3_d_p/tests/kda/perf/test_request_initialization_profile.py -q
scripts/run_safe_pytest.sh --profile models/demos/deepseek_v3_d_p/tests/kda/perf/test_request_initialization_profile.py -q
```

## Completion status

G1–G6 and the bonus pass locally. The final release rebuild passed, and all required LB test selections have passing evidence with no remaining skipped required case. Python syntax checks passed for all 22 changed/new Python files, clang-format passed for all 24 changed C++ files, and `git diff --check` passed.

The result is limited to KDA-layer implementation validated on LB plus the synthetic transformer bonus. Full checkpoint/AttnRes correctness, Galaxy/SC4 behavior, cross-host migration ordering, and production throughput remain deferred. The traced multi-slot restriction from #59977 remains enforced, and migration prefixes ending in KDA now fail until a completion protocol covers their final exports.


## Draft-PR cleanup validation

- The host contract selection passed 66 cases, including the actual migration entry-point failure paths and non-K3/MTP acknowledgement counts (`review-fix-host.log`). The final rerun after adopting the repository's `expect_error` fixture is recorded in `review-fix-host-final.log`.
- The separate `test_request_initialization_perf_loudbox` passed in 15.02 seconds (`review-fix-lb-perf.log`). Median enabled/control ratios: 1.000531 at absolute zero, 0.999799 at positive start, and 0.984927 including the historical reset versus device initialization (about 1.51% faster locally). These differences do not establish production throughput or statistical significance.
- The shared synthetic LB gate retains its original 8.758 ms reference and Galaxy behavior. The original two-sided gate can still fail on this faster local machine; it has not been rebaselined to force a pass.
- Pre-commit checks cover every changed/new file. The first pass removed trailing whitespace in the review document; the final result is recorded in `pre-pr-precommit-final.log`.
- The native C++ and device algorithms are unchanged from the completed release builds and LB compatibility matrix above.

## Recommended CI validation for this branch

Use the PR branch as the workflow ref, Release builds, Ubuntu 22.04 and LTO disabled. No manual workflow has been dispatched as part of PR preparation.

| Priority | Workflow and selection | Setup / coverage |
| --- | --- | --- |
| Required PR checks | `all-static-checks.yaml` (automatic on PR) | Formatting, source policy, licenses and static checks. No accelerator needed. |
| First hardware run | `tt-metal-l2-nightly.yaml`: Blackhole only, `additional_test_categories=experimental`, optional suites off | Includes Blackhole P100 and P150 CIv2 experimental rows. Exercises default-off compatibility and new single-device KDA policy tests. The 2x4 mesh rejection case skips on one device, so this alone does not cover distributed KDA. |
| First model run | `blackhole-e2e-tests.yaml`: `system-type=LoudBox (8xP150)`, `model=disaggregated_prefill`, `test-selection=all` | Eight-device LB, including the accuracy job with the added synthetic KDA transformer, and the existing perf job. BH e2e builds with Tracy. The synthetic transformer itself needs no checkpoints; the surrounding jobs require their existing model/cache mounts. |
| Focused perf run if separating jobs | Same BH e2e workflow, `test-selection=bh-lb-disaggregated-prefill-perf` | Existing LB absolute latency gate, with its original baseline. The new same-run request-initialization benchmark is currently local/manual only; adding it to this row is a follow-up CI choice. |
| Optional broader qualification | `blaze-models-prefill-tests.yaml`, `test-type=k3_accuracy_suite,k3_transformer_suite` | Galaxy SC1, TorusXY and staged K3 weights/goldens. Useful for validating the production path beyond LB; outside the agreed local acceptance scope. SC4/cross-host migration qualification is a separate follow-up. |

The BH e2e dispatch dropdown currently does not expose the accuracy row by itself. The `all` selection with `model=disaggregated_prefill` runs the LB prefill group. Adding `bh-lb-disaggregated-prefill-accuracy` to that dropdown would allow an accuracy-only dispatch without changing the jobs.

Before promoting this draft, consider adding the adapter restart matrix, allocation test and migration host tests to the LB accuracy row, and the LB-only request benchmark to the perf row. The new transformer is already wired; the full local acceptance matrix is not automatically run by either selected workflow.

Suggested dispatches, to be run after selecting the desired CI scope:

```bash
gh workflow run tt-metal-l2-nightly.yaml --ref pjosipovic/kda-device-request-initialization \
  -f run_blackhole=true -f run_wormhole=false -f run_triage_tests=false \
  -f additional_test_categories=experimental -f build-type=Release -f 'platform=Ubuntu 22.04'

gh workflow run blackhole-e2e-tests.yaml --ref pjosipovic/kda-device-request-initialization \
  -f 'system-type=LoudBox (8xP150)' -f model=disaggregated_prefill -f test-selection=all \
  -f build-type=Release -f 'platform=Ubuntu 22.04'
```
