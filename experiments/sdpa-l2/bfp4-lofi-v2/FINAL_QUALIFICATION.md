# Final current-source qualification

The initial post-format device checks completed on 2026-09-15, approximately 12:34–12:42 UTC, on reservation 220027, `yyzo-bh-08`, Blackhole P100A. Additional late-window stress checks completed through approximately 13:33 UTC and are reported separately below. These are experimental-kernel qualification results, not production integration or model-quality acceptance.

## Outcome and scope

**85 complete attention output hashes match their pre-format controls exactly.** All current runtime source pins checked by the corresponding audits match the files used for this qualification. The result is split into two deliberately different reference scopes:

| Qualification | Commands / attention rows | Reference coverage | Outcome |
|---|---:|---|---|
| Final small suite | 20 / 43 | N1024, H2, D128, all Q rows and all K/V | Complete output hashes match; finite, applicable preparation, replay and input-integrity gates pass |
| Block-order suites | 2 / 42 | N4096/32768, H2, 128 sampled Q rows per head, all K/V | Complete output hashes and all recorded accuracy/interorder metrics match |
| BFP4 primitive | 4 / 0 | Four N1024/four-core normal, thresholds, wide-exponent and zero probes | Exact device/native-oracle agreement |

The block-order interorder differences cover **every output element** even though their original-input reference metrics sample Q rows. Their preparation contract treats signed zeros as value-equivalent and permits no nonzero-bit mismatches. Original/permuted FP64 references agree within the stated tolerance. Both suites record two bit-exact replays, input immutability and stable source pins.

The four primitive probes do **not** record complete-output hashes, correctness trace replays with zero timing iterations, or during-run source-stability booleans. Their exact-oracle checks and current pins pass; they are not included in the 85 bitwise attention comparisons. Older primitive controls had different shapes/core counts and are not asserted to be identical-output baselines.

Evidence:

- [24-command small-suite plan](final-smoke-plan.json), [saved strict audit](final-smoke-audit-v1.json), and [gate implementation/scope](FINAL_SMOKE_VALIDATOR.md).
- [24-row block-order records](block-permutation-current-v1.jsonl), [18-row exp/order records](block-permutation-exp-current-v1.jsonl), and [saved current-source audit](block-permutation-current-audit-v1.json).

The plan was written before execution and retains its original planning label and SHA. This document and the completed records report execution status. No timing iterations were performed in these final numerical reruns; historical performance measurements remain separately identified in the research reports.

## Source and formatting checks

The raw numerical/evidence checkpoint is `52b288a3213`. Formatting processed 96 Python and 129 C++/header files, with the [source-format audit](source-format-audit.json) recording the exact pre-format archive hash and comparison results:

- No non-docstring Python AST changes from formatting; one module docstring had whitespace changes.
- 127 C++/header non-comment token streams were identical. Two files had reviewed brace insertions around existing single-statement bodies, with no declarations or control-flow changes.
- A later **test-only** repair replaced formatting-sensitive reciprocal/scale assertions with AST and whitespace-tolerant checks. Numerical producers were unchanged.
- Black 26.3.1 with Python 3.10 target and repository configuration was available; the repository hook pins 23.10.1, which was not installed. Clang-format 19.1.4 used the repository style. The full pre-commit suite was not run.

Historical `.source` witnesses retain byte-exact producer line endings and EOF spacing. Generic end-of-file fixers can alter those bytes; the scoped `.gitattributes` disables Git line-ending normalization and controls its whitespace warning, not pre-commit behavior. The [witness archival note](witness_sources/README.md) documents this limitation. No global hook configuration was changed, and the research checkpoint is not presented as an already-clean production PR.

The final device reruns provide the numerical verification for their selected kernel paths. They do not imply that all 129 historical/experimental C++ files, including preserved negative branches, were JIT-compiled again. No host C++ or CMake changes required a fresh whole-repository host build, and none is claimed.

A final read-only formatting check also passes for all 102 current Python files and 129 C++/header files, including the six late-window CPU/audit/test additions. Those additions do not change the earlier numerical producers or the scope of the original 96-file Python AST comparison.

## Test and evidence checks

**86 distinct tests passed**, counted once each:

| Environment | Tests | Scope |
|---|---:|---|
| Local Python 3.14.6 | 33 | Historical checkpoint validator |
| Local Python 3.14.6 | 12 | Strict current-source final validator |
| Local Python 3.14.6 | 17 | V layout (3), combined recipe (3), paired timing (3), mean correction (5), reciprocal/final scale (3) |
| Local Python 3.14.6 | 8 | Late-window exp stress validator, including 256K plan coverage |
| Remote Python 3.10.19 | 13 | Captured-input CPU interface |
| Remote Python 3.10.19 | 3 | Idealized CPU exp/order control unit tests |

The three reciprocal/final-scale tests and eight exp-stress validator tests also passed under remote Python 3.10.19; these are repeat executions, not additional distinct tests. The CPU quantization-noise model separately passed its five self-checks: dense reference, mean-correction identity, uniform-attention K invariance, uniform-attention mean correction, and softmax-Jacobian finite difference. The finite-difference relative discrepancy was `4.875774975611193e-06`.

The frozen selected-variant validator passes for four variants, 14 source pins, 34 shared-input cases and 136 stored repeat outputs/metrics. This validates pinned records; it is not a fresh rerun of the entire frozen suite. The new captured-interface test uses an explicitly synthetic fixture and a private FAST Q256 correction-address reset. It is not labeled an unmodified frozen FAST execution.

The historical checkpoint audit covers 169 entries representing 173 files. Recorded gates pass, while current-source drift warnings are expected after formatting and are disclosed separately. The warning count is a count of audit entries, not necessarily distinct files. Historical witnesses are never substituted into the strict current-source final gate.

Recheck from the repository root:

```sh
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_final_smokes.py
python3 -B experiments/sdpa-l2/bfp4-lofi-v2/validate_research_checkpoint.py
python3 -B experiments/sdpa-l2/validate_selected_variants.py
git diff --check
```

## Additional late-window diagnostics

The separate [exp stress study](EXP_STRESS.md) adds 80 small configurations across four routes, ten distributions and two seeds, each repeated in a fresh process. All 80 complete output hashes match across repetitions; the [small-suite audit](exp-stress-small-audit-v1.json) passes 160 records, 80 native/LUT pairs across repetitions and 80 process comparisons. These are additional accuracy-only runs, not part of the 85 pre-format hash comparisons above. The LoFi producer does not record a correctness trace replay, original/prepared input hashes or input-immutability gates with zero timing iterations. Exact preparation and reference/source assertions are witnessed in matching current source; those assertions do not invent missing runtime hashes. The HiFi2 producer additionally records two correctness trace replays and input integrity.

The 32K extension adds 48 configurations (four routes, six distributions, two seeds), also repeated in fresh processes. Its [saved audit](exp-stress-long-audit-v1.json) passes 96 records, 48 native/LUT pairs and 48 process comparisons. Accuracy references 128 explicit Q rows per head and all K/V, while output hashes and finite checks cover all outputs. These are not timing results; the same LoFi evidence omissions remain disclosed.

The targeted 256K extension adds 24 configurations (four routes, normal/outliers/common-K, two seeds), each repeated in a fresh process. Its [saved audit](exp-stress-256k-audit-v1.json) passes 48 records, 24 native/LUT pairs and 24 process comparisons. It uses the same 128-Q/all-KV reference scope and full-output finite/hash coverage. Across all three late-window device suites, 304 result records, 152 exp-pair checks and 152 process-repeat checks pass. These counts are distinct from the 85 pre-format output comparisons.

The [36-case FP64 CPU exp/order model](EXP_ORDER_CPU_MECHANISM.md) and [84-case lattice extension](EXP_ORDER_LATTICE_CPU.md) separately pass their model assertions and source/input/replay checks. They are explicitly idealized CPU studies, not device tests or performance measurements. Their numerical conclusions and different sampled-input scope are reported separately.

The [16-row common-V derived-metric study](COMMON_V_DERIVED_METRICS.md) passes its source/evidence pins, regenerated reference-RMS checks, available HiFi2 original-input hashes and centered-reference algebra. It computes centered error and output-floor ratios from stored error norms; it does not reconstruct or newly verify actual device outputs or replay.

## Limits

Finite/replay/provenance PASS is not a common L2 cutoff, a performance claim, an exhaustive compiler/environment closure, or an end-to-end model evaluation. Long-context performance was not rerun after formatting. No real activation captures, NVIDIA execution, backward, causal/masked attention, GQA/MQA, arbitrary head dimensions or tails were qualified. Existing frozen variants and production numerical code remain unchanged.
