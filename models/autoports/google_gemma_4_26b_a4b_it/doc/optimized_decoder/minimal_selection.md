# Minimal prefill candidate selection

Current selected source is `5ff391a2efb6096e7d9eaa499c6d19a8548cf9a60d108425e773d4ded72ee898`. [The v8 source proof](source_delta_v8.md), [completed validation](validated_v8_validation_summary.json) and [native audit](final_roofline_audit.md) bind fullM2/HiFi2/producerL1 QKV and slidingM4/HiFi4/DRAM. Full packed output and tiedKV slice inheritL1; concat returnsDRAM. All17 current gates and both native profiles pass. Historical candidate matrices below retain original hashes; independent final review remains separate.

Select minimal QKV K8 for sliding and K16 for full, using HiFi4 sliding/HiFi2 full, with unchanged FP32 activation/output and BFP8 weights. Select minimal output K8 only for full, preserving LoFi, BF16 input and FP32 output. Sliding retains multicast2D output K16. All selected minimal configs use grid 11×8, N8, subblock 1×4 and setup-built Mblock=min(2,padded_M_tiles) for full QKV; sliding QKV and full output retain min(4,padded_M_tiles).

## Actual 4096-prefill/128-decode matrix

Whole-prefill host values are medians of three synchronized samples. They include the full decoder and are not isolated GEMM times.

| Candidate | Layer | Prefill PCC | Minimum decode PCC | Whole-prefill median µs | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| [baseline](minimal_baseline_layer0.json) | 0 | 0.9991528513 | 0.9953353194 | 222538.749 | pass |
| [baseline](minimal_baseline_layer5.json) | 5 | 0.9991236270 | 0.9950555611 | 190377.647 | pass |
| [output K16](minimal_output_k16_layer0.json) | 0 | 0.9991528513 | 0.9953353194 | 222297.721 | pass |
| [output K16](minimal_output_k16_layer5.json) | 5 | 0.9991236270 | 0.9950555611 | 190071.574 | pass |
| [output K8](minimal_output_k8_layer0.json) | 0 | 0.9991528513 | 0.9953353194 | 222723.370 | pass |
| [output K8](minimal_output_k8_layer5.json) | 5 | 0.9991236270 | 0.9950555611 | 189892.093 | pass |
| [qkv K16](minimal_qkv_k16_layer0.json) | 0 | 0.9991563938 | 0.9952796328 | 221464.144 | pass |
| [qkv K16](minimal_qkv_k16_layer5.json) | 5 | 0.9991234309 | 0.9950717323 | 187922.907 | pass |
| [qkv K8](minimal_qkv_k8_layer0.json) | 0 | 0.9991560943 | 0.9952574859 | 221364.003 | pass |
| [qkv K8](minimal_qkv_k8_layer5.json) | 5 | 0.9991233638 | 0.9949893235 | 188109.431 | FAIL |

Full QKV K8 completes the workload but fails at 4211 PCC 0.994989323531; K16 is selected. Output candidates preserve every headline decode PCC relative to their matched baselines. The QKV change affects the prefill cache, so its decode results were checked explicitly.

## Stress and maximum-context acceptance

| Candidate | Workload | Prefill PCC | Minimum decode PCC | Sampled prefill rows / minimum PCC | Result |
| --- | --- | ---: | ---: | --- | --- |
| [qkv K16, layer5](minimal_long_k16_layer5.json) | 262144 + 3 boundary checks | 0.9990340764 | 0.9978274142 | 291 / 0.9960844903 | pass |
| [qkv K8, layer0](minimal_long_k8_layer0.json) | 262144 + 3 boundary checks | 0.9993039684 | 0.9995932677 | 291 / 0.9951303778 | pass |
| [qkv K16, layer0](minimal_qkv_k16_stress_layer0.json) | 1025/512 | 0.9990994208 | 0.9955550619 | — | pass |
| [qkv K16, layer5](minimal_qkv_k16_stress_layer5.json) | 1025/512 | 0.9990415840 | 0.9962529030 | — | pass |
| [qkv K8, layer0](minimal_qkv_k8_stress_layer0.json) | 1025/512 | 0.9990982367 | 0.9955550619 | — | pass |

The maximum-context runs compare 291 specified input rows, aggregate/tail results and boundary decode checks. They do not compare every prefill row. The full maximum-context candidate uses minimal QKV with the previous output backend; the combined full QKV/output policy has now passed the separately attributed integrated maximum/near-maximum checks below.

## Alternating output control

Eight pairs execute in one process with alternating baseline/minimal order. Both paths are warmed; samples measure the complete synchronized prefill.

| Output candidate | Baseline median µs | Minimal median µs | Minimal faster pairs | Selection |
| --- | ---: | ---: | ---: | --- |
| [Layer0, K16](minimal_alternating_k16_layer0.json) | 221598.1425 | 221729.5825 | 1/8 | retain baseline |
| [Layer5, K8](minimal_alternating_k8_layer5.json) | 189347.9410 | 189143.1020 | 7/8 | minimal selected |

Sliding minimal output loses 7/8 pairs despite a small apparent advantage in the initial three-sample matrix, so it is disabled by default. Full minimal output wins 7/8 pairs; its observed median improvement is 204.839 µs (about 0.108%). This is a bounded measured selection, not a claim of an absolute optimum or of additive speedups when combined with minimal QKV.

All four commands in [minimal_acceptance_commands.json](minimal_acceptance_commands.json) return zero. [minimal_selection.json](minimal_selection.json) retains hashes, raw paired samples, individual sample arrays, PCCs and exact failure positions. The source-resolution proof in [final_policy.json](final_policy.json) separates the selected defaults from their pre-integration hardware evidence.

## Archived completed v5 evidence

The following artifacts record ancestor v5 runtime `169c0d97d7d0e9f35d97633f133305f1088b987ef625693d3faa100d25c3e67b`, not current v7. The [final v5 summary](validated_v5_validation_summary.json) reports `all_final_default_correctness_gates_passed`: all 18 journal commands return zero. Both kinds pass B32, prefix continuation, request reuse, BF16-cache compatibility, headline, strict sampled maximum/near-maximum rows and separate Watcher checks; all four pytest cases pass. Integrated 1025/512 stress passes both exact-HF and optimized-versus-fused preservation gates. Optimized HF minima are 0.995555061903 sliding and 0.996252903034 full; direct preservation minima are 0.995158301497 and 0.996327176492. Both v5 native profiles are archived at that same source hash. Archived v6 regressions remain source-bound. Current v7 validation includes fresh full public/max gates and both-kind stress, with explicit inheritance only for unchanged sliding broad contracts. Current native profiles remain separate.

| Artifact | Minimum decode PCC | Sampled prefill rows / minimum PCC |
| --- | ---: | --- |
| [validated_v5_long_262144_layer5.json](validated_v5_long_262144_layer5.json) | 0.9978274142 | 291 / 0.9960844903 |
| [validated_v5_long_262143_layer5.json](validated_v5_long_262143_layer5.json) | 0.9976669813 | 291 / 0.9960844903 |
| [validated_v5_headline_layer5.json](validated_v5_headline_layer5.json) | 0.9950717323 | — |
| [validated_v5_long_262144_layer0.json](validated_v5_long_262144_layer0.json) | 0.9995932677 | 291 / 0.9951303778 |
| [validated_v5_long_262143_layer0.json](validated_v5_long_262143_layer0.json) | 0.9997222924 | 291 / 0.9951303778 |
| [validated_v5_headline_layer0.json](validated_v5_headline_layer0.json) | 0.9952574859 | — |

## Full-only HiFi2 selection and current integration

[Same-process paired controls](minimal_pairs_v6_summary.md) favor HiFi2 by median paired−473.456µs sliding/−545.573µs full, with7/8 and8/8 faster pairs. Grid11×10 has no demonstrated gain and is rejected. Both HiFi2 stress checks pass, but sliding maximum-context row32 fails the unchanged .995 bar; only full is adopted. Full candidate and current integrated262144/262143 runs pass all291 sampled rows (minimum.996029643684647). Current validation also passes all full public contracts, both headline/Watcher kinds, four pytest cases, both512-step stress gates and five affected-prefill boundary controls. See [current short/tail timings](prefill_boundary_v7_summary.md). No old device/profile measurement is rebound to this source.
