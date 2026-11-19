> Current disposition: The owner subsequently accepted the recorded linear performance shortfall for stage5 and requested no further tuning. See [owner_decision.md](owner_decision.md) and [limitations](../eval_readiness/limitations.json). The technical failure assessment below is preserved; its earlier incomplete/no-waiver status is superseded by this explicit owner decision. No repair success or numerical waiver is claimed.

# AutoFix: linear FUNCTIONAL floor — unresolved

Evidence snapshot before owner acceptance: 2026-11-14. **AutoFix did not meet the original floor.** The current default passes the full-attention FUNCTIONAL floor; linear attention still fails it. Native linear alternatives have additional sensitive-layer diagnostic anomalies. Neither the passing full-attention row nor the default layer34 aggregate stress establishes stage acceptance. The completed AutoFix attempts failed to restore the complete contract; the stage is not accepted and there is no waiver or completion claim.

## Contract, diagnosis and measurement correction

The true 1×4 mesh must retain the accepted whole-output/state PCC gates, deterministic evolving trace replay, and ≥10% of the useful per-rank DRAM bound for one user with 128 input /128 decode tokens. Ordinary restored-position medians and single-chip efficiency are not this gate.

[AUTODEBUG_functional.md](AUTODEBUG_functional.md) identified ordered norm/projection cost, input normalization, selector and recurrence work. It also found BF16 linear router weight storage despite FP32 logits, and audit-context construction inside the original timer. The corrected byte model excludes resident L1 gamma and charges BF16 router weights. The corrected timer excludes audit-context setup/teardown while retaining device-only auditing. These reporting corrections do not remove the failed floor.

Corrected latency ceilings are **0.518283125 ms/token linear** and **0.344432500 ms/token full attention**, using 512 GB/s per rank. No padding, redundant reads, intermediate buffers or CCL traffic inflate the useful-byte numerator. Each FUNCTIONAL run below has 384 samples across three evolving 128-step sequences. State resets, input/position copies, presynchronization and host output checks are outside the replay clock.

## Hypotheses and measured outcomes

All paths below are experimental wrappers with explicit policy in their `.json.provenance.json` sidecars. Times are corrected FUNCTIONAL means in ms/token, not final-default medians. Layer 0 uses seed42 unless marked seed154. PCC is minimum whole-output decode PCC against the paired single-chip baseline.

| Hypothesis / evidence | Mean ms | Floor fraction | Min PCC | Result |
|---|---:|---:|---:|---|
| [Corrected linear control](repair_corrected_default_l0.json) | 0.993848357 | 5.21491% | 0.999158705 | Floor fails |
| [Native linear norm + projection](repair_corrected_native_both_l0.json) | 0.641690534 | 8.07684% | 0.999129197 | Floor fails; sensitive-layer regression below |
| [Above + native recurrence](repair_native_recurrence_l0.json) | 0.615891656 | 8.41517% | 0.998372623 | Floor fails; additional sensitive-layer regression |
| [Native recurrence family + FP32 selector](repair_native_selector_recurrence_l0.json) | 0.577010844 | 8.98221% | 0.998372623 | Floor fails |
| [Native family + SiLU once/page, seed154](repair_native_silu_pages_l0.json) | 0.561513518 | 9.23011% | 0.997745464 | Floor fails |
| [Native family + SiLU tile workers, seed154](repair_native_silu_tiles_l0.json) | 0.549172300 | 9.43753% | 0.997745464 | Floor fails; matches once/page tensors |
| [Native DRAM projections + selector + tile workers](repair_dram_native_l0.json) | 0.552731700 | 9.37676% | 0.999131380 | Floor fails |
| [Same family, DRAM HiFi2](repair_dram_hifi2_l0.json) | 0.551243200 | 9.40208% | 0.999124661 | Floor fails |
| [Same family, DRAM LoFi](repair_dram_lofi_l0.json) | 0.551229100 | 9.40232% | 0.996622928 | Floor fails; less accuracy margin, negligible time gain |
| [Exact compensated norm tail + selector](repair_exact_epilogue_selector_l0.json) | 0.759227388 | 6.82645% | 0.999158705 | Floor fails |
| [Exact tail + selector + SiLU tile workers](repair_exact_norm_silu_l0.json) | 0.721752700 | 7.18090% | 0.999158705 | Floor fails; matches control outputs/states |
| [Full input norm on residual shards](repair_corrected_norm_l3.json) | 0.372412604 | 9.24868% | 0.999834106 | Floor fails |
| [Full selector + projection-aligned input norm](repair_selector_projectionnorm_l3.json) | 0.341464075 | 10.08693% | 0.999762756 | Candidate passes, not integrated/default-qualified |
| [Full selector + residual-aligned input norm](repair_selector_residualnorm_l3.json) | 0.341138911 | 10.09655% | 0.999834106 | Candidate passes, preserves established residual layout |

The residual-aligned full-attention candidate has now been integrated and independently measured as the current default: [restored_functional_l3.json](restored_functional_l3.json), 0.340836807 ms/token, 10.10550%, minimum decode PCC 0.999834106, passing. The current linear default [restored_functional_l0.json](restored_functional_l0.json) remains below the floor: 0.719504474 ms/token, 7.20333%, minimum decode PCC 0.999158705. Both retain passing accuracy/replay checks. The full-attention pass does not waive the linear failure. Earlier `repair_topk1024_*` and `repair_topk2048_l3.json` preserve unsuccessful selector-padding controls; 2048-padding yielded 0.413026977 ms full-attention FUNCTIONAL latency.

[Norm component evidence](norm_epilogue_component.json) is bit-equal across 129 scalar rows/all four ranks. [SiLU retry](silu_component_retry.json) passes exhaustive 65,536 BF16 Z codes, all ranks, debug-gate/output comparisons and tracked replay with program-cache allocations included. The first SiLU trace failed because copy signatures were not warmed; the retry warms them rather than exempting allocations. [Integrated SiLU comparison](silu_integrated_tensor_comparison.json) establishes equality between the two seed154 variants. CPU comparison of `repair_exact_norm_silu_l0.pt` against `repair_corrected_default_l0.pt` also found identical inputs, prefill, all decode outputs, and all saved rank-wise convolution/recurrent states. This is measured-case equality, not universal proof.

## Sensitive layer 34: aggregate accuracy and diagnostic anomalies

Matched controls use layer34, seed154, input128/decode192. This additional sensitivity stress evaluates aggregate whole-output PCC ≥0.995 and per-step diagnostics with floor0.99, consistent with the functional-decoder and stage-review method. It does **not** add a per-step0.995 gate. Existing stage4 mandatory per-forward fixtures and their checker remain unchanged. Token top-1 agreement is not applicable to a standalone hidden-state output; hidden-channel argmax is not token argmax.

[Matched HF comparison](layer34_hf_comparison.json) verifies exact input equality, checkpoint identity, current HF/reference/cache source hashes and the exact historical oracle-runner Git source. No new HF compute or TT hardware execution was required. All listed aggregate HF comparisons exceed0.995:

| Evidence | Aggregate PCC vs HF | Minimum step PCC vs HF (diagnostic) | Interpretation |
|---|---:|---:|---|
| [Default mesh](repair_default_sensitive_l34.json) | 0.999702065 | 0.991128479 | Meets aggregate and diagnostic criteria; no HF waiver needed |
| [Selector only](repair_selector_sensitive_l34.json) | 0.999702065 | 0.991128479 | All192 outputs bit-equal to default |
| [Restore full baseline router policy](repair_baseline_router_l34.json) | 0.999841150 | 0.992242348 | Meets both criteria; alternative policy control |
| [Native norm](repair_isolate_norm_l34.json) | 0.999496290 | 0.987882354 | Aggregate passes; diagnostic anomaly below0.99 remains |
| [Native projection](repair_isolate_projection_l34.json) | 0.999616499 | 0.988056859 | Aggregate passes; diagnostic anomaly below0.99 remains |
| [Native family + full baseline router policy](repair_baseline_router_native_l34.json) | 0.999498392 | 0.987963329 | Router restoration does not remove native anomaly |
| [Native recurrence family + full baseline router policy](repair_baseline_router_recurrence_l34.json) | 0.998917611 | 0.966245666 | Aggregate passes; larger diagnostic anomaly remains |

The prior claim that the default fails this extended stress or needs an HF-controlled exception was an incorrect per-step gate interpretation and is superseded here. The paired single-chip reference itself has aggregateHF PCC0.999897753 and minimum0.999667588; these differences do not turn the default's passing aggregate/diagnostic result into a failure.

[Selector comparison](selector_layer34_comparison.json) establishes exact default/selector equality. [Candidate-to-default comparisons](layer34_default_candidate_comparison.json) isolate additional native changes: diagnostic minima0.988462994 for norm,0.988639016 for projection and0.968895326 for recurrence. These native alternatives remain unaccepted pending repair; passing aggregate alone does not dismiss their below0.99 anomalies. Source confirms the prior mesh selected FP32 logits while the single-chip path rounded to BF16. This is useful causal context, not a waiver. Terminal states cannot locate the first divergence; an intermediate residual/norm/router/expert capture is a justified diagnostic for native anomalies, not a new default acceptance requirement.

## Final isolated native CCL / exact-recurrence attempts

The CCL repair passed its hardware-pack component oracle (40/40 eager/changing-trace comparisons), and the exact recurrence component passed48/48 bit-equality comparisons. These component results did not close the complete decoder contract:

| Isolated layer0 128-in/128-out control | Mean ms/token | Floor fraction | Minimum decode PCC | Outcome |
|---|---:|---:|---:|---|
| [Exact recurrence + repaired FP32 CCL](repair_fp32_ar_exact_l0.json) | 0.869184612 | 5.96287% | 0.999162636 | Accuracy/replay pass; floor fails; slower than retained default |
| [Native norm/DRAM projection + repaired FP32 CCL](repair_fp32_ar_native_l0.json) | 0.554099732 | 9.35361% | 0.999129693 | Layer0 accuracy/replay pass; floor still fails |

Both are isolated whole-layer timing controls with384 changing-input replay samples; neither supplies accepted optimized numbers. The combined native norm/projection/exact-recurrence path still has a matched-HF layer34 diagnostic anomaly: aggregate0.999559383, minimum0.987942862 at step174 ([comparison](repair_fp32_ar_native_exact_l34_hf.json)). Aggregate passes, but the below0.99 diagnostic remains unqualified. Its earlier timing overlapped CPU profile postprocessing and is not used for ranking. The FP32 CCL change does not repair the upstream native-family anomaly.

The native CCL patch/build experiment has been removed from live C++ source and retained only as a patch artifact. [Restoration verification](native_restore_verification.json) confirms that both C++ files exactly match their pre-experiment source hashes. Both layer kinds reproduce pre-experiment prefill and all 128 decode outputs bit-for-bit. The rebuilt `_ttnncpp.so` has new SHA256 `cab28abb4f0c3b4f493c25909cc82835adda813c0a64d7c8a26a579c1cdcfbb1` ([binary provenance](fp32_ar_restored_pack_oracle.json)); this is source restoration and measured output equivalence, not binary bit-identity. The retained default's recorded linear FUNCTIONAL result is0.719504474ms/7.20333%, versus the required maximum0.518283125ms. **The AutoFix attempt failed the mandatory contract.** The default layer34 aggregate stress still passes, so no HF waiver or new per-step0.995 gate is introduced.

## Reproduction and remaining work

Exact commands, environment and source identities are the matching provenance sidecars. Drivers are `tests/run_multichip_b1_repair.py` and `tests/run_multichip_recurrence_repair.py`, using `--length 128 --decode-steps 128 --functional-sequence`, three sequence repeats and named `--mesh-build-kwargs` hypotheses. Sensitive runs use `--layer 34 --seed 154 --decode-steps 192`. Commands run under bounded timeouts with serialized device ownership. Original JSONs and nonzero exits remain evidence; do not overwrite their `passed` flags. The historical repair runner applied per-step0.995 to the additional layer34 stress; its false flags are preserved, with the corrected aggregate/diagnostic interpretation explicitly recorded above. Mandatory fixture gates are not changed.

The current exact-arithmetic linear default is **0.2012213 ms/token above its ceiling**, requiring another **27.97% latency reduction (1.3882× speedup)**. There is no evidenced single small change that closes this gap. Exact ordered-scan coefficient reuse/fewer redundant activation reads remains a justified unmeasured kernel hypothesis from AutoDebug, but has no current whole-layer gain estimate. The tested exact-recurrence fusion passes component equivalence but loses complete-consumer latency (0.869184612ms); further recurrent data-movement variants are untried handoffs, not an earned repair. The native recurrence shortcut has unresolved below0.99 diagnostic anomalies. Neither is an earned speedup or a reason to claim repair success. First-divergence capture remains a possible repair investigation for the unselected native family; the default layer34 aggregate stress passes and needs no waiver.

Optional optimization stopped under the last-two-gains and four-hour Stop Rules. These rules do **not** waive the hard FUNCTIONAL floor or accuracy findings. AutoFix attempts so far have failed to restore the full stage contract. The current-default four clean timing fixtures and both FUNCTIONAL rows now exist; linear FUNCTIONAL fails while full attention passes. Boundary, stress, natural B32, watcher, mixed ownership, allocation-positive-control and context subsets have passed for the retained source. B1/B32 profile collection and report generation are complete ([B1](perf_summary.json), [B32](perf_b32/perf_summary.json)); MoE attribution and restored-native verification are complete. Strict evidence consolidation passes33/34, failing only linear FUNCTIONAL. Applicable checkpoint hooks pass with production Black/isort formatting explicitly deferred to preserve measured source bytes. Independent round1 requested the unchanged linear floor and targeted non-aligned/resumed watcher coverage; that watcher subset passes and independent round2 closes it. Round2 remains more-work-needed solely for the unresolved linear floor. The local checkpoint is explicitly incomplete; its SHAs are recorded in work_log.md. Reported default timings belong only to the retained path, not to a rejected faster candidate. No stage completion or push is claimed.
