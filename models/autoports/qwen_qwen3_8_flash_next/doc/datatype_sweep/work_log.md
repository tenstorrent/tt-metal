# Datatype sweep work log

Stage 8, Qwen/Qwen3.8-Flash-Next, pinned revision 812d96e10a0dd82b69d749e68693698cf6c36daa. Started from completed stage-7 main@ccb44bb0cb36efb5eb26718a5520dbdf1ec41dc5. No vLLM integration or pushes authorized/performed. Unrelated untracked prior-stage evidence remains untouched.

## Contract and thresholds

Per reference top1 >=90%, top5 >=98%, top100 =100%; full48, 1x4 Blackhole p300c, advertised context262144. Rank only trace-verified teacher-forcing decode, using repeated serial baseline spread for ties. P1 required; required-to-fit expert BFP4 must carry measured loss. Token-out128/128 benchmark separately after selection. All candidates use explicit build kwargs files. Final independent review and local checkpoint remain pending.

## Startup and baseline

Installed/enabled plugins verified with `codex plugin list`: tt-model-bringup0.1.24 and tt-autodebug0.1.8. Package paths in per-run provenance; environment.py exports used. `/dev/tenstorrent/0..3` were unowned (fuser empty), bounded tt-smi lists4Blackhole devices. No resets. `logs/device_list.txt`. Repository hooks run before device jobs; `logs/baseline_hooks.log`, `logs/plumbing_hooks_v2.log`.

Initial inherited runtime refresh: `baseline_refresh_1.json`, `.provenance.json`, `logs/baseline_refresh_1.log`, exit0. AIME24 chat100: prefill97/100/100%, traced teacher forcing96/100/100%, TTFT7863.243ms, decode8.729855t/s/u. It used original source modules loaded before policy edits; provenance lists initial source hashes. CPU HF reference job overlapped loading only, ended01:34:30UTC, before prepared/scored timing. No ranking claim from setup time.

`natural_text.refpt` contains128prefix+512actual consecutive Tale-of-Two-Cities tokens at offset2048. CPU exact mmap original HF load, no TTNN, no quantization, top100 per position. `natural_text.provenance.json`, `natural_text_hf/` contain command, tokens, loader/control evidence; completed01:34:30UTC. Numerical teacher forcing uses raw book text and is not a chat quality verdict.

## Capacity and candidate matrix

`initial_precision_capacity_audit.md`: all-BFP8 experts+PLE alone39,766,491,136B/rank exceed34,178,731,008allocatable by5,587,760,128B. All-BFP8 starting policy physically rejected before load, even at zero KV. BFP4 is required for fit; P1 still required. Layer0/47 BFP8 upgrades cost629,145,600B/rank and will be measured. Fused expert gate/up/down require identical weight dtype and share fidelity per invocation (explicit op contract); bundle is one material group.

`configs/` enumerates inherited BFP4/LoFi, expertHiFi2, BFP8edge LoFi/HiFi2 controls, BFP8projection LoFi/HiFi2, shared and QSA inner-layer BFP4LoFi/HiFi2, activation8, CCL8 and KV8. These files are planned candidates, not completed results. No winner selected yet.

## Active work

Policy construction and readback added. Inherited precision is unchanged by default until a selected artifact exists. `run_datatype_candidate.py` runs installed readiness scorers, passes/validates identical build kwargs, records actual operands/compute configs and trace replay deltas. `inherited_e4_lofi_v1` is the first new-path repeated baseline on both references. Pending: candidate runs, capacity updates, P1, final capability/resume/qualitative/token-out gates, charts, review, commits.

## Timing anomaly resolved before ranking

`inherited_e4_lofi_v1.json`: first AIME TF8.642884t/s/u includes2trace captures; second AIME TF9.675963t/s/u has0captures. Natural TF9.659099 has0captures. The difference is first-use capture overhead, not baseline timing noise. `run_datatype_candidate.py` now performs2token forced warmup per reference outside timing, and requires model replay counts plus zero captures/releases in each measured TF row. The forthcoming v2 baseline is repeated serially in this warmed regime; v1 cold TF rows are not Pareto/ranking evidence. This changes measurement only, not model math. Initial prefill97% and natural97.65625% are retained as correctness evidence.

CPU audit followup:49tests+40subtests pass with TTNN import actively prohibited (`cpu_checks.json`, exact invocation in `initial_precision_capacity_audit.md`). Fixed global-head/per-layer misuse, mixed fused expert dtypes, layer-kind-inapplicable overrides, caller-owned cache dtype mismatch and shared-pool dtype byte accounting. Model and generator defaults now both resolve selected artifact. Sample/logit buffers consume their policy fields; residual/logit outputs enforce the fixed validated storage contract. These fixed BF16 contracts are not advertised as tunable BFP8 residuals.

The final constructor dtype assertions/field propagation are algebraically identical for the inherited baseline: only host metadata checks and unchanged BF16 allocation arguments; no device op or math change. They were added while v2 baseline had already imported its modules; exact initial source hashes and runtime policy are preserved in that run's provenance. Subsequent candidate runs use final constructor source. A selected-default final run is mandatory.

Queue stop file requested between jobs to add a four-layer BFP8/HiFi2 higher-precision control before the lower-precision trials. No active job interrupted. Conservative2GiB pool planning rejects even2expert upgrades at full context, although stage6 measured max-context free memory permits up to4upgrades arithmetically. The candidate uses the full private262144cache; actual load/prepare/replay free memory determines whether it is offered to P1. No shrink of context or claim that conservative pool planning proves physical failure.

Timing followup: v2 completed accuracy-pass for all8rows but harness marked TF trace checks false: capture0, release6(AIME)/3(book). Source inspection finds the newly exposed P1 `teardown()` is called by readiness runners AFTER timing; its cleanup increments release counters. No untraced decode or wrong result observed. Harness now defers runner teardown when reusing its one resident generator; outer finally still closes it. v2 remains immutable diagnostic evidence, not the final sweep baseline. v3 reruns the same warmed regime with counters scoped correctly. Two v2 warmed AIME values9.677739/9.680351 (spread0.002612); natural9.664311/9.657415. No model implementation repair required.

### Baseline v3 accepted

All eight baseline reference/path/repeat rows pass with traced teacher forcing and no capture/release inside the measurement. AIME24 100-token prefill top-1 0.97, teacher forcing 0.96, top-5/top-100 1.0. Warmed AIME decode repeats [9.679690005440683, 9.67923230752992] t/s/u; spread 0.0004576979107628887 t/s/u. Natural 512-token continuation prefill top-1 0.9765625, teacher forcing 0.98046875, top-5/top-100 1.0. Exact evidence `inherited_e4_lofi_v3.json` and matching provenance/log. P1 adapter explicitly resets persistent sampling to greedy for teacher-forced callbacks; sampled reasoning-text generation retains model-card parameters. Runtime implementation remains frozen during matrix.

### Four-layer higher-precision trace failure / AutoFix investigation

`e8_four_hifi2_v3` constructed/prepared at full262144 capacity, with131090432B/bank free (1048723456B/rank), but the first readiness prefill failed at end_trace_capture: buffer4210685952 overlaps another live trace high-water4218759104. No accuracy row or performance ranking is accepted for this run. Queue exited1 and released devices; fuser showed no owner. No reset/kill. AutoDebug launcher first sandbox preflight failed because unprivileged namespaces are unavailable; retried inspection-only with explicit AUTODEBUG_SKIP_CHILD_SANDBOX=1 under the user-authorized subagent scope. Logs `logs/autodebug_trace_v1.log` and `logs/autodebug_trace_v2.log`. A fixed trace-region experiment is pending the isolated diagnosis; no assertion is disabled. Required KV decoder smoke/control runs now use the normal explicit precision constructor and retain host outputs for paired PCC; full-model token agreement remains separate required work.

AutoDebug v2 could not execute tools: the selected local Codex binary lacked its adjacent code-mode host. Terminated only that own CPU-only process PID2530818; no device process touched. Relaunched the bundled skill with a PATH scoped to the installed vendor Codex directory containing its matching host, preserving global configuration. The launcher writes root AUTODEBUG.md; saved pre-existing root report as `prior_root_AUTODEBUG.md` and will restore it after copying the new diagnosis into this stage directory. Both KV decoder runs exited0; paired minimum short-prefix rowPCC0.9990308742 and batch-userPCC0.9997351219. Full48 B32 vocabulary control is running separately.

B32 full48 BF16 short-prefix control `kv_b32_inherited_e4_lofi_v1.json` passed: 32 identical natural33-token prefixes,2forced decode inputs, all slots position35, model_replays2/capture0/release0 during decode, 50,331,648B/rank sharedKV diagnostic pool. Full vocabulary `[32,3,248320]` logits saved locally. BFP8 pair remains pending. Probe uses capacity128 solely for short-prefix/B32 diagnostics; advertised262144 unchanged and full-capacity candidate remains mandatory. Added `e8_three_hifi2` (expert BFP8 layers0,1,47) as a lower-memory higher-precision bracket; not yet measured/offered to P1.

### KV BFP8 diagnostic rejection

Both B32 full48 short-prefix snapshots completed with tracked public traces, same32identical natural-text inputs, sameforcedtokens, and no decode recapture/release. Top1 matched all32users/all3phases. However fullvocabularyPCCs perphase are0.9718483(prefill33),0.9729653(position33),0.9841406(position34), flattened0.9771782. This fails the0.99 per-user diagnostic floor despite layer3 PCC passing; KV8 is not eligible for selection. `kv_short_prefix_b32_comparison_v2.json` records perphase values. Fullcapacity/fullreference KV8 accuracy and speed remain pending and will be captured for rejection evidence. CPU source/paired-input audit requested to distinguish any implementation defect from accumulated quantization loss; no guard or threshold weakened.

AutoDebug finished successfully via the matching vendor CLI. New report copied to `datatype_sweep/AUTODEBUG.md`; prior root report restored byte-for-byte. AutoFix is running the fixed768MiB trace-region test with all protections/fullcontext unchanged; `AUTOFIX.md` records hypotheses and commands/results as they complete.

### Three-layer precision reference fit experiment

Both fixed-region four-layer trials closed normally after allocation errors: 768 MiB and 640 MiB left insufficient indexer workspace. Exact errors and provenance are in `AUTOFIX.md` and the matching `e8_four_fixed*_v1` artifacts. Started `e8_three_hifi2_v1` with all 48 layers, full 262144 context, global expert HiFi2, BFP8 expert layers 0/1/47, both readiness references, and `--trace-diagnostics`. All allocation checks remain enabled. No reset, context reduction, or model arithmetic change.

Manually inspected `natural_text_hf/text.txt`: the offset-2048 readiness reference is narrative prose from the Dover mail passage. This is distinct from the offset-256 table-of-contents prefix used by the paired KV diagnostic.

### Full-capacity P1 reference accepted

`e8_two_hifi4_v1` passed with BFP8 routed experts in layers0/47, BFP4 in the other46 layers, expert HiFi4 globally, and unchanged BF16 cache/activations/CCL. It achieved97% AIME prefill/teacher-forcing top1; natural-text98.828125% prefill and98.2421875% teacher forcing; all top5/top100 are100%. AIME warmed traced speed9.661007913 tokens/s/user. All six prefill buckets,31/32/33-token prompts and sampled-to-greedy transitions passed at262144 capacity. Three upgrades fail retained16-bucket trace placement; four fail earlier placement or reserved-region workspace. See `AUTOFIX.md` for the accepted practical bracket.

Started unchanged installed P1 scoring through `run_datatype_p1.py` using `p1_candidates_two.json`, reference `e8_two_hifi4`, comparison/selected-for-this-provisional-run `inherited_e4_lofi`. This is not final selection. Two retained sampled reasoning answers,6144-token cap, at least512 tokens/answer and1000 scored positions, pinned chat/thinking template, model-card temperature1/top-p0.95/top-k20/seed1000,256-token traced decode on each answer, original-HF slice threshold2048. All default noise bounds remain unchanged. Exact command/source evidence: `p1_initial_two_v1.provenance.json`; log: `logs/p1_initial_two_v1.log`.

### P1 text generation complete

The unchanged runner retained bundled prompts r01/r02 with1213/2222 answer tokens (3435 total), exceeding the1000-position gate. `p1_texts_decoded.json` preserves exact token IDs, pinned tokenizer, rendered prompts, sampled completions and generation metadata. Both sampled completions close thinking and terminate at EOS; provenance inspection found no obvious repetition or truncation. This is not a substitute for the final matched-HF qualitative suite. Numerical P1 scores are still pending at this log entry.

### Final validation accuracy assertion

CPU inspection found the reused optimized validation driver asserted top-5/top-100 only. Added the stage-required top-1 ≥0.90 assertion and recorded threshold before the selected-default run. This changes acceptance reporting only, not model execution or P1 scoring.

### CPU report tri-state gates and dominance proof

Updated only `tests/report_datatype_sweep.py`, `tests/test_datatype_report_cpu.py` and this log. P1 and capability outcomes now preserve `null` for unscored/unevaluated gates, with explicit status/reason. `failed_gates` contains actual false outcomes; `pending_gates` records unknowns; only explicit all-gate acceptance sets `eligible=true`. Missing capability decisions no longer claim measured capability failure. Actual capability failure still independently rejects KV8 even with unknown/passing P1. Scored P1 decisions require boolean outcome, compatible status, reason and evidence.

Final selection refuses unresolved contenders unless their unscored P1 decision explicitly uses `status="not_scored_dominated"`, names an actually eligible `dominated_by` config, cites both exact measured result artifact paths (absolute paths or paths relative to checkout working directory), and its traced speed is lower by more than repeated-baseline noise. The reporter recomputes this difference rather than trusting a claimed speed gap. Unknown faster or noise-tied candidates, unknown capability on contenders, ineligible/missing comparators, uncited artifacts, duplicate config IDs and invalid traced speeds cannot silently pass selection. Valid pruning preserves `passed=null`, `eligible=false`, `pending_gates`, and `status="not_selected_unscored"`, with a recorded numeric dominance check. Explicit failed capability/execution/accuracy remains sufficient rejection without inventing P1 failure. Plot diamond frontier and missing-speed count remain; pending gates are labeled on measured points.

CPU validation: 21 report tests passed with `TT_VISIBLE_DEVICES=0,1,2,3` and an active import guard that raises on every attempted TTNN import. The temporary-fixture CLI test generated JSON, CSV and both PNG charts, verifying null P1, separate pending/failure fields and ineligible dominance pruning; it did not overwrite production sweep artifacts. Other tests cover faster/tied unscored rejection, both measured evidence citations, missing/ineligible comparators, unknown capability versus false structured gate, actual capability rejection, duplicate IDs, and numerical accuracy distinct from late execution failure. Repository pre-commit hooks ran only on the three touched files and all applicable hooks passed (exit 0). No runtime edits or devices.

### Routed expert workload invariant

The target configuration has 512 routed experts and exactly 10 experts per token. The measured native active-expert path passes `num_experts_per_tok=10` in both prefill and decode (`tt/multichip_decoder.py`); it does not execute a dense all-expert debug substitute. TP4 stores 128 local experts per rank. Router logits/top-k remain FP32, router weights BF16, and routing weights BF16 throughout the coarse sweep. Thus active expert count is fixed across policy comparisons; policy-dependent route IDs may change numerically and are covered by full-model accuracy/P1.

### Initial P1 completed — all checks pass

`p1_initial_two_v1.provenance.json` exited 0 after 2581.37 s, including fresh generation and original HF scoring. Preserved canonical result and runtime sidecar as `p1_initial_two_v1.json` and `p1_initial_two_v1.runtime.json`. Two retained answers provide 3435 scored positions; inherited expert BFP4+LoFi vs the fitting two-layer-BFP8+HiFi4 control has ΔNLL −0.00081 ±0.00078, KL×1000 1.25709, and traced-decode top-1 0.98828 for both over 512 positions. Original HF control KL×1000 is 3.62 (bound4). All critical/advisory checks pass; errors empty; no waiver. Selection remains pending the full coarse matrix and likely combinations.

### Coarse queue v3 started

After P1 exited0 and `fuser` reported no holders on devices0–3, removed the between-job stop file and launched `tests.run_datatype_queue --suffix v3 proj8_lofi proj8_hifi2 shared4_lofi shared4_hifi2 qsa4_lofi qsa4_hifi2 e4_hifi2 e8_edges_lofi e8_edges_hifi2 activation8 ccl8 kv8 head8_lofi head8_hifi2 gr8_lofi gr8_hifi2 gdn_zab8_lofi gdn_zab8_hifi2`. Exact child commands/source/environment are in each `_v3.provenance.json`; queue log is `logs/coarse_queue_v3.log`. Every child has timeout5400 and runs serially on full48/full262144. First two projection candidates complete/pass; AIME warmed traced t/s/u 9.6853631098 (LoFi) and9.6894039579 (HiFi2). These are provisional comparisons pending remaining matrix, P1 and close-result repeats. Validation-driver hooks passed in `logs/validation_hooks_v3.log`.

### Reusable CPU precision telemetry audit

Added `tests/audit_datatype_precision.py` with `--inputs ... --output ...` and optional `--selected <policy.json>`. It revalidates policy/layer exceptions, requires all 48 ordered layer summaries and correct projection roles/counts, and compares actual recorded projection/expert/shared-down/head weight dtypes and compute configurations with the resolved policy. It checks both decode and rows paths, activation configuration, CCL/KV selectors and fixed accumulation flags. Output separates tensor introspection, constructed-config introspection, source/declaration-only assumptions, structure and policy identity; fixed residual/logit/sampling contracts and literal output/workload fields are not mislabeled as tensor readbacks. Candidate overrides resolve their own artifact. Recognized default final-validation/resident-benchmark paths compare requested policy with `--selected`, including nested `performance.runtime_precision`. Audit success is not an accuracy, capability, P1 or performance verdict.

Completed baseline/reference plus projection/shared candidates were checked with the equivalent CLI:

```bash
python_env/bin/python models/autoports/qwen_qwen3_8_flash_next/tests/audit_datatype_precision.py \
  --inputs models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/inherited_e4_lofi_v3.json \
  models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/e8_two_hifi4_v1.json \
  models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/proj8_lofi_v3.json \
  models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/proj8_hifi2_v3.json \
  models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/shared4_lofi_v3.json \
  --output models/autoports/qwen_qwen3_8_flash_next/doc/datatype_sweep/precision_propagation_recheck_v1.json
```

The invocation used a CPU `runpy` wrapper with an import guard rejecting any TTNN import: five reports pass, 22,490 equality checks, zero mismatches. Existing `precision_propagation_audit.*` evidence was preserved. `precision_audit_cpu_controls_v1.json` records an in-memory altered layer-3 QSA decode fidelity correctly rejected at the exact field, selected-default positive/mismatch controls for both report shapes, explicit-policy isolation, missing-layer and empty-dtype rejection, and requested-only-report rejection. These synthetic controls do not claim a selected-default device run. The module-name default-driver recognition issue exposed by the CPU control was fixed before final hooks.

`python_env/bin/pre-commit run --files models/autoports/qwen_qwen3_8_flash_next/tests/audit_datatype_precision.py` passed, exit 0; exact hook output: `logs/precision_audit_hooks_v1.log`. Hooks targeted only the new Python file. No runtime edits, device access, canonical-result edits or prior-evidence overwrite.

### Completed inner shared/attention BFP4 pairs

All four candidates completed the same full48/full262144 AIME100 and natural512 prefill/teacher-forcing runs with zero capture/release during measured decode. These are short-reference results, not P1 approvals.

| Config | Minimum top-1 | Minimum top-5 | AIME traced t/s/u | Full-reference gate |
|---|---:|---:|---:|---|
| shared4_lofi | 0.96000000 | 1.00000000 | 9.690366405 | True |
| shared4_hifi2 | 0.96000000 | 1.00000000 | 9.689034064 | True |
| qsa4_lofi | 0.91000000 | 1.00000000 | 9.686859755 | True |
| qsa4_hifi2 | 0.91000000 | 1.00000000 | 9.685994469 | True |

### Routed expert fidelity / edge-layer controls completed

All three controls passed every short-reference gate at full capacity. The inherited LoFi baseline also passes P1 and is faster than each of these controls. Higher precision remains useful as an accuracy control, not an observed speed improvement.

| Config | Minimum top-1 | Minimum top-5 | AIME traced t/s/u |
|---|---:|---:|---:|
| e4_hifi2 | 0.96000000 | 1.00000000 | 9.670870340 |
| e8_edges_lofi | 0.95000000 | 1.00000000 | 9.666055426 |
| e8_edges_hifi2 | 0.97000000 | 1.00000000 | 9.669162662 |

### Activation, CCL and KV switches completed

All three pass the full short-reference numerical gate, but all are slower than the traced baseline. KV8 additionally fails its mandatory short-prefix/B32 PCC gate and cannot be selected.

| Config | Minimum top-1 | Minimum top-5 | AIME traced t/s/u | Disposition |
|---|---:|---:|---:|---|
| activation8 | 0.94000000 | 1.00000000 | 9.653394595 | slower than inherited P1-passing baseline |
| ccl8 | 0.97000000 | 1.00000000 | 9.546785963 | slower than inherited P1-passing baseline |
| kv8 | 0.96000000 | 1.00000000 | 9.293589017 | slower; capability PCC failed |

### Coarse GR comparison and combined-policy plan

GR BFP8+LoFi and HiFi2 both completed all four full-model numerical gates. AIME traced decode is 9.7246062063 and 9.7172467262 t/s/u respectively; natural-text top-1 is at least 0.974609375. These remain provisional speed results pending combined-policy tests, repeat measurements, and P1. Created `configs/combined8_lofi.json` and `combined8_hifi2.json`: all eligible nonexpert projection groups use BFP8 with the named fidelity, experts retain fitting BFP4+LoFi, and activation/residual/CCL/KV/logits/sampling remain BF16. After the GDN pair, test mixed best-fidelity policies with shared BFP4 and optionally QSA BFP4, preserving first/last exceptions. No candidate is selected yet.

### CPU selected-default construction audit

`default_construction_audit.md` records exact final selected-file/runtime-policy/cache/planner assertions and distinguishes private validation/resident benchmark evidence from public shared-pool planning. Pure-host arithmetic found a concrete conditional issue: two expert-layer BFP8 upgrades add629145600B/rank and cause the current conservative public full262144/B1 planner to reject (7100 blocks available versus8194 needed), despite successful private-cache construction. A winning policy in that family requires evidenced planner resolution; no context reduction is justified by this conservative rejection. Inherited and tested memory-saving projection policies admit the full profile and return262176 pre-worker tokens including the null page. No selected artifact existed at audit time. Final validation/resident harnesses consume defaults and record actual policy, but do not call the public planner or finish qualitative content review. No runtime/config edits, TTNN imports or devices; no vLLM adapter claim.

### Final default-construction proof instrumentation

Added `tests/selected_precision_snapshot.py`, called by datatype validation after construction and resident benchmark outside timed windows when a selected artifact exists. It asserts full dictionary equality, full262144/B1 capacity,12distinct K/V pairs at8192x1x32x256 with selected dtype, and public cache-budget agreement with the static-weight-aware pure planner. It records selected-file SHA256, cache spec and planner ledger. These harness-only changes do not change TT numerical/runtime source or invalidate P1 score caches. Applicable pre-commit hooks passed (`logs/selected_snapshot_hooks_v1.log`). The queue now accepts `--repeats`; combined policies will use2 warmed serial repeats.

### Readable Pareto charts for the full candidate matrix

Updated only the plotting implementation in `tests/report_datatype_sweep.py`; numerical gates, eligibility, ranking, frontier calculation and JSON/CSV measurements remain unchanged. `pareto_figure(records, metric, threshold, selected)` now separates short stable point IDs from full policy names in a speed-ordered side key. The key includes every record, full config ID, pass/fail/pending state, accuracy and exact displayed timing; missing-timing attempts remain listed. Greedy label placement avoids overlapping ID boxes and uses leader lines without moving measured coordinates. The selected point is red and painted last. The dotted accuracy threshold and measured frontier remain in the overview. A large slow-outlier gap automatically adds a clearly labeled detail inset for the dense upper timing band, preserving the full overview and all points. Figure height grows with the key rather than crowding 30 labels onto one axis.

CPU rendering used a temporary synthetic 30-policy fixture with 28 policies at top-5=100%, near-identical timings, exact coincident measurements, a slow rejected policy and one attempt lacking timing. A second case places all 29 measured policies at exactly the same coordinates. Both top-1/top-5 renders passed geometry checks: all 29 unique ID labels present, all 30 key entries present, zero numeric-label or key-text overlaps, measured scatter coordinates unchanged, selected marker red, and threshold/frontier preserved. Dense previews were inspected locally after placing the inset below overview markers; readable names/status/values are in the side key. Synthetic previews and fixture are under `/tmp/qwen_datatype_chart_8718svju/`, not production chart paths. These fixtures are not measured-model results.

Evidence: `chart_readability_audit_v1.json`, `logs/chart_readability_fixture_v1.log`, `logs/chart_readability_tests_v1.log` and `logs/chart_readability_hooks_v1.log`. All 21 existing CPU reporter tests pass, including the temporary full CLI/JSON/CSV/PNG test; test and fixture runs actively reject TTNN imports. `python_env/bin/pre-commit run --files models/autoports/qwen_qwen3_8_flash_next/tests/report_datatype_sweep.py` passed, exit 0. No final sweep results/charts were overwritten, no runtime/config changes and no devices.

### Combined-policy matrix

Prepared five complete policies after coarse group timings: `combined8_lofi`, `combined8_hifi2`, `combined8_mixed`, `combined_shared4_mixed`, `combined_shared4_qsa4_mixed`. Mixed uses GR/GDN ZAB LoFi and head/shared/QSA HiFi2; the shared4 variant restores layers0/47 shared projections to BFP8+HiFi4, and the additional QSA4 variant restores layer47 QSA to BFP8+HiFi4. Every expert remains BFP4+LoFi. The three persistent-weight savings versus inherited are540057600,579624960 and651714560B/rank. All keep BF16 activation/residual/CCL/KV/logits/sampling and full262144/B1. Each combined policy will run both references with two warmed repeats through explicit build-kwargs files. Policy source was validated with `load_policy`; this is a search plan, not a selection claim.

### Cross-process repeat aggregation for final ranking

`tests/report_datatype_sweep.py` now accepts multiple completed input artifacts for one config ID and emits one policy row/point. It pools every valid serial AIME teacher-forcing timing sample, including each process's resident repeats, rather than averaging process means. Baseline noise is the full min/max spread across all baseline timing samples supplied. The singleton workflow retains its measurement/gating behavior, including failed-attempt reporting.

Repeated inputs must have identical requested precision policy, hardware, mesh, complete environment dictionary, measurement regime, capacity, construction kwargs apart from the policy-file path, and reference/path coverage with identical scored counts. Each process must declare serial measurement, complete/pass, and contain complete local repeats with passing verified rows and clean teacher-forcing replay counters (no captures/releases, positive model replays). Failed/incomplete artifacts, failed rows, changed regimes/policies, partial references/repeats, duplicated input paths and byte-equivalent parsed reports are refused rather than silently omitted or averaged into success. A present provenance sidecar reporting unsuccessful exit or unfinished status also blocks merging.

`evidence_paths` and `source_measurements` preserve every source path, command, git head, build kwargs, environment, hardware/mesh/regime, execution outcome, provenance path and recorded provenance execution metadata. Every original per-run field, score row and counter is retained in `references`, tagged with its source artifact and source-local run index. The legacy singular `evidence`/`command` fields retain the first input for compatibility; the plural provenance fields are authoritative for aggregated runs. Dominance pruning must cite every constituent artifact of both compared policies. Citations support absolute paths, checkout-relative paths and unambiguous paths relative to the P1 decision file; ambiguous relative citations fail.

Usage is the existing reporter command with every fresh and retained compatible measurement included under `--inputs`; no special merge flag or separate mean file is needed. Accuracy/P1/capability gates remain policy-specific and unchanged. Process provenance remains necessary evidence of isolation; pooling does not manufacture proof of hardware exclusivity.

Validation: all 26 CPU reporter tests pass with TTNN imports actively forbidden. The new full CLI fixture merges two synthetic processes with timing samples `[10,12]` and `[14]`, verifies pooled mean12 (not process-mean12.5), baseline spread4, one policy row, all three samples, intact commands/provenance/rows, CSV and both PNGs in a temporary directory. Focused refusal tests cover mismatched metadata/policy/coverage, failed/incomplete/untraced rows, failed or unfinished provenance, missing replay counters, partial local repeats, copied artifacts and incomplete dominance citations. A read-only check of actual `inherited_e4_lofi_v3.json` confirms every original run field is unchanged by singleton loading. Evidence: `repeat_aggregation_audit_v1.json`, `logs/repeat_aggregation_tests_v1.log`, and successful targeted hooks in `logs/repeat_aggregation_hooks_v2.log`. No production sweep results/charts overwritten, runtime changes or device access.

### First three combined policies completed

All eight accuracy rows (two repeats times two references times prefill/traced teacher forcing) pass for each policy. AIME traced means: combined8_lofi9.7706457642, combined8_hifi2 9.7726328830, combined8_mixed9.7660101678 t/s/u. Within-process spreads are0.0043391186,0.0034498629,0.0020644747 respectively, larger than the initial inherited baseline0.0004576979. No winner is selected; cross-process baseline/finalist repeats and P1 are planned. Individual group gains did not add: the mixed policy is slower than either uniform policy in these runs. Two further combined BFP4 policies are running/queued.

### User cap: search frozen

User instructed: "Stage time/token cap reached. Keep the best version that passes the required gates, run only the remaining required gates once, write untried ideas as hand-off notes in the stage work log, and complete the stage (pass-with-carried is acceptable for non-blocking items)." Set STOP_QUEUE to finish the active combined_shared4_mixed job and prevent the queued combined_shared4_qsa4_mixed search. No further cross-process performance search or adaptive precision combinations will run. Remaining work is P1 qualification, selected default/token-out/context/resume/qualitative validation, report/gate/review, and local commits. Untried combinations and additional noise characterization are hand-off items under this superseding cap.

### Search stopped; final P1 qualification

Active `combined_shared4_mixed_v1` completed all eight numerical rows, mean AIME traced9.7768296659t/s/u (repeats9.7769403501 and9.7767189818). Queue observed STOP_QUEUE and closed normally; combined_shared4_qsa4_mixed was not run. The fastest measured policy exceeds the next policy by0.0041967830t/s/u, above the accepted initial baseline spread0.0004576979. Under the user cap, final P1 tests this fastest candidate once against the cached e8_two_hifi4 reference, with originalHF comparison enabled; slower policies need no additional P1 scoring if this candidate qualifies. `p1_candidates_final.json`, `p1_final_v1.provenance.json`, `logs/p1_final_v1.log` preserve the invocation. Final selected artifact remains unwritten until qualification.

### Search freeze and final documentation scaffold

The user's stage time/token cap freezes new precision search. The active `combined_shared4_mixed` process may finish normally; the queued `combined_shared4_qsa4_mixed` experiment is stopped before launch. README now records the completed 19-policy coarse matrix and the first three completed combined policies, their actual traced means/repeat ranges and all-reference accuracy minima. It explicitly keeps selection, finalist P1, selected-default runtime evidence, capability/qualitative/context checks, post-selection token-out, final artifacts, independent review and local commit pending. No acceptance or selected-policy claim was made. This documentation update launched no tests or experiments.

## Carried to next stage

These are unperformed optimization/noise-investigation ideas under the user's cap, not failed gates, measured regressions or waivers of required finalization work:

- **Unrun `combined_shared4_qsa4_mixed`:** combines interior shared/QSA BFP4 changes with the BFP8 projection/head adjustments. Its config is preparatory only; no numerical, timing, P1 or capability result is claimed. Preserve it as an untried follow-up rather than fabricating a rejected/accepted point.
- **Adaptive combinations and further per-group/layer tuning:** additional combinations inferred from coarse wins, alternative fidelity mixes, and sensitivity-driven layer exceptions are not explored after the freeze. Completed LoFi/HiFi2 pairs and current combinations supply the measured candidate set; optional search expansion must not displace required selected-config gates.
- **Additional cross-process repeats and extended noise characterization:** the combined LoFi run's observed within-process range is 0.004339118634 t/s/u, versus 0.000457697911 for the initial baseline. More fresh baseline/finalist repetitions would better characterize close-policy ordering, thermal/host variation and process-to-process spread. Those extra experiments are deferred, not treated as measurements. The reporter is ready to pool compatible genuine repeats while retaining all sources if a later stage collects them.

The parent must still complete the required gates once and report any unresolved blocking accuracy/correctness or missing-gate finding. Nonblocking carried work may accompany the permitted review outcome; this handoff list itself is not a stage-review verdict. No vLLM integration has been started by this stage.

### Selected config and final P1 pass

Selected `combined_shared4_mixed`, the fastest observed fully accurate candidate at9.7768296659 traced AIME t/s/u. All four final P1 runner checks pass:3435positions, ΔNLL0.00212≤0.004,KL×1000=2.47593≤5,originalHFreferenceKL×1000=3.62242≤4,and selected traceddecode top1=0.99023 versus reference0.98828 over512positions. No loss waiver. Canonical `doc/eval_readiness/precision_reasoning.json` and preserved `p1_final_v1.json` name the shipped ID. Copied the complete measured policy to `selected_precision_config.json`; normal constructors consume this artifact. Frozen SHA recorded in `selected_artifact_before_final_runs.json`. Remaining finaldefault capability/tokenout/qualitative checks are pending.

### Final readiness runner gate

The first final stage gate (`logs/stage_gate_v1.log`) passed all four precision checks but rejected the historical limitations registry because it lacked the runner's `items` list. Added an empty runner waiver list while preserving every historical anomaly under `limitations`; no gate waiver is requested. The corrected invocation passed (`logs/stage_gate_v2.log`). Command: `MODEL_DIR=models/autoports/qwen_qwen3_8_flash_next bash $TT_MODEL_BRINGUP_ROOT/prompts/model_bringup_multigoal/08-datatype-sweep.check.sh`.

### Post-selection default token-out benchmark

`post_selection_token_out_v1.json` completed successfully (exit0; 282.664 s). The normal constructor loaded selected SHA `68dbcc6ca583ac3a97770654f95a81222dc186e269fa32d04dbc6be3e015105d`; actual weight/fidelity policy, BF16 KV tensors, public admission plan and 262144 capacity all matched. Runtime sources remained unchanged throughout measurement.

The stage7-equivalent warmed B1 128-input/128-output no-readback window measured **9.801183224430 token/s/user**, 127 model+sampling replays, with no timed token uploads/readbacks/captures/releases. Public `decode_forward` measured 9.803960292545 token/s/user separately. Warmed TTFT was 4711.342390 ms and includes setup/reset, full prefill, first sample and caller read. The 9.801183224430 token/s/user resident figure is the post-selection reporting/comparison baseline; it is separate from the 9.776829665936 traced teacher-forcing selection mean.

Exact command/environment/source hashes: `post_selection_token_out_v1.provenance.json`; raw local log: `logs/post_selection_token_out_v1.log`. Final selected default accuracy,257-token continuation,66 chat outputs and32769-token smoke now run once in `final_validation_v1` under trace allocation tracking.

### Selected shared-suite blind comparison and bounded limitations

All66selectedTT outputs completed; the fresh no-history judge `/root/final_blind_judge` read only the frozen label-hidden120sample packet, rubric and rating template. After its ratings were saved, the parent joined labels with the existing comparator. Every prompt q01–q06 has0/10TT and0/10HF failures under the unchanged rubric; `final_validation_v1/qualitative/blind_comparison.json` is `within_matched_control_margin`. The six matched greedy pairs were directly inspected in `greedy_review.md`. The pinnedHF control was reused with unchanged checkpoint/template/settings and byte-identical copies recorded by the final driver.

The blind judge separately flagged unsupported call/instruction remarks. `qualitative_anomaly_audit.md/.json` finds6/60selectedTT,4/60previous optimizedTT and0/60HF occurrences, including q02seed6 asserting an absent MUST-call instruction. This is not an HF-matched known limitation. Source inspection found no concrete prompt concatenation or missing reset bug; origin is unproven and runtime defects are not ruled out. The remarks remain on-topic, do not replace the answer under the frozen rubric, and the class predates selection. Carry fresh-process/reordered-request controls and broader factual/form checking under the user cap, subject to independent stage review. The new limitations registry section preserves this evidence without requesting any runner waiver. Expected thinking/EOS markers and both-backend verbosity remarks are separately classified.

Exact artifacts: `final_validation_v1/qualitative/{blind_packet.json,blind_packet_status.json,blind_ratings.json,blind_judge_notes.md,blind_comparison.json,greedy_review.md,qualitative_anomaly_audit.md,qualitative_anomaly_audit.json}`. No rubric/scorer/outputs were changed after blind scoring. Context smoke is still running.

### Owner resume: final smoke completed

The owner resumed stage 8 with no further precision search. The existing validation child had completed all phases before resume: `final_validation_v1.json` passed; the 32769-token smoke passed in1248.605318s (1248.722308s including its extra validation replay), with capacity262144 unchanged,1025 prefill replays, finite terminal logits, no new capture/release and correct positions32770/32771. The runtime audit and selected-default snapshot passed. The complete validation took2618.874349s.

The interrupted wrapper left its provenance status at running and did not capture a return code. `final_validation_v1.completion_recovery.json` preserves this limitation and records the semantic pass, VALIDATION_DONE log marker, orderly UMD close, absent job processes and no device holders. No exit0 is invented, and the valid completed smoke is not rerun. Final tables/docs and independent stage review follow.

Final health listing (`timeout 60 tt-smi -ls --local`) returned0 and showed all four Blackhole p300c chips; no holders or reset. `final_device_health.json` records the compact result. Final reports contain26 evaluated configs; selected policy eligible/pass. Preserved the immutable natural reference text in commit hooks with one exact end-of-file-fixer exclusion in `.pre-commit-config.yaml`; adding a newline would alter its recorded content. No runtime source changed.

Final commit hooks initially rejected trailing spaces in a raw device-list log and four bounded evidence files above500KB. Restored the raw log byte-for-byte from the index and kept it local-only with other raw logs. Added an exact four-path size-check exclusion for the required pyplot charts and full-model KV runtime JSONs (all below650 KiB), preserving the numerical evidence instead of rewriting it. No global size limit or numerical gate changed. Tensor score caches remain local under existing repository ignores; canonical P1 and compact audit/result JSON are committed.

Final pre-commit hooks passed (`logs/final_hooks_v3.log`); no runtime or test source changed. Whitespace check passed with `core.whitespace=cr-at-eol` for the intentionally preserved book-reference CRLF. Both actual Pareto PNGs were visually inspected: selected red point, dotted accuracy threshold, measured frontier, complete policy key and explicit no-timing failure. `final_checks.json` records these checks.


### Independent stage review and closure

Fresh xhigh reviewer `/root/stage8_review_final` returned **pass-with-carried**, no blocking Required Work, in [stage_review_v1.md](stage_review_v1.md). The explicit owner cap/resume permits nonblocking carry without spending additional review rounds. Review independently checked the full matrix, selected P1/default propagation, plots, source hashes, qualitative outputs, continuation and completed32769-token smoke. The interrupted wrapper exit remains unknown; complete driver output and orderly device close establish semantic completion without inventing exit0.

The clean post-selection token-out timing used allocation tracking off. Unchanged-source candidate/final-default validation with tracking on and program-cache skipping disabled supplies separate correctness evidence. Preserve that distinction with the9.801183224430t/s/u reporting baseline. The four hook-exempt evidence files are below650 KiB, not650000decimalbytes; exact sizes are in the review.

#### Carried to next stage — final review disposition

- Unsupported process/call remarks remain unresolved: selectedTT6/60, priorTT4/60, HF0/60. The bounded frozen-rubric gate passes; investigate fresh-process/request-order/state controls later. No HF-equivalence, leakage diagnosis or repair is asserted.
- The128-token suite does not validate complete long answers, runnable code or broad factual/form quality. Preserve direct qualitative notes and the unchanged rubric.
- Close-policy ordering is observed, not statistically established; additional cross-process repeats and the already-listed untried precision ideas remain deferred. No further precision search ran.
- Future integration must consume the selected artifact and use the post-selection token-out baseline; vLLM construction/serving remain outside stage8. KV8 stays rejected.
- Preserve the unknown wrapper exit, local-only raw artifacts, conservative memory residual accounting, and separate trace-tracking/timing regimes. Platform/packet tuning and exact98304-byte residual attribution remain optional follow-up.

All required stage8 evidence is complete. Local checkpoint SHAs follow below; no push.

### Local checkpoint

Repository `/home/ttuser/qwen38-flash/tt-metal`, branch `main`: stage8 implementation/evidence checkpoint **`fe54653c95baf9cb13e7dac080c58c4446fce651`** (`Select validated Qwen3.8 full-model precision policy`). All commit hooks passed. This follow-up records the checkpoint and final closure metadata; no push. Unrelated prior-stage untracked artifacts were left untouched.
