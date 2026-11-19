# Optimized decoder work log

Stage start: 2026-11-13 15:42:38 UTC. Base commit 3cd9937d. Scope: optimized_decoder.py plus tests/docs. Chip0 serialized; four Blackhole p300c devices listed, no holders. No resets. Enabled tt-autodebug verified in Codex config; installed environment.py setup validated. Existing untracked fused artifacts preserved.

## Initial operation-topology audit (before tuning)

The completed fused per-op reports are the measured baseline (`../fused_decoder/per_op_findings.md`, six `tracy/wrap_final_ops_layer{0,1,3}_{prefill,decode}/report` directories). Decode sparse gate/up+down costs 2.039/2.034/2.038 ms (41/32/22 percent of kernels). Baseline real-weight dtype is BF16 for GDN/QSA experts and BFP8 for PLE, all HiFi4. This stage evaluates BFP8 as default plus BFP4/LoFi candidates.

| Current ops | Candidate | Constraint / initial action |
|---|---|---|
| Packed indexed sparse gate/up40 cores, down8, K-block1/subblock1 | Larger K blocks, more down cores, larger subblocks, HiFi2/LoFi | Preserve exact routing, active10 decode and variable prefill union; first sweep |
| Packed shared gate/up/route and GDN z/a/b | DRAM-sharded reader geometry and tuned split control | BF16 sensitive operands; measure whole group including movement |
| Packed QSA QKV + precise indexer | DRAM sharding; maintain ordered indexer arithmetic | Existing whole-output accuracy contract retained |
| DRAM expert input/intermediates | L1 input/output placements | Full resident PLE memory and physical sparse shapes must fit |
| Exact GR read, precise products/reductions, exact residual writer | Carry width-sharded layout across consumers | Exact BF16 boundaries and sensitive router ties; adaptation needed |
| Paged native SDPA plus custom selection | Reduced KV dtype candidate | Existing BF16 state/context retained unless candidate validated at max |
| 32-row physical chunks | Large prefill config / chunk opportunities | Logical non-aligned tails stay public; no new alignment restriction |

No candidate accepted yet. Inherited packed topology must be compared with tuned separate controls. CCL/LM-head/token-feedback work is inapplicable to this single-device decoder stage.

## Initial sweeps

`initial_sweep` failed before device open because TT_MESH_GRAPH_DESC_PATH was missing; restored the predecessor p150 descriptor in run_optimized_job.py. No reset. initial_sweep_v2: BFP8 baseline5.0155ms, down40/block10/subblock2=3.7385ms; geometry_v1 combined gate40/block20 down40/block10/subblock2=3.1547ms (37.1% gain), prefill78.28→58.84ms. Real checkpoint; same input/state; bitwise equal HiFi4 candidate. HiFi2/LoFi3.1495/3.1482ms are within sub-percent spread, so precision selection remains open. Geometry scans include larger divisors through gate80 and down20 and 8/20/40/80 down cores.

L1-output experiment reached a DRAM-only Python guard in fused_packed_silu. This is not a rejection: optimized helper now accepts either interleaved buffer type, using the existing TensorAccessor metadata for both operands. Retrying this adapted consumer. Sparse indexed mode already obtains exact10/20 loop count from indices; nnz is forbidden with indices by sparse_matmul_device_operation.cpp:399 and is left unset. Prefill unions have variable exact counts, so no constant nnz is asserted.

## Dense and residual audit

Completed dram_plan: real GDN/shared/PLE matrix shapes, BF16 and BFP8, packed versus legal separate, interleaved versus DRAM-sharded,1/2/3 readers and input-core/K-block sweeps. `dense_component_summary.json` retains best controls; full JSON retains all candidates/errors. Some wide BF16 large-block configurations exceed L1; smaller-block/core alternatives in the same family measured successfully. These failures do not reject the family. GDN BFP8 packed58.19us vs separate65.01us; shared packed30.10us vs separate41.63us; PLE separate93.68us vs packed96.34us, so PLE remains separate. Component input distribution is seeded representative BF16; subsequent actual full-decoder HF runs, not component PCC alone, decide precision.

`integrated_plan` runs dense8 and DRAM-sharded implementations on real HF GDN/PLE/QSA33. All six pass with tracked replay. Shared, attention, and PLE ordinary matrix weights are BFP8 candidates; exact residual/router/GDN QKV/output and ordered-indexer arithmetic retain inherited high-precision representations.

`residual_v1` compares real-weight read→write→next-read contracts. L1 and width-sharded L1 retain exact output and replay behavior at1/32rows. The sharded consumer was explicitly adapted to L1 interleaved normalization and bounded DRAM projection workspace; no immediate restoration of the stored residual. Decode DRAM1.342ms, L1 1.295ms, width-sharded1.301ms, width-sharded+DRAM consumer1.350ms. Integrated residual ranking includes external admission/return costs; no component-only selection.

Clean ranking so far: sparse/L1 3.136ms; denseBFP8 3.1166ms; integrated DRAM-sharded3.0776ms for GDN32. Baseline rows bracket each selected variant. Final selection and same-harness default rerun remain pending.

## Cumulative correctness investigation (2026-11-13)

The initial integrated default failed synthetic GDN length1 at traced step1 (whole PCC0.9947856), GDN tuple batch32 request3 decode (0.9927617), GDN fresh97-row live-prefill diagnostic row (0.9895512; whole0.9999657), and PLEbatch5 row2 prefill contribution diagnostic (0.9949988; whole0.9981965). Raw failed results remain failed; no acceptance bar was changed. Literal HF0 lengths1/65, HF1 length33, HF3 lengths1/2/31/32/33/65, PLE/QSA continuation and QSA32-request batch passed. The real single-token GDN check gives prefill0.9999703/decode minimum0.9999431; it does not by itself resolve other failures.

`precision_failure_controls_plan.json` and `layout_failure_controls_plan.json` isolate the GDN accuracy problem. HiFi4 experts, BF16 shared weights, DRAM residual storage, and disabling expert L1 each leave request3 decode near0.99276. BF16 GDN z/a/b with DRAM HiFi2 also fails (0.992799); BFP8 with ordinary HiFi4 matmul fails (0.992738). BF16 GDN z/a/b plus DRAM HiFi4 passes the unchanged batch32/backend-control gate, request3 decode0.9998763. BF16 separate interleaved GDN arithmetic gives the same0.9998763. Thus both z/a/b weight precision and fidelity need preservation on this real model fixture. This is a real-weight rejection, not a synthetic-only veto. The faster BFP8 GDN candidates are not eligible for final selection. The rest of the BFP8/LoFi policy remains under validation.

The batch control switches match the completed fused stage: `--gr-backend-control` for layer0 and `--ple-quantization-control` for layer1. These classify only inherited fresh-prefill contribution differences; literal whole-output and decode gates remain0.995. Initial optimized runs omitted those switches, so the PLE tiny contribution miss will be rerun with its independent packed-lookup HF control. Fused BFP8 control reproduces the inherited strict fresh-prefill failure but request3 decode passes0.9999832, separating it from the optimized GDN regression.

AutoDebug launched inspection-only via the installed launcher. Container namespace preflight failed before model start; existing unrestricted authorization permits the documented skip. The configured0.147 CLI lacked its matching code-mode host. A local host symlink exposed an IPC-version mismatch, so the successful investigation uses `/usr/local/bin/codex`0.153 with its packaged host. No repository source or device state was changed by those failed launches, and the incompatible inspection process was stopped. Hardware jobs remained serialized, with no reset.

## Optimization stop rule

Tuning stopped after the last two retained structural candidates each gained less than3%: GDN DRAM integration3.1166→3.0776ms, then L1 residual3.0776→3.0364ms (rank_plan). QSA SDPA refinement is also less than3%. Accuracy repair and required final evidence continue; stop rule does not waive gates. Final eligible-default ranking is pending.

### Not tried (stop rule)

- Reduced KV-cache dtype: state remains BF16 and custom selected-page consumers are BF16-specific. No reduction is claimed rejected without an implementation. KV layout/dtype is unchanged, so only32K–64K smoke is required.
- Larger physical prefill chunks and large-M 2D matmuls: the inherited exact recurrence/indexer/sparse contracts use physical chunks at most32. Public arbitrary logical lengths remain supported. A wider-chunk redesign is deferred, not a public alignment restriction.
- Exhaustive shared-down/attention-output DRAM geometry, activation precision, FP32-destination/packer combinations, and a fully sharded normalization implementation beyond the measured interleaved consumer. Expected incremental gain is unproven.
- CCL, row-parallel multi-device output, LM head, sampling, token feedback and serving are absent from this isolated single-device decoder stage. No full-model TTFT/throughput claim is made.

## Final precision repair and cumulative validation

AutoDebug's source inspection and AutoFix experiments are in `autofix_accuracy/`. The final source resolves GDN attention_dtype=None to BF16 and GDN DRAM matmul fidelity to HiFi4; QSA resolves to BFP8/HiFi2. This fixes real-weight batch/continuation failures as well as the synthetic CI regression. No quality threshold was changed to repair these failures. Final synthetic test suite:14/14; real-checkpoint CI lengths1,2,31,33,34,65,66 all pass; real GDN128prefill/128decode repeated stress passes with minimum decode PCC0.9998453. Real-weight inputs are seeded Gaussian fixtures, not claimed recorded natural-language activations. An optional embedding-input attempt failed before device open because the local snapshot lacks tokenizer/embedding payload; the installed-checkpoint tests use the established HF fixture.

Matched fused/optimized HF PCC appears in `final_accuracy_summary.json`. GDN65 prefill0.9999890→0.9999838 and decode minimum0.9999908→0.9999780. PLE33 prefill0.9981529→0.9981104 and decode0.9978529→0.9978223. QSA six lengths1/2/31/32/33/65 all retain whole-output>=0.995; BFP8 ordinary attention/shared weights account for the small change from the predecessor's BF16 policy. Final batch limits GDN32/QSA32/PLE5 pass. GDN request16 and PLE fresh continuation contribution differences use the predecessor's independently matched backend controls, recorded in `../eval_readiness/limitations.json`; new real-weight GDN regressions were fixed rather than classified as inherited.

All32768 context smokes pass with final default construction. They exercise reduced prefix components (GDN recurrence; PLE lookup/hash plus recurrence; QSA attention) and complete terminal public decode where supported by that probe. They are not full-MoE32768-token end-to-end timing claims. KV remains BF16 and paged layout/capacity262144 unchanged; no new maximum is advertised. The full resident PLE envelope remains4768.

### PLE4768 diagnostic disposition

`final_ple_capacity4768.json` deliberately retains its legacy `passed:false`: one residual-subtracted prefill row has PCC0.9948805. Literal whole-output global PCC0.9980767 and every whole-output row>=0.9951833 pass; all three decode steps and reset/replay/runtime audits pass. The independently saved HF packed-lookup control is bound to the exact device output identity and literal reference; TT versus that control passes all rows (minimum whole0.9985678, contribution0.9985477). Literal HF itself does not fail here: row7 HF-packed versus literal contribution0.9951070, versus TT0.9948805. This is documented as an explained diagnostic, not an original-HF failure. `assess_optimized_capacity.py` independently assesses the user's explicit whole-output contract, enforces diagnostic floor0.99, strict decode and determinism, and writes `final_ple_capacity4768_contract.json`; it never changes the legacy result.4767 additionally passes the original strict probe. `ple4768_row7_diagnostic.json` retains the exact comparison. No capacity is reduced.

### Profiler advice and remaining scope

Final reports verify BFP8/LoFi expert rows, BF16/HiFi4 GDN packed projection, BFP8/HiFi2 shared/QSA/PLE ordinary projections, exact active10 decode and actual memory/program settings. Gate/down geometry, L1 inputs/intermediates, fidelity and sharding advice were exercised. BFP4 gate/up LoFi measured77.827us versus default105.719us; BFP4 down39.156us versus default49.747us in separate representative captures. These are candidate kernel measurements, not final full-decoder BFP4 deployment claims. Stage8 owns the final precision choice.

Additional untried items under stop rule: full-prefill expert intermediates in L1; shared-down L1/HiFi2/subblock variants; wider or fused exact-GR prefill reductions; BFP4 attention projections; exhaustive other dense-role fidelity and geometry. Expected gains are unproven. Exact-GR binary/reduction work now dominates prefill; fully optimizing it would require a new exact arithmetic implementation. Existing required row-major router/indexer and DRAM-sharding conversions remain included in timings. No zero-conversion or full-model roofline claim is made. Some generic report FLOP percentages exceed100 because physical tile/core accounting does not represent useful decoder FLOPs; actual kernel times and active traffic are used.

The serialized final queue was paused between jobs to separate split-expert build inputs from output filenames. No device process was interrupted. `final_evidence_remaining_plan_progress.json` records that boundary; `final_evidence_finish_plan.json` resumes only unfinished jobs. The final BFP8/LoFi geometry sweep repeats the material gate/down choices under the selected precision, avoiding a comparison based solely on earlier HiFi4 geometry. No reset has been needed.

## Final comparisons and long-QSA control

All25 final baseline/default/control timing rows are in `final_performance_summary.json`; the complete short/boundary/long table is `performance_results.md`.32-token fused→optimized prefill/decode milliseconds: GDN78.284/5.019→58.806/3.052; PLE95.877/6.349→63.110/4.248; QSA184.062/9.526→149.206/7.296.1024-token results preserve the improvement; see the exact table. All measurements are serialized chip0, real checkpoint, warmed; traced decode includes host submission/synchronization, prefill is eager whole-call latency.

The final27-geometry BFP8/LoFi sweep passes. Its fastest single observation3.05136ms differs negligibly from the selected final default3.05184ms. Alternating the selected default and down-Kblock20 three times with64replays (`final_geometry_tie_repeat.json`) yields medians3.052530/3.052430ms and overlapping samples; both directions occur. The existing default is retained as a measurement tie, not claimed significantly faster than the tied knob. The final separate-expert12-geometry family gives3.07663ms, and the native fused-SiLU family3.07589ms. The legal separate winner passes HF65 with identical final selected PCC. Native fused-SiLU fails real HF decode0.992139 and is rejected; it is also slower than the packed selected path. No first-API-error rejection is involved.

`final_perf_optimized_3_1024.json` exits1 because its legacy same-construction timing comparison gates every row at0.995: whole-row minimum0.993039, contribution minimum0.992520, at rows572/1009. All decoded outputs/state replay exactly and floating prefix-state buffers have PCC1 against the control. Follow-up `qsa_long_diagnostic.json` saves exact inputs/outputs and adds whole-global metrics without changing the strict predicate. Same experts with SDPA Kchunk32 pass the strict local rows; same Kchunk128 with expert HiFi4 still misses. The selected whole-global comparison is0.999936.

Independent CPU-only full real-weight HF evaluation of those exact saved inputs (`qsa_long_hf_control.json`) passes the user contract: prefill0.99960095, minimum whole row0.99242079, minimum contribution row0.99196941, decode minimum0.99980420. Every compared variant passes the whole-output/diagnostic-floor contract, including conservative SDPA32. `qsa_long_assessment.json` verifies exact reproduction of every saved timing prefill diagnostic, prefix-state PCC and overlapping decode metrics; no raw failed status changes. This is a controlled per-row diagnostic under the explicit owner contract, not a literal-HF-failure claim. Token top1 is inapplicable to these hidden streams. The limitations registry records the exact scope. The independent HF job is CPU-only and ran after hardware timing stopped, avoiding timing contention.

Final plans intentionally preserve failures: `final_controls_plan_progress.json` includes the expected strict QSA diagnostic exit1; `final_hf_controls_plan_progress.json` stops on the rejected optional fused-SiLU candidate; `final_tie_plan_progress.json` completes the remaining confirmation. These scheduler statuses are not conflated with selected-path acceptance. The selected implementation did not change during these controls; all earlier final optimized correctness/watcher evidence remains current.

## Review round 1 closure: two-row Watcher coverage

Round 1 (`stage_review_round1.md`) required only the missing physical two-row compact-expert Watcher path. `review_watcher_tail2_plan.json` runs final-default public prefill34 (32+2) and three changing traced decode inputs for real-weight GDN, resident PLE and QSA. All use Watcher10, allocation tracking1 including program-cache allocations, profiler disabled. Every watcher log detaches cleanly with no error; reset/replay checks pass. GDN/QSA raw probes exit0. Implementation is unchanged from final timing and correctness evidence.

PLE's legacy strict probe exits1 solely for prefill row5 (in the first32-row chunk): whole/contribution row PCC0.99462019/0.99381730. Whole-output global PCC0.99813887 and contribution global0.99800983 pass; all three decode steps pass0.995, as do exact prefill reset, replay/state equality and runtime audits. The same seeded fixture through the fused predecessor (`review_tail2_control_plan.json`) reproduces row5 at0.99446413/0.99364443, with global0.99820047/0.99807578. The optimized row diagnostic improves slightly. This establishes inherited sensitivity, not a claim that literal HF fails. The user's whole-output bar and0.99 diagnostic floor are unchanged; token top1 is inapplicable to decoder hidden streams. Both raw failed statuses are preserved.

Commands: `python -m models.autoports.qwen_qwen3_8_flash_next.tests.run_optimized_plan --plan models/autoports/qwen_qwen3_8_flash_next/doc/optimized_decoder/review_watcher_tail2_plan.json`, followed by the same runner with `review_tail2_control_plan.json`; exact environments and commands are in each provenance sidecar. CPU-only `python -m models.autoports.qwen_qwen3_8_flash_next.tests.assess_optimized_watcher` verifies source identity, tracked watcher environments, raw-log hashes/clean detach, matched fused diagnostics and the explicit acceptance predicates, writing `review_watcher_tail2_summary.json` (pass). Logs remain unmodified under `logs/` and `watcher/`.

Metadata notes from review: real-CI properties inherit a synthetic-fixture label, but the runner loads the real checkpoint recorded at report/checkpoint; earlier QSA expert labels can retain BF16 while selection sidecars and profiler rows establish BFP8. The allocation-tracker positive control is linked directly from `acceptance_manifest.json`. No gate or runtime policy changed during review closure.

## Independent acceptance and checkpoint

Fresh round2 inspection returned **clean-pass**, with no Required Work (`stage_review_round2.md`). It independently closed the two-row compact-expert Watcher coverage finding and classified the PLE34 diagnostic against its matched fused control. Round1's broader accuracy/performance/context conclusions remain accepted. No runtime implementation changed during review follow-up. Pre-commit checks pass on all stage-owned staged paths; final log `logs/final_checkpoint_precommit.log`.

Stop-rule handoffs remain under `Not tried (stop rule)` and the later profiler handoff paragraph; they are not failed gates. No multichip/full-model/vLLM work was begun. Only this tt-metal checkout contains stage-owned changes. Local commits follow this acceptance; nothing is pushed.

Stage checkpoint: repo `/home/ttuser/qwen38-flash/tt-metal`, branch `main`, commit `acdff236c6e336109c39dcb1ba20531e2192f48d`. This commit contains the optimized runtime, tests, evidence and passing reviews. A documentation-only follow-up records this SHA; no push was run. Pre-existing untracked fused-stage raw artifacts remain untouched.
