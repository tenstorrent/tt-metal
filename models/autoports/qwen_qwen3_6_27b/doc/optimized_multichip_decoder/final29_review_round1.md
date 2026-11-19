# Stage Review

Verdict: more-work-needed

Independent round **1 of 3**, stage 5 `optimized-multichip-decoder`, Qwen/Qwen3.6-27B, live checkout on `main` at HEAD `8502f42bca20dcad1507fbc8cebf7032220b7d8e`. The final decoder SHA256 is `2d4ba9dd20c86f6ae9d710c5fe79c259bdd66fe6007b6af4be5437ca01554c0b`.

No blocking numerical, trace, watcher, runtime-fallback, context, or inter-layer-layout defect was found in the inspected final29 evidence. The remaining substantive concern is **nonblocking B1 prefill performance uncertainty**, together with a source-explained profiler annotation limitation. I classify neither as a demonstrated model regression nor as a waived numerical gate. The user's cap stops further optimization and optional timing. The installed review skill permits `pass-with-carried` only in round 3 and says that before that round unresolved carry items yield `more-work-needed`; this report therefore does not issue an early conditional pass. This is review closure work, not an instruction to restart device search or mark the goal blocked.

Paths below are relative to `models/autoports/qwen_qwen3_6_27b/`; **D** means `doc/optimized_multichip_decoder/`.

## Required Work

- **None for model correctness, accuracy, or missing required device evidence.** No new device run, model change, relaxed PCC threshold, or replacement of a failed result is requested.
- Finish the review workflow by recording the nonblocking dispositions below and using fresh, diff-only follow-up reviewers. Preserve `final_gate.json` as the actual failed artifact-check result. Passing review must precede the local checkpoint; pending review/commit fields are correctly pending at this point and are not an independent implementation defect.

## Other Concerns

- **P2: B1 prefill confidence is inconclusive for both layer kinds; carry the uncertainty explicitly.**
  Evidence: all five `D/final29_pair_l{0,3}_p{0..4}_{control,final}.json` pairs, `D/final29_b1_statistics.json`, `D/final29_b1_uncertainty.json`, and the unchanged `tests/check_optimized_multichip_evidence.py:629` decision rule. Independent arithmetic from all process means gives GDN control/final **2.395564131/2.194059826 ms**, median paired difference **−0.216594121 ms**, maximum/one-sided bound **+0.037668118 ms**. Full attention gives **1.761019820/1.701234902 ms**, median **−0.062272574 ms**, bound **+0.057086038 ms**. Four of five pairs favor final for each kind. With five pairs the declared exact 95% procedure uses the maximum difference and has 96.875% coverage. It does not establish zero-margin nonregression. The checker stops this group at GDN, so its one-entry statistics list must not be read as an attention pass.
  Disposition: nonblocking performance uncertainty under the explicit cap, rather than output wrongness or evidence of population regression. The original goal requires matched before/after timings and `$optimize` requires a B1 regression row; neither specifies this additional exact confidence procedure. All observations and the stricter predeclared procedure remain part of the record. Do not call this test passed, select favorable pairs, change the estimator/margin, or obtain optional repetitions now. Carry a prespecified confirmation series to a later authorized stage if needed. The remaining numerical/trace/watcher/context gates are separate and passed.

- **P2: DRAM-sharded matmul core/FLOP annotations assume one reader per bank and overstate compute utilization.**
  Evidence: `D/perf/final29_profile_l0_b1_s512/decode_perf_report.csv`, for example `32 x 5120 x 7168`, reports 8 cores and **175.6527%** FLOPs. Its raw `ops.csv` attributes explicitly select `num_workers_per_dram_bank=2`; the production DRAM-sharded factory derives its worker set from bank reader assignments and divides N across that worker set (`ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:124–190`). The installed `python_env/lib/python3.12/site-packages/tt_perf_report/perf_report.py:1337` unconditionally replaces this geometry with `arch_spec.dram_sharded_matmul_cores`, eight for Blackhole. This explains the apparently impossible percentage and affects the tool's compute-bound annotations.
  Disposition: add this exact caveat to final documentation and carry tool/accounting correction. Preserve raw tool output. Do not use those `Cores`, `FLOPs [%]`, or compute-bound labels as measured utilization for this two-reader path. The actual latency, dtype/fidelity, DRAM-byte arithmetic, and separately derived **43–45% traffic-lower-bound/host-time** values are not changed by this diagnosis. No hardware rerun is needed to classify it, and no 70% stop-rule claim is made.

- The handoff inventory contains untried final-HiFi4 geometry, fused-norm capacity, recurrence compositions, and long-prefill movement work. Those are permitted handoff items after the user's cap, not proof of rejection or required new experiments. The selected minimal/DiT norm path is **off**; the faster minimal candidate's failed 64-owner L1 allocation was correctly excluded. Import-order/unused-import lint findings and stale preparation-tense sentences in the final manifest/inter-layer/handoff notes are documentation/cleanup items, not reasons for new measurements. Update their status using the completed artifacts without changing frozen measured sources.

## Hard-Check Gaps

- `D/final_gate.json` accurately has `pass: false`, 25 checked groups, and errors for `b1_prefill_regression` and `review_commit`. The package-aware `D/check_final29.py` only supplies the planner's qualified `run_name`; it restores `runpy.run_path` and calls the unchanged checker. I found no assertion suppression. If final review accepts carry, use a separate acceptance/disposition record and retain the original checker outcome; a passing review must not silently turn this checker into a pass.
- `tests/run_multichip_contract.py` uses `zip` in mixed-list comparisons without first asserting returned count. Both expected outputs and both contribution arrays are actually present in each final receipt, and `flat_prefill_forward` constructs one output per validated input entry. This is a defensive test improvement, not missing output evidence here.
- Resource tests share two immutable representative weight sets across 64 independent mutable owners. They do not prove allocation or inference with 64 distinct real layer weights. `D/final29_capacity.json` and the context contract correctly separate that test from source-derived full-model memory estimates. Stage 6 must retain that distinction and perform its own full-model acceptance.
- The runtime audit's executed sentinel covers six public transfer APIs in initial eager prefill/decode and capture; the static reachable-call audit and 48 mutation controls broaden its evidence. It is not a universal detector of arbitrary Python/Torch behavior or every timed invocation. The documented limitation is accurate; inspected selected forward code did not reveal a host fallback.

## Anomaly Ledger

- Observed anomaly: inherited GDN LoFi HF contribution PCC 0.986013771 at S31 row12.
  Evidence: `D/final_validation28_contract_l0_tp4.json`, frozen-stage4 control, `D/debug_contract28/AUTODEBUG.md`, `D/AUTOFIX_contract29.md`, and final29 HF contract.
  Affected path: GDN attention-role prefill arithmetic.
  Control or comparison: frozen stage4 reproduces the failing contribution vector; HiFi4 intervention passes; final29 uses that same policy by default.
  Likely subsystem: attention-role precision/fidelity, without proof of a single faulty kernel.
  Investigation performed: checked fidelity propagation, retained controls, final-source HF length/resume/list results and final profile rows.
  Resolution: **fixed**. Final GDN minimum HF length contribution PCC is 0.997941968; final attention minimum is 0.997976478. The reported final timings include the HiFi4 repair.

- Observed anomaly: B1 paired confidence bounds exceed zero despite lower point estimates.
  Evidence: the twenty fixed paired-arm artifacts and unchanged checker rule cited above.
  Affected path: B1/S512 warmed eager prefill latency inference.
  Control or comparison: alternating frozen original TP4/current-default arms, current common harness/native build, three warmed calls per process, five independent pairs per kind.
  Likely subsystem: performance sampling uncertainty; this evidence does not isolate its cause.
  Investigation performed: recomputed each paired difference, both means, medians and maximum-based bounds; checked estimator/provenance code and preserved outcomes.
  Resolution: **controlled as disclosed nonblocking uncertainty**, not a confidence pass; carry disposition awaits the review sequence.

- Observed anomaly: profiler compute utilization exceeds 100%.
  Evidence: the CSV/raw attributes, installed reporting function and native reader-worker construction cited above.
  Affected path: reported DRAM-sharded compute-core/FLOP annotations.
  Control or comparison: fixed eight-worker reporting versus selected two workers per each of eight DRAM banks.
  Likely subsystem: performance-report model.
  Investigation performed: traced the annotation calculation to the unconditional core-count override; verified separate byte-based accounting does not consume that number.
  Resolution: **controlled by source explanation**; document/carry the reporting limitation, retain measured latency.

- Observed anomaly: minimal fused norm is faster but fails simultaneous 64-owner L1 allocation.
  Evidence: `D/norm_resources26_minimal_b1.json`, `D/debug_norm_capacity27/AUTODEBUG.md`, selected policy, and passing final B1/B32 resource receipts.
  Affected path: unselected native fused norm owner storage.
  Control or comparison: selected `distributed_decode_norm=off` has zero fused norm owners and passes 64 independent CCL/state owners.
  Likely subsystem: lockstep L1 allocation capacity.
  Investigation performed: inspected retained diagnosis, selected dispatch and actual final owner counts.
  Resolution: **controlled by excluding the candidate**. No capability reduction or unsupported claim that DiT/ephemeral alternatives were measured.

- Observed anomaly: allocation tracker deliberately reports a buffer that would be corrupted; final logs also contain motherboard-ID fallback, small AICLK deviations and semaphore-fragmentation warnings.
  Evidence: final trace/resource positive-control fields, final logs and watcher logs.
  Affected path: validation controls and platform discovery/allocation diagnostics.
  Control or comparison: positive control must reject, then tracked final replay succeeds with no acknowledgments; final resource tests and context reservation succeed.
  Likely subsystem: expected tracker negative test; platform metadata/clock/allocation diagnostics.
  Investigation performed: compared warning classes to control sequencing and final receipts. The motherboard warning is a tray-ID discovery fallback, not model computation fallback. Baseline-only subset-device warnings do not change the measured TP4 path.
  Resolution: **controlled**; no unclassified final watcher assertion, device corruption, or fallback failure found.

## Scope Inspected

- Goal/skills: original `/home/ttuser/qwen36-27b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`, supplied final29 brief and user wrap-up; installed 0.1.24 `stage-review`, `optimize`, `model-bringup` startup, `multichip`, `tt-device-usage`, and applicable `tt-enable-tracing` requirements. No generated-text/full-model/vLLM/MoE requirements were applied to this dense decoder-only stage.
- Artifacts: final README/work log, topology/candidate/inter-layer/handoff/selection notes, final manifest/gate, validation manifest/summary and selected raw result/command/log/receipt records across all seven phases, both HF contracts, both stack/resource/context families, repeated stress and watcher families, all B1 pair values, B32 before/after rows, all six final profile CSV/accounting families, advice inventory, runtime audit, capacity/context contract, native build/AGMM/norm validation evidence, AutoFix/AutoDebug controls and read-only lint results.
- Re-derived acceptance observations: all **8,935** validation source/native pins match the current filesystem; the 68-job summary and final TP4 receipts cover both kinds. HF contracts each contain 12 lengths, seven resumes and two mixed tails; every dropped-history control fails as intended. B32 stress has two 512-step trajectories per process and two processes per kind; B1 has 128-step trajectories. Final 64-owner resource evidence has tracking1/skip-cache0, zero acknowledgments, one main capture, zero recaptures and zero late program-cache growth. Five-layer B2/B32 stack instrumentation passes the identical residual object/address across boundaries, with no boundary conversion and traced/eager equality. Both final 32K context smokes pass with 20 GiB explicit reservation; the advertised 262144 context remains unchanged with historical maximum evidence accurately identified.
- Performance/source observations: measured final paths have no policy/source override, mesh `[1,4]`, sharded compact decode `[1,B,1280]`, and current source pins. B32 prefill/decode improve in every reported matched row. All six profile families show BF16 activation/BFP8 weights, GDN attention HiFi4 and MLP HiFi2, full-attention LoFi. Decode CCL category sums are 66–69 microseconds inside layers. The final corrected GDN latency is reported rather than an earlier faster LoFi candidate.
- Code: selected default/configuration, matmul/collective/persistent-buffer ownership, norm/reduction, flat prefill scheduling, page-table certification and cache dispatch, compact decode and recurrence hooks in `tt/`; validation/contract/decoder/stack/resource/evidence/accounting paths in `tests/`; stage-owned native MMRS ownership and bounded Linear AGMM/RMS changes and their test references. This is not an exhaustive review of every historical opt-in candidate.
- Commands: read-only `git status/diff`, `rg`, bounded source/log reads, standard-library JSON/CSV parsing, source SHA256 comparison, paired-statistic arithmetic and profiler/native-source comparison. One exploratory capacity printer stopped on a list-valued field; no artifact was changed. No TTNN import, device open/reset/reservation, build, server, hardware experiment or subagent was used. The only write is this report.

## Residual Risk

- The required decoder evidence is strong, but it does not establish full-model accumulated quality, all-distinct-layer allocation, request scheduling or serving trace lifetime. Those are later-stage gates.
- The final 32K smoke and source capacity plan satisfy the unchanged-KV stage5 context cadence; they are not a newly executed 262144-token final-source run.
- Performance uncertainty and profiler compute annotations must remain visible in the handoff. Do not claim statistically established B1 nonregression, 70% achieved roofline, or tested performance for capped candidates.
- The run plan authorizes **local checkpoints only**, after passing review, with shared native fixes isolated as required by startup guidance. The supervisor owns pushing; this stage must not push. No checkpoint SHA exists yet, consistently with the required order.
