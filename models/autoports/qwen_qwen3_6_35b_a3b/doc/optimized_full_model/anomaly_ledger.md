# Stage 7 anomaly ledger

This ledger distinguishes rejected experiments from defects in the selected path.
Final correctness and clean performance pass; native profile passes; long-context smoke passes; independent review remains pending.

| Observation | Evidence and investigation | Resolution |
|---|---|---|
| Starting full-model performance missed the required useful-byte floor | `before.json`:25.081181ms versus21.631299ms ceiling. AutoDebug and exact structural/scan/recurrence repairs are recorded in `AUTODEBUG.md` and `AUTOFIX.md` | Fixed: candidate2 passes21.424392ms and frozen final default reproduces21.416868ms. Denominator and numerical policy unchanged |
| Candidate1 still missed the floor | `candidate1_performance.json`:22.272769ms; remaining0.641471ms. Full state-slab fusion preserves exact operation order and state bits | Fixed by candidate2; no denominator or numerical-policy change |
| Head reader3/block32 and initial one-reader geometry exceeded static L1 | `head_r3_b32_c16.json`, `head_r1_b1_c16.json`; allocation sizes recorded in work log | Rejected. Legal split4 one-reader control passes but is slower (`head_pressure_r1_b1_s4.json`) |
| A health check accidentally overlapped reset completion | Recovery commands/logs described in work log | Overlapping result discarded. Fresh serialized health and mesh check are the acceptance evidence |
| AutoDebug child sandbox preflight failed | `logs/autodebug_launch.log`, `logs/autodebug_retry.log` | Authorized isolated read-only retry completed; no diagnosis inferred from the failed preflight |
| Early scan probes used a different output shard geometry | Corrected production-layout read32/read64/NOP artifacts plus subsequent Watcher/caching checks | Early results remain exploratory; selected kernels qualified with actual8×[32,128] output geometry |
| A combined-cache probe stopped on its source-anchor assertion | `combined_cache32_sum_l0.json`; concurrent source change invalidated an experimental source rewrite | No numerical result accepted. Subsequent integration uses the explicit production selector, not source rewriting |
| New head constructor sliced a TTNN Shape directly | First sampler acceptance log; failure preceded kernel execution | Fixed with tuple conversion; sampler acceptance/timing v2 pass all three greedy choices |
| Linear batched decoder called `.get` on an optional None position | `logs/state_slab_integrated_l0_b3.log`, baseline side before candidate execution | Minimal optional-position guard; `production_slab_l0_b3.json` passes exact nonaligned B3 output/state comparison |
| Persistent sampler prototype rejected a valid 1D mesh axis spelling | First watched persistent-sampler log; no candidate native operation executed | Prototype normalizes None/1 to physical axis1 for the exact1×4 mesh |
| Persistent broadcast sampler returned poison-derived token127615 | `logs/persistent_sampler_l1_watched_v2.log` and `persistent_sampler_l1_watched_v3.diagnostics.json`: local32 columns written, remote96 columns untouched | Experimental candidate excluded. Native partial-packet diagnosis is recorded separately; selected split sampler remains unchanged and passes correctness |
| Candidate2 TTFT is higher than baseline in one measurement | `candidate2_performance.json`:2880.454ms versus2787.782ms; candidate1 was2768.434ms | Controlled by final same-harness2729.973ms and six boundary samples; no robust prefill gain/regression inferred; see prefill_performance_analysis.md |
| Standalone linear-layer floor remains below10% | `production_slab_functional_l0.json`: exactness passes but standalone useful-byte floor is false | Preserved explicitly. Stage7 acceptance uses the complete40-layer useful-byte calculation; no standalone floor pass claimed |

Inherited stage5 numerical-policy rejections and the stage6 owner-accepted,
HF-reproduced resumed-prefill limitation remain in their original evidence and
the context/serving contracts. This stage does not manufacture a new waiver or
reinterpret those historical measurements as fresh stage7 gates.

## All40 serving page-table assertion

Observed anomaly: serving command gate saw30 updates but asserted3. Evidence: final/serving_commands_all40.json and its log at the page_changed case. Affected path: opt-in all40 test harness; production generator unchanged. Control/comparison: reduced0/3 hasone full layer×3slots; all40 has10×3slots (30); _bind_pages increments once per changed layer/slot. Investigation: inspected cache iteration, prior checkpoints and failed counter delta. Resolution: test derives cache-group count and checks every saved page table; final/serving_commands_all40_v3.json passes corrected30/0 counters and saved-table equality; fixed.

## Serving verification loop shadowing

Observed anomaly: v2 fails TypeError at state() after both corrected page-table cases pass. Evidence: final/serving_commands_all40_v2.json/log. Affected path: new harness verification loop shadows local helper with a dictionary. Investigation: Python local binding and exact failing line. Resolution: renamed loop item cache_slot; hooks pass; final/serving_commands_all40_v3.json passes all9 command checkpoints; fixed. Production unchanged.

## Candidate2 TTFT increase and isolated-layer timing mismatch

Observed anomaly: candidate2 TTFT2880.454ms exceeded baseline2787.782ms; isolated blocking-layer stack21.297628ms exceeded the candidate model-only trace20.942603ms that includes terminal work. Controls/investigation: final unchanged-production same-harness all40 measurement, plus retained-carrier queued representative layer controls with boundary-only synchronization. Resolution: controlled. Final TTFT2729.973ms, candidate1 previously2768.434ms; no robust TTFT improvement or regression inferred from these single samples. Queued layer budget20.668310ms removes0.629318ms of per-layer fencing/measurement-regime cost; final whole token-out21.416868ms is3.62% above this engineering estimate before terminal allowance. Separate fixed-input/router controls are not an exact additive decomposition; completed reduced native attribution is separately scoped in final_profile_audit.md. Artifacts: final/performance.json, final/queued_layer0.json, final/queued_layer3.json, final/layer_stack_budget.json, perf_summary.json.

## Native metadata and profiler warnings

The final reduced capture reuses cached embedding-gather metadata with height64,
while the captured tilize producer has padded32. The native audit preserves
this discrepancy and does not derive runtime payload bytes from the cached
height. No output, source-policy or collective-width contradiction was found.
The optional Tracy WASM-copy warning names a different default directory;
actual retained raw capture converted and29,440 execution rows validated.
Custom/new operations lacking tt-perf-report classification retain native timing
and bounded source-role attribution; no row is silently excluded from totals.

Clean timing/context runs disable trace allocation tracking and therefore print
the generic active-trace allocation warning. They do not qualify lifetimes by
suppressing it: separate final tracking1/skip0 replay, serving and B32 gates plus
the deliberate unsafe-allocation positive control supply that evidence. The
runtime audit names generator-wide preparation and exact persistent ownership.
The `to_layout` already-in-requested-layout warning describes its documented
no-op memory conversion behavior; explicit `to_memory_config` and guarded
operand layouts supply required layouts. No new layout policy is inferred from
that warning. These classifications do not waive an actual tracker rejection.

## Optional router probe lifetime failure

`router_l1_tracked.json` rejected one live trace-local score allocation before
replay. The probe retained loop variable x after deleting out. Deleting both
references fixes the exact lifetime; no tracker exemptions or broad scopes were
introduced. Three separate tracking1/skip0/Watcher10 v2 controls and three clean
v2 timings pass192 comparisons each. Initial untracked rows are superseded.
This is outside frozen production/tests, so no full-model gate is invalidated.
The device process closed normally; no hardware reset was needed. Logs and the
failed/final Watcher files remain at manifest-linked paths.
