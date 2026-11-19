# Stage Review

Verdict: **more-work-needed**

Independent round 1 review of stage 5, `optimized-multichip-decoder`, for
`Qwen/Qwen3.6-35B-A3B`, dated 2026-11-14. This reviews an incomplete checkpoint,
not an accepted stage. No hardware, server, reset, implementation edit, or
additional reviewer was used. The only file written is this report.

## Required Work

- **P1: The retained linear-attention default fails the mandatory FUNCTIONAL floor.**
  Evidence: Recomputed the mean of all 384 samples in
  `restored_functional_l0.json`: **0.719504473863708 ms/token**. The per-rank
  useful-byte model in `tests/multichip_functional.py` gives
  **0.0518283125 ms/token**, hence **7.203334292%**, below the unchanged 10%
  floor. The maximum permitted latency is **0.518283125 ms/token**: another
  **27.97% reduction** is needed. Prefill PCC is 0.999991977; minimum decode
  PCC is 0.999158705; changing-input replay and saved-state checks pass.
  `gate_summary.json` fails `functional_l0` and passes its other 32 rows.
  Fresh restored full attention passes at 0.3408368071783722 ms/token and
  10.105496%; that does not satisfy the linear row.
  Why this matters: The optimize Stop Rule bounds optional tuning, but
  explicitly does not waive this hard gate. Faster B32 results and an earlier
  candidate cannot replace a qualifying final default.
  Required next step: Restore the linear floor while preserving the accepted
  numerical and trace contracts, then rerun the affected final-default gates
  and obtain a passing independent review. The documented bounded AutoFix
  attempt has **failed to restore this contract**: exact recurrence plus
  repaired FP32 CCL takes 0.869184612 ms, and native norm/DRAM projection plus
  repaired CCL takes 0.554099732 ms. The native family also retains a
  sensitive-layer anomaly. These are substantive failed repair experiments;
  they do not prove that every possible kernel design is exhausted. If work
  ends on that failed AutoFix outcome, preserve this as an incomplete stage,
  not a pass or a waiver.

- **P2: Watcher coverage omits changed non-aligned/resumed prefill paths.**
  Evidence: All four `watcher_l{0,3}_b{1,32}.json.provenance.json` commands use
  `--length 32`; `run_multichip_decoder.py` prepares those prefills at
  `start_pos=0`. `run_optimized_multichip_validation.sh` uses the same fixed
  watcher cases. The stage changes the packed linear projection used by
  prefill (`tt/multichip_decoder.py:406`) and singleton helpers also reachable
  from singleton/lead/tail paths. Boundary lengths 1/31/65/257, S129 stress,
  and the mixed stack's prefixes 33/35 plus resumed lengths 32/34 do pass,
  but their provenance has watcher unset. The optimize evidence checklist
  explicitly requires watcher coverage of changed prefill paths, including
  continuation at nonzero start, lead rows and block-size groups.
  Why this matters: The existing watcher dumps establish clean aligned S32
  execution; they do not inspect the changed partial-row and resumed
  branches exercised by the non-watcher correctness tests. This is a missing
  required validation subset, not a claim of a detected memory error.
  Required next step: Add a **small targeted watcher run** on the selected
  default covering the changed non-aligned/lead/tail and nonzero-start
  prefill branches, preferably reusing the existing mixed-stack fixture and
  a boundary fixture only where its branch is otherwise absent. Keep
  profiler off and preserve the completed-dump/error checks. A full sweep,
  another maximum-context run, or a new profiler capture is unnecessary for
  this finding.

## Other Concerns

- Optional tuning is legitimately capped. The recorded successive B32 gains
  (2.47%, then 0.60% linear / 2.69% full) support Stop Rule 2, and the work log
  records the four-hour cap. Untried batched residual interfaces, canonical
  fused AG-MM, HiFi2 B32 geometry cross-products, shared packing, grouped
  expert tuning and additional precision candidates remain **handoffs**, not
  new Required Work. They must not be described as measured rejections.
- The topology rejections inspected are earned within their stated scope.
  Hidden512 residuals feed the following norm, rather than an immediate
  restore solely for the old interface. Raw candidate commands and results
  give linear/full sharded B32 decode 21.585338/5.831879 ms and repaired fused
  output 5.892128 ms; the LoFi/BFP8-CCL fused family gives 5.886388 ms. The
  combined fused input/output path fails exact paired K-cache equality;
  canonical separate input projection repairs that boundary. This does not
  reject every future canonical-order fused kernel.
- Full B32 prefill regresses **5.47%** (42.404445 to 44.724137 ms), while its
  decode improves **23.32%**. The README discloses this two-phase tradeoff.
  All four short-run decode medians improve; B1 prefill also improves. The
  separate FUNCTIONAL means are correctly kept out of the short-run table.
- The retained FP32 all-reduce has a demonstrated synthetic cancellation
  precision discrepancy. The experimental patch fixes 40/40 component
  checks with the documented hardware BF16 pack oracle, but whole-decoder
  promotion is not established. The restored implementation is controlled
  by passing model PCC and exact pre/post-restoration output/state equality
  in the recorded cases. Do not advertise universal full-FP32 arithmetic or
  a successful native repair; further promotion needs model-level evidence.
- Formatting remains deferred for the measured production module.
  `logs/checkpoint_precommit.log` explicitly skips Black/isort;
  `logs/final_python_precommit_except_measured_module.log` passes them for
  the other changed Python files. `deferred_formatting.patch` contains the
  production import ordering/formatting delta. This is accurately disclosed
  and is not an additional numerical blocker; complete routine cleanup
  before an eventual accepted checkpoint without silently relabeling older
  source provenance.
- No local stage checkpoint exists yet. The run plan says local commits only
  and **never push**; supervisor push policy is not inferred from Git history.
  No full-model or vLLM work is required by this decoder-only stage.

## Hard-Check Gaps

- The strict checker reports 32/33 but its watcher gates only validate the
  four fixed S32 fixtures; it does not enforce changed-path coverage. This
  accounts for Required Work P2 despite those watcher rows passing.
- `gate_summary.json` checks `functional_l*.json`, not the newer
  `restored_functional_l*.json`. The earlier and newer means have the same
  fail/pass classification, and independent saved-tensor comparison found
  **no tensor differences**, including prefill, every decode output and
  saved state. The fresh artifacts contain native source and loaded-library
  provenance. This does not change today's verdict, but future qualification
  after a native/runtime edit must use the final loaded build, not only the
  Python decoder hash.
- No extra generation, qualitative text, serving trace-reuse or maximum
  context artifact is demanded here. They are outside this stage, or the
  unchanged BF16 KV/head-layout rule explicitly permits the 32769-token
  context smokes.

## Anomaly Ledger

- Observed anomaly: Linear FUNCTIONAL below 10%.
  Evidence: `restored_functional_l0.json`, `repair_fp32_ar_{exact,native}_l0.json`.
  Affected path: Selected B1 linear decoder on the true 1x4 mesh.
  Control or comparison: Full-attention pass; original/default and isolated
  complete-decoder repair sequences with 384 samples each.
  Likely subsystem: Ordered projections/norms and recurrent computation/data
  movement; no single unmeasured gain is asserted.
  Investigation performed: AutoDebug, exact selector/norm/SiLU integration,
  native projection/fidelity experiments, exact recurrence and FP32 CCL controls.
  Resolution: **more-work-needed**, Required Work P1.

- Observed anomaly: Additional layer34 default steps below 0.995.
  Evidence: `layer34_hf_comparison.json` and original saved tensors.
  Affected path: Additional S128/D192 seed154 sensitivity stress.
  Control or comparison: Independently recomputed matched-HF aggregate
  **0.999702065354939**, minimum **0.991128479144610**; inputs match exactly.
  Selector-only is bit-equal to default over all 192 outputs.
  Likely subsystem: Inherited mesh precision/rounding differences; the exact
  cause of each excursion is not established.
  Investigation performed: Matched HF and single-chip controls, router-policy
  controls and component isolation.
  Resolution: **controlled** under the specified whole-output aggregate and
  0.99 diagnostic floor. No HF waiver or invented per-step 0.995 gate is needed.

- Observed anomaly: Native alternatives have diagnostics below 0.99.
  Evidence: `repair_fp32_ar_native_exact_l34_hf.json`; independently
  recomputed aggregate **0.999559383397452**, minimum **0.987942861710545**
  at step174. Earlier native norm/projection/recurrence controls also regress.
  Affected path: Experimental, unselected native linear family.
  Control or comparison: Matched HF inputs and selected/default control.
  Likely subsystem: Native norm/projection-to-routing arithmetic; repaired
  collective alone does not explain or fix it.
  Investigation performed: Isolated controls, router restoration, exact
  recurrence and repaired FP32 collective.
  Resolution: **controlled by rejection**; unresolved if this family is revived.

- Observed anomaly: Combined fused AG-MM/MM-RS breaks duplicated K equality.
  Evidence: `logs/combined_fused_l3.log`, `AUTODEBUG.md`, `AUTOFIX.md`.
  Affected path: Unselected hidden-sharded fused-input family.
  Control or comparison: `autofix_fused_rs_l3.json` passes exact paired caches
  after separate canonical input projection, retaining distributed norm/fused output.
  Likely subsystem: Rank-dependent projection accumulation order.
  Investigation performed: Source audit and adapted next-consumer whole-layer run.
  Resolution: **controlled by rejection and passing adapted control**.

- Observed anomaly: Original SiLU component replay was rejected by allocation tracking.
  Evidence: `silu_component.json`, `silu_component_retry.summary.json`,
  `AUTOFIX_functional.md`; selected stress runs include program-cache allocations.
  Affected path: Component-copy warmup, not an accepted exemption.
  Control or comparison: Exact copy signatures warmed before capture; exhaustive
  BF16-code checks and repeated tracked replay succeed.
  Likely subsystem: Late program-cache allocations.
  Investigation performed: Corrected warmup; no broad allocation scope introduced.
  Resolution: **fixed in the retry**, selected-path tracked stress passes.

- Observed anomaly: CCL native library changed during repair experiments.
  Evidence: Native provenance in `fp32_ar_*` and restored FUNCTIONAL sidecars.
  Affected path: Native all-reduce experiment and subsequent default restoration.
  Control or comparison: Both C++ hashes match pre-experiment sources; independent
  CPU comparison found all tensors in both restored FUNCTIONAL artifacts equal
  to the earlier defaults. The new library hash is explicitly different.
  Likely subsystem: Native build/install identity.
  Investigation performed: Restored cancellation probe and fresh complete-decoder runs.
  Resolution: **controlled**; source restoration and tested output equivalence,
  not binary bit-identity or blanket qualification of future changes.

## Scope Inspected

- Goal/skill paths: `/home/ttuser/qwen36-35b-a3b/multigoal_logs/05-05-optimized-multichip-decoder.prompt.txt`;
  installed `tt-model-bringup/0.1.24/skills/{stage-review,optimize,multichip,tt-device-usage,model-bringup}/SKILL.md`;
  optimize method/evidence checklist and relevant tracing lifetime requirements.
- Live worktree on `main`, base `5ebe699604445e7b30ea0813e15aa913efbd3188`,
  with staged/unstaged stage changes. Current production decoder SHA256:
  `13b403518fb8b71dcf5b502bddeb9be79bcd6acf27b1bb51ab47c4ef902d952f`.
  Selected helper hashes match measured provenance; the observed later
  `mesh_exact_recurrence_probe.py` formatting change is outside the selected path.
- Artifact paths: Stage README/work log, topology/decision/residual documents,
  strict gate JSON/CSV, before/final/restored and repair JSON/provenance/saved
  tensors, matched HF cache and sensitivity comparisons, context contract and
  smokes, natural B32, boundary/stress/mixed/watcher controls, allocation
  positive control, B1/B32 perf summaries and detailed report CSVs, MoE
  attribution, AutoDebug/AutoFix reports, component controls and hook logs.
- Code paths: `tt/multichip_decoder.py`, selected `mesh_fp32_topk.py`,
  `mesh_compensated_norm.py`, `mesh_decode_silu.py`; decoder/validation/profile
  runners, FUNCTIONAL byte model and gate checker, runtime audit,
  HF comparison and MoE attribution logic; trace acknowledgement/lifetime
  sites in the ordinary and mixed-stack runners.
- Commands run: Read-only `git diff/status`, `rg`, `sed`, bounded file reads;
  small CPU Python JSON/CSV/hash analyses and Torch saved-tensor comparisons.
  Recomputed all short-run medians, FUNCTIONAL means, restored tensor equality
  and selected matched-HF PCCs. Inspected current-source default profile
  identities and actual matmul rows: BFP8/HiFi2 full/shared projections,
  BFP8/LoFi indexed experts, and FP32 router operands where expected.
  The multi-minute full CSV checker was not rerun; its completed log was inspected.

## Residual Risk

- This is inspection and saved-evidence analysis, not independent device execution.
  Recorded tracking stresses have 512 evolving replays per kind, program-cache
  inclusion, clean runtime audits and live-trace prefill checks. Mixed-stack
  coverage has two requests, disjoint shared pages, one capture/24 replays,
  and a serial CQ0 workspace retained through release. The source uses exact
  final-output acknowledgements, not blanket forward/warmup exemptions.
- Residuals are replicated BF16 activations but computation is genuinely TP4/EP4;
  no additional inter-layer collective is required by the selected interface.
  Shared CCL and selector buffers require the documented serial consumption.
  Concurrent reuse of the same workspace is not qualified.
- The context contract preserves 262144 aggregate cached tokens and 32 recurrent
  request slots; this is not 32 simultaneous full-context requests. S32769
  smokes use capacity65536 and two following decode steps. Full-stack memory
  remains planning arithmetic rather than a loaded full-model demonstration.
- Profile kernel sums and MoE firmware envelopes are appropriately distinguished
  from clean wall latency. B32 profile prefill covers one representative request;
  clean B32 prefill timing covers all requests. These scopes cannot be interchanged.

The stage cannot pass this review. The bounded AutoFix failure is supported by
the recorded experiments; the missing targeted watcher subset is additional
required evidence before eventual acceptance.
