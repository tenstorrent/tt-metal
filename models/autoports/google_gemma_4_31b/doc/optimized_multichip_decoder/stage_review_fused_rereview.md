# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None.

## Hard-Check Gaps

- `perf_accounting.json` is an explicitly named historical `final_noop_removed`
  accounting snapshot (0.42814175/0.48092775 ms device and
  0.464383/0.5175075 ms host).  The current final-default accounting is not
  inferred from that file: it is independently present in
  `tracy/final_post_fused_review`, whose filtered CSVs sum to
  0.42830025/0.48162025 ms and reconcile to the source-current
  0.463813/0.5166275 ms host medians.  The provenance label prevents this
  retained historical snapshot from contradicting the final headline.
- The exact-position optimized test allocates the advertised cache and reaches
  absolute position 262,143 but does not itself repopulate every historical
  token.  The linked prior-stage populated-context evidence covers the unchanged
  BFP8 KV layout/capacity, and this stage adds exact-position traced replay,
  non-aligned input, page-table, cache-layout, and batch-32 coverage.  No
  capability reduction was introduced, so this is not a blocking gap.
- Stage-owned commits and the administrative status transition are properly
  pending until this review; they are post-review handoff actions rather than
  evidence defects.

## Anomaly Ledger

- Observed anomaly: The prior final review found that fused all-gather+matmul
  had been rejected from a bound rather than an exact Gemma coherent-family
  measurement.
  Evidence: `stage_review_final.md`; exact primitive logs
  `candidates/fused_agmm_model_shapes.log`,
  `candidates/fused_agmm_full_o.log`, and
  `candidates/fused_agmm_down_bfp8.log`; source-current coherent run
  `candidates/fused_agmm_coherent_final.log`; and 12 signpost-filtered report
  families under `tracy/fused_agmm_coherent/`.
  Affected path: Rejected alternate decomposition for attention O and MLP down,
  not the final default.
  Control or comparison: The production replicated spine and fused H/TP-local
  spine have matching next-QKV endpoints and use the same real checkpoint
  O/down/packed-gate-up/QKV weights.  The lower-movement spine keeps H/TP local
  through fused O, distributed norms/residuals, fused packed gate/up, fused
  down, post-MLP distributed norms/residual, and the next fused QKV; the
  immediate-gather candidates are measured separately.
  Likely subsystem: Ring `AllGatherMatmulAsync` geometry and distributed norm
  overhead for Gemma's narrow TP-local output widths.
  Investigation performed: The first output-subblock validation error was
  adapted from width 4 to 1 on an 8x6 program grid.  Exact TP4 sliding O
  K8192/N1344 and full O K16384/N1344 passed at PCC 0.999963534 and
  0.999964025.  Down K21504/N1344 BFP4 scored 0.993138794, then the same
  shape/layout was adapted to BFP8 and passed at 0.999964537.  The real-weight
  coherent family passed at 0.999751962/0.999694822 but warmed at
  0.9525755/1.0556255 ms versus 0.3844085/0.413468 ms.  Independent report
  sums reproduce 925.49775/1026.021 us versus 349.765/384.9575 us.  Persistent
  gathered-input buffers, explicit Ring/one-link/dim-3 configuration, and
  separate trailing-gather variants were included.
  Resolution: controlled; exact adapted evidence rejects this family as
  2.478x/2.553x slower without using an immediate old-contract restore.

- Observed anomaly: The selected packed MLP partial was already BFP8 but an
  earlier helper typecast it to BFP8 again before the async reduction.
  Evidence: `stage_review_rereview.md` and the before/after profiler roots
  `tracy/final_selected` and `tracy/final_noop_removed`.
  Affected path: Earlier traced decode default.
  Control or comparison: Current `_tp_allreduce` typecasts only when
  `partial.dtype != communication_dtype`; current decode reports retain the
  needed BF16-to-BFP8 attention conversion and two BFP8-to-BF16 boundary
  restores but contain no BFP8-to-BFP8 conversion.
  Likely subsystem: Collective dtype preparation.
  Investigation performed: Source guard, source-current final latency/suite/
  context/watcher reruns, and post-remediation profiler inspection.
  Resolution: fixed.

- Observed anomaly: Current-topology BFP4 attention is faster for sliding
  attention but fails the declared full-attention accuracy gate.
  Evidence: `candidates/review_attention_bfp4_final.log` records
  0.997252806 sliding and 0.992305137 full versus the declared 0.995 gate;
  `review_cumulative_low_precision.log` independently records full PCC
  0.992102801.
  Affected path: Rejected precision candidate only.
  Control or comparison: Final BFP8 attention/QKV7/BFP8-decode-CCL passes at
  0.999802416/0.999718188.
  Likely subsystem: Full-attention projection sensitivity to BFP4 weights.
  Investigation performed: Isolated and cumulative real-weight, current-
  topology trials for both meaningful layer kinds.
  Resolution: controlled.

- Observed anomaly: The first Linear fused MM+RS run hung, and later Ring/L1
  candidates exposed persistent-scratch placement collisions.
  Evidence: `candidates/fused_mmrs_hang_triage.txt.gz`,
  `fused_mmrs_model_shape*.log`, `ring_l1_AUTODEBUG.md`, and
  `ring_l1_AUTOFIX.md`.
  Affected path: Rejected topology candidates, not the final Linear async-
  reduction default.
  Control or comparison: Ring exact-shape MM+RS passed at PCC 0.999963; the
  adapted complete Ring family and the final Linear family both pass, while
  the source-current final suite, separate watcher run, and terminal pool
  teardown are clean.
  Likely subsystem: Fused Ring-kernel topology contract and L1 placement.
  Investigation performed: Triage, bounded reset/health recovery, Ring retry,
  shared tail-24 physical scratch, two semaphore epochs, M>1 projection
  adaptation, and current-policy Ring versus Linear rerun.
  Resolution: controlled/fixed for the selected default.

- Observed anomaly: Final full-prefill latency is 1.16% slower than the
  current-run Stage 05 baseline while decode improves materially.
  Evidence: `evidence/baseline_latency.log` versus source-current
  `evidence/final_latency.log`: full prefill 2.180140 -> 2.205460 ms; traced
  decode 0.575653 -> 0.5166275 ms.
  Affected path: Final performance headline.
  Control or comparison: Sliding prefill improves 2.418039 -> 2.4064085 ms;
  decode improves 11.91%/10.25%; Ring trades a 211.99 us sliding-prefill loss
  for only run-noise decode/full-prefill differences.  Phase-specific BF16
  prefill communication avoids the much larger BFP8-both-phases regression.
  Likely subsystem: Run variation and topology-dependent communication.
  Investigation performed: Phase-specific dtype adaptation, final default
  rerun after all source/test remediation, and coherent current-policy Ring
  comparison.
  Resolution: controlled; Linear is the strongest complete workload family.

## Scope Inspected

- Goal/skill paths: supplied Stage 05 optimized-multichip-decoder contract;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: Stage 05 `README.md`, `work_log.md`,
  `perf_accounting.json`, `candidates/summary.csv`, all three prior review
  reports, candidate logs/XML relevant to topology/precision/geometry/CCL,
  final latency/suite/exact-context/watcher logs and XML, context contract,
  `tracy/fused_agmm_coherent`, and `tracy/final_post_fused_review`.
- Code paths: live `tt/multichip_decoder.py`,
  `tests/test_multichip_decoder.py`, the Gemma dtype-suffix change, the generic
  fused AGMM helper's optional subblock override, the fused MM+RS helper's
  optional output-block override, and stage-owned diffs from the Stage 05
  starting checkpoint.
- Commands run: read-only `git status/log/diff/diff --check`, `find`, `rg`,
  `sed`, `stat`, SHA-256 verification, XML parsing, CSV parsing, filtered
  device-time summation, and source inspection.  No TT device, server,
  profiler, watcher, reset, or other hardware command was run by the reviewer.

## Residual Risk

- Final hardware evidence postdates the last live test-source edit and is
  internally consistent: 12 standard passes (plus 48 intentional env-gated
  skips), 4 latency passes, 2 advertised-position passes, 4 separate watcher
  passes, and 4 profiler windows.  The final enriched CSV hash verifies, and
  independent sums reproduce 428.30025/481.62025 us device time for
  sliding/full decode.
- Final correctness covers both meaningful layer kinds, all four replicas,
  non-aligned length 33, sliding 1025/1057 wrap, permuted page tables, mutable
  and repeated traced replay, batch 32, BFP8 paged-cache/layout contracts,
  terminal shared-pool cleanup, runtime fallback audit, and absolute position
  262,143.  PCC remains above the declared 0.995 gate.
- The final inter-layer contract is explicitly replicated BF16 DRAM with no
  inter-layer collective or conversion.  The two material reductions are
  owned inside each layer, use a mesh-shared tail-24 persistent scratch pool,
  and preserve the 262,144 context contract without requiring aligned public
  sequence lengths.
- The only remaining actions are administrative: record this clean pass,
  transition status, and create isolated local stage commits while excluding
  the documented unrelated dirty files.  They do not change this technical
  verdict.
