# Stage Review

Verdict: clean-pass

## Required Work

- None.

## Other Concerns

- None.

## Hard-Check Gaps

- The current advertised-position tests allocate the full TP4 cache and run
  traced decode at absolute position 262,143, but do not populate all prior
  history or compare that output to the single-chip oracle. This is not a
  current blocker: the cache dtype, local-head layout, page geometry, and
  262,144 allocation contract are unchanged; linked prior populated-context
  evidence covers the storage contract; and the current suite separately
  covers real-weight paged prefill/decode, permuted page tables, non-aligned
  lengths, mutable replay, and both layer kinds.
- The current source CSV, filtered profiler CSVs, and console/test logs match
  the documented live-worktree evidence but are hidden by repository-wide
  `*.csv`/`*.log` ignore rules. The stage-owned README, accounting, work log,
  XML, reports, hashes, PNGs, and this review are cleanly separable from the
  unrelated dirty paths. At checkpoint time the owner must explicitly include
  any ignored evidence it intends the commit itself to retain; this is the
  normal post-review administrative action, not a technical evidence defect in
  the present shared workspace.

## Anomaly Ledger

- Observed anomaly: Current full-attention prefill wall time is 2.4464875 ms,
  12.22% slower than the historical Stage 05 starting-path number of
  2.180140 ms.
  Evidence: `evidence/current_head_latency.log`; `README.md`; current
  `tracy/current_head/full_attention_PREFILL/perf_report.csv` sums to
  1,418.152 us, while the prior reviewed final report sums to 1,422.2695 us.
  Affected path: Untraced warmed prefill wall-time reporting, not the traced
  decode target or on-device prefill graph.
  Control or comparison: The current and prior final profiles have the same 30
  operation signatures, dtypes, fidelities, program fields, and memory
  contracts as multisets. The current device sum is 0.29% faster, and the
  same-process current TP4 prefill remains 1.396x faster than its single-P150
  control (3.415086 ms).
  Likely subsystem: Host scheduling/harness variance outside the stable device
  graph.
  Investigation performed: Independent CSV summation, exact operation-
  signature multiset comparison against `tracy/final_post_fused_review`, and
  reconciliation with all 12 host samples and the same-process single-chip
  control.
  Resolution: controlled; the report correctly uses 2.4464875 ms as the
  authoritative current wall number rather than replacing it with the older
  faster sample.

- Observed anomaly: Later repository stages changed
  `tt/multichip_decoder.py` substantially after the original Stage 05 commit,
  so copied Stage 05 candidate evidence could have become stale.
  Evidence: `git diff 96d41fdf0c5..b68b16df75d --
  tt/multichip_decoder.py`; current source CSV and four filtered reports under
  `tracy/current_head`.
  Affected path: Applicability of the retained projection, topology,
  collective, precision, geometry, and placement candidate matrix.
  Control or comparison: For sliding/full prefill and decode, current versus
  prior reviewed-final profiler rows have identical operation-signature
  multisets, including 33/30/59/58 op counts, shapes, weight/input/output
  dtypes, math fidelities, DRAM-sharded flags, memory contracts, inner blocks,
  and output-subblock fields. Fresh correctness and latency tests exercise the
  live default at HEAD `b68b16df75d121dcbe0128d6fa4cea98f993b870`.
  Likely subsystem: Potential source/evidence drift after full-model, datatype,
  and integration work.
  Investigation performed: Live source inspection, commit diff inspection,
  current/prior profiler multiset comparison, and current artifact parsing.
  Resolution: controlled; the measured Stage 05 layer topology and contracts
  are unchanged, so the exact historical candidate-family evidence remains
  applicable and the fresh profile corroborates it.

- Observed anomaly: The current full-attention BFP4 projection candidate is
  faster locally but misses the declared accuracy gate.
  Evidence: `candidates/review_attention_bfp4_final.log` records full-layer
  PCC 0.992305137 and
  `candidates/review_cumulative_low_precision.log` records 0.992102801, both
  below 0.995; current BFP8 attention passes at 0.999718188.
  Affected path: Rejected attention precision candidate only.
  Control or comparison: Both sliding and full layer kinds use real checkpoint
  weights under the cumulative selected topology; current profiler rows prove
  BFP8/LoFi reached QKV and O at runtime.
  Likely subsystem: Full-attention sensitivity to BFP4 projection weights.
  Investigation performed: Isolated and cumulative real-weight trials followed
  by current default PCC/profile revalidation.
  Resolution: controlled; the higher-precision selection is based on a real-
  weight model-visible failure, not a synthetic veto.

- Observed anomaly: A fused Linear matmul-plus-reduce-scatter probe hung, while
  the first exact fused all-gather-matmul/down path needed shape/dtype
  adaptation.
  Evidence: `candidates/fused_mmrs_hang_triage.txt.gz`,
  `candidates/ring_l1_AUTODEBUG.md`, `candidates/ring_l1_AUTOFIX.md`, exact
  fused AGMM logs, `candidates/fused_agmm_coherent_final.log`, and
  `tracy/fused_agmm_coherent`.
  Affected path: Rejected fused/lower-movement alternatives, not the current
  Linear persistent-all-reduce default.
  Control or comparison: Ring MM+RS passes exact shape at PCC 0.999963. Exact
  TP4 fused AGMM O shapes pass at PCC at least 0.999963; down was adapted from
  BFP4 PCC 0.993139 to BFP8 PCC 0.999965. The real-weight coherent H/TP-local
  spine carries the lower-movement layout through distributed residual/norm,
  packed gate-up, down, and next QKV without an immediate old-contract restore,
  but is 2.478x/2.553x slower at PCC 0.999752/0.999695.
  Likely subsystem: Ring fused-CCL geometry, narrow TP-local matmul output, and
  distributed norm/residual overhead.
  Investigation performed: Hang triage/recovery, Ring retry, output-subblock
  adaptation, dtype adaptation, persistent gathered-input buffers, immediate-
  gather compatibility controls, and coherent next-QKV endpoint measurement.
  Resolution: controlled; the adapted exact family evidence earns retention of
  the current replicated-boundary topology.

- Observed anomaly: Ethernet watcher instrumentation is disabled in the
  current watcher run.
  Evidence: `evidence/current_head_watcher.log` records only `ETH` disabled;
  the linked Stage 04 watcher failure records a 27,792-byte active-Ethernet
  firmware program exceeding the 25,600-byte kernel-config buffer.
  Affected path: Watcher instrumentation only; Linear fabric remains active.
  Control or comparison: Four current target-mesh tests pass with worker and
  NoC checks enabled, the device watcher dump completes cleanly, and the
  profiler is a separate run.
  Likely subsystem: Firmware/watcher instrumentation capacity, not decoder or
  CCL correctness.
  Investigation performed: Scoped `TT_METAL_WATCHER_DISABLE_ETH=1` rerun with
  decode for both layer kinds and non-aligned sliding-wrap stress.
  Resolution: controlled by an exact infrastructure limitation and clean
  remaining watcher coverage.

## Scope Inspected

- Goal/skill paths: supplied Stage 05 optimized-multichip-decoder contract;
  `.agents/skills/stage-review/SKILL.md`;
  `.agents/skills/optimize/SKILL.md`;
  `.agents/skills/multichip/SKILL.md`.
- Artifact paths: current `README.md`, `work_log.md`,
  `perf_accounting.json`, `candidates/summary.csv`, all prior Stage 05 review
  reports, relevant candidate logs/XML/autodebug/autofix evidence,
  `evidence/current_head_*`, `tracy/current_head` including the 4,066,432-byte
  enriched source CSV and matching SHA-256, prior reviewed profiler roots,
  fused coherent-family evidence, and `doc/context_contract.json`.
- Code paths: live `tt/multichip_decoder.py`, inherited public
  prefill/decode wrappers in `tt/functional_decoder.py`, and
  `tests/test_multichip_decoder.py`. The prompt-supplied
  `tt/multichip_decoder_config.py` path does not exist in this checkout; its
  relevant configuration is embedded in `multichip_decoder.py` and the
  optimization-policy helpers inspected there.
- Commands run: read-only `git status/log/show/diff/diff --check/check-ignore`,
  `find`, `rg`, `sed`, `stat`, `sha256sum`, JSON/XML/CSV parsing, independent
  profiler summation, operation-signature multiset comparison, and artifact
  path verification. No TT device, server, profiler capture, watcher, reset,
  or hardware command was run by this reviewer.

## Residual Risk

- The authoritative current default is supported by 12 standard-suite passes,
  2 advertised-position passes, 2 warmed-latency passes, 4 separate watcher
  passes, and 4 profile windows on real `MeshShape(1, 4)` hardware. PCC covers
  both meaningful dense layer kinds, all four replicas, non-aligned prefill and
  sliding wrap, permuted page tables, mutable/repeated trace replay, batch 32,
  BFP8 paged cache, and terminal shared-pool teardown.
- The current source CSV hash verifies. Independent filtered-CSV sums reproduce
  427.9625/481.827 us device decode, which reconciles to authoritative current
  host medians 0.4635775/0.5181375 ms and the documented 35.615/36.3105 us
  remainder. Runtime rows prove BFP8/LoFi attention, BFP4/LoFi packed gate-up
  and down, BFP8 cache, and two persistent BFP8 async reductions restored to
  the replicated BF16 DRAM layer boundary.
- No implementation, test, or context-contract path is dirty relative to the
  measured source HEAD. Stage-owned dirty state is confined to current
  revalidation documentation/evidence plus this report and can be checkpointed
  without the unrelated requirements deletion, `.exp_run`, `fusion_tests`,
  full-model artifacts, or `vllm` tree.

CLEAN PASS
