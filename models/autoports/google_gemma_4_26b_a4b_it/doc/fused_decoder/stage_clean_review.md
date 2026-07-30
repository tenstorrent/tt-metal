# Stage Review

Verdict: more-work-needed

## Required Work

- P1: The dedicated MoE gate/router family is not assessed or earned as a
  rejection.
  Evidence: `FusedDecoder._router_weights` still spells out FP32 linear,
  `topk`, selected-score softmax, scatter, per-expert scaling, and typecasts.
  The graph-fusing skill explicitly requires exploration of dedicated MoE gate
  ops (`generalized_moe_gate`, `moe_gate_mm`, `deepseek_moe_gate`, or the
  reusable gate module). The candidate ledger says TopK is already a dedicated
  op but does not assess these gate kernels, despite this checkout containing a
  Blackhole generalized gate with top-k 8 and selected-score softmax support.
  Why this matters: The user requires every applicable graph-fusing pattern to
  be assessed/tried when possible and no remaining decoder graph optimization.
  A component op being dedicated does not exhaust a larger dedicated fused-op
  candidate.
  Required next step: Check the current Gemma router contract against the
  available gate kernels (128 experts, top-k 8, selected-score softmax,
  per-expert post-scale, prefill/decode shapes). If expressible, adapt and
  measure it with PCC/performance; if not, record the exact shape, layout,
  output, or semantic blocker. Also explicitly classify the single-device
  `moe_compute`/`TTMoEDecode` family; the current common module appears
  mesh-oriented and its checked-in Gemma config does not match this layer's
  128-expert single-device contract, but that rejection is absent from the
  audit.

- P2: Final evidence freeze still predates the accepted manifest.
  Evidence: `final_manifest.md` declares a frozen window ending at 00:51 UTC,
  while both `final_manifest.md` and `final_manifest.sha256` have filesystem
  timestamps of 00:52:21 UTC. The checksum inventory and fail-closed host test
  pass, but the prior review's stale-window finding is therefore not fully
  repaired.
  Why this matters: The final provenance record claims an immutable time window
  that does not contain its own final inventory.
  Required next step: Freeze the documentation after all accepted artifacts
  are final and update the window to include final manifest generation. Rerun
  checksum verification and the seven-test host gate afterward.

- P2: README correctness and measurement descriptions contradict the accepted
  artifacts.
  Evidence: README says the final performance regime is a median of 3 prefill
  runs and 11 decode replays, then says the same final table uses 7 and 101.
  The accepted timing JSONs record 7/101. README also reports sliding PCC
  0.998617/0.999655 and full PCC 0.997773/0.999861, while the manifest-hashed
  final PCC artifacts record sliding 0.9986195/0.9996651 and full
  0.9984834/0.9998653.
  Why this matters: README is the final stage handoff and currently mixes
  obsolete and accepted-run values.
  Required next step: Replace the stale rows with values and repeat counts from
  the accepted JSONs, then freeze and revalidate the final record.

## Other Concerns

- The full-attention batch-1 traced-decode median win is 0.000132 ms
  (approximately 0.004%). It satisfies the literal final-source comparison,
  but the sequential sample distributions overlap substantially, so the
  performance separation remains fragile.
- Several correctness JSONs do not embed source/build provenance. Their
  post-source timestamps, checksum inventory, and later watcher rerun provide a
  coherent final-source chain, so this is not a separate blocker.

## Hard-Check Gaps

- The manifest validator derives the accepted artifact families and now fails
  closed; it does not validate prose values in README or the declared freeze
  time.
- No automated gate checks that every dedicated fused-op family named by the
  graph-fusing skill has a candidate-ledger disposition.

## Anomaly Ledger

- Observed anomaly: The final freeze ends before final manifest generation.
  Evidence: Declared end 00:51 UTC; manifest files timestamped 00:52:21 UTC.
  Affected path: Final evidence provenance.
  Control or comparison: `sha256sum -c` and the fail-closed manifest test pass.
  Likely subsystem: Evidence lifecycle/documentation.
  Investigation performed: Compared manifest declaration, file timestamps,
  checksum verification, and host validator.
  Resolution: more-work-needed

- Observed anomaly: README mixes old and accepted PCC/performance regimes.
  Evidence: README 3/11 versus JSON 7/101; README PCC values differ from all
  manifest-hashed final PCC JSONs.
  Affected path: Final correctness/performance handoff.
  Control or comparison: Direct JSON inspection and checksum verification.
  Likely subsystem: Documentation refresh.
  Investigation performed: Compared README tables to accepted artifacts.
  Resolution: more-work-needed

- Observed anomaly: The previously reported fused-runtime defects are repaired.
  Evidence: The live subclass overrides router, dense MLP, prefill MoE, and
  decode MoE; prefill preserves 32-token splitting; all three GeGLU sites use
  operand activation on the consuming multiply. The host gate passes 7/7.
  Boundary artifacts cover both layer kinds at logical lengths
  1/31/32/33, 63/64/65 or 127/128/129, and 1023/1024/1025, all above PCC
  0.995. Real-weight, trace, cache, context, and watcher artifacts pass.
  Affected path: Fused runtime correctness and dispatch.
  Control or comparison: Functional source, test dispatch, manifest hashes,
  watcher 9/9, PCC/boundary/trace/context JSONs.
  Likely subsystem: Stage-02 implementation.
  Investigation performed: Static code inspection, artifact parsing, checksum
  verification, and host-only pytest.
  Resolution: fixed

- Observed anomaly: Traced replay has no TTNN per-op report.
  Evidence: `profiler_summary.md` separates capture topology from replay and
  retains Blackhole/110-worker raw device trace reports with two nonzero rows
  for sliding/full at batches 1 and 32. Prefill has 523-row Blackhole
  `tt-perf-report` CSVs for each layer kind.
  Affected path: Decode profiler evidence.
  Control or comparison: Synchronized 101-replay host timing JSONs and raw
  device trace firmware/kernel duration.
  Likely subsystem: Tracy retained-trace join limitation.
  Investigation performed: Read raw CSV headers/rows, reports, logs, and the
  limitation summary.
  Resolution: controlled

## Scope Inspected

- Goal/skill paths: supplied Stage 02 contract,
  `.agents/skills/stage-review/SKILL.md`,
  `.agents/skills/graph-fusing/SKILL.md`, and
  `.agents/skills/tt-device-usage/SKILL.md`.
- Artifact paths: fused README/work log/manifests/profiler summary, all prior
  stage reviews, AutoDebug report, PCC/boundary/cache/trace/context/timing JSONs,
  watcher output, prefill reports, and decode raw profiler CSVs.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`,
  `tests/test_fused_decoder.py`, `tests/test_functional_decoder.py`, and
  relevant common/experimental MoE gate and compute locations.
- Commands run: read-only `sed`, `find`, `grep`, `git status`, `git diff`,
  `stat`, `sha256sum`, small read-only JSON/CSV analysis, and host-only
  `pytest -q .../tests/test_fused_decoder.py`. No TT device, watcher, profiler,
  reset, or server command was run.

## Residual Risk

- Runtime correctness, non-aligned lengths, cache behavior, advertised context,
  determinism, watcher cleanliness, profiler-equivalent evidence, and six
  literal median wins are supported by retained artifacts.
- The stage cannot receive `clean-pass` until the unassessed dedicated router
  family is dispositioned and the final documentation/provenance
  contradictions are repaired.
