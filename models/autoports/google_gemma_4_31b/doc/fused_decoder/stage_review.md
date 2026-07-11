# Stage Review

Verdict: more-work-needed

## Required Work

- P1: Prove mutable-input trace replay on the fused decode graph.
  Evidence: `tests/test_fused_decoder.py:59-135` captures and repeatedly replays only one unchanged batch-1 token/position. The reused batch-32 test at `test_functional_decoder.py:769-777` copies the original token and original positions back into the captured allocations, then checks identical output; it does not mutate either input. The existing functional regression at `test_functional_decoder.py:400-524` does mutate token and position across random/non-block-aligned and window-wrap positions, but the fused suite never invokes it with `FunctionalDecoder` replaced by `FusedDecoder`. `FusedDecoder._decode_attention` changes QKV placement, cache-update operations, cache grids, and the traced graph itself, so the functional-stage run does not prove the delivered fused trace consumes changed buffers.
  Why this matters: preserving trace replay, paged-cache ownership, and current-position semantics is an explicit stage contract. Identical replay proves determinism but cannot detect a captured stale token, stale RoPE position, or stale cache-update index.
  Required next step: add/run a fused-path version of `test_changed_trace_buffers_random_and_boundaries` for both layer kinds and both random/non-aligned and sliding-window-wrap cases, require PCC against the second HF decode plus a stale-output negative control and repeated determinism, and retain a refreshed watcher-clean log for that fused mutable-input trace path.

- P1: Close or measure the unfused long-prefill GELU path before claiming graph-fusion exhaustion.
  Evidence: `_FusedSharedMLP.__call__` in `tt/fused_decoder.py:53-71` installs `fused_activation` only when `m_tiles <= 4`; otherwise it executes a separate `ttnn.gelu`. Long prefill is split into `MLP_CHUNK = 4096`, so advertised 262144/262113-token execution repeatedly takes the standalone-GELU branch. `README.md` says forcing small fused chunks would regress end-to-end latency, but there is no long-prefill profiler/candidate artifact, adapted larger-M program configuration, minimal op-contract/L1 blocker, or measured chunk-family comparison supporting that rejection. The retained activation candidate covers only sequence 128 and establishes that `activation=` without an explicit program config does not fuse; it does not reject a legal explicit fused configuration for the 4096-row chunks.
  Why this matters: the original contract requires exhausting dedicated fused ops/merges and leaving no remaining applicable decoder graph optimization within current TTNN/P150 capabilities. The graph-fusing skill requires adapting a rejected candidate and measuring it or recording an exact contract blocker; an unmeasured prose assertion is not closure.
  Required next step: try legal explicit fused-GELU program configurations for the long MLP chunk geometry (including adapted chunk sizes if necessary), PCC-check the real long path, and retain comparable warmed performance evidence. Keep the fastest correct family, or document an exact TTNN/L1/divisibility blocker or a measured end-to-end regression.

- P2: Refresh evidence against the delivered test source and make profiler/candidate provenance self-consistent.
  Evidence: the current `tests/test_fused_decoder.py` mtime is 2026-07-11 08:37:12 UTC, later than `standard_suite.log` (08:29:04), `watcher_refresh_decode.log` (08:29:31), all final Tracy artifacts (08:32:22), and `long_nonaligned_262113.log` (08:36:23). The work log records the current test hash `62baf70f...`, but no recorded run embeds that hash or postdates the file. In addition, final Tracy console logs say they wrote to `doc/fused_decoder/tracy_refresh/...`, while the delivered canonical files are under `tracy/...`; two rejected-candidate report texts say they wrote to the canonical selected paths rather than their candidate directories. Independent CSV recomputation is internally consistent, but file movement is undocumented.
  Why this matters: the review cannot determine from retained evidence whether the post-run test edit was import-only or changed coverage/assertions. Stage-review treats a required check that may be stale as required work, and ambiguous report destinations weaken the claimed candidate-to-final lineage.
  Required next step: rerun the complete standard suite after the final test edit; record current implementation/test hashes with the command/result; refresh any affected watcher/long evidence if the edit was behavioral. Document the artifact move/copy procedure or regenerate report console text with final destinations so each raw CSV, filtered CSV, and rendered report has unambiguous provenance.

## Other Concerns

- The selected sliding-decode total recomputes to 2556.330 us, while the post-projection candidate is 2557.140 us: only 0.810 us (0.032%). This satisfies the literal single-sample ordering but is far below the other reported gains and has no repeated-run distribution. Treat the claim that the final path beats every candidate as fragile; preferably repeat both variants under the same warmed regime and report median/spread.
- The long 262144 full-attention populated-cache parity is PCC 0.996631, above the 0.995 bar but materially below short decode (0.999624) and the 262113 control (0.998192). The artifact records the value but does not investigate the exact-limit delta. This is not independently a failure because it passes the declared bar and the inherited functional stage documented similar exact-limit risk, but it should remain visible.
- The checkout contains a same-family production Gemma 4 implementation under `models/demos/gemma4`, including shared MLP/attention wiring, but no same-model, same-stage, single-P150 fused-decoder profiler reference was found. Thus there is no demonstrated faster reference regression, but the search/result should be stated explicitly in the stage work log.

## Hard-Check Gaps

- The four final performance tables are single signposted device executions. Independent sums reproduce 3426.833 us/26 ops (sliding prefill), 4196.175 us/23 ops (full prefill), 2556.330 us/40 ops (sliding decode), and 2883.920 us/39 ops (full decode); the functional baselines reproduce 3521.305, 4254.306, 2576.819, and 2911.473 us respectively.
- No dynamic interposition test forbids host fallback. Static source inspection covers all local fused helpers and the signposted reports show zero host ops; this is adequate for the measured paths, but it does not dynamically audit every inherited/transitive helper on long paths.
- The context contract JSON still points to functional-stage evidence. Capacity did not change, so leaving its numeric contract unchanged is appropriate, but adding fused evidence references would make stage-level provenance clearer.

## Anomaly Ledger

- Observed anomaly: traced correctness replays only unchanged inputs.
  Evidence: `tests/test_fused_decoder.py:120-133` and `test_functional_decoder.py:769-777`.
  Affected path: fused traced decode for sliding and full layers.
  Control or comparison: the functional suite has a real mutation regression at `test_functional_decoder.py:400-524`, but it was not run through `FusedDecoder`.
  Likely subsystem: trace input-buffer/current-position/cache-update binding.
  Investigation performed: inspected the fused and reused functional tests, decode implementation, watcher command, and retained logs.
  Resolution: more-work-needed.

- Observed anomaly: long prefill leaves GELU as a standalone device operation by construction.
  Evidence: `tt/fused_decoder.py:53-71`; `MLP_CHUNK = 4096` in `tt/functional_decoder.py:37`; no matching long-path candidate/profiler artifact under `doc/fused_decoder/candidates` or `tracy`.
  Affected path: 262144/262113 prefill MLP chunks and any prefill with more than 128 rows.
  Control or comparison: seq-128 explicit program config fuses GELU and removes the unary row; the only failed activation candidate is also seq 128.
  Likely subsystem: matmul program configuration/chunk geometry.
  Investigation performed: traced the MLP branch condition, enumerated retained candidates, searched Gemma/common MLP fused-activation configurations, and compared the graph-fusing rejection standard.
  Resolution: more-work-needed.

- Observed anomaly: delivered tests postdate every recorded validation artifact.
  Evidence: filesystem mtimes listed in Required Work; the current hash is recorded only in prose.
  Affected path: evidence provenance for all fused test gates.
  Control or comparison: implementation source predates the refreshed standard, watcher, Tracy, and non-aligned long runs, and its current SHA-256 matches the work log.
  Likely subsystem: post-validation formatting or test maintenance; the retained artifacts cannot distinguish it from a behavioral edit.
  Investigation performed: compared mtimes, hashes, worktree status, logs, and compiled-cache times.
  Resolution: more-work-needed.

- Observed anomaly: profiler console destinations do not match delivered directories, and some candidate report texts name canonical destinations.
  Evidence: `tracy/*/*/*console.log` names `tracy_refresh/...`; candidate report texts for `activation_argument_not_fused...` and `folded_mlp_scalar_only...` name canonical `tracy/...` outputs.
  Affected path: artifact lineage, not recomputed values.
  Control or comparison: raw and filtered CSV row counts/op types agree, signpost filtering is shown, and independent sums reproduce all reported totals.
  Likely subsystem: post-run artifact relocation/copying.
  Investigation performed: read every report console/text file and recomputed CSV totals/op counts.
  Resolution: controlled numerically, but provenance remediation is included above.

- Observed anomaly: runtime logs warn that firmware 19.9.0 is newer than fully tested 19.5.0.
  Evidence: standard, long, watcher, and profiler runtime logs.
  Affected path: hardware environment.
  Control or comparison: required correctness runs pass; refreshed watcher attaches/checks/detaches without fatal/assert/NOC/overflow/sanitizer findings.
  Likely subsystem: environment compatibility warning.
  Investigation performed: compared warnings across runs and scanned watcher/runtime logs for fault signatures.
  Resolution: controlled.

## Scope Inspected

- Goal/skill paths: supplied Stage 02 fused-decoder contract; `.agents/skills/graph-fusing/SKILL.md`; `.agents/skills/tt-device-usage/SKILL.md`; `.agents/skills/stage-review/SKILL.md`.
- Artifact paths: `doc/fused_decoder/README.md`, `work_log.md`, `standard_suite.log`, both long-context logs, both watcher trees/logs, all functional/fused Tracy raw and filtered reports, all retained candidate reports, and `doc/context_contract.json`.
- Code paths: `tt/fused_decoder.py`, `tt/functional_decoder.py`, `tests/test_fused_decoder.py`, `tests/test_functional_decoder.py`, and searched Gemma/common MLP/attention implementation references.
- Commands run: read-only `sed`, `nl`, `wc`, `find`, `rg`, `stat`, `sha256sum`, `git status`, `git diff --check`, and small read-only Python CSV analyses. No TT device, reset, server, profiler, vLLM, or hardware experiment was started.

## Residual Risk

- Current short/batched/long PCC evidence is above the 0.995 acceptance threshold, nonidentity page tables and wrapped sliding-cache ownership are exercised, final measured windows contain no host ops, and raw CSV totals/op counts substantiate the headline tables.
- The verdict remains `more-work-needed` because mutable trace semantics and fusion exhaustion are stage-critical requirements, and final test-source provenance is not tied to a retained passing run.
- Scope under `models/autoports/google_gemma_4_31b` is contained to the allowed fused implementation, fused tests, and fused documentation; `doc/context_contract.json` is unchanged. Unrelated dirty/untracked workspace entries must remain excluded from the eventual stage checkpoint.
