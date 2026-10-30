# Stage review, round 7: iterations 11-12 (2026-10-30)

Independent review (fresh subagent, read-only, no device use) of commits 884e0eedff6 (docs restructure),
7f0a7f1f896 (iteration 11), 2c3a0d63846 (collective sweep, rejected), a95fcbdf37d (iteration 12) against the
optimization-loop skill (hard constraints, document format, exit and gate), the docs and the evidence.

## Verdict: more-work-needed

## Findings (as returned)

1. **P2. The iteration-11 commit bundles an unrelated UMD submodule bump** (`7eb549a29db -> 292fe208659`, the
   worktree's pre-existing checkout, picked up by `git add -u`); not mentioned anywhere.
2. **P2. `perf_analysis.md` deviates from the document format**: the ranked table was still "TP=4 after
   iteration 6" with deltas appended in cells; the iteration-12 attribution named the wrong rows and understated
   the net (-1.07 vs -1.15 ms per 4 layers from the CSV: the `[32[16]] x [32[16],5120]` row only halved, the two
   adapter-forward rows went too); the rejected-collectives cell was multi-sentence and over-reached ("nothing
   kernel-level to gain": overlap untried); the resumed-state bullet stopped at iteration 10; README still said
   SDPA ~39 and "inherent collectives"; the HiFi2 row sat in the Rejected table against the hard-constraints
   block; the iteration-11 attribution bullet was a paragraph duplicating the details file.

Other concerns: env knobs for a rejected candidate left in the shared CCL wrapper (behaviour unchanged); no
guard against `max_runtime_args` in the WORK_LIST factories; the `lowrank_update` Python router does not check
the op's M <= 4096; the SDPA ranked-table cell should not read as a near-roofline claim (~30 ms vs ~3.5 of FLOP
roofline); the WORK_LIST compute kernels read the item count through a variable named `start_idx`.
Hard-check gaps: the loss match was last run after iteration 10 (iteration 12 changes the rounding of every LoRA
update); the zero-padding of the rank-16 K lanes is an unchecked invariant shared with the previous path.
Re-derived clean: MFU 30.92 / 31.29 % from the logs with the cool-downs observed, per-chip matmul rows equal,
Tracy diffs 10 -> 11 -> 12 (only the new op codes appear), harness 35 / 38, gradient checks PASS, the standalone
checks, HiFi4 + fp32 DEST, CB block discipline, HEAD_SPLIT / reduce page indexing, LPT item coverage, hash.

## Response (same day)

- Finding 1: `415ec0f1d3e` restores the pointer in the tree (the worktree keeps the local checkout that every run
  used; `work_log.md`, "Build environment note").
- Finding 2: ranked table regenerated from `O/iter12_tp_ranked_ops.txt` (chip 0, per-layer rows x10, class and
  remaining rung per row); iteration-12 attribution rows restated from the CSV (-1.15 / -1.18 ms); rejected
  cell shortened, collectives labelled "transfer at the library optimum, overlap untried"; resumed-state bullet
  and README ranking updated; HiFi2 row moved to the Ledger; iteration-11 bullet trimmed.
- Other concerns: `TT_FATAL` on the list length vs `max_runtime_args` in the three factories; the Python router
  falls back to the two ops above M = 4096; the SDPA cell names the re-read pattern, not a roofline; the
  `start_idx` variable commented. The env knobs stay (unset = library defaults; the sweep is reproducible).
- Hard-check gaps: the loss match reruns in the iteration-14 chain; the zero-pad invariant is recorded in the
  Ledger (shared with `ttnn.matmul` on the same tensors).
