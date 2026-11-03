# Phase Review

Stage review round 15: optimisation loop, iterations 48-49 plus the answers to review round 14 (commits `95e7ebd6bd9`
(review 14 answers, the overlap window's hazard fixed and the window rejected), `a0ef2fe9a45` (48), `a6ff2f8375b` (49) on
`agobeljicTT/mistral-bringup`). Independent reviewer, read-only; no device opened, no implementation file modified.
Paths: `D` = `doc/`, `O` = `doc/optimization/evidence/`. The review covers the committed state at `a6ff2f8375b`; during
the review the live worktree acquired uncommitted iteration-50 edits (`frozen_linear.py`'s qkv dX^T config, `*_PENDING`
placeholders in the three documents, untracked `O/iter50_*`), which are outside this round.

Verdict: more-work-needed

Three P2, no P1. The round-14 required work is done and re-derives: the TP-capable loss comparison exists
(`train.py --host-loss` gathers the vocab shards; the 8-layer TP=4 match found the committed window wrong at 0.73 / 0.62
and every serial control right at 0.0066-0.0068, then 0.0066 with the fixed window), the determinism rerun and the
isolating control are bit-identical over 10 real-weights steps (`O/iter48b_real_trajectories.txt`, re-derived from the
three JSONs: all deltas +0.0000), the 2-block compare is tracked with a verdict (worst rel 1.03e-4 on the two A gradients
that take the owned product), the Ledger entry is rewritten with the cause (the async reduce-scatter's host-side frees
under sub-device concurrency) and a protocol, the 32-pair alternating run is reported as mean +- SE (+2.4, SE 0.9;
re-derived: +2.43 +- 0.85 with the (off, on) pairing, +1.89 +- 0.97 with the (on, off) pairing, unpaired +2.43 +- 1.18,
second half +1.3 / +1.6: the sign holds under every pairing), the 47 Result / Attribution carry the per-window device
region and the pipelined baseline, `cycle_diff.py` prints op counts, `O/iter44_cycle_diff_full.txt` carries the vanished
64-core rows and chip 2, iteration 46 is re-classed with its reason. Iterations 48 and 49 re-derive: 48c kernels without
collectives -0.186 / -0.453 ms per 4 layers (chip 0 / 2) at 440.8 ms wall against 46c's 440.5; 49 -0.260 / -0.354 at
436.1 ms (the K-concats 0.372 -> 0.047 at 8 -> 1 op, the pads 0.088 -> 0.036, the h skinny rows 0.359 -> 0.357,
transpose_split +0.035); harness 70 / 71, TP gradient check PASS (rel 1.15-1.49e-2, unchanged), 8-layer loss match PASS
on one chip (0.0234, bit-identical to 48's) and at TP=4 (48: 0.0068; 49: 0.0094, tol 0.1), the 40-layer real-weights
trajectories 46 -> 48c within 0.006 (mixed signs) and 48c -> 49 within 0.002 (steps 4, 5, 6, 9: +0.001, +0.002, -0.002,
+0.002). The wide-norm kernels are sound (below). What does not hold: the ranked cost table describes the rejected
window-on path and predates 49; iteration 48's Attribution cell does not add up to its Result and its "quarter of the
estimate" reason points at the wrong side of the estimate; the iteration-47 record still carries the pre-fix prose and
the bold "best" marker on a path recorded as faulty.

## Required Work

- P2: The ranked cost table is sourced from the rejected window-on profile and predates iteration 49
  Evidence: `D/perf_analysis.md` "Ranked cost table" at `a6ff2f8375b` (line 125): "Source: TP=4 after iteration 48, chip
  0 of `O/iter48_tp_ranked_ops_full.txt`"; that profile ran with the (faulty) window on (`O/iter48_measure.log`:
  "collective overlap (iteration 47): installed", "collective overlap windows 80"). The table's rows: "8 reduce-scatters
  run inside the overlap windows (iterations 47-48)", the collectives row "3.88 (window rows, 8 ops)", the LoRA row
  "gate_up dB 0.41 (77 cores, in the window)", "the K-concats 0.39 (gate_up + qkv) + 0.04 (out)", "pads / fills 0.15",
  and the LoRA row's remaining rung "the norms write a wide `[T, in + 64]` output ... (~-0.5 ms per 4 layers)", the norm
  row's rung "the wide output (above) is the next change". On the landed path (`O/iter48c_tp_ranked_ops_full.txt`,
  `O/iter49_tp_ranked_ops_full.txt`): no window, the reduce-scatters on 36 cores (10 + 8 ops), dB on 110 cores
  (`SkinnyContract ... 16384x1024 | 110`), the K-concats 0.047 (1 op), the pads 0.036, the wide output landed. The 49
  commit changed `perf_analysis.md` by 10 lines (two history rows, the 49 section) and not the table; the 48 commit did
  not touch it.
  Why this matters: the skill updates the table every iteration and the table is the loop's steering instrument; round
  14 (P2) asked for exactly this regeneration, and the answer regenerated it from a profile of a configuration the same
  commit rejected. The next item (50, in flight) is being costed from the raw table files again.
  Required next step: regenerate the table from `O/iter49_tp_ranked_ops_full.txt` chip 0 (with chip 2 / the critical
  chip's collectives as before), the collectives rung restated as "overlap on sub-devices tried and rejected (iteration
  47 / 48b: +2.4 ms per step, ~10 us of real overlap per window)", the LoRA row's rung advanced past the wide output
  (what remains: transpose_split 0.52, the dB / dA products, the first layer's concat), the matmul row's qkv dX^T rung
  pointing at iteration 50; the source line naming the landed configuration (overlap off).

- P2: Iteration 48's Attribution does not reconcile with its Result, and the "quarter of the estimate" reason is misplaced
  Evidence: `D/perf_analysis.md` Iteration 48 Attribution (chip 0 per 4 layers): "the composite attention glue gone
  (-1.64 ...), the fused glue in (+0.84 ...), the base matmuls +0.28" = -0.52, against the Result's "-0.186 (chip 0)".
  The cell's own listed items sum to -1.50 (LowrankUpdate 0.24 + 0.13, matmuls 0.19 + 0.17 + 0.12 + 0.10 + 0.10, adds
  0.22, skinny 0.14 + 0.03), not -1.64; the rows that vanished or shrank in `O/iter48c_cycle_diff_vs46c.txt` chip 0
  (re-derived: LowrankUpdate 0.237 + 0.128, BinaryNg 0.222, the `[T, 32]`-class matmuls 0.188 + 0.173 + 0.117 + 0.096 +
  0.051 + 0.037, SkinnyMatmulPartial / Reduce 0.144 + 0.034) sum to -1.43; the fused glue +0.83 and the base matmuls
  +0.28 re-derive; the norm rows +0.05 and the sub-threshold rows (+0.09) are not in the cell. The work log's reason
  ("a quarter of the expected -10 because the fused route's transpose_split and K-concat cost ~115 us per layer and
  its qkv dX^T is 50 % slower") blames glue that the estimate already carried ("the fused glue per layer qkv ~200 us
  (h 36, the K-concat 50, transpose_split 61, ...) ... 1.1 ms per 4 layers"): the fused glue landed at 0.83 against 1.1
  expected. The shortfall is on the other side: the estimate removed "2.35 ms per 4 layers" of composite glue
  (`O/iter47_tp_ranked_ops_full.txt`) while the rows the swap actually deleted are 1.43 (the 2.35 counted rows the
  swap does not touch: the heads layout ops, the kv concat, the slices, the transposes of other layers), plus the qkv
  dX^T +0.28.
  Why this matters: the format's Attribution cell is "the rows that moved, as numbers"; a cell whose rows sum to three
  times its Result, and a Lesson line that names the wrong cause, mis-steer the next estimates (the same "dozen small
  ops" base was quoted in the 48 commit message and the README row).
  Required next step: the Attribution cell restated with rows that add to -0.186 (composite -1.43, fused +0.83, base
  matmuls +0.28, norms +0.05, remainder +0.09) and the reason corrected (the composite-glue base was 1.43, not 2.35;
  the fused glue came in under its estimate); the work log's "Expected / Measured" and Lesson lines and the README row's
  "(the composite attention glue -1.64, ...)" corrected likewise. No measurement needed.

- P2: The iteration-47 record and the commits table were not corrected cell by cell after the hazard finding
  Evidence: (a) `D/perf_analysis.md` Iteration 47 Verification cell (line 493) still ends "... the rounding-level dA
  difference amplified over 40 layers of sign-like early AdamW steps (Ledger)" while the Ledger entry (line 603) says
  "a real hazard, not rounding" and the Attribution cell says "the window as committed here was wrong": the finding
  was answered by adding sentences elsewhere, not by correcting the cell. (b) The MFU history row 47 (line 95) keeps
  the bold best markers "**433.3 (431.6)** | **37.44 %**" and the 47 section's Result cell likewise, on a path the
  same document records as rejected and incorrect; the README State line says "measured on a faulty overlap window".
  (c) The work log's Commits table at `a6ff2f8375b`: the 48 row is "| 48 | `a0ef2fe9a45` |" with no description, and
  the 49 row carries two descriptions (its own and 48's in an extra cell); the worktree edit only replaces "(this
  commit)" with the SHA. (d) The README Conventions harness-count line stops at "70 (47)" (71 at 49 missing; round 14
  flagged the same line). (e) The Rejected table's iteration-7 row ("qkv / out TP shards on the fused path ... 551.3 vs
  547.5") has no pointer to its adapted retry (48), although the 48 section calls itself that retry.
  Why this matters: the skill's format rule ("a review finding is fixed by correcting the bullet or table cell, not by
  adding a caveat paragraph") and the MFU history's bold convention (the loop's best correct number) both fail here;
  the history now marks a result the loop itself says it cannot stand behind, while the best correct number is 49's.
  Required next step: the 47 Verification cell's last clause replaced by the hazard finding; the 47 history row and
  Result cell un-bolded with "(faulty window, rejected)" in the change cell; the commits table rows 48 / 49 repaired;
  the harness-count line extended to 71 (49); the iteration-7 Rejected row pointing at 48.

## Other Concerns

- The overlap's rejection is earned: the fixed window measured in process over 32 pairs (+2.4 +- 0.9; both pairings
  positive; 22 of 32 pairs positive), explained mechanically (28 us of switch waits, the clone off the persistent output,
  dB on 77 cores at 132 vs 95 us, against ~10 us of real overlap per window measured in round 14), and the mechanism
  kept opt-in. Two things belong in the Rejected row: the run's trend (first 8 pairs +5.0, last 8 +1.0; the off steps
  warm faster, 0.196 vs 0.120 ms per step) and the ceiling that closes the question: at ~10 us of real overlap per
  window and 80 windows, even a free window returns ~0.8 ms per step (0.2 %), so the untried cheaper variants (one
  buffer set per window to drop the clone, a 1-2 row collective split so dB keeps 88-99 cores) cannot pay.
- Commit bisection: `95e7ebd6bd9` (the review answers) carries iteration 48's wiring (`out_linear.transposed_output`,
  the opt-in overlap on the fused qkv, `fused_forward` handing the attention the shard) and 48's document sections,
  while `a0ef2fe9a45` (48) carries only `frozen_linear.py`; at `95e7ebd6bd9` the documents describe code the tree does
  not yet run (the wiring is inert until the shapes join `TRANSPOSED_SHAPES`). The review commit also mixes the hazard
  fix, the window's rejection, `--host-loss` and the controls. It bisects, but the 48 section sits one commit early.
- Iteration 49 moved the TP=4 numerics at rounding level (8-layer TP=4 match 0.0068 -> 0.0094; the 40-layer trajectory
  +0.001 / +0.002 / -0.002 / +0.002 at steps 4-6, 9; the 1-layer gradient check's rel 1.49e-2 -> 1.48e-2 on qkv B) while
  the kernels are bit-identical standalone and the single-chip 8-layer match is bit-identical to 48's (0.0234, every
  step equal; the single-chip model also sets the tail since its qkv and gate_up are fused). The only candidates are
  the collectives on the wide `[.., 5184]` shapes (the gather and the reduce-scatter of the wide gradient); the
  Verification cell says "within one bf16 ulp (0.002)" (step 5 moved 0.0019 at 0.14, two ulps at that magnitude) and
  does not say where the difference comes from. Rounding-level and controlled by the TP match; name the source.
- The wide-norm kernels check out: the writer's tail zeros come from `cb::zero`, which the reader fills before anything
  else and the compute waits on before producing `cb::out`, so slice 0's tail write (after its last `cb::out` block)
  reads a filled tile; the op enforces bf16 (`check_device_tensor` with the default `{BFLOAT16}` dtype list), so the zero
  tile's size matches the output page; every slice writes at the wide stride `Wt_out`; `out_tail_tiles` is in the
  forward hash and `dL_dout.logical_shape()` in the backward hash; the backward reader reads `dL_dout` at one site with
  `dy_page = r * Wt_dy + col0`; the two-phase paths refuse a wide output / gradient loudly (`TT_FATAL(out_tail_cols ==
  0)`; the partial op's `dL_dout must match the input shape`); the first layer's qkv input comes from the plain first
  norm (narrow: one K-concat per step remains, 0.047 ms), the final norm (`ln_fc`) stays plain, and `use_fused_
  residual_norm` sets the tail only when every qkv / gate_up is a fused LoRA layer (a composite consumer would see a
  wider input). Nits: `add_rmsnorm_fw_allcore` truncates `out_tail_cols / 32` silently (a non-multiple of 32 would
  misplace the tail; `TAIL_ROWS` = 64 today); no harness case combines the wide gradient with `transposed_grad` (the
  mlp_norm's backward on the landed path does both; covered only by the TP loss match and the gradient check).
- `CollectiveOverlap._persistent_buffers` hard-codes the TP degree (`self.tp = 4`, `out_shape[dim] //= 4`); opt-in only.
- Evidence hygiene: `O/iter47_loss_comparison_tp4_{overlap,no_overlap}.txt` (the round-14 tracebacks) are still untracked
  in the worktree and cited nowhere; delete them. The 48b chain log ends at "iter48b_tp TP=4 MFU after 600s cool-down"
  with no rc line (the chain was cut once the alternating run rejected the window; say so in the log or the work log).
- The MFU history's 48 alternating row (on 467.3 / off 464.9) is a warm-box run (465 ms level against 440 cooled) without
  the "warm" label the table's header promises; the evidence cell says "4-minute cool-down".
- The 48 / 49 cells give the MFU run's step-phase clock range (0.99-1.26, 0.97-1.35 GHz) but not the profile's
  critical-chip clock the contract asks for (`O/iter48c_tp_analysis.txt`: chip 1 at 0.929 GHz; `O/iter49_tp_analysis.txt`:
  chip 3 at 0.941); the cycle-time rows are clock-independent, so the omission costs nothing, but the cell should carry it.
- The worktree moved during the review (iteration 50 in flight): `frozen_linear.py` already carries 50's qkv dX^T config,
  the three documents carry `PA50_*` / `IT50_*` placeholders. This review is of the committed state.

## Hard-Check Gaps

- An attribution checksum: `cycle_diff.py` (or a sibling) printing per-category sums (composite glue, fused glue, base
  matmuls, norms, remainder) so an Attribution cell is copied from a script output that adds to the Result.
- The ranked table regenerated by script from the landed profile each iteration, with the source profile named and its
  configuration (overlap off) stated in the source line.
- A harness case for the wide gradient with `transposed_grad` (the landed mlp_norm backward).
- The Rejected row's ceiling for overlap work (us of real overlap per window x windows per step) as a standing number.

## Anomaly Ledger

- Observed anomaly: the committed iteration-47 window corrupted the gradients.
  Evidence: `O/iter48_loss_comparison_tp4_overlap_one_semaphore_set.txt` (max 0.73), `..._rotating_semaphores.txt`
  (0.62), the serial controls `..._no_overlap.txt` 0.0068, `..._serial.txt` 0.0066, `..._serial_async.txt` 0.0066, the
  fixed window `..._overlap.txt` 0.0066; `O/iter47_full_tp4_real.json` drift (+0.047 at step 10).
  Affected path: `CollectiveOverlap.reduce_scatter` (`reduce_scatter_minimal_async` with host-freed staging buffers under
  sub-device concurrency).
  Control or comparison: the three serial controls; the fixed window against torch; two fixed-window 40-layer runs
  bit-identical to each other and to the serial-routes control.
  Likely subsystem: the async op's staging buffers freed on the host right after the enqueue.
  Investigation performed: loss matches re-read; the six JSONs and the three real-weights JSONs diffed step by step.
  Resolution: fixed (persistent staging / output buffers, rotating sets, the clone after the window); the window rejected
  on time.
- Observed anomaly: the fixed window is slower than no window.
  Evidence: `O/iter48b_mfu_alternate.json` (32 + 32 steps): on 467.3 (sd 4.3) / off 464.9 (sd 5.1); paired +2.43 +- 0.85
  ((off, on) pairs), +1.89 +- 0.97 ((on, off) pairs), unpaired +2.43 +- 1.18; halves +2.5 / +1.3; slopes 0.120 (on) /
  0.196 (off) ms per step.
  Affected path: `--ccl-overlap` (opt-in); the Rejected row for 47.
  Control or comparison: the three warm 40-layer runs (474.2 / 475.2 overlap, 474.1 serial) equal within noise.
  Likely subsystem: the switch waits, the clone, dB on 77 cores, against ~10 us of real overlap per window.
  Investigation performed: the statistics re-derived under both pairings; the trend measured.
  Resolution: controlled (rejected, off by default); the ceiling to be recorded (Other Concerns).
- Observed anomaly: iteration 48 landed at a quarter of its estimate.
  Evidence: `O/iter48c_cycle_diff_vs46c.txt` (-0.186 / -0.453), the work log's estimate (-1.1 ms per 4 layers).
  Affected path: the 48 Attribution cell, the work log's Lesson, the README row.
  Control or comparison: the vanished rows summed from the diff (composite -1.43 against the quoted 2.35; the fused glue
  +0.83 against 1.1 expected; the base matmuls +0.28).
  Likely subsystem: the estimate's base counted rows the swap does not remove.
  Investigation performed: per-row categorisation of the diff file.
  Resolution: more-work-needed (P2: the cell and the reason restated; no measurement).
- Observed anomaly: iteration 49's TP=4 numerics moved at rounding level while its kernels are bit-identical standalone
  and the single-chip match is bit-identical to 48's.
  Evidence: `O/iter49_loss_comparison_tp4.txt` 0.0094 vs `O/iter48_loss_comparison_tp4_no_overlap.txt` 0.0068;
  `O/iter49_full_tp4_real.json` vs `O/iter48c_full_tp4_real.json` (max 0.002); `O/iter49_norm_tests.log`.
  Affected path: the wide shapes through the all-gather and the reduce-scatter.
  Control or comparison: the single-chip 8-layer match (every step equal to 48's); the 1-layer TP gradient check (rel
  unchanged to the third digit).
  Likely subsystem: the collectives' chunking / accumulation on the 162-tile-column shape.
  Investigation performed: the loss files and JSONs diffed; the kernel diffs read.
  Resolution: controlled (within the match's tolerance and the trajectory's past rounding moves); the source to be named
  in the Verification cell.
- Observed anomaly: the ranked cost table describes the window-on configuration.
  Evidence: `D/perf_analysis.md` line 125 against `O/iter48_measure.log`.
  Affected path: the next iterations' costing.
  Control or comparison: `O/iter49_tp_ranked_ops_full.txt`.
  Likely subsystem: the regeneration ran before 48c / 49 and was not repeated.
  Investigation performed: the source line, the measure log and the 48 / 49 commit diffs read.
  Resolution: more-work-needed (P2).
- Observed anomaly: the 48c "cooled" MFU run started with the chips at 67-69 C (the 49 run at 66-68 C); step-phase clocks
  0.99-1.26 and 0.97-1.35 GHz.
  Evidence: `O/iter48c_tp_mfu_tp4_tt-smi.txt`, `O/iter49_tp_mfu_tp4_tt-smi.txt`.
  Affected path: the wall numbers 440.8 / 436.1 (the difference, 4.7 ms, is inside the 1-2 % noise).
  Control or comparison: the cycle-time rows (-0.26 / -0.35 ms per 4 layers: -2.6 to -3.5 ms per step), which carry the
  result.
  Likely subsystem: the box's idle temperature after a 600 s cool-down.
  Investigation performed: tt-smi tails read.
  Resolution: controlled (the Result cells cite the cycle-time rows).

## Scope Inspected

- Goal/skill paths: `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`,
  `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/optimization-loop/SKILL.md`.
- Artifact paths (under `/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/`):
  `perf_analysis.md` (MFU history, Ranked cost table, Iterations 47-49, Rejected, Ledger; worktree and `git show
  a6ff2f8375b:`), `optimization/README.md`, `optimization/work_log.md` (review round 14 answers, Iterations 48-49, the
  Commits table; worktree and HEAD), `optimization/review_round14.md`; `optimization/evidence/`:
  `iter48_loss_comparison_tp4_{overlap_one_semaphore_set,overlap_rotating_semaphores,no_overlap,serial,serial_async,overlap}.txt`
  and their `iter48_loss_8layers_tp4_*.json`, `iter47_tp_loss_check.log`, `iter48b_chain.{sh,log}`,
  `iter48b_real_trajectories.txt`, `iter48b_full_tp4_real_{overlap_a,overlap_b,serial}.json`,
  `iter48b_tp_grad_check_2layers_compare.txt`, `iter48b_tp_grad_check_tp4.log`, `iter48b_alternate_summary.txt`,
  `iter48b_mfu_alternate.{json,log}`, `cycle_diff.py`, `iter44_cycle_diff_full.txt`, `iter48_cycle_diff.txt`,
  `iter48c_cycle_diff_vs46c.txt`, `iter49_cycle_diff.txt`, `iter48_measure.log`, `iter48_chain.log`, `iter48c_chain.log`,
  `iter49_chain.log`, `iter48_tp_mfu_tp4.log`, `iter48c_tp_mfu_tp4.log`, `iter49_tp_mfu_tp4.log` and their `_tt-smi.txt`,
  `iter48_tp_tracy_tp4_4layers_tt-smi.txt`, `iter48c_tp_tracy_tp4_4layers_tt-smi.txt`, `iter49_tp_tracy_tp4_4layers_tt-smi.txt`,
  `iter48_tp_analysis.txt`, `iter48c_tp_analysis.txt`, `iter49_tp_analysis.txt`, `iter46c_tp_ranked_ops_full.txt`,
  `iter47_tp_ranked_ops_full.txt`, `iter48c_tp_ranked_ops_full.txt`, `iter46_full_tp4_real.json`,
  `iter48c_full_tp4_real.{json,log}`, `iter49_full_tp4_real.{json,log}`, `iter48_loss_comparison.txt`,
  `iter49_loss_comparison.txt`, `iter49_loss_comparison_tp4.txt`, `iter48c_tp_grad_check_tp4.log`,
  `iter49_tp_grad_check_tp4.log`, `iter48_harness.log`, `iter48c_harness.log`, `iter49_harness.log`, `iter49_norm_tests.log`.
- Code paths (under `/home/ttuser/agobeljic/tt-metal/`): `tt-train/sources/ttml/ttml/modules/ccl_overlap.py` (diff
  `95e7ebd6bd9`), `tt-train/sources/ttml/ttml/modules/fused_lora.py` (diff and the forward's wide-path block),
  `tt-train/sources/ttml/ttml/modules/frozen_linear.py` (diff `a0ef2fe9a45`; the worktree's iteration-50 edit noted),
  `tt-train/sources/ttml/ttml/models/mistral/fused_residual_norm.py` (diffs `95e7ebd6bd9`, `a6ff2f8375b`; whole file),
  `tt-train/sources/examples/mistral_train/train.py` and `tp_gradient_check.py` (diff `95e7ebd6bd9`),
  `tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_allcore/add_rmsnorm_fw_allcore.{cpp,hpp}`,
  `tt-train/sources/ttml/metal/ops/add_rmsnorm_fw_rows/**` (device operation, types, program factory, the CB map, the
  reader, the writer, the compute kernel's `cb::zero` use), `tt-train/sources/ttml/metal/ops/rmsnorm_bw_rows/device/**`
  (device operation, program factory, the reader), `tt-train/sources/ttml/metal/ops/rmsnorm_bw_allcore/**` (the
  dispatcher and the two-phase validation), `tt-train/sources/ttml/metal/common/tensor_validation.hpp`,
  `tt-train/sources/ttml/metal/common/dataflow_utils.hpp` (the zero-tile generator),
  `tt-train/sources/ttml/nanobind/nb_ops.cpp` (diff), `tt-train/tests/python/test_mistral.py` (diff:
  `test_add_rmsnorm_wide_output_matches_the_plain_one`).
- Commands run: `git show --stat` and `git show <commit> -- <path>` on the three commits; `git status`, `git diff`
  (documents, `frozen_linear.py`), `git show HEAD:<doc>`; grep over the logs and documents; python3 (`-I`) over the loss
  JSONs (per-step deltas between 46 / 48c / 49 / 48b a / b / serial), the alternating-run JSON (means, sds, both
  pairings' mean +- SE, halves, slopes, positive-pair counts), the cycle-diff files (per-category sums of the 46c ->
  48c rows) and the ranked tables (rows present in 46c and absent in 48c, rows new in 48c, the small-op rows of 47). No
  device opened; no training, profiling or test run; no implementation file modified.

## Residual Risk

- The loop's best correct number (49: 436.1 / 37.20 %) rests on 1 % of wall against 46c (440.5) and on cycle-time rows of
  -0.45 to -0.8 ms per 4 layers across 48 + 49; the wall is within the stated noise, the rows carry the claim.
- The collectives (4.8-10 ms per 4 layers per chip, skew-dominated) are now a measured floor: overlap on sub-devices was
  tried and lost; the remaining lever is the chips' clock skew, which no kernel change addresses.
- Every TP-keyed numerics change is now checked by the 8-layer TP=4 torch match (the round-14 gap closed); the
  single-chip match stays bit-identical across 48 / 49 and no longer discriminates.
- Iteration 50 is already in flight on a library program config (the qkv dX^T on 110 cores, a swept-config "reuse" rung
  with its reason stated in the README row); the kernel-level item of that iteration is not yet visible.
