# Phase Review

Stage review round 14: optimisation loop, iterations 44-47 (commits `5223e39f0e9` (44), `551e4e8bd1c` (45), `263dfc67576`
(46), `4c96f9fde82` (47) on `agobeljicTT/mistral-bringup`). Independent reviewer, read-only; no device opened, no
implementation file modified. Paths: `D` = `doc/`, `O` = `doc/optimization/evidence/`. The review covers the committed
state at `4c96f9fde82`; during the review the live worktree acquired uncommitted iteration-48 edits (see Scope), which
are outside this round.

Verdict: more-work-needed

One P1 and four P2. The kernel-level parts re-derive: the norm rows' slice cap (44: 0.768 -> 0.713 ms per 4 layers on
chip 0, 0.753 -> 0.690 on chip 2, standalone 46 -> 43 / 50 -> 46 / 60 -> 56 us), the swept 2D configs (46: the four rows
-0.244 ms per 4 layers on both chips, the op count 3 -> 4 -> 3 confirmed), the trace replay's "no gain" (45: ABBA 444.9
vs 445.2 cooled, in-process -0.5 %, losses equal), the sub-device window (47: the three programs overlap on device, the
20-core reduce-scatter bit-identical in the scoping, `grid_rows` in the hash, factory and a bit-exact harness case), the
harness counts (66 / 66 / 68 / 70), the SwiGLU op tests (35), the 1-layer TP gradient checks (PASS, unchanged rel
1.4-1.7e-2) and the single-chip 8-layer loss matches (0.0234 / 0.0229 / 0.0234 / 0.0234). What does not hold: the landed
iteration-47 path (the loop's result, 433.3 ms) has no passing torch-referenced loss evidence that exercises it, its
40-layer trajectory drifts one-signed to +0.047 while the scripted TP=4 loss check failed on a flag and went unrecorded,
and its measured gain is inside the step noise with the device profile showing a tenth of the claimed amount on the
critical chip.

## Required Work

- P1: Iteration 47's correctness evidence for the landed (overlap) path is incomplete, its scripted TP=4 loss match
  failed unrecorded, and the trajectory drift is marked "controlled" without a control that separates rounding from a
  fault
  Evidence: (a) `O/iter47_tp_loss_check.sh` (committed) runs the 8-layer TP=4 real-weights loss match with and without
  the window; both runs died at start-up (`O/iter47_train_8layers_tp4_{overlap,no_overlap}.log`: `ValueError:
  --host-loss needs the full logits on one chip; not available with --tp > 1`, `train.py:199`), the two comparison
  outputs are tracebacks (`O/iter47_loss_comparison_tp4_{overlap,no_overlap}.txt`, untracked), and no document mentions
  the attempt or its failure. (b) The single-chip 8-layer loss match (0.0234) never builds a window ("single chip: no
  window", `D/perf_analysis.md` Iteration 47 Verification); the torch-referenced checks that do run the window are the
  1-layer TP gradient check (bars PCC >= 0.99 / rel <= 5e-2, measured 1.4-1.7e-2, identical to 46c's printed columns) and
  nothing else; the 2-block check compares on against off only (`O/iter47_tp_grad_check_2layers_compare.txt`, untracked;
  the two logs end at "saved 32 gradients" with no verdict). (c) `O/iter47_full_tp4_real.json` against
  `O/iter46_full_tp4_real.json` (the no-overlap control `O/iter47_full_tp4_real_no_overlap.json` is bit-identical to 46):
  per-step loss deltas +0.0000, +0.0020, +0.0040, +0.0069, +0.0098, +0.0078, +0.0234, +0.0078, +0.0293, +0.0469: positive at
  every step from 2 and growing; the previous rounding-level changes moved the same trajectory by at most 0.006 with
  mixed signs (46 vs 44: +0.001, +0.002, -0.006, +0.002; 45's device scalars: three steps at -0.002; 41: <= 0.005 per
  step). The overlap run was not repeated, so the path's determinism is untested. (d) The Ledger entry's control is the
  2-block on/off diff (gate_up A at rel 7e-5, `O/iter47_da_accuracy_check.log`: op vs ttnn 4.8e-5, both 1.69e-3 against
  fp64), which establishes the step-1 gradient difference is rounding-level but says nothing about steps 2-10 of a
  40-layer run with 40 windows; "AdamW's early sign-like steps amplifying" is prose.
  Why this matters: the loop's result (433.3 ms / 37.44 %) is the overlap path; the contract requires the 8-layer loss
  match and the real-weights run on the landed path; the skill rejects a material anomaly dismissed with prose, and the
  one 40-layer artefact that exercises the path is the anomaly.
  Required next step: a TP-capable loss comparison (the host loss at TP > 1 from the gathered logits shard, or the
  device loss at the 0.1 bar as in the composite phase) run with the window on and off at 8 layers against the torch
  trajectory, recorded in the Verification cell; a second 40-layer real-weights run with the overlap (determinism: the
  two overlap runs must be bit-identical); one isolating control (the owned dA route without the window, or the window
  with dA on ttnn's route) so the drift is attributed to the route's rounding or to the window; the 2-block compare file
  committed and its logs carrying the verdict; the Ledger line rewritten with these controls (resolution "controlled"
  only if the loss match passes and the repeat is bit-identical).

- P2: Iteration 47's measured gain is inside the step noise and the device profile supports about a tenth of it on the
  critical chip; the Attribution's serial baseline was not serial
  Evidence: in-process alternating run (`O/iter47_mfu_alternate.log`, `O/iter47_alternate_summary.txt`): on 439.25 ms
  (sd 2.8) against off 443.12 (sd 5.8) over 6 + 6 steps, difference -3.9 with SE 2.6 (t = 1.5); the on steps rise
  435.5 -> 443.4 while the off steps fall 451.1 -> 436.4: steps 3-8 give -10.5, steps 9-14 give +2.8. The ABA between
  processes (`O/iter47_measure.log`): 433.3 -> 437.8 (off) -> 438.0 (on): the second overlap leg equals the no-overlap
  leg. Device side (`O/iter47_tp_tracy_tp4_4layers/.../ops_perf_results*.csv` against 46c's, steady steps, per window
  from the end of the preceding 32-core transpose to the end of the following `RMSNormBackwardRows`): on the critical
  chip 0 (0.955 / 0.990 GHz in the two profiles) the region is 279 us (46c, median) -> 269 (47): -10 us per window,
  ~-0.4 ms per step for 40 windows, against the Attribution's "~-60 us per layer, -2.5 ms expected, -3.8 measured". In
  46c the 36-core reduce-scatter already ran concurrently with the following norm kernel's other cores (the dispatcher
  launches per core: the norm's firmware span 153 us for a 44 us kernel, the reduce-scatter 117 us inside it), so "110 +
  95 + 30 serial" overstates the baseline; in 47 the window costs 19 us before and 9 us after it on every chip (the
  bare switch pair is 7-8 us standalone), dB on 77 cores 132 us against 95 on 110, and the exit's device-side wait
  serialises the norm behind the reduce-scatter instead.
  Why this matters: the skill asks small changes to be reported in device time and a change to beat the best correct
  measured path; a 0-1 % wall move from 6 + 6 interleaved steps with a trend is not that, and the Result / README cells
  state -3.9 as the measurement.
  Required next step: an in-process alternating run long enough for the SE to drop below 1 ms (>= 30 pairs), reported as
  mean +- SE per mode; the Result cell carrying the range (0 to -4.5 between processes, the alternating figure with its
  SE); the Attribution cell restated with the per-window device-side region (anchor to the norm's end) on the critical
  chip and the reason the expected gain was overstated (the pipelined baseline, the 77-core dB, the two waits).

- P2: The ranked cost table is five iterations stale and its collectives rung is contradicted by iteration 47
  Evidence: `D/perf_analysis.md` "Ranked cost table": "Source: TP=4 after iteration 42" (line 119 at `4c96f9fde82`);
  the collectives row's remaining rung "a floor at TP=4 (iteration 33: no overlap on one sub-device, ETH dispatch
  unavailable)" (line 126) while iteration 47 landed an overlap on sub-devices with a 7 us switch pair; the RMSNorm row
  "(iteration 44, in verification ...)" (line 128) after 44 landed; the qkv / out cells (0.76 + 0.59, 0.59 + 0.48) predate
  46 (0.69 + 0.51, 0.52 + 0.43, `O/iter46c_cycle_diff.txt`); the LoRA row's gate_up dB 0.38 / dA on ttnn predates 47
  (0.527 on 77 cores, 0.137 on the op, `O/iter47_cycle_diff.txt`); the Rejected row for iteration 33 ("a collective
  sub-device (10-20 cores) would cost the matmuls 9-18 % ... more than the ~19 ms it hides") is now answered by the
  dynamic switch and carries no pointer. The README "State" line for 33 says the same.
  Why this matters: the skill updates the table every iteration; it is the loop's steering instrument and the next
  iteration (48, already in flight on disk) is being costed from `O/iter47_tp_ranked_ops_full.txt` directly.
  Required next step: regenerate the table from `O/iter47_tp_ranked_ops_full.txt` chip 0 (with the window rows and the
  per-chip collectives minimum), the collectives rung restated (overlap works on sub-devices; the remaining collectives
  have no independent work behind them except the qkv dx, which 48 targets), the norm and LoRA rows at 44 / 47, the
  iteration-33 Rejected row and the README State line pointing at 47.

- P2: Evidence hygiene: untracked artefacts cited by the documents, a committed check whose failure is nowhere, an
  incomplete cycle diff
  Evidence: `git status`: `O/iter47_tp_grad_check_2layers_compare.txt` (cited by the Ledger and the Verification cell),
  `O/iter47_loss_comparison_tp4_{overlap,no_overlap}.txt` (tracebacks), `O/iter46b_tp_ranked_ops.txt` untracked;
  `O/iter47_tp_loss_check.sh` committed with no result recorded; `O/iter44_cycle_diff.txt` lists only the four new
  104-core rows and no chip-2 rows (the vanished 64-core rows 0.175 / 0.172 / 0.421 and chip 2's 0.753 -> 0.690 are in the
  full tables `O/iter42_tp_ranked_ops_full.txt` / `O/iter44_tp_ranked_ops_full.txt` but not in the diff file the cells
  cite); `O/cycle_diff.py` prints ms without the op counts the Ledger's own iteration-46 protocol asks for ("read the
  ranked table's op count with its ms"); the 2-layer gradient-check logs carry no PASS / FAIL.
  Why this matters: the committed history is the bisect record; a cited file that is not committed and a failed check
  with no record are the pattern round 13 flagged (the untracked iteration-28 evidence, the Commits table).
  Required next step: commit the compare file (and the script that produced it); either delete the two traceback files
  or replace them with the rerun under the P1; regenerate `O/iter44_cycle_diff.txt` with the full output; add the op
  count to `cycle_diff.py`'s rows.

- P2: Iteration 46 carries no kernel-level change and no deleted op chain, and no reason is stated
  Evidence: `263dfc67576` changes `frozen_linear.py` (four `PROGRAM_CONFIG_2D` entries, `ConfiguredFrozenLinear`),
  `lora.py` routing, a harness test; no kernel, factory or op; the first chain added an op (3 -> 4 dX matmuls). The
  skill's exit section calls "a sweep of a library kernel's program configs" tuning, not a kernel attempt; the README's
  hard-constraints line reads "a kernel-level item every iteration"; the README row classes it "math (kernel data
  flow)" and the work log's checklist gives the roofline fraction (60-75 %) but not why the kernel rung was not taken.
  Why this matters: the skill lists "an iteration carries no kernel-level change ... with no reason stated" as
  more-work-needed; iteration 45's kernel change is opt-in and off the default path, with its reason stated (the one
  direct test of the dispatch row), which is the form 46 lacks.
  Required next step: the Change cell and README row state the reason (the kernel rung for these four shapes is the
  ttnn matmul fork; the config is the "reuse" rung taken first, -2.4 ms/step) or the row is re-classed as config
  tuning; no measurement needed.

## Other Concerns

- Iteration 45's "losses identical to eager" is at bf16-loss granularity (0.0625 at 11.9); the real-weights trace run
  differs from 44's eager at three steps by one ulp and the single-chip loss match moved 0.0234 -> 0.0229: the device-side
  scalars are a rounding-level numerics change, as the Verification cell says; the Result cell's "identical" should say
  "at bf16 granularity". The opt-in AdamW kernel path (`DEVICE_STEP_SCALARS`) is exercised only when
  `TTML_ADAMW_DEVICE_STEP=1` is set (the 45 harness run); the 46 / 47 chains run the default path, so the kept code has
  no standing test.
- The sub-device switch semantics check out: with one hardware CQ `reset_worker_state` takes `reset_launch_msg_state =
  true` and skips `finish_nolock`, so the switch is a device-side dispatch wait per sub-device (the host keeps its
  dispatch lead); `TT_FATAL(... "Cannot reset worker state during trace capture")` matches "--trace disables it". The
  documents' "drains the queue" is right at the device level; the Attribution cell should say the wait is device-side
  (the window's 19 + 9 us in the profile are those waits plus dispatch, not host syncs).
- `fused_forward` hands the shard to `block.mlp` whenever the model holds a `collective_overlap`, while
  `use_fused_residual_norm` installs it when any fused gate_up exists; a block whose gate_up is not fused would receive
  the shard without a gather. Not reachable in the model (every gate_up is fused), but an assertion costs nothing.
- The 4-layer TP Tracy runs are host-bound (inter-op gap p90 240-280 us, the profiled step 30 s for 4 steps), so the
  profile's critical chip and its idle % do not describe the 40-layer step; the cycle-time rows stay valid, the window
  gaps carry some host latency.
- The README Conventions line's harness counts stop at "66 (review 12)"; 68 (46) and 70 (47) are missing. The Commits
  table lists 47, 46, 45 above 41; fine, but the 47 row said "(this commit)" until the worktree edit replaced it.
- Iteration 44's MFU row (460.8 on a throttled box) is labelled and the result is the cycle-time rows; the 46c profile
  flipped the critical chip (chip 0 at 0.955 GHz, chips 2-3's collectives +6 ms per 4 layers as waiting), handled with
  cycle-time rows as before.
- The worktree moved during the review: uncommitted edits in `fused_residual_norm.py`, `frozen_linear.py`, the three
  documents (an iteration-48 README row and sections with `*_PENDING` placeholders) and an untracked
  `O/iter48_measure.sh`. The brief's "only uncommitted change is the umd submodule" no longer holds; this review is of
  the committed state.

## Hard-Check Gaps

- A TP-capable loss comparison in `train.py` (the host loss from the gathered logits shard at `--tp > 1`), so the
  8-layer loss match can run on TP-only paths; every TP-keyed change since iteration 24 has relied on the 40-layer
  trajectory alone.
- A per-window device-side measurement as the standard evidence for overlap changes: from the Tracy CSV, the region
  from the preceding op's end to the following op's end per window on the critical chip (the script used here is in the
  reviewer's scratch, not in `O/`).
- A determinism protocol for landed paths: two real-weights runs of the same code must be bit-identical before a
  trajectory move is attributed to numerics.
- `cycle_diff.py` printing op counts next to the ms.

## Anomaly Ledger

- Observed anomaly: the 40-layer real-weights trajectory with the overlap is higher than 46's at every step from 2
  (+0.002 to +0.047).
  Evidence: `O/iter47_full_tp4_real.json`, `O/iter46_full_tp4_real.json`, `O/iter47_full_tp4_real_no_overlap.json` (bit-
  identical to 46).
  Affected path: `FusedLoraLinearT` backward with `overlap` (`fused_lora.py`), `CollectiveOverlap`, the owned dA route.
  Control or comparison: the no-overlap run (exact); the 2-block on/off diff (gate_up A rel 7e-5, everything else 0);
  the standalone dA accuracy (op vs ttnn 4.8e-5, both 1.69e-3 vs fp64); earlier rounding-level changes moved the same
  trajectory by <= 0.006 with mixed signs.
  Likely subsystem: the dA route's rounding amplified, or a window-related fault visible only at depth / across windows;
  not separable from the artefacts.
  Investigation performed: per-step deltas re-derived; the scripted TP=4 loss check found failed on `--host-loss`.
  Resolution: more-work-needed (P1).
- Observed anomaly: the scripted TP=4 8-layer loss check for 47 failed at start-up and is unrecorded.
  Evidence: `O/iter47_train_8layers_tp4_{overlap,no_overlap}.log`, the two untracked traceback files.
  Affected path: Verification cell of iteration 47.
  Control or comparison: none.
  Likely subsystem: `train.py` guard (`--host-loss` at TP > 1).
  Investigation performed: logs read, `train.py:198-199` read.
  Resolution: more-work-needed (P1).
- Observed anomaly: iteration 47's wall gain 0 to -4.5 ms between processes, -3.9 +- 2.6 in process with a trend.
  Evidence: `O/iter47_measure.log`, `O/iter47_mfu_alternate.log`, the Tracy region analysis (-10 us per window on chip 0).
  Affected path: Result / Attribution cells, README row 47, the MFU history's bold row.
  Control or comparison: the 46c cooled run (440.5) at a different clock mix (0.955 vs 0.990 GHz critical chip).
  Likely subsystem: the pipelined baseline (per-core launch overlapped the 36-core reduce-scatter already), the 77-core
  dB, the two switch waits.
  Investigation performed: statistics over the 6 + 6 steps; the per-window device regions on all four chips.
  Resolution: more-work-needed (P2).
- Observed anomaly: iteration 46's first chain ran 4 qkv dX matmuls per 4 layers (0.583 -> 0.684).
  Evidence: `O/iter46_cycle_diff.txt`, `O/iter46b_cycle_diff.txt` (0.780 with 4 ops, no config), `O/iter46c_cycle_diff.txt`
  (0.513 with 3 ops).
  Affected path: `ConfiguredFrozenLinear.backward`.
  Control or comparison: the 46b rerun without the config.
  Likely subsystem: a Python Function without the C++ linear's gradient skip.
  Investigation performed: the op counts of the three tables re-derived (12 -> 16 -> 12 per 4 layers on both chips).
  Resolution: fixed (`ctx.x_needs_grad`), harness case added; the lesson applied in 47 (the 46c -> 47 op counts: RS36
  14 -> 10, RS20 0 -> 4, SkinnyContract 110 -> 77 at 4 each, the dA matmuls 12 -> 8, transposes 8 -> 12: +4 ops).
- Observed anomaly: iteration 44's MFU run on a throttled box (460.8 against 448.7).
  Evidence: `O/iter44_tp_analysis.txt` (chip 2 at 0.988 GHz), the harness at 20 minutes against 4.
  Affected path: MFU history row 44.
  Control or comparison: the cycle-time rows (norm rows -0.055 chip 0 / -0.063 chip 2), the 45 eager repeat on the cooled
  box (444.9).
  Likely subsystem: the box's thermal state.
  Investigation performed: tables and tt-smi samples read.
  Resolution: controlled (labelled in the row).
- Observed anomaly: iteration 45's first eager run 466.8 against 444.9 ten minutes apart.
  Evidence: `O/iter45_eager_tp4.log` (no cool-down after the 44 chain, 75-81 C), `O/iter45_eager2_tp4.log`.
  Affected path: the ABBA.
  Control or comparison: the cooled pair 444.9 / 445.2 and the in-process alternate.
  Likely subsystem: thermal.
  Investigation performed: tt-smi files read.
  Resolution: controlled (the cooled pair is the result).
- Observed anomaly: the 46c profile's collectives moved from chip 0 (9.3 -> 4.1 ms per 4 layers) to chip 2 (4.2 -> 10.2).
  Evidence: `O/iter46c_cycle_diff.txt`, `O/iter46c_tp_analysis.txt` (chip 0 at 0.955 GHz, chips 2-3 at 1.14).
  Affected path: the per-chip collectives rows.
  Control or comparison: the matmul rows x1.00 on every chip; kernels without collectives -0.44 / -0.02.
  Likely subsystem: clock skew (the slowest chip waits least).
  Investigation performed: the per-chip effective clocks read.
  Resolution: controlled (cycle-time rows without collectives used).

## Scope Inspected

- Goal/skill paths: `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`,
  `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/optimization-loop/SKILL.md`.
- Artifact paths (under `/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/`):
  `plan.md`, `perf_analysis.md` (MFU history, Ranked cost table, Iterations 44-47, Rejected, Ledger; working tree and
  `git show 4c96f9fde82:`), `optimization/README.md`, `optimization/work_log.md` (Iterations 44-47, review round 13,
  Commits), `optimization/review_round13.md`; `optimization/evidence/`: `iter44.{sh,log}`, `iter44_cycle_diff.txt`,
  `iter44_norm_slices_check.log`, `iter44_tp_analysis.txt`, `iter4{2,4}_tp_ranked_ops_full.txt`,
  `iter44_tp_mfu_tp4.log`, `iter44_full_tp4_real.json`, `iter44_loss_comparison.txt`, `iter44_tp_grad_check_tp4.log`,
  `iter44_harness.log`; `iter45.sh`, `iter45_{eager,trace,alternate,eager2,trace2}_tp4.log` and `_tt-smi.txt`,
  `iter45_loss_eager_vs_trace.txt`, `iter45_real_trace_vs_eager44.txt`, `iter45_full_tp4_real_trace.json`,
  `iter45_loss_comparison.txt`, `iter45_tp_grad_check_tp4.log`, `iter45_harness.log`; `iter46c.sh`,
  `iter46_attn_configs_abab.log`, `iter46{,b,c}_cycle_diff.txt`, `iter46{,b,c}_tp_mfu_tp4.log`, `iter46{,c}_tp_analysis.txt`,
  `iter46c_tp_ranked_ops_full.txt`, `iter46c_tp_mfu_tp4_tt-smi.txt`, `iter46_full_tp4_real.json`,
  `iter46_loss_comparison.txt`, `iter46{,c}_tp_grad_check_tp4.log`, `iter46_harness.log`; `iter47_measure.{sh,log}`,
  `iter47_tp_loss_check.sh`, `iter47_train_8layers_tp4_{overlap,no_overlap}.log`,
  `iter47_loss_comparison_tp4_{overlap,no_overlap}.txt`, `iter47_alternate_summary.txt`, `iter47_mfu_alternate.log`,
  `iter47_tp_mfu_tp4.log`, `iter47_mfu_no_overlap.log`, `iter47_mfu_overlap2.log` and their `_tt-smi.txt`,
  `iter47_cycle_diff.txt`, `iter47_tp_ranked_ops_full.txt`, `iter47_tp_analysis.txt`, `iter47_chain.log`,
  `iter47_ccl_overlap_scoping.log`, `iter47_subdevice_switch_scoping{,2}.log`, `iter47_da_accuracy_check.log`,
  `iter47_tp_grad_check_tp4.log`, `iter47_tp_grad_check_2layers_{overlap,no_overlap}.log`,
  `iter47_tp_grad_check_2layers_compare.txt`, `iter47_full_tp4_real{,_no_overlap}.{log,json}`,
  `iter47_loss_comparison.txt`, `iter47_harness.log`, `iter4{4,7}_swiglu_op_tests.log`, `cycle_diff.py`,
  `tp_measure.sh`, the Tracy CSVs `iter46c_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv` and
  `iter47_tp_tracy_tp4_4layers/reports/*/ops_perf_results*.csv`.
- Code paths (under `/home/ttuser/agobeljic/tt-metal/`): `tt-train/sources/ttml/ttml/modules/ccl_overlap.py`,
  `tt-train/sources/ttml/ttml/modules/fused_lora.py` (diff), `tt-train/sources/ttml/ttml/modules/lora.py` (diffs 46, 47),
  `tt-train/sources/ttml/ttml/modules/frozen_linear.py` (diff 46), `tt-train/sources/ttml/ttml/models/mistral/fused_residual_norm.py`
  (diff 47), `tt-train/sources/ttml/metal/ops/skinny_contract/device/*` (diff 47),
  `tt-train/sources/ttml/metal/ops/{add_rmsnorm_fw_allcore,add_rmsnorm_fw_rows,rmsnorm_bw_allcore,rmsnorm_bw_rows}` (diff 44),
  `tt-train/sources/ttml/metal/optimizers/adamw/*`, `tt-train/sources/ttml/optimizers/adamw.cpp`,
  `tt-train/sources/ttml/core/mesh_device.cpp` (diff 45), `tt-train/sources/examples/mistral_train/train.py` (diffs 45-47,
  lines 198-199, 273), `tt-train/sources/examples/mistral_train/tp_gradient_check.py` (diffs 46, 47, the compare block),
  `tt-train/sources/examples/mistral_train/tp_rank_ops.py` (duration column), `tt-train/tests/python/test_mistral.py`
  (diffs 46, 47), `tt_metal/impl/sub_device/sub_device_manager_tracker.cpp`, `tt_metal/distributed/fd_mesh_command_queue.cpp`
  (`reset_worker_state`), `tt_metal/impl/program/dispatch.cpp` (`reset_worker_dispatch_state_on_device`).
- Commands run: `git show --stat` and `git show <commit> -- <path>` on the four commits; `git status`, `git diff` on the
  documents and the two modified modules; `git check-ignore`; grep over the logs; python3 (`-I`) over the loss JSONs
  (per-step deltas), the alternating-run steps (means, sds, SE, halves), the two ranked tables (op counts per row), the
  two Tracy CSVs (window spans, overlap of the three programs' firmware spans, the gaps before and after each window, the
  per-window region from the preceding transpose to the norm's end against 46c's serial chain, per device, steady steps
  only). No device opened; no training, profiling or test run; no implementation file modified.

## Residual Risk

- Until the P1 controls exist, the loop's best number (433.3 ms) rides on a path whose only 40-layer artefact drifts;
  if the drift turns out to be a window fault, iterations 47 and the in-flight 48 (a second window per layer) both move.
- The overlap's gain is at most 1 % of the step and may be nil; the collectives' remaining cost is the chip skew, which
  no overlap addresses.
- The loop still has no loss evidence that runs on TP-only code paths other than the 40-layer trajectory; every TP-keyed
  change since iteration 24 shares this gap.
- The 4-layer TP profiles are host-bound under Tracy; attributions that depend on inter-op timing (switch costs, overlap
  spans) are measured in a regime the 40-layer step does not have.
