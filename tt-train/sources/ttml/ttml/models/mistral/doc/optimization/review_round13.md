# Phase Review

Stage review round 13: optimisation loop, iterations 41-42 and the iteration-43 scoping (commits `87778c395bd`,
`efcc16ecdf9`, `13c9d66dcf2`, `9e460d03d40`, `a7c2b6edac9`, `8f3213e0267` on `agobeljicTT/mistral-bringup`, live
worktree). Independent reviewer, read-only; no device opened. Paths: `D` = `doc/`, `O` = `doc/optimization/evidence/`.

Verdict: more-work-needed

No P1. The kernels (fp32 DEST four tiles per acquire, fixed-size CB pushes, the resident CBs never wrapping), the
sequence-parallel autograd contract (`comm_ops.cpp`: reduce-scatter backward = all-gather, SHARDED gather backward =
reduce-scatter of the column-parallel partial dx, REPLICATED gather backward = local partition behind the LM head's
own broadcast, the scatter exact for a replicated bf16 input), the TP gradient check (PASS, every adapter and the
loss), the harness counts (64 / 66, the two round-12 cases present), the memory probe (27.06 -> 26.07 GiB) and the
cycle-time sums all re-derive from the artefacts. Six P2 items: numbers in the documents that contradict their
artefacts, a result selected between two runs inside the noise band, a stale ranked table, an unbacked ceiling
statement, two rejection cells that overstate or skip the adapted retry, and a Commits table four commits behind.

## Required Work

- P2: Iteration 41's loss-match figure (0.0286) contradicts its artefact (0.0236) and iteration 42's move is
  misattributed
  Evidence: `O/iter41_loss_comparison.txt`: `max |diff| 0.0236 (tol 0.1), mean +0.0048`; `O/iter41.log` tail says
  the same. `D/perf_analysis.md` Iteration 41 Verification ("PASS at 0.0286"), Ledger ("moved 0.0285 -> 0.0286"),
  `work_log.md` Iteration 41 ("loss match PASS (0.0286)") and Iteration 42 Verification ("0.0234 ... moved it from
  0.0286 by the out-projection dB back on ttnn") all carry 0.0286. The trajectory moved at iteration 41 (0.0285 ->
  0.0236; host fp32 loss at every step, e.g. step 3 11.3175 -> 11.3126, the bf16 loss at steps 5 and 9), which is
  the norm numerics (the squares from the bf16 sums, the 19-bit FPU operands); iteration 42 moved it 0.0236 ->
  0.0234, not the out-projection dB route.
  Why this matters: the same class of cell error round 12 corrected for iteration 40; the Ledger now records a
  numerics change as "0.0285 -> 0.0286" when the artefact shows the first real trajectory move since iteration 32.
  Required next step: correct the four cells to 0.0236 / 0.0234 and the Ledger line to the norm attribution.

- P2: Iteration 41's result is the better of two cooled runs inside the noise band, and the "throttled first run"
  evidence does not hold
  Evidence: `O/iter41_tp_mfu_tp4.log` 461.8 (best 457.6) after a 600 s cool-down; `O/iter41b_tp_mfu_tp4.log` 454.9
  (best 448.1) after 900 s: 1.5 % apart, the stated step-time noise. The case for discarding the first rests on the
  profiles' effective clocks (`O/iter41_tp_analysis.txt` critical chip 1.011 GHz; `O/iter41b_tp_analysis.txt`
  0.986 GHz: the repeat's profile is the worse one), while the MFU runs' own tt-smi files do carry the step phase:
  the last 3-4 samples (16:01:39-43 for 41, 16:32:21-24 for 41b, matching the device-close timestamps of the logs)
  read AICLK 0.98-1.27 GHz at 160-208 W for 41 and 0.96-1.16 GHz for 41b; the earlier "1350 on every steady
  sample" readings are the model build at 60-90 W. The Ledger's round-11 line "the MFU runs themselves carry no
  clock reading" is therefore wrong for every run 40-42 (`O/iter4{0,1,1b,2}_tp_mfu_tp4_tt-smi.txt`). The
  README's own convention ("a 4-layer TP profile checked per chip for a stalled chip before a TP number is
  believed") rejects 41 on chip 2-3 at 1.01 and would reject 41b on chip 3 at 0.986.
  Why this matters: the iteration's cycle-time gain (-0.23 / -0.32 ms per 4 layers on chip 0, -0.32 / -0.29 on chip
  2; re-derived from the full tables) is ~0.6 % of the step, below the wall noise; the skill says such changes are
  reported in device time, not by picking the lower wall. Bolding 454.9 as "the loop's best wall" is a selection.
  Required next step: the MFU history and README row carry both runs as the iteration's range (461.8 / 454.9) with
  the cycle-time rows as the result; the Ledger line about missing clock readings is replaced by the step-phase
  AICLK samples per chip (40: 1.03-1.26, 41: 0.98-1.27, 41b: 0.96-1.16, 42: 0.98-1.21 GHz); from here on the
  per-run clock is read from those samples (or the sampler runs at a higher rate through the step phase).

- P2: The ranked cost table is two iterations stale and its norm row's remaining rung is no longer true
  Evidence: `D/perf_analysis.md` "Ranked cost table": "Source: TP=4 after iteration 40", RMSNorm row "(iteration
  41, in verification)". After 42 (`O/iter42_tp_ranked_ops_full.txt`, chip 0): norms 0.77 ms per 4 layers (was
  2.05), the collectives are RS `[1024, 5120]` x14 + AG `[256, 5120]` x13 (+ 2 AG `[1024, 1280]`), the LoRA row
  lost the out dB op (0.137 -> ttnn 0.117 + 0.009), the glue row gained 21 `BinaryNg [256, 5120]` (0.10). The norm
  kernels on the shard run on 64 of 110 cores: `kMaxSlices = 8U` in both `*_rows_program_factory.cpp` and 8 tile
  rows give 8 x 8 items; `O/iter42_norm_sp_check.log` 46 us for 10 MB (~217 GB/s) against the 130 / 4 = 33 us the
  1024-row efficiency would give, i.e. ~50 % of the DRAM roofline, not the "within 10-30 %" of the README State.
  Why this matters: the table is the loop's steering instrument; the README's "what remains with a positive sign"
  list (~-10 / -5 / -5 / -3 ms) has no rows behind it, and a 7.7 ms/step norm item at half its roofline (a cap
  constant) is not in it.
  Required next step: regenerate the table from `O/iter42_tp_ranked_ops_full.txt` chip 0 (with the per-chip
  collectives minimum), one row per remaining item with its cost and rung, including the norm kernels' slice cap at
  the shard shape (8 -> 13 slices: 104 cores) and the four State items with their derivations.

- P2: The README's ceiling statement is arithmetically wrong, contradicts the previous State line and has no
  artefact behind its components
  Evidence: `D/optimization/README.md` "State after iteration 42": "50 % (323 ms) sits below the matmul FLOPs at
  the delivered clock (~200 ms) plus the link-bound collectives (~40) plus the owned kernels' DRAM floors (~60):
  not reachable". 200 + 40 + 60 = 300 < 323: the target is above that floor, not below it; the "State after
  iteration 41" line says so ("the floor ... is ~300 ms, so 50 % needs nearly all of it"). The "~60 ms of owned
  kernels' DRAM floors" has no derivation (the same State says ~110 ms of owned kernels at 10-30 % above roofline,
  i.e. ~85-100 ms of floors; after 42 the owned rows sum to ~97 ms/step). "~40 % is the realistic ceiling" while
  the listed remaining items sum to ~-20 ms -> ~38 %.
  Why this matters: a ceiling claim re-sets the user's target; the skill's exit section requires the gap
  decomposition (step = roofline + math inefficiency + overhead + dispatch) as numbers before any such claim, and
  the document format forbids prose longer than three lines.
  Required next step: a gap-decomposition table from the iteration-42 profile (per class: measured cycle-time at
  the critical chip, roofline figure, headroom), the floor line corrected (with the ttnn matmuls at their measured
  82-88 % the floor is ~335 ms wall at 1.1 GHz, which is the argument the text wants to make), and the two State
  lines reconciled.

- P2: Two iteration-43 rejection cells are weaker than their logs
  Evidence: (a) `all_gather_matmul_async(transpose_a=True)`: one call per shape, refused at
  `ccl_fusion_mcast_2d.cpp:1716` (`a_shape_padded[-1] == b_shape_padded[-2]`, a check that runs before the
  transpose the op itself advertises), `O/iter43_mm_rs_scoping.log`; no adapted retry (the validation is one line
  in an experimental op the loop has patched before, iteration 35) and no inference from the (a) measurement
  (the gather workers' core rows cost the mcast matmul 164 us of 746 at 11x8, more than the 109 us `[256, 5120]`
  gather it could hide for every shape but qkv / out). (b) `minimal_matmul`: the README State says "37-93 % behind
  the committed mcast configs"; `O/iter43_minimal_matmul_bench.log` has +10.2 % (gate_up dX^T M4 K16 N4) and
  +15.0 % (w2 dX^T) with N4 configs, while `O/iter43_minimal_matmul_sweep.log` swept N in {1, 3} only (65 configs,
  not 48) and never revisited N4, N2 or 2D subblocks beyond 1x3 / 4x1.
  Why this matters: the skill counts a first shape error without an adapted retry as unearned, and a cell must
  carry the best measured number. The `matmul_reduce_scatter_async` rejection itself is earned (fair grid: the
  committed `(8256, 5120)` config is 11x10 / K 6 / 1x3 with per_core_M 16, the fused runs used K 6 / 1x3 with
  per_core_M 20 on 11x8; the matmul alone on that grid 910 vs 746 x 110/88 = 932; the fused op hid 89 of the 238 us
  reduce-scatter at 11x8 and none at 11x9; bit-identical outputs).
  Required next step: (a) one adapted retry (the validation patched for `transpose_a`, or the op run on a shape it
  accepts to measure what the fused gather hides) or the cell stating the (a)-derived inference with the numbers;
  (b) the README State and the Rejected cell carrying +10 % as the best measured and the sweep's N4 gap.

- P2: The Commits table is four commits behind, one of them bundled
  Evidence: `work_log.md` "Commits": no row for `13c9d66dcf2` (41 docs + round-12 answers), `9e460d03d40`
  (round-12 code items), `a7c2b6edac9` (42), `8f3213e0267` (43 scoping). `9e460d03d40` bundles a perf routing
  change (`SKINNY_CONTRACT_MAX_N` 8192 -> 5120) with a validation change, two harness cases, a docstring and an
  accessor fix; it was measured only inside iteration 42's chain (`O/iter42_cycle_diff.txt`: SkinnyContract 0.137
  -> 0, Matmul +0.117, Transpose +0.009).
  Why this matters: round 11 and 12 both asked for the table to match the history; the loop's bisect record is the
  table.
  Required next step: the four rows, the bundling noted as it was for iteration 40.

## Other Concerns

- Iteration 42's wall gain is 41-47 % of the expected: the work log expected ~-15 ms/step, the cycle-time rows gave
  -13.3 (chip 0) / -10.8 (chip 2), the result moved 454.9 -> 448.7 (-6.2); the real-weight warm run moved -10.5
  (472.3 -> 461.8). The Result cell gives no reason. The step-phase AICLK samples of the 42 MFU run (chips 0-2 at
  0.98-1.06 GHz against 41b's chip 0-1 at 1.08-1.16) are the likely one; state it.
- "Same collective count" (perf_analysis Iteration 42, the commit message): chip 0 runs 43 -> 46 collective ops
  per 4-layer step (+1 reduce-scatter for the embedding scatter, +2 all-gathers: the first norm's and the final
  norm's). The scatter is a reduce-scatter of an all-reduced tensor plus a `multiply` (0.12 + 0.10 ms/step): the
  vocab-parallel embedding's all-reduce could end at its reduce-scatter half (exact, no multiply, one fewer
  program). 0.2 ms/step, but it is an "ask why" item.
- The Attribution cell's "the scatter's `BinaryNg` +0.10" covers 21 ops of `[256, 5120]` per 4-layer step; the
  scatter's multiply is one of them, the other 20 (5 per layer, `O/iter42_tp_ranked_ops_full.txt` line 52) are not
  attributed to model code.
- Iteration 42 carries no kernel-level change against the README's own constraint ("a kernel-level item every
  iteration"); the reason (a layout change that removed three quarters of the norm and residual-add work and the
  broadcast nodes) is implicit in the Change cell, not stated as the reason.
- Iteration 41 landed at 43-54 % of its expected gain (-0.15 / -0.19 vs the work log's ~-0.35 ms per 4 layers).
  The work log states the reason (serial read / exchange / write phases, one item per core) and the next fix; the
  README row and the Iteration cell call it "landed" without that note. The standalone numbers re-derive
  (clock-corrected -5 % / -7.5 % / -2 % from the reference matmul ratios).
- The TP gradient check is a one-layer model (`num_hidden_layers: 1`) at bars PCC >= 0.99, rel <= 5e-2 (attention
  8e-2) against a single-chip reference npz that predates iteration 41's norm numerics; the measured rel errors are
  1.4-1.65e-2, unchanged. It caught the 4x double reduction of the first iteration-42 run, so it does its job for
  structural errors; the stronger evidence for the 40-layer sequence-parallel path is the TP real-weight trajectory
  (per-step losses within 0.006 of iteration 41's, `O/iter42_full_tp4_real.log`), which the Verification cell
  should name as such.
- The REPLICATED gather's backward (`mesh_partition` along dim 2) relies on the partition index matching the
  reduce-scatter / all-gather chunk order; validated only through the TP gradient check (every gradient flows
  through the LM head's path). Fine as evidence; worth one sentence in the code comment.
- `D/optimization/evidence/perf_analysis_details.md` stops at iteration 27; the Rejected table's "Full text of
  each: `D`" no longer holds for rows after that. Not new to this round.

## Hard-Check Gaps

- A two-layer TP gradient check (`num_hidden_layers: 2`) would exercise the residual stream crossing a layer
  boundary in sharded form (the fused norm's `sum` output as the next layer's residual under SP), which the
  one-layer check does not; the single-chip harness cannot (TP-only mode).
- The MFU runs' step phase is 3-4 tt-smi samples; a sampler at ~4 Hz through the step phase (or the train script
  printing per-step AICLK) would give every TP wall its clock, which the Ledger currently says is unavailable.
- The `minimal_matmul` sweep with N4 / N2 and 2D subblocks on 11x10 (the configs that were best in the first bench)
  before the "37-93 %" figure is cited anywhere.
- The norm row kernels at 13 slices (104 cores) on the 256-row shard: a standalone measurement before the ranked
  table carries the rung.

## Anomaly Ledger

- Observed anomaly: iteration 41 loss-match figure 0.0286 in four document cells.
  Evidence: `O/iter41_loss_comparison.txt` max |diff| 0.0236; `O/iter40_loss_comparison.txt` 0.0285;
  `O/iter42_loss_comparison.txt` 0.0234.
  Affected path: `D/perf_analysis.md` Iteration 41 / 42 Verification, Ledger; `work_log.md` Iteration 41.
  Control or comparison: the chain logs' own tails (`O/iter41.log`, `O/iter42.log`).
  Likely subsystem: transcription; the real move is the iteration-41 norm numerics.
  Investigation performed: artefacts read; the per-step host fp32 losses compared across 40 / 41 / 42.
  Resolution: more-work-needed (cells).
- Observed anomaly: iteration 41 reported from the lower of two cooled runs 1.5 % apart.
  Evidence: `O/iter41_tp_mfu_tp4.log` 461.8, `O/iter41b_tp_mfu_tp4.log` 454.9; profile critical chips 1.011 /
  0.986 GHz; MFU-run step-phase AICLK 0.98-1.27 / 0.96-1.16 GHz.
  Affected path: MFU history row 41, README row 41 and State, Iteration 41 Result cell.
  Control or comparison: the cycle-time rows (-0.15 / -0.19 norm rows; -0.23 / -0.32 kernels without collectives),
  which agree between the profiles.
  Likely subsystem: run-to-run spread of the box; the "throttled" label is not supported by either run's samples.
  Investigation performed: tt-smi files of runs 40-42 parsed; sample timestamps aligned with the logs' device-close
  times.
  Resolution: more-work-needed (report the range; correct the Ledger's "no clock reading" line).
- Observed anomaly: iteration 42 wall -6.2 ms against cycle-time -13.3 ms/step.
  Evidence: `O/iter42_cycle_diff.txt`, `O/iter42_tp_mfu_tp4.log`, `O/iter42_full_tp4_real.log` (-10.5 warm).
  Affected path: Iteration 42 Result cell.
  Control or comparison: the warm real-weight pair (472.3 -> 461.8) and the step-phase AICLK samples.
  Likely subsystem: the box's clock mix during the MFU run.
  Investigation performed: the samples compared (42's chips 0-2 at 0.98-1.06 GHz).
  Resolution: more-work-needed (state the reason in the cell).
- Observed anomaly: 21 `BinaryNg [256, 5120]` ops per 4-layer step after iteration 42 (0.10 ms).
  Evidence: `O/iter42_tp_ranked_ops_full.txt` line 52; none at `[1024, 5120]` before.
  Affected path: the glue row; the Attribution cell names the scatter alone.
  Control or comparison: 41b table.
  Likely subsystem: autograd gradient accumulation or the scatter's multiply (one of the 21).
  Investigation performed: tables diffed; the 20 remaining ops not traced to model code.
  Resolution: controlled (0.02 % of the step); attribution owed in the table.
- Observed anomaly: collective count 43 -> 46 under "same collective count".
  Evidence: the 41b / 42 full tables (RS `[1024, 5120]` 13 -> 14; AG 13 + 4 -> 13 + 4 + 2).
  Affected path: Iteration 42 Change cell, commit message.
  Control or comparison: the per-call times (RS 120 us, AG 109-112 us at the library optimum).
  Likely subsystem: the embedding scatter and the two end-of-stream gathers.
  Investigation performed: counted.
  Resolution: controlled (0.2 ms/step); the cell should say +3.
- Observed anomaly: the norm row kernels at 64 of 110 cores on the shard.
  Evidence: `kMaxSlices = 8U` in both factories; cores column 64 in the 42 table; 46 / 50 us per call for 10 MB.
  Affected path: norm rows (7.7 ms/step), the State's "within 10-30 % of their DRAM rooflines".
  Control or comparison: 1024 rows at 96 cores, 130 us for 40 MB (~310 GB/s).
  Likely subsystem: the slice cap, sized for 32 tile rows.
  Investigation performed: factories and standalone log read; not measured at 13 slices (no device).
  Resolution: more-work-needed (a table rung with a number).
- Observed anomaly: the Ledger states the MFU runs carry no clock reading.
  Evidence: `O/iter4{0,1,1b,2}_tp_mfu_tp4_tt-smi.txt` last 3-4 samples at 0.96-1.27 GHz, 130-208 W.
  Affected path: Ledger (round 11 entry), the iteration 36-41 clock arguments.
  Control or comparison: the device-close timestamps of the MFU logs.
  Likely subsystem: the sampler's 1 Hz rate against a ~4 s step phase; the earlier samples are the model build.
  Investigation performed: timestamps aligned.
  Resolution: more-work-needed (correct the line; use the samples).

## Scope Inspected

- Goal/skill paths: `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`,
  `/home/ttuser/agobeljic/skills/plugins/tt-train-bringup/skills/optimization-loop/SKILL.md`.
- Artifact paths (all under `/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/ttml/models/mistral/doc/`):
  `perf_analysis.md` (MFU history, Ranked cost table, Iterations 41-42, Rejected, Ledger), `optimization/README.md`,
  `optimization/work_log.md` (Iterations 41-43, Stage review round 12, Commits), `optimization/review_round12.md`
  (headers only); `optimization/evidence/`: `iter41.{sh,log}`, `iter41b.{sh,log}`, `iter42.{sh,log}`,
  `tp_measure.sh`, `iter4{0,1,1b,2}_tp_analysis.txt`, `iter4{0,1,1b,2}_tp_ranked_ops_full.txt`,
  `iter41_cycle_diff.txt`, `iter42_cycle_diff.txt`, `iter4{0,1,2}_loss_comparison.txt`,
  `iter4{1,2}_tp_grad_check_tp4.log`, `iter42_harness.log`, `iter4{1,2}_full_tp4_real.log`,
  `iter4{0,1,1b,2}_tp_mfu_tp4.log`, `iter4{0,1,1b,2}_tp_mfu_tp4_tt-smi.txt`,
  `iter4{1,1b,2}_tp_tracy_tp4_4layers_tt-smi.txt`, `iter41_norm_fw_check.log`, `iter41_norm_bw_check.log`,
  `iter42_norm_sp_check.log`, `iter43_fused_ccl_scoping.{py,log}`, `iter43_minimal_matmul_bench.log`,
  `iter43_minimal_matmul_sweep.log`, `iter43_mm_rs_scoping.{py,log}`.
- Code paths (under `/home/ttuser/agobeljic/tt-metal/tt-train/sources/ttml/`):
  `metal/ops/add_rmsnorm_fw_rows/device/kernels/compute/add_rmsnorm_fw_rows_kernel.cpp`,
  `.../dataflow/reader_add_rmsnorm_fw_rows.cpp`, `.../add_rmsnorm_fw_rows_cbs.hpp`,
  `.../add_rmsnorm_fw_rows_program_factory.cpp`; `metal/ops/rmsnorm_bw_rows/device/kernels/compute/rmsnorm_bw_rows_kernel.cpp`,
  `.../dataflow/reader_rmsnorm_bw_rows.cpp`, `.../rmsnorm_bw_rows_cbs.hpp`, `.../rmsnorm_bw_rows_program_factory.cpp`;
  `ttml/models/mistral/fused_residual_norm.py`, `ttml/models/mistral/rmsnorm_allcore.py`,
  `ttml/models/mistral/frozen_embedding.py` (line 57), `ttml/modules/lora.py`, `ttml/modules/frozen_linear.py`
  (`FORWARD_TRANSPOSED_CONFIG`), `ops/distributed/comm_ops.cpp`, `ttnn_fixed/distributed/ttnn_ops.cpp` (grep),
  `metal/ops/skinny_matmul/device/skinny_matmul_device_operation.cpp` (diff),
  `/home/ttuser/agobeljic/tt-metal/tt-train/tests/python/test_mistral.py` (diff, bars),
  `/home/ttuser/agobeljic/tt-metal/tt-train/sources/examples/mistral_train/tp_gradient_check.py` (bars, layer count).
- Commands run: `git log`, `git show --stat` and `git show` on the six commits; `git log -S"0.0286"`; two python3
  scripts over the ranked-ops tables (kernel sums with and without collectives per chip for 40 / 41 / 41b / 42; the
  41b -> 42 row diff and collective counts); a python3 parse of the tt-smi sample files (per-chip AICLK / temperature
  / power, samples below 1350 MHz, timestamps aligned with the MFU logs' device-close lines); grep over the docs and
  code. No device opened, no training or profiling run, no implementation file modified.

## Residual Risk

- The sequence-parallel path is exercised against the single-chip reference only at one layer; a boundary-crossing
  error in the sharded residual stream (its gradient accumulation across layers) would show up only in the
  40-layer real-weight trajectory, which currently matches iteration 41's within 0.006 per step.
- The TP walls of iterations 36-42 are compared on profile clocks that are measured minutes after the wall; the
  step-phase AICLK samples now shown to exist have 3-4 points per run, so the per-run clock remains a coarse proxy
  and the loop's small wall deltas (41: 0.6 %) stay inside the noise.
- The norm kernels' numerics on the FPU (bf16-sum squares, 19-bit operands) moved the single-chip loss trajectory by
  up to 0.005 per step; within tolerance, but every further FPU move in an owned kernel should carry its loss-match
  delta rather than a PCC alone.
- The iteration-43 conclusion that collectives stay separate programs is well supported for the big matmuls; for
  the sub-millisecond ones (qkv, out, the backward's dX reduce-scatters under the adapter products) it is an
  estimate (~0.3 ms per layer) without a measurement.
