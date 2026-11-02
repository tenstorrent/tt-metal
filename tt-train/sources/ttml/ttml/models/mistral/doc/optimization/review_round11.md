# Phase Review

Review round 11: `optimization-loop`, iterations 28-37 (commits `a891b66ff30` .. `6c81785747f`, HEAD `6c81785747f`),
2026-11-02. Reviewer: fresh subagent, read-only (no device, no runs; artefact analysis scripts only). Previous round:
`review_round10.md` (iterations 23-27). Paths: `O/` = `doc/optimization/evidence/`.

Verdict: more-work-needed

No P1. The code of iterations 28-37 re-derives clean (kernels, autograd contract, tests, standalone checks, rejections
with numbers). The required work is documentation and evidence hygiene that the contract makes mandatory: an
attribution overstated by a pre-existing op, clock-mix cells contradicted by the profiles they cite, a ranked table ten
iterations stale, iteration 28's cited evidence not in git and its docs in another commit, a memory change not probed,
and the recurring "8-layer loss match" cell for changes the single-chip path never runs.

## Required Work

- P2: Iteration 37's attribution counts a pre-existing transpose and omits the slower dB product (-0.11, not -0.18 ms per 4 layers)
  Evidence: full chip-0 tables regenerated from the Tracy CSVs with `tp_rank_ops.py --top 400`
  (`O/iter36b_tp_tracy_tp4_4layers`, `O/iter37_tp_tracy_tp4_4layers`): `Transpose [5120,1024]` 5 ops 0.257 -> 1 op
  0.050 (the one op exists in `O/iter35_tp_ranked_ops.txt` too: 0.049, 1 op, before iteration 36 added four); the w2
  dB product `Matmul [1024,32]x[1024,5120]` 16 ops 0.469 -> 12 ops 0.351 plus a new `Matmul [1024,32]x[5120,1024]`
  4 ops 0.140 (+0.022; `O/iter37_scoping.log` had predicted +5 us per call); `RMSNormBackwardRows` +0.076;
  `AddRMSNormForwardRows` +0.001. Net -0.108 ms per 4 layers; kernels-minus-collectives on chip 0 37.74 -> 37.63.
  `perf_analysis.md` "Iteration 37" Attribution says "-0.18 ms per 4 layers (~-1.8 ms/step), above the ~-1.5
  expected"; the commit message says the same.
  Why this matters: the landed gain is 73 % of the checklist's -0.15 expectation, not above it; the cell is the
  iteration's result at the attribution level.
  Required next step: correct the Attribution cell to the four removed ops (-0.207), the dB product's +0.022 and the
  net -0.11 ms per 4 layers (~-1.1 ms/step, below the expected -1.5); add the dB row to the cell. Same for iteration
  36: "the new `Transpose` row 0.255" is four new ops at 0.206 plus the pre-existing one; its net is -0.36 (w2 fwd
  -0.592, transposes +0.206, `WriteTileBlock` +0.012, `Transpose [32,5120]` +0.010), which only helps the claim.

- P2: The clock-mix explanations of iterations 36 and 37 are contradicted by the profiles they cite
  Evidence: `O/iter35_tp_analysis.txt` effective clocks 1.091 / 1.011 / 1.129 / 1.158 GHz (chips 0-3);
  `O/iter36_tp_analysis.txt` 1.148 / 1.122 / 1.029 / 1.016; `O/iter36b_tp_analysis.txt` 1.124 / 1.060 / 1.111 / 1.078;
  `O/iter37_tp_analysis.txt` 1.136 / 1.165 / 1.123 / 1.102. The step follows the slowest chip (every matmul row
  x1.00 across chips); the slowest chip ran at 1.011 (35), 1.016 (36), 1.060 (36b), 1.102 (37). `perf_analysis.md`
  "Iteration 36" Result: "a bad clock mix (profile chips 2-3 at 1.02-1.03 GHz effective against 1.13-1.16 in iteration
  35's)" names chips 2-3 and omits iteration 35's chip 1 at 1.011: the critical chip's clock was the same in both
  profiles. "Iteration 37" Result: "clocks 1.10-1.17 GHz effective, a mix comparable to iteration 35's 462.9": the
  critical chip is 9 % faster than in 35's profile, so the -2.4 ms wall (462.9 -> 460.5) and the "best 453.7" are at
  least as much clock as change (the cycle-time gain is -0.11 ms per 4 layers, item 1). The MFU runs themselves have
  no clock evidence (tt-smi AICLK reads 1350 on every steady sample of `O/iter3{5,6,6b,7}_tp_mfu_tp4_tt-smi.txt`;
  only the dips at teardown show throttling), so the profile clocks are the sole proxy. The tt-smi files do show
  that 36b and 37 ran cooler (T max 66-68 C, no dips) than 35 and 36 (70-73 C, dips to 937-1206 MHz).
  Why this matters: the result metric's cells carry explanations that the cited artefacts do not support; the
  direction of the bias changes between the two cells (the clock excuses 36's loss, is silent on 37's gain).
  Required next step: in both Result cells state the critical-chip clock of the profile (35: 1.011, 36: 1.016, 36b:
  1.060, 37: 1.102 GHz) and the run's T max, report the wall as "within the run spread" and the cycle-time rows as
  the attribution; drop the chips-2-3 wording. Add the per-chip clocks of 28-37 to `D` "MFU history", which the MFU
  history header points to and which stops at iteration 27.

- P2: The ranked cost table is ten iterations stale
  Evidence: `perf_analysis.md` "Ranked cost table": "Source: TP=4 after iteration 27, chip 0 of
  `O/iter27_tp_ranked_ops.txt`"; rows still carry gate_up fwd 6.15 (now 5.84), w2 fwd 3.25 at 88 cores (now 2.66 on
  110), `RMSNorm bwd rows 1.33` (1.19), `SDPABackwardKV 0.72` (0.63), the fixed row "forward stats in one pass
  (~1.3 ms)" as a remaining rung (landed in 32, `VpCeRowStats` 0.539 + 0.188), the SDPA row "block the KV backward
  (iteration 28)" and the norm row "pass 2 of the backward on the FPU" as remaining rungs (landed in 28 and 29), the
  layout row without the 36/37 transposes.
  Why this matters: the skill's step 8 updates the table every iteration; the table is what ranks the next item,
  and the current one would rank items already taken.
  Required next step: regenerate from `O/iter37_tp_ranked_ops.txt` (chip 0; the full table, not the top 28) with the
  class per row and the remaining rung per row updated for 28-37; keep the collective rows with the critical-chip
  clock of that profile.

- P2: Iteration 28's cited evidence is not in git and its documentation is in the review-10 commit
  Evidence: `git status` shows 21 untracked files under `O/`, all iteration 28: `iter28.sh`, `iter28b.sh`,
  `iter28_patch.py`, `iter28{,b}_tp_ranked_ops.txt`, `iter28{,b}_tp_analysis.txt`, `iter28{,b}_loss_comparison.txt`,
  `iter28{,b}_tp_mfu_tp4.json`, `iter28{,b}_full_tp4_real.json`, `iter28{,b}_loss_8layers_hostloss.json`, the tt-smi
  samples, `iter28c_build_check.sh`; `git check-ignore` matches none of them. `perf_analysis.md` "Iteration 28"
  cites `O/iter28_patch.py`, `O/iter28b_tp_{mfu_tp4,analysis,ranked_ops}.*`, `O/iter28b_loss_comparison.txt`. Commit
  `a891b66ff30` (28) touches two kernel files and 26 evidence files but no doc; the Iteration 28 section, README row
  and work-log checklist are in `85fc26239b7` (review round 10), which also edits seven SDPA kernel / factory files.
  Why this matters: the history must bisect with its docs (round-8 finding on iteration 13, repeated); a reviewer or
  a clone cannot re-derive iteration 28's numbers from the repository.
  Required next step: add the 21 files in a docs-and-evidence commit labelled iteration 28; note in the commits table
  that 28's docs live in `85fc26239b7`.

- P2: `README.md` iterations table stops at 34; `perf_analysis.md` sections 35-37 are out of order and their Verification cells carry no numbers
  Evidence: `optimization/README.md` "Iterations" table: last rows 33, 34, "12 candidate"; 35, 36, 37 appear only in
  "State". `perf_analysis.md` section order: 32, 37, 36, 35, "State after iteration 20". Verification cells of 35-37
  read "harness 48; TP gradient check; 8-layer loss match; real weights" without PASS / counts / max |diff| while
  28-32 give "PASS (0.0301)". Commits table row 37: "(this commit)" instead of `6c81785747f`.
  Why this matters: the document format is a contract item; the README table is the per-iteration index.
  Required next step: add rows 35-37 (class, harness count, ms/step, MFU, status); reorder the sections; fill the
  Verification cells (harness 48 / 48 / 49 passed, TP gradient check PASS at PCC >= 0.99989 and rel <= 1.67e-2,
  loss match PASS 0.0301, real weights 476.5 / 477.3 / 475.3); write the SHA.

- P2: The +3.4 GB per chip of iteration 36 is a claim, not a probe; the plan's memory budget is not updated
  Evidence: `O/iter36_scoping.log` "TP=4 memory probe": 17.58 GiB after the forward, 11.69 after the backward, taken
  before the copy; no probe after `d1ed2496368`. `perf_analysis.md` "Iteration 36" Attribution: "Memory: TP=4 peak
  after the forward 17.6 GiB before the copy"; "Loop exit": "TP=4 40 layers ~11 GiB weights + ~2.3 GiB activations
  per chip (scaled, not probed)"; `plan.md` §4 has no TP=4 row and no `W_cat^T` copy.
  Why this matters: the loop's state must hold the memory peak; a second 84.5 MB weight copy per layer is the
  largest allocation change since the fused weights and bounds any further copy (the rejected qkv / out
  orientation is argued on exactly this budget).
  Required next step: the next chain run prints the allocator peak after the forward (the probe of
  `O/iter36_scoping.log` on the current build); record it in "Iteration 36", the README State and `plan.md` §4.

- P2: "8-layer loss match PASS" is cited for three changes the single-chip run never executes
  Evidence: iteration 35's config key `(5120, 16384)` and iteration 36/37's `(8256, 5120)` are the TP shard shapes
  (`frozen_linear.py` `FORWARD_PROGRAM_CONFIG`, `FORWARD_TRANSPOSED_CONFIG`; `use_fused_residual_norm` sets
  `transposed_output` only where `weight_cat_fwd` exists); the single-chip 8-layer model has gate_up `(5120, 65536)`
  and w2 `(32832, 5120)`, so `O/iter3{5,6,7}_loss_comparison.txt` are byte-identical to `O/iter32_loss_comparison.txt`
  (max |diff| 0.0301, signs +7/-3). Round 10 raised the same for iteration 24 and carried "a TP=4 torch-referenced
  loss comparison" as a hard-check gap; the Ledger entry covers 24 only.
  Why this matters: the Verification cell must tie the claim to the measured path (skill: "do not tie the claim to
  the measured path, or contradict it" is a more-work trigger).
  Required next step: in the 35-37 Verification cells say the single-chip match does not exercise the change and
  cite what does: the harness cases (`test_mlp_fused_lora_wide_w2`, `test_fused_residual_norm_reads_the_transposed_w2_output`,
  PCC >= 0.99998, rel <= 6.3e-3 vs torch fp64, `O/iter3{6,7}_harness_subset.log`), the TP gradient check against the
  2026-10-29 single-chip reference (`O/tp_grad_check_single.npz`), and the TP=4 real-weight trajectories: 35 vs 37
  per-step losses differ by at most 0.004 over 10 steps (`O/iter35_full_tp4_real.log`, `O/iter37_full_tp4_real.log`:
  0.3223 / 0.3242, 0.2031 / 0.2051, 0.3379 / 0.3418, 0.4453 / 0.4453). Add a Ledger bullet generalising the
  iteration-24 entry to every TP-shard-keyed change.

## Other Concerns

- `AddRMSNormFrozenGamma.forward` infers the transposed delta from shapes (`fused_residual_norm.py`: `ds[2] == rs[3]
  and ds[3] == rs[2]`); for a square `[T, C]` it is undetectable and a transposed delta would be added untransposed
  without an error. The module already carries `transposed_output`; pass it explicitly (or `TT_FATAL` on square
  shapes). Not reachable at seq 1024 x 5120.
- `FusedLoraLinearT.backward` on the transposed path runs `dcat_t` with `core_grid=full_core_grid()` and drops
  `dx_matmul_kwargs` (iteration 19's 2D config); measured equal (`Matmul [8256,5120]x[5120,1024]` 2.804 vs 2.808 ms
  per 4 layers), so no regression, but the two paths now use different configs for the same product; say so in the
  code comment or route both through one helper.
- Iteration 36's rejected qkv orientation (`perf_analysis.md` Rejected): a measured -0.6 ms/step not taken "for
  +0.6 GB of copies" on a chip at ~21 GiB of 31.83. The row has numbers, so it is earned; the reason is thin once the
  memory peak is probed (item 6). The out orientation is slower and stays rejected.
- Ledger, iteration 35: "the valid runs are the ones with `max|diff|` 4-7 for the math-free variants" is the no-pack
  variants; the math-free (`nomath`) rows read `max|diff| 0.00` because the sweep's reference is ttnn's own matmul
  on the same patched kernel (`O/iter35_matmul_sweep1d.py:97-104`), and both outputs are the same garbage. The
  validity of the runs rests on the fresh `TT_METAL_CACHE` per variant and the time deltas (1.70 -> 1.09 ms), which
  hold (`O/iter35_attrib.log`; `SKIP_COMPUTE` exists in `bmm_large_block_zm_fused_bias_activation_metal2.cpp:448-475`).
  `O/iter35_attrib.sh` says the JIT cache does not hash the kernel text, `O/iter35_attrib_patch.py` says it does;
  the Ledger already records that this was not established. Correct the Ledger wording.
- The `Transpose [5120,1024]` row of iterations 36 / 36b is "5 ops": one is a pre-existing per-step op (present in 35
  and 37 at 0.05 ms); its origin is not attributed anywhere.
- `perf_analysis.md` "Iteration 36" Result and Change bullets run five to seven rendered lines; the format caps prose
  at three. Move the clock discussion to a Ledger bullet and the memory figure to its own bullet.
- The review-10 commit `85fc26239b7` bundles doc answers, two kernel clean-ups and iteration 28's documentation.
- Working tree at review time: `O/iter35_dx_depth.py` modified (an `--out-block-h` option) and
  `metal/ops/swiglu_packed_fw/device/swiglu_packed_fw_device_operation_types.hpp` modified, neither committed; the
  phase owner's concurrent work, outside this review's scope, but the sweep script's committed version is not the
  one on disk.
- Iteration 37's MFU run: first step 973 ms against ~700 in every other run (`O/iter37_tp_mfu_tp4.json`); excluded
  by the two-step warm-up, consistent with the JIT of the two new kernel variants; no action.
- Iteration 29 lands iteration 20's variant A under the 2026-10-31 allowance ("bf16 intermediates where the PCC
  holds", README row 21). The README "Hard constraints" bullet still reads "HiFi4 + fp32 accumulate (no lower
  fidelity evaluated)"; the matmul fidelity is unchanged, but the bullet should cite the allowance so the 19-bit FPU
  rounding in `rmsnorm_bw_rows` (rel 4.1e-3 vs fp64) is visibly inside the constraints.

## Hard-Check Gaps

- A TP=4 torch-referenced loss comparison (carried from rounds 8-10); the real-weight trajectory comparison above is
  the available substitute.
- The allocator peak after the forward on the current build (item 6); needs a device run.
- The critical-chip effective clock of each MFU run cannot be measured (AICLK reads 1350); the profile's clock is a
  proxy from a different run. Recording "profile critical-chip clock" in the MFU history cells makes the proxy
  explicit.
- The pre-existing `[5120, 1024]` transpose (one op per step): attribute it to its line of model code.

## Anomaly Ledger

- Observed anomaly: iteration 36's wall (468.2 / 463.8 ms) is worse than 35's (462.9) while its cycle-time rows improve by 0.36 ms per 4 layers.
  Evidence: `O/iter3{5,6,6b}_tp_mfu_tp4.json` steps 3-8 means 462.9 / 468.2 / 463.8 (re-derived); full chip-0 tables kernels-minus-collectives 38.07 -> 37.79 -> 37.74 ms per profiled step.
  Affected path: the result metric.
  Control or comparison: profile critical-chip clocks 1.011 / 1.016 / 1.060 GHz; tt-smi T max 70 / 72 / 66 C.
  Likely subsystem: box clock and temperature; the expected -3.6 ms/step is inside the 1-2 % step-time noise (4.6-9 ms).
  Investigation performed: the reviewer's per-chip clock and tt-smi summaries (this report); the doc's chips-2-3 wording does not match them.
  Resolution: more-work-needed (item 2: correct the cells; the wall stays "within spread", the attribution is the cycle-time rows).
- Observed anomaly: iteration 37's wall gain (-2.4 ms, best 453.7) exceeds its cycle-time gain (-1.1 ms/step).
  Evidence: `O/iter37_tp_mfu_tp4.json`; full chip-0 tables 37.74 -> 37.63 ms per profiled step.
  Affected path: the result metric's "best" row.
  Control or comparison: 37's profile critical chip at 1.102 GHz vs 35's 1.011; 37's run T max 67-68 C vs 70-72.
  Likely subsystem: clock / temperature.
  Investigation performed: as above.
  Resolution: more-work-needed (items 1-2: the cells state the clock and the corrected attribution).
- Observed anomaly: 21 iteration-28 evidence files untracked, including files the docs cite.
  Evidence: `git status --porcelain -- O/`; `git check-ignore` empty.
  Affected path: evidence integrity of iteration 28.
  Control or comparison: iterations 29-37's `.txt` / `.json` / `.sh` files are tracked.
  Likely subsystem: the iteration-28 commit was made before the chain's files were added; the review-10 commit took the docs.
  Investigation performed: `git show --name-only` on `a891b66ff30` and `85fc26239b7`.
  Resolution: more-work-needed (item 4).
- Observed anomaly: `Transpose [5120,1024]` row shows 5 ops after a change that adds 4.
  Evidence: `O/iter35_tp_ranked_ops.txt` (full table) has the same row at 1 op / 0.049 ms.
  Affected path: iteration 36 / 37 attribution.
  Control or comparison: iteration 37's table keeps the 1 op at 0.050.
  Likely subsystem: a per-step transpose outside the blocks (not attributed).
  Investigation performed: full-table diff 35 -> 36 -> 36b -> 37.
  Resolution: more-work-needed (item 1; attribute the op).
- Observed anomaly: iteration 37's first MFU step 973 ms (others ~700).
  Evidence: `O/iter37_tp_mfu_tp4.json`.
  Affected path: none (warm-up step).
  Control or comparison: steps 2-8 in the usual range.
  Likely subsystem: JIT compile of the new kernel variants.
  Investigation performed: step list inspection.
  Resolution: controlled.

## Re-derived clean

- MFU: steps 3-8 of every `*_tp_mfu_tp4.json` reproduce the logged ms/step, best and MFU (96.37 TFLOP / 594 TFLOP/s):
  28 464.7 / 34.92 %, 28b 468.4 / 34.63, 29 472.2 / 34.36, 30 468.4 / 34.63, 32 464.2 / 34.95, 35 462.9 / 35.05,
  36 468.2 / 34.65, 36b 463.8 / 34.98, 37 460.5 / 35.23; `args`: 40 layers, tp 4, seq 1024, batch 1, random init,
  no profiler flag. Real-weight warm runs 486.9 / 488.4 / 489.2 / 483.5 / 476.5 / 477.3 / 475.3 as documented.
- Loss match: 28b 0.0299, 29-37 0.0301 (tol 0.1), PASS in every `*_loss_comparison.txt`.
- Harness: 48 passed (28b, 29, 30, 32, 35, 36), 49 (37); subsets 1 / 5 / 1 passed; the new test checks sum, out, dx,
  dresidual and both adapters' dA / dB vs torch at PCC 0.999981-0.999998, rel 1.9e-3-6.3e-3.
- TP gradient check: PASS on 28b-37 against the 2026-10-29 single-chip reference (PCC >= 0.999892, rel <= 1.67e-2,
  loss identical); iteration 33's check failed with the documented `cq_shared_state.cpp:18` TT_FATAL.
- Iteration 28: `O/iter28c_noncausal_check.log` NONCAUSAL PATH PASS (PCC >= 0.999994); iteration 30: six shapes
  bit-exact (`O/iter30_build_check.log`); iteration 29: `O/iter29_norm_bw_check.log` 148 us, PCC 0.999993, rel
  4.09e-3; iteration 32: `O/iter32e_ce_stats_check.log` row max exact, chain 215 vs 500 us; iteration 36:
  `O/iter36_write_tile_block_check.log` three modes bit-exact; `O/iter36_scoping.log` 0.883 -> 0.730 ms + 60 us,
  qkv / out numbers as in the Rejected row; `O/iter37_scoping.log` collective 121 / 115 vs 125 / 117 us, dB product
  identical at 42 vs 37 us.
- Iteration 35: `O/iter35_kdepth.log` depth 9 1.59 vs depth 2 1.66 ms (max |diff| 3.12e-2, one bf16 ulp); the gate_up
  forward row 6.149 -> 5.845 on every chip; the sharded-in0 configs that fit ran at 1.668-1.692 ms
  (`O/iter35_sharded_in0_gateup.log`); the writer patch 1.681 -> 1.677 (`O/iter35_writer_check_*.log`); the 2D depth
  sweep within noise (`O/iter35_depth2d.log`); `O/iter35_tracy_threads.log` exposed tail 168 -> 29 us.
- Rejections 31 / 33 / 34: `O/iter31_skinny_{baseline,check}.log` 48 / 67 / 48 -> 47 / 68 / 47 us at identical
  outputs; `O/iter33_tp_grad_check_tp4.log:207` and `O/iter33_eth_dispatch.log` carry the refusals;
  `O/iter34_ttnn_matmul_fullsync_validation.txt` and the sweep logs (1.67 vs 1.69 ms at 4-tile subblocks). All rows
  in the Rejected table carry paths and numbers.
- Iteration 37 kernels: `writer_rmsnorm_bw_rows.cpp` scatters tile (r, k) to page `k * Ht + r` of the `[C, T]`
  output with `Ht = T / 32` from the factory (correct for row-major tile pages); the compute packs each `dx` block a
  second time into `dx_copy`, transposes `n` tiles through DEST (`transpose_tile`, exact) and pushes full `block`
  tiles (tail padded with tile 0) so the CB discipline holds; the pass-2 loop re-issues `reconfig_data_format` per
  block, so the srcA switch for the transpose does not leak; `rmsnorm_bw_rows_device_operation.cpp` validates
  `[1, 1, S, C]`, emits the second output spec `[.., C, S]` and hashes `transposed_grad`. `reader_add_rmsnorm_fw_rows.cpp`
  fetches delta's page `(col) * Ht + r`; the compute reads delta only in pass 1 (pass 2 uses `cb::sum`), hashes
  `delta_transposed`. `AddRMSNormFrozenGamma.backward` returns `dsum` for the residual and the transposed output for
  the delta; `FusedLoraLinearT.backward` consumes `g^T` without the in1 transpose and routes dB through
  `skinny_contract(big_transposed=True)` (`[r, N]` via `transpose_a` + `transpose_b`, then the output transpose).
- Iteration 36: `write_tile_block` maps `(row_offset + i / Wt_src) * Wt_dst + col_offset + i % Wt_src`, validated
  against the destination's tile columns; `write_tail_columns` writes the scaled B into column block `in / 32` of
  `W_cat^T`; the harness asserts the copy exists only for w2.
- Clocks of the profiles and tt-smi samples: matmul rows x1.00 on every chip in 28b-37 (no stalled chip); every MFU
  run after a 600 s cool-down (chain timestamps), start temperatures 49-53 C.

## Scope Inspected

- Goal/skill paths: `skills/plugins/tt-train-bringup/skills/optimization-loop/SKILL.md`,
  `skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`; the contract in the review prompt.
- Artifact paths: `doc/perf_analysis.md` (MFU history, ranked table, sections 27-37, State after 20, Rejected, Ledger),
  `doc/plan.md` §3-4, `doc/optimization/README.md`, `doc/optimization/work_log.md` (28-37, commits), `review_round10.md`,
  `O/perf_analysis_details.md` (headers), `O/iter28{,b}*`, `O/iter29*`, `O/iter30*`, `O/iter31*`, `O/iter32*`,
  `O/iter33*`, `O/iter34*`, `O/iter35*` (chain, attribution, sweeps, writer check, sharded in0, depth 2D, Tracy
  threads), `O/iter36*`, `O/iter36b*`, `O/iter37*` (chain, scoping, harness subset, MFU JSON, tp_analysis, ranked ops,
  tt-smi, Tracy CSVs).
- Code paths: `ttml/modules/{frozen_linear,fused_lora,lora}.py` (HEAD and the 36 / 37 diffs),
  `ttml/models/mistral/fused_residual_norm.py`, `metal/ops/rmsnorm_bw_rows/` (writer, compute, cbs, factory, device
  op), `metal/ops/add_rmsnorm_fw_rows/` (reader, compute, factory, device op), `metal/ops/write_tile_block/` (writer,
  device op), `tests/python/test_mistral.py` (the two tests), `examples/mistral_train/{tp_analysis,tp_rank_ops,tp_gradient_check}.py`,
  `ttnn/.../bmm_large_block_zm_fused_bias_activation_metal2.cpp` (`SKIP_COMPUTE`), `git show --name-only` of the 11
  commits.
- Commands run: read-only `git show / log / status / check-ignore / ls-files`, `grep`, `sed`; Python over the JSON,
  tt-smi and tp_analysis files; `tp_rank_ops.py --devices 0 --top 400` over the four existing Tracy CSVs (35, 36, 36b,
  37) into the scratchpad. No device opened, no run started, no file in the repository modified except this report.

## Residual Risk

- The result metric moves inside the box's clock spread for iterations 28-37 (the cycle-time attribution over the ten
  iterations is about -1.3 ms per 4 layers on chip 0, ~-13 ms/step, against -7.9 ms of wall); the wall ranking of
  single iterations is not reliable, and the "best" rows are partly climate.
- Iterations 35-37 exist only on the TP shard shapes; the single-chip harness and loss match never run them, so the
  TP gradient check (one step, against a pre-loop single-chip reference) and two shard-width harness cases are the
  whole correctness evidence for the transposed w2 route.
- `rmsnorm_bw_rows` with `transposed_grad` writes 10 MB more per call; the kernel's bandwidth fraction after 37 is
  not restated (126 / 165 us rooflines in the stale ranked table).
- The loop is open: the README State's "~36-37 % ceiling under the constraints" is a claim; the remaining items
  (ttnn matmuls at 72-85 %, collectives ~40 ms/step) now carry kernel attempts (33, 34, 35), but no gap decomposition
  has been written since the loop exit of iteration 3.
