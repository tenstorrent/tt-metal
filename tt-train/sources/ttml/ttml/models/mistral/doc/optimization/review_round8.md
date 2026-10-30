# Stage review, round 8: iterations 13-18 (2026-10-30)

Independent review (fresh subagent, read-only, no device use, no build) of commits a1fc69ebed8 (13, `transpose_split`),
8b86016e718 (14, `vp_cross_entropy_bw`), 7a21a21258a (15, SDPA forward chain), f045fb8eba8 (16, SDPA backward chain),
f07712081ba (17, `rmsnorm_bw_rows`), 2a75ffce1bd (18, `skinny_contract` + `[r, N]` router) against the
optimization-loop skill (hard constraints, kernel boost checklist, document format), the docs, the `O/` evidence and
the kernels. Round-7 items re-checked. Paths below are relative to `doc/` (`O/` = `optimization/evidence/`); kernel
paths relative to `tt-train/sources/ttml/`.

## Verdict: more-work-needed

No correctness bug and no hard-constraint breach found; the four P2 items are attribution / evidence-provenance /
format / commit-hygiene, two of them repeats of round-7 findings.

## Findings

1. **P2. Iteration-18 attribution counts collective waiting as kernel gain and explains the A/B wall gap with a
   clock figure that does not distinguish A from B** (`perf_analysis.md:247-248`, `work_log.md:167`, commit message).
   - "chip 0 kernels 45.9 vs 46.0 (A) vs 48.2 (17) ms per profiled step (`O/iter18b_tp_analysis.txt`)": the
     `kernels ms` column of `tp_analysis.txt` includes collectives, which on chip 0 were 6.0 (17) / 4.4 (A) / 4.4 (B)
     ms. Collectives excluded: 42.2 / 41.6 / 41.5, so 1.6 of the quoted 2.3 ms vs iteration 17 is box skew, not the
     change (the round-6 convention: same chip, collectives excluded).
   - "The two runs differ by the clock mix (chip 0 at 1.03-1.05 GHz effective in both profiles vs 1.16 in iteration
     17's)": these are the *profile* runs' effective clocks (`O/iter18_tp_analysis.txt` 1.033, `O/iter18b_...` 1.046),
     not the MFU runs' (tt-smi reads 1350 on all chips in `O/iter18{,b}_tp_mfu_tp4_tt-smi.txt`; no per-chip clock of
     the MFU runs exists), and they are the same for A and B, so they do not explain 498.0 vs 503.9. What the evidence
     supports: the 5.9 ms (1.2 %) gap is inside the 1-2 % run-to-run noise and the per-step spread (A 484.6-507.0,
     B 495.1-510.6; `O/iter18{,b}_tp_mfu_tp4.log`).
   - The router choice itself is justified by cycle-time: with all six product rows plus the new transposes, chip 0
     per 4 layers is 1.63 (17) -> 1.47 (A) -> ~1.32 (B) ms (`tp_rank_ops.py --top 70` on the on-disk CSVs:
     17: 0.575 + 0.389 + 0.289 + 0.187 + 0.118 + 0.075; A: 0.472 + 0.387 + 0.234 + 0.186 + 0.118 + 0.075; B: 0.468 +
     0.369 + 0.187 + 0.118 + 0.088 `SkinnyContract` N = 1536 + ~0.09 transposes / pads), consistent with the standalone
     table (110 vs 123 us, 41 vs 53). The doc's "1.11 vs 1.28 vs ~1.45" omits the qkv dB and out dA rows (same delta).
     Unattributed: the dA `[1024,32[16]] x [1024,5120]` row dropped 0.289 -> 0.234 in run A although its code path
     (ttnn `transpose_a` matmul, 88 cores) did not change except for the dropped `core_grid=full_core_grid()`
     (`modules/lora.py:78`, previously passed in `fused_lora.py`).
   - Required: restate the Result / Attribution bullets with the collectives-excluded kernel totals, drop the
     "clock mix" sentence (or measure it), list the six rows, and note the dA row move.

2. **P2. The iteration-18 run-B profile failed once and was re-taken by hand; the committed artefacts mix the two
   attempts and nothing records it** (`O/iter18b_chain.log`: "TP=4 Tracy 4 layers ... rc=1";
   `O/iter18b_tp_tracy_tp4_4layers_failed1/.logs/` holds a 1.4 GB `tracy_ops_times.csv` and no report; the committed
   `O/iter18b_tp_tracy_tp4_4layers.log` starts 18:54:33, three minutes after the chain finished at 18:51:42, while the
   committed `O/iter18b_tp_tracy_tp4_4layers_tt-smi.txt` (mtime 18:51:41) is the *failed* attempt's sampling). The
   failed attempt ran 63 s against 86 s for the healthy 17 / 18A profiles. Not mentioned in `perf_analysis.md`,
   `work_log.md` or the Ledger. The Anomaly Rule applies (what failed: the training process, the capture or the
   post-processing; control: the rerun's identical kernel rows). Required: Ledger entry with the classification, the
   rerun command, and the provenance of the two files.

3. **P2. The ranked cost table is still "TP=4 after iteration 12" with six iterations of deltas appended in cells**
   (`perf_analysis.md:66-82`: "bwd 1.32 after iteration 17 (~27)", "fwd 0.56 after 15, bwd Q 0.78 and KV 1.10 after
   16", "~23 after iteration 13", "~8 after iteration 14"; the LoRA-adapter row's remaining rung is still "dA / dB
   products on 33-88 cores (K-split like `skinny_matmul`)", which iteration 18 tried and closed; `README.md:71`
   "Remaining ranking ... after iteration 12"). Round-7 finding 2 was exactly this ("after iteration 6" with deltas in
   cells) and was fixed by regenerating from `O/iter12_tp_ranked_ops.txt`; the fix lasted one iteration. The format
   rule is "the current table"; `O/iter18b_tp_ranked_ops.txt` exists. Required: regenerate (chip 0, per-layer rows
   x10, class and remaining rung per row, the rooflines and measured fractions of the rows iterations 15-18 touched).

4. **P2. Commit hygiene: the iteration-13 documentation was committed inside the round-7 review commit, and a stray
   file rode along with iteration 18.** `git show 456f9cf7a8e` (review round 7 items, 16:02:09) adds the
   `perf_analysis.md` "Iteration 13" section and MFU row, the README row and note, and the work-log section with a
   `| 13 | (next commit) |` placeholder; `a1fc69ebed8` (iteration 13, 16:02:24) touches no doc. The history does not
   bisect iteration 13 with its docs (round-7 finding 1 was the same class). `2a75ffce1bd` adds
   `O/iter18_cmake.log` (a 13 KB cmake configure log referenced by no document; the 13-17 equivalents are untracked),
   and `work_log.md:231` still reads `| 18 | (this commit) |`. Required: note the split in the work-log commits table
   (or fold the iteration-13 doc hunk into the row's description), drop or reference the cmake log, fill the hash.

## Other concerns (P3)

- `perf_analysis.md` format: the Iteration 15 and 18 **Change** / **Attribution** bullets are 4-6-line paragraphs
  (`:222`, `:246-248`; rule: no paragraph over three lines, the rest into `evidence/`); `O/perf_analysis_details.md`
  ends at Iteration 12, so there is no overflow file for 13-18. `:258` "Resumed loop (iterations 4-14)" lists through
  18 B.
- `perf_analysis.md:240` "Landed within the expected 130-150 us band plus the exchange ...": 165 us is outside the
  band; state it as 165 vs 130-150 expected, -0.51 vs -0.6..-0.8 ms (above half the expected gain, so a win).
- Iteration-18 roofline bookkeeping disagrees with itself: `work_log.md:160` "1.8 ms per 4 layers per chip across the
  six rows of `O/iter17_tp_ranked_ops.txt`" vs `perf_analysis.md:248` "~1.45 (17)"; the six rows in that file sum to
  1.63. `work_log.md:168` then calls the remaining time "the bandwidth floor"; per row after B: gate_up dB 92 us vs
  81 roofline (88 %), N = 5120 dB / dA 29 vs 25 (86 %), w2 dA 47 vs 40 (85 %), out dA 29 vs 5 (17 %), qkv dB 22 vs
  7.5 (34 %). The two small ones are launch-latency-bound, not bandwidth-bound; the "near roofline" rule wants these
  fractions in the row and the next attempt named (one launch for all six products of a layer, or folding dA / dB
  into the `lowrank_update` / `transpose_split` passes that already stream the same operands).
- Iteration 16's **Expected** (`work_log.md:145`) assumed bandwidth-bound kernels although the same checklist's
  roofline line (`:143`) already had them at 310-350 GB/s, i.e. within 15 % of DRAM bandwidth; the compute floor
  should have been estimated before the rewrite. The post-hoc reason and next fix are recorded as the skill requires.
- Harness gaps for `skinny_contract`: `test_skinny_contract_matches_torch` (`tests/python/test_mistral.py:438-458`)
  covers N = 16384 / 5120 / 1024 but not the one shape the router sends to the op (N = 1536, 48 cores, `cols` = 1), nor
  a shape with a short last item (`Nt % cols != 0`, e.g. N = 3552: `cols` = 2, last `ncols` = 1), which is the only
  path through the `pack_tile(j < ncols ? j : 0)` / full-block padding code
  (`metal/ops/skinny_contract/device/kernels/compute/skinny_contract_kernel.cpp:64`, reader `:43-57`). N = 1536 is in
  the standalone check only (`O/iter18_contract_check.log`).
- `modules/lora.py:78` (ttnn path of `skinny_contract`) drops the `core_grid=full_core_grid()` the fused backward
  passed before iteration 18 (`fused_lora.py` diff). ttnn chose 88 / 110 cores on its own and the row got faster, so
  no action beyond attribution, but the iteration-2 lesson ("microbenchmarks must mirror the exact call") says the
  standalone and the model should use the same call.
- README `:63` and the commit message give run A's 498.0 / 32.58 % alongside run B as if both were results of the
  landed code; run A measured a router that is not on the branch. Keep it in the MFU history (it is labelled) but the
  README / state line should carry B only.

## Hard-check gaps

- Loss match last rerun after iteration 14 (`O/iter14_loss_comparison.txt`, 0.0294). Iterations 17 and 18 change the
  rounding path of every norm backward (PCC 0.999999 / rel 1.7e-3 vs fp64) and every LoRA gradient (rel 1.2e-3 between
  orientations); 15 / 16 are bit-identical / one ulp. Below the 0.10 bar by construction, but the final path has not
  been loss-matched; rerun in the next chain (as round 7 asked and iteration 14 did).
- Rank-16 zero-pad invariant (Ledger) now also carries `skinny_contract`'s resident `small^T` (pad rows 16..31 of
  `h` / `dh` must be zero or the output pads are non-zero and feed the optimizer) and the ttnn `[r, N]` + transpose
  path; still not validated on device.

## Anomaly ledger

- Observed anomaly: iteration-18 run-B 4-layer Tracy profile returned rc=1 inside the chain; a second profile was
  taken by hand. Evidence: `O/iter18b_chain.log`, `O/iter18b_tp_tracy_tp4_4layers_failed1/`, file mtimes. Affected
  path: the run-B attribution (`O/iter18b_tp_ranked_ops.txt`, `O/iter18b_tp_analysis.txt`, both from the rerun).
  Control: the rerun's compute rows equal run A's within 2-step noise (matmul rows identical, `SwigluPackedBw`
  1.072 / 0.969 / 0.958 across 17 / A / B with no change). Likely subsystem: profiler capture / post-processing (1.4 GB
  partial `tracy_ops_times.csv`, run 63 s vs 86 s). Investigation performed: none recorded. Resolution:
  more-work-needed (finding 2).
- Observed anomaly: wall A 498.0 < B 503.9 while cycle-time favours B. Evidence: the two MFU logs, the two profiles.
  Control: per-step spread 484-511 across the two runs; run-to-run noise 1-2 % (Ledger). Resolution: controlled, once
  the "clock mix" sentence is replaced by the noise statement (finding 1).

## Re-derived clean

- UMD submodule pointer `7eb549a29db` in the tree of all six commits (`git ls-tree`); the worktree's `M umd` is the
  documented local checkout.
- MFU history rows 13-18 B match the logs (516.4 / 512.5 / 506.0 / 508.4 / 505.6 / 498.0 / 503.9 ms; best 512.0 /
  503.3 / 502.4 / 502.2 / 499.4 / 484.6 / 495.1; MFU 31.42 / 31.66 / 32.06 / 31.91 / 32.09 / 32.58 / 32.19 %), 8 steps,
  2 warm-up, 40 layers, random init, cool-down 600 s in every chain; per-chip matmul rows equal (x1.00) in every
  `tp_analysis.txt`; the 14 real-weight run 526.1 ms with the baseline trajectory.
- Harness 40 / 42 / 42 / 42 / 42 / 45 / 45 passed; TP=4 gradient check PASS in all seven chains (PCC >= 0.9999,
  rel <= 1.42e-2, params identical).
- Attribution rows 13-17 against the ranked tables: 15 `SDPAForward` 0.911 -> 0.559; 16 bwd Q 0.862 -> 0.783, KV
  1.151 -> 1.097; 17 `RMSNormBackward` 1.093 + `Partial` 0.740 -> `Rows` 1.320; ops 488 -> 487 -> 487 -> 479 -> 479 /
  491 (B: 12 new transposes per 4 layers, as stated).
- Kernel boost checklists for 15-18 present with roofline, bytes, cores, structural fix, expected and a standalone
  measurement before each full-model run; 16 and 18 landed below half the expected gain and say so with the reason;
  18 names a next fix ("none worth its cost", see P3 above).
- HiFi4 + fp32 DEST in every new op (`fp32_dest_acc_en=true` in all five factories; `lora_matmul_config` HiFi4);
  no HiFi2 / LoFi anywhere in the new code or the Python; the HiFi2 Ledger entry stays out of the Rejected table.
- Round-7 fixes still in place: `max_runtime_args` guards in the three SDPA factories (work-list and chain:
  `+16` / `+24`), the `lowrank_update` router's M <= 4096 check (`lora.py:45`), the collectives cell wording.
- SDPA chain protocol (`sdpa_fw_reader_kernel.cpp:78-158`, `sdpa_bw_q_reader_kernel.cpp:125-205`,
  `sdpa_bw_kv_reader_kernel.cpp:79-185`): ready counter -> data write to the receiver's reserved slot at the same CB
  address -> VALID relay after `async_writes_flushed`; the receiver sets INVALID before signalling ready, so the sender
  semaphore never exceeds 1 and a stale VALID cannot be observed; every core of a program runs the same `rounds`
  (lists padded with `0xFFFFFFFF`, `sdpa_fw_program_factory.cpp:378-418`, `chain_schedule` in both backward
  factories) and chains never cross a stream; `chain_chunk` falls back to 2 / 1 when `St % 4 != 0`; CBs hold two
  chunks and pops are whole rows of a chunk (no straddle). Injector helpers reserve / push full chunks.
- `rmsnorm_bw_rows` (`reader_rmsnorm_bw_rows.cpp`, `writer_...`, compute): partial written into every row core's
  slot `s`, barrier, then S semaphore increments; the reader waits for S and resets; one item per core by
  construction (dispatch `rmsnorm_bw_allcore.cpp:31-40` requires rows x slices <= cores); the shorter last slice
  (52 of 54 tiles at C = 5120) is covered by the model shape and `test_rmsnorm_allcore_backward_with_frozen_gamma`
  (1-batch case; the 2-batch case falls back to the two-phase op).
- `transpose_split`: one tile per item, items = output tiles, scale in fp32 DEST only for tail items, bit-exact at
  four shapes; hash covers rows / tail / scale / shape.
- `vp_cross_entropy_bw`: per-device shard window from the mesh coordinate (`shard_window`), one-hot patched in L1
  only when `first_v <= target < last_v`, full-block pushes (`kBlock` = 2) with a padded tail, `grad_reg` = 2 inside
  the four fp32 DEST slots; the padded-gradient-tile bug is fixed in `losses.cpp` by repeating g into a tile.
- `skinny_contract`: full-block pushes / pops (`prefetch x cols`, CB of two batches), `cols <= 4` fp32 accumulators,
  in0 = resident `small^T`, in1 = streamed `big` (correct operand order for `small^T @ big`), hash covers scale /
  transpose / shapes, T <= 4096 and r <= 32 validated; standalone PCC 0.999999 at seven shapes incl. r = 32.

## Scope inspected

- Skills: `stage-review/SKILL.md`, `optimization-loop/SKILL.md`; `review_round7.md`.
- Docs: `perf_analysis.md`, `optimization/README.md`, `optimization/work_log.md`, `O/perf_analysis_details.md`
  (headings).
- Evidence: `O/iter1{3,4,5,6,7,8}{,b}*.{log,json,txt,sh,py}`, `O/iter18b_tp_tracy_tp4_4layers_failed1/`,
  `O/tp_measure.sh`; `tp_rank_ops.py --top 70` run host-only on the on-disk CSVs of 17 / 18 / 18b.
- Code: `metal/ops/{transpose_split,vp_cross_entropy_bw,rmsnorm_bw_rows,skinny_contract}/` (all files),
  `metal/ops/sdpa_fw/` and `metal/ops/sdpa_bw/` chain sections and factories, `metal/common/dataflow_utils.hpp`
  helpers, `metal/ops/rmsnorm_bw_allcore/rmsnorm_bw_allcore.cpp`, `modules/lora.py`, `modules/fused_lora.py`,
  `ops/distributed/losses.cpp` (diff), `tests/python/test_mistral.py` (diffs and norm tests).
- Git: `git ls-tree` of the six commits, `git show --stat` of 456f9cf7a8e and the six, file lists vs tracked set.

## Residual risk

- Chain-mode SDPA correctness rests on the standalone comparisons (bit-identical forward and dQ, one-ulp dK / dV)
  and the harness; the protocol has no watchdog, so a future shape where a stream has fewer than 2 cores or a
  mismatched round count would hang rather than fail.
- The result-metric gain of iterations 15-18 (512.5 -> 503.9 ms, 1.7 %) is of the order of the run-to-run noise;
  the per-chip cycle-time rows are the evidence, and they depend on 2-step profiles whose row noise is ~0.1 ms.
