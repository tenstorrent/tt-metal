# Stage review, round 10: iterations 23-27 (2026-11-01)

Independent review (fresh subagent, read-only, no device use, no build) of commits 0be1385dbde (23, chain forwarding on
the writer RISC), 4041c62353d (review-9 P1 fix), b88de55dd41 (24, wide w2 input), 0cd8e1213a8 (25, low-rank update
fused into `transpose_split`), c193bd73925 (26, chain credits, 8-row forward chunks, forward chains of 8) and
46dca91d991 (27, per-round stream bounds, backward chains of 7) against the optimization-loop skill (hard constraints,
kernel boost checklist, rejection standard, document format), the docs, the `O/` evidence and the kernels. The
worktree carries uncommitted iteration-28 edits to `sdpa_bw_kv_program_factory.cpp` / `sdpa_bw_kv_compute_kernel.cpp`
and a changed UMD pointer; every code statement below is about the committed HEAD (`git show 46dca91d991:...`).
Paths relative to `doc/` (`O/` = `optimization/evidence/`); kernel paths relative to `tt-train/sources/ttml/`.

# Phase Review

Verdict: more-work-needed

No P1. The five landed numbers (480.3 / 33.78 %, 481.2 / 33.72 %, 476.7 / 34.03 %, 472.5 / 34.33 %, 468.1 / 34.66 %),
the real-weight walls, the loss matches, the harness counts, the TP gradient checks, the bit-exactness claims and every
attribution row re-derive from the logs; the chain credit protocol, the per-round stream bounds, the 8-row forward
chunk, the wide w2 path and the fused transpose read as correct. Five P2 items: an unearned rejection (bf16 DEST), a
bundled commit (26), stale and contradicted state documents (ranked table, "State after 20", README), the document
format of the new sections, and kernel-boost checklists without expected times (27's forward landed at a third of the
implied gain with no reason, and two evidence citations point at files that do not exist).

## Required Work

- P2: The bf16-DEST rejection of iteration 26 is a single attempt at a straw-man variant
  Evidence: `perf_analysis.md:395` ("bf16 DEST in the SDPA forward (same algorithm) ... 145 vs 132 us and PCC 0.9999
  ... the 8-tile DEST buys nothing without a reorganised algorithm"); `O/iter26_attrib_fw_bf16dest.log` (one run,
  fw 145, out PCC 0.99994, dQ/dK/dV 0.9999, max |diff| 0.04-0.09); `O/iter26_attrib_patch.py` (the patch keeps
  `kPass`/4-tile passes). The iteration's own attribution (`perf_analysis.md:322`: "the rest is the per-stage pipeline
  latency of eight CB-serialised stages per chunk") names exactly what an 8-tile DEST would halve, and the harness bars
  the PCC would have to hold are 0.999 forward / 0.98 attention gradients (`tests/python/test_mistral.py:73-76`), which
  the measured 0.9999 clears.
  Why this matters: the skill's rejection standard ("a single failed attempt at one rung of the ladder") is not met;
  the row closes the loop's largest remaining owned-kernel lever (the Change bullet of 23 called it "the compute lever")
  on a variant that could not win by construction. Separately, the row must say which side of the hard constraint the
  candidate sits on: "HiFi4 + fp32 accumulate mandatory" (`README.md:79`) versus the 10-31 allowance of "bf16
  intermediates where the PCC holds" (`README.md:88`); bf16 DEST changes the matmul accumulation, not only an
  intermediate.
  Required next step: either (a) state in the Rejected row that bf16 accumulation is outside the allowance and strike
  the "measured slower" reasoning (then the candidate was never admissible and the measurement is not a rejection), or
  (b) measure the candidate as intended: `kPass = 8` with `dst_size = 8` (one DEST pass per 8-row chunk in QK^T, exp
  and PV), the six-shape check against `O/iter26_sdpa_baseline.npz` with the PCC and relative error per output, and the
  rejection or acceptance on that number.

- P2: Commit c193bd73925 (iteration 26) bundles three separable changes, and support code for a rejected experiment
  landed with it
  Evidence: `git show --stat c193bd73925`; the commit message itself lists credits (`O/iter26_patch.py`, measured
  bit-exact at 132 / 387), 8-row forward chunks (`O/iter26b_patch.py`, measured with a numerics change: 37 % of the
  forward elements, 49 % of dK/dV at one ulp, `O/iter26b_sdpa_check.log`) and the chain-length knobs (`O/iter26c_*`).
  `metal/common/program_utils.hpp:38-64` gained `lpt_schedule(..., initial_load)` for the position-aware LPT that the
  same commit rejects; no factory at HEAD passes it (`git grep initial_load HEAD -- tt-train/sources/ttml/metal` hits
  only the helper). `work_log.md:328` reads `| 27 | (this commit) |` (rounds 8 and 9 flagged the same placeholder).
  Why this matters: the one-change-per-commit rule exists so the history bisects; the numerics change (26b) and the
  pure data-flow change (credits) now share a commit, and a regression in either cannot be isolated. Dead experiment
  code carries a rejected idea into the tree.
  Required next step: split c193bd73925 into three local commits (credits; 8-row chunks; chain-length knobs), or at
  minimum record the bundling in the commits table row as was done for iteration 19; drop `initial_load` from
  `lpt_schedule` (or keep it with a one-line reason); fill `work_log.md:328` with the hash in the next commit.

- P2: The state documents are stale and contradict iterations 22-27
  Evidence: `perf_analysis.md:80-96` "## Ranked cost table ... Source: TP=4 after iteration 18 (run B)": its SDPA row
  still reads "bwd KV 1.08 + reduce 0.11, bwd Q 0.78, fwd 0.56" (now 0.722 / 0.435 / 0.397, `O/iter27_tp_ranked_ops.txt`),
  its LoRA row lists `lowrank_update 0.49 + 0.35 + 0.13` (the fused layers' calls are gone since 25) and its layout row
  "forward K-concats 0.31 + 0.19 + 0.06" (the w2 concat is gone since 24). `perf_analysis.md:357` and `README.md:89`
  still say "took the owned kernels to their exact-fp32 ceilings ... not the remaining ttml kernels (~10 ms in total at
  their ceilings)", while iterations 22-27 removed 4.2 + 1.6 + 7 + 4.6 + 1.2 + 1.8 = 20.4 ms/step cycle-time from owned
  kernels with no numerics change but 26's rounding order; the round-9 answer added a caveat paragraph
  (`perf_analysis.md:343`) instead of correcting the cells, which the skill forbids. `README.md:15` lists the harness
  count up to "47 (19)" although it is 48 since 24.
  Why this matters: the skill's step 8 updates the ranked table every iteration; the table is the loop's ranking
  instrument and is nine iterations old, and the summary line a reader takes as the loop's state is wrong by 2x.
  Required next step: regenerate the ranked table from `O/iter27_tp_ranked_ops.txt` (class per row, full-depth cost,
  remaining rung), restate `perf_analysis.md:357` and `README.md:89` as cells ("after 27: owned kernels ~X ms/step at
  ~Y % of their rooflines"), add "48 (24)" to `README.md:15`.

- P2: `perf_analysis.md` sections 23-27 do not follow the document format
  Evidence: bullets of 431-1333 characters (`perf_analysis.md:295, 304, 313, 322, 323, 334`; the three-line rule);
  bullets outside the fixed set (`Standalone`, `What is left in SDPA`, `Rejected on the way`, `Roofline / bytes`,
  `Attribution first`, `Attribution in the step` at `:300, 305, 314, 322, 325, 328, 333`); sentences in Rejected cells
  (`:395-396`). `O/perf_analysis_details.md` still ends at iteration 12 (round-8 and round-9 P3).
  Why this matters: the skill names this a more-work-needed condition; the long bullets hold the only record of what
  was measured where, and are unskimmable.
  Required next step: per section the five bullets (Change, Result, Attribution, Verification, Evidence) of at most
  three lines each; the standalone tables, attribution and rejection prose into `evidence/perf_analysis_details.md`
  (or per-iteration notes under `evidence/`), linked from the Evidence bullet.

- P2: Kernel-boost checklists without expected times; iteration 27's forward landed at a third of the implied gain
  with no reason; two evidence citations do not resolve
  Evidence: `work_log.md:158-163` (27) and `:165-172` (26) carry no "expected time after the fix" line (checklist item
  5); 27's roofline line says the forward streamed 12 chunks per core for 6.1 needed and the iteration-26 skeleton puts
  the forward's stream at 24 us (`perf_analysis.md:322`: `nofwd,noread` 51 vs 75), so halving the stream implies ~12 us
  and the forward moved 114 -> 110 (`O/iter26c_build_check.log`, `O/iter27b_build_check.log`); nothing says why.
  Iteration 23 (`work_log.md:192-193`) implied targets of ~110 / ~100 / ~170 us and landed 132 / 137 / 198 (18-41 % of
  the gain) with the reason found only in 26's attribution, not recorded in 23. `perf_analysis.md:322` cites
  `O/iter26_tracy*` ("scratch") for the per-core spans 78 / 101 / 122 us and the sink / injector offsets: no such file
  exists (`ls O/iter26_tracy*`); `perf_analysis.md:333` and `work_log.md:160` quote 6.1 / 12.2 / 12.0 chunks needed from
  a "replicated schedule (python)" that is not in `O/`; `work_log.md:169` "S=2048 ... 1304 -> 1174" has no source among
  the cited logs (1174 is `O/iter26_sdpa_baseline.log`; 1304 appears nowhere, the pre-26 S=2048 backward in
  `O/iter26_build_check.log` is 1153).
  Why this matters: the checklist's expected number is what makes "below half the expected gain" decidable; the
  position-aware-LPT rejection and the chain-length choice rest partly on the per-core span analysis that cannot be
  inspected.
  Required next step: add the expected-time line to 26 and 27 (and 23, back-filled from 26's attribution); state the
  reason for 27's forward (the forward's stream is already hidden by the credits, the hop count halved but the sink's
  fill did not; or whatever the data says) and the next structural fix; copy the Tracy per-core analysis and the
  schedule-replication script into `O/` or strike the citations; correct or source "1304".

## Other Concerns

- Iteration 24's "8-layer loss match PASS (unchanged numerics)" (`perf_analysis.md:308`) is true but does not exercise
  the change: the single-chip 8-layer run fuses 24 linears = qkv, gate_up, out per layer and never w2
  (`O/iter24b_train_8layers_hostloss.log` "fused into the base matmul: 24 linears"; w2 is 32768 -> 5120, not a
  transposed shape), so `O/iter24_loss_comparison.txt` is byte-identical to 23's. The wide path's torch-referenced
  evidence is `test_mlp_fused_lora_wide_w2` at the shard width (PCC 0.99999, rel 4-5e-3, `O/iter24b_harness.log:190-197`)
  and the TP gradient check; say so in the bullet (round-9 finding 2 class). Iteration 25's fused transpose is
  exercised by the loss match (gate_up / qkv / out fused on single chip), its wide variant only at TP.
- `SwigluPackedBw` fell 1.096 -> 0.745 ms per 4 layers in 24 with the same bytes and the gradient operand's row stride
  8192 -> 8256 tiles (`O/iter23_tp_ranked_ops.txt`, `O/iter24b_tp_ranked_ops.txt`); the doc attributes it to "its wide
  gradient operand" without a mechanism (a power-of-two row stride hitting the same DRAM banks is the likely one). No
  standalone A/B of the backward exists (`O/iter24_wide_check.log` times the forward only: 136 -> 140). If the
  mechanism is the stride, every `[1024, 8192]`-strided operand in the step is a candidate; worth a Ledger entry and a
  standalone number.
- The kernel totals without collectives on chip 0 are flat at 39.5 ms per profiled step for 25, 26 and 27
  (`O/iter25_tp_analysis.txt` 47.6 - 8.1, `O/iter26_tp_analysis.txt` 44.0 - 4.5, `O/iter27_tp_analysis.txt` 53.0 - 13.5)
  while the SDPA rows fell 1.858 -> 1.737 -> 1.554 (`O/iter2{5,6,7}_tp_ranked_ops.txt`): the per-op gains of 26 and 27
  (-0.12, -0.18 ms per 4 layers) are at the 2-step profile's noise floor (`SwigluPackedBw` alone moves 0.737 -> 0.811
  between 26 and 27 with no change). The result metric moved more (476.7 -> 468.1) than the attribution, which the doc
  rightly assigns to the clock mix. Quote the kernels-without-collectives total for 26 and 27 as was done for 24 / 25.
- The forward's non-chain modes (standard, arbitrary mask, work-list when cores < 2 x streams) now run 8-row chunks
  (`sdpa_fw_program_factory.cpp:92-100` picks 8 for every mode; CBs at `:457-473`, mask CB `2 * Sk_chunk_t` at `:486-491`;
  the `USE_ATTN_MASK` compute path pushes the score tiles one at a time into the 16-tile CB and the exp stage reserves
  8, `sdpa_fw_compute_kernel.cpp:207-228`, `sdpa_compute_utils.hpp:174`). I find no straddle (every push total per chunk
  is 8 and the CB is 16), but nothing on the branch runs these modes (the four attention tests and the six standalone
  cases all take the chain); same class as round-9 P1. One non-causal standalone case would close it.
- `sdpa_fw_compute_kernel.cpp:131-138` still describes the pre-26 "pop+reserve ... CB is single-buffered, size =
  Sk_chunk_t" aliasing that no longer exists (the stamp now lands before the push into a two-chunk CB); `chain_valid_sem`
  is created and relayed nowhere (`sdpa_fw_program_factory.cpp:443`, `[[maybe_unused]]` in all three writers).
- The gate_up concat kept in 24 (`perf_analysis.md:304`, `work_log.md:186`: a 128-row tail at +2.5 % K or a slice before
  the collective "each cost about the 65 us concat") is an estimate, not a measurement, and is not in the Rejected table;
  at 0.206 ms per 4 layers (~2 ms/step) it is below the threshold, so a Rejected row with "estimated" suffices.
- Ledger entries missing: the two device hangs of the iteration-26 attribution (`noinit` variants, `tt-smi -r` 16:30;
  only in the Evidence bullet and the work-log lesson) and the iteration-27 profile's clock mix (chip 3 at 0.976 GHz
  effective, chip 0 collectives 13.5 ms per profiled step against 4.5 in 26, `O/iter27_tp_analysis.txt`; tt-smi 1350 MHz
  and 66-70 C on all chips, `O/iter27_tp_tracy_tp4_4layers_tt-smi.txt`).
- The failed first chain of 24 left `O/iter24_tp_ranked_ops.txt` (204 bytes) and `O/iter24_tp_analysis.txt` ("only 1
  steps") in the tree; they are labelled only through the Evidence bullet's "`O/iter24_*` (the failed first chain)".
- `README.md:81` "Reviews" and the commits table should record this round once answered.

## Hard-Check Gaps

- The bf16-DEST candidate as intended (8-tile passes) needs a device run (P2 item 1, option b).
- A standalone `swiglu_packed_bw` A/B, wide vs narrow gradient operand, to confirm the -32 % and its mechanism.
- One non-causal (or `B x kv_heads > 55`) SDPA forward case with `--ref` to cover the 8-row chunk in the non-chain
  modes.
- The schedule-replication script behind "6.1 / 12.2 / 12.0 chunks needed" and the per-core Tracy spans of iteration
  26 into `O/`.
- Carried from rounds 8 and 9: the rank-16 zero-pad invariant on device; a TP=4 torch-referenced loss comparison (the
  wide w2 path, the fused transpose's scaled-tail variant and the 2D w2 config are covered by the TP gradient check and
  the shard-width harness case only).

## Anomaly Ledger

- Observed anomaly: iteration-27 profile at a 0.98-1.21 GHz effective clock mix, chip 0's reduce-scatter row 10.08 ms
  (593 us per call) against 2.33 ms (137 us) in iteration 26.
  Evidence: `O/iter27_tp_analysis.txt`, `O/iter26_tp_analysis.txt`, `O/iter27_tp_tracy_tp4_4layers_tt-smi.txt`.
  Affected path: the profile only (matmul rows x1.00 on every chip; the MFU run 464-472 ms per step, tight).
  Control or comparison: iteration 26's profile at 1.16-1.18 GHz on all four chips.
  Likely subsystem: box clock skew (Ledger `perf_analysis.md:411`), chip 3 this time.
  Investigation performed: none beyond the per-chip ratio check; the doc notes "(chip 3 at 0.98)".
  Resolution: controlled (cycle-time rows used); a Ledger bullet wanted.
- Observed anomaly: `SwigluPackedBw` -32 % in iteration 24 without a bytes change.
  Evidence: `O/iter23_tp_ranked_ops.txt` 1.096, `O/iter24b_tp_ranked_ops.txt` 0.745, stable since (0.737-0.811).
  Affected path: the SwiGLU backward with the `[1024, 8256]`-strided gradient.
  Control or comparison: the forward (+4 us standalone) and `TransposeSplit` (equal) with the same stride change.
  Likely subsystem: DRAM bank interleaving of a power-of-two row stride.
  Investigation performed: none.
  Resolution: more-work-needed (a standalone A/B; P3, see Other Concerns).
- Observed anomaly: chip-0 kernel totals without collectives flat at 39.5 ms over 25-27 while the SDPA rows fell 0.30.
  Evidence: the three `tp_analysis.txt` and `tp_ranked_ops.txt` files.
  Affected path: the attribution claims of 26 and 27.
  Control or comparison: the result metric fell 8.6 ms over the two iterations; row noise ~0.07 (`SwigluPackedBw`).
  Likely subsystem: 2-step profile noise.
  Investigation performed: per-row diff 25 -> 27 (this review): SDPA -0.304, `SwigluPackedBw` +0.042, `Reduce` +0.018,
  `TransposeSplit` +0.012, the rest under 0.01.
  Resolution: controlled (the result metric is the TP wall); quote the totals (P2 item 5 / Other Concerns).
- Observed anomaly: two device hangs during the iteration-26 attribution (`noinit` skeleton variants).
  Evidence: `O/iter26_attrib3.log` "rc=1 (no result)" twice, `O/iter26_attrib_fw_skeleton_nopack_noinit.log`,
  `O/iter26_attrib_fw_skeleton_nopack_noregs_noinit_nocopy.log`; `perf_analysis.md:329` "tt-smi -r 16:30".
  Affected path: attribution only (the kernels with the init calls removed are not valid programs).
  Control or comparison: the same skeletons with the init calls kept (`skel_all`, `skel_noreconfig`: 72-75 us).
  Likely subsystem: unpacker / packer left unconfigured.
  Investigation performed: lesson recorded (`work_log.md:171`); reset; the remaining variants ran.
  Resolution: controlled; a Ledger bullet wanted.
- Observed anomaly: iteration 24's first chain crashed in the TP gradient check (`SwigluPackedWide.backward` returned
  two gradients) after the harness passed.
  Evidence: `O/iter24.log`, `O/iter24_tp_grad_check_tp4.log`.
  Affected path: the wide path (TP only).
  Control or comparison: the second chain with the fix and the new harness case (`O/iter24b_*`).
  Resolution: fixed, lesson recorded (`work_log.md:188`).
- Observed anomaly: iteration 25's first two builds measured the plain transpose with the fused term missing
  (`FUSED LOWRANK FAIL`, max |diff| 1.2-5.2).
  Evidence: `O/iter25_fused_check.log` (binding without `w`), `O/iter25b_fused_check.log` (factory patch not applied).
  Control or comparison: `O/iter25c_fused_check.log` bit-exact.
  Resolution: fixed, lesson recorded (`work_log.md:179`).
- Observed anomaly: iteration 23's first build forwarded upward in core index on NOC_1 (backward 427 -> 592 us).
  Evidence: `O/iter23_build_check.log`; `O/iter23b-d_build_check.log` after the reversal.
  Resolution: fixed; the direction is now derived in the factory (`sdpa_fw_program_factory.cpp:409-414`).

## Re-derived clean

- MFU rows (`O/iter2{3,4b,5,6,7}_tp_mfu_tp4.log`, 8 steps, 2 warm-up, 40 layers, random init, 600 s cool-down,
  `useful FLOPs/step 96.371 T`, peak 594.0): 480.3 (473.1) / 33.78 %, 481.2 (470.6) / 33.72 %, 476.7 (468.2) / 34.03 %,
  472.5 (465.9) / 34.33 %, 468.1 (465.4) / 34.66 %; real weights warm 502.2 (495.1), 497.9 (490.7), 492.2 (489.6),
  493.2 (481.4), 487.5 (483.3) with the baseline trajectory (0.4434 -> 0.1436/0.1426 -> 0.4492; step 4-9 values move
  by one bf16 ulp from 26 on, consistent with the forward's rounding-order change). The 24 row correctly cites
  `iter24b` (the first chain's MFU run has rc=1).
- Harness 47 / 47 / 48 / 48 / 48 / 48 passed (`O/iter2{3,4,4b,5,6,7}_harness.log`); TP gradient check PASS in every
  chain, eight LoRA gradients at PCC 0.9999 / rel 1.3-1.5e-2, loss 12.8125, parameters bit-identical; loss comparisons
  0.0292 (23, 24, 25, identical to 22) and 0.0298 (26, 27, identical to each other) at tol 0.1 with the baseline
  trajectory; the 26 -> 27 pair is byte-identical as claimed ("same numerics").
- Standalone bit-exactness: 23 (`O/iter23d_build_check.log`), 26 credits (`O/iter26_build_check.log`) and 27
  (`O/iter27_build_check.log`, `O/iter27b_build_check.log` against the 26 baseline) show max |diff| 0 on out / dK / dV
  at all six shapes and the carried 139 / 555 / 144 / 76 / 390 / 18 one-ulp dQ elements from iteration 22; 26b/26c show
  the 8-row chunk's numerics change (out 37 %, dQ 27 %, dK/dV 49 % of elements, max |diff| 3.9e-3 / 7.8e-3 / 3.1e-2,
  PCC >= 0.9999964). 24: every wide piece max |diff| 0 (`O/iter24_wide_check.log`). 25: fused vs two ops max |diff| 0
  at gate_up, w2 wide and w2 scaled tail, fp64 error unchanged (3.103e-2 / 2.958e-2 / 3.134e-2,
  `O/iter25c_fused_check.log`).
- Standalone timings as quoted: 23 fw 145 -> 137, bw 427 -> 393 (592 upward); 24 pad + concat 101 us, tail write 67 vs
  68, wide SwiGLU 140 vs 136, wide transpose_split 86 = 86; 25 gate_up 180 -> 74, w2 277 -> 116; 26 fw 136 -> 132 -> 131
  -> 115, bw 393 -> 387 -> 386 -> 388, chain sweep 147 / 123 / 120 / 119 / 114 / 122 / 126 / 129 / 131 for 4 / 5 / 6 / 7
  / 8 / 9 / 11 / 12 / 14 (`O/iter26c_sweep*.log`), hop-load variants 137-142 / 397-420 slower; 27 fw 110, bw 354 (chain
  14) / 345 (chain 7), sweep 343-346 vs 355-359 (`O/iter27_sweep.log`); 32 heads 1518 -> 1210 and S=2048 1174 -> 1046
  from `O/iter26_sdpa_baseline.log`.
- Attribution rows (chip 0, per 4 layers): 22 -> 23 KV 0.896 -> 0.824, Q 0.574 -> 0.523, fwd 0.545 -> 0.512 (-0.156);
  23 -> 24b `Concat` 0.321 and its `Pad` gone, `SwigluPackedBw` 1.096 -> 0.745, 483 -> 475 ops, kernels without
  collectives 40.7 -> 40.0; 24b -> 25 `LowrankUpdate` 8256 x4 (0.370) gone and 5120 x8 -> x4 (0.513 -> 0.259),
  `TransposeSplit` 0.301 -> 0.399 and 0.185 -> 0.253 (-0.46), 475 -> 467 ops, 40.0 -> 39.5; 25 -> 26 fwd 0.511 -> 0.422,
  Q 0.523 -> 0.508, KV 0.824 -> 0.807 (-0.12); 26 -> 27 KV 0.807 -> 0.722, Q 0.508 -> 0.435, fwd 0.422 -> 0.397
  (-0.18); 467 ops on every chip in 25 / 26 / 27 with the same op set; matmul rows x1.00 on all chips in every profile.
- Iteration-26 attribution (`O/iter26_attrib*.log`): skeleton deltas 49 / 53 / 91 us of 136 / 132 / 198 (the doc's "50 /
  53 / 91" are the skeleton deltas, not the per-stage sums 51 / 62 / 101), math-free pack-free skeletons 75-78 / 62 /
  99, `nofwd,noread` 51, `noreconfig` 72, `nocopy` 72, `noregs` 74; chain length 7 / 28: 129 / 150 forward.
- Code: UMD pointer `7eb549a29db` in all six commits; no `HiFi2` / `LoFi` in the new code (the Wormhole HiFi3 fallback
  comment is pre-existing); `fp32_dest_acc_en = true` in every SDPA and transpose_split program; the attribution patches
  left nothing in the committed kernels; every new parameter is in the program hash (`skinny_matmul_device_operation.cpp
  :173-183` tail_col / tail_tiles / tail_out shape and `:109-118` x shape for the partial; `swiglu_packed_fw` hashes
  `args` with `out_cols`; `swiglu_packed_bw` hashes the dL_dh shapes; `lowrank_update` the acc shape;
  `transpose_split_device_operation.cpp:411-420` `dx_cols` and `w.has_value()`).
- Chain credits (26): the receiver reserves chunk c's slots (`cb_reserve_back` = the compute popped c - 3) after
  `forwarded >= c - 2` (its writer sent c - 3), credits c, then credits ahead up to c + 2 only when
  `forwarded + 3 >= credited + 1` and `cb_pages_reservable_at_back((credited - c + 1) chunks)` both hold (the slot's
  previous occupant popped and forwarded); the sender's writer waits `received >= c + 1` (own reader) and
  `sender >= c + 1` (downstream credit) before the two/four unicast writes to the same CB addresses (CBs sized exactly
  `kChainSlots` chunks in all three factories), `noc_async_writes_flushed`, remote `receiver_sem.up`, local
  `forwarded = c + 1`; semaphores created at 0 (`INVALID`); the sink skips the `forwarded` gates, the injector never
  credits; a one-core chain is injector and sink. No cycle: credits flow upstream, data downstream, each gated on local
  pops (`sdpa_fw_reader_kernel.cpp:107-167`, `sdpa_fw_writer_kernel.cpp:129-159`, `sdpa_bw_q_reader_kernel.cpp:153-229`,
  `sdpa_bw_q_writer_kernel.cpp:88-156`, `sdpa_bw_kv_reader_kernel.cpp:113-198`, `sdpa_bw_kv_writer_kernel.cpp:98-143`).
- Per-round stream bounds (27): forward / Q `max` over the chain's rows of `ceil((q + 1) / chunk)` with `empty = 0`
  (`sdpa_fw_program_factory.cpp:422-434`, `sdpa_bw_q_program_factory.cpp:365-370`), KV `min` of the items' K rows
  floored to the chunk with `empty = St` (`sdpa_bw_kv_program_factory.cpp:345-351`); every core of a chain gets the
  chain's vector; readers, writers and compute loop over the same counts, the chunk counter runs over the rounds, the
  compute drops the chunks past the diagonal / above the K row (`sdpa_fw_compute_kernel.cpp:275-283, 393-401`,
  `sdpa_bw_q_compute_kernel.cpp:326-335, 382-390`, `sdpa_bw_kv_compute_kernel.cpp:138-150, 298-310`); the KV compute's
  per-row pops of a chunk-pushed CB never straddle (CB = 3 chunks, chunk = 4 rows).
- 8-row forward chunk (26): QK^T in two `matmul_block` passes of `kPass = 4` columns packed in order into the reserved
  8-tile chunk, the causal stamp out of order at offsets `diag_pos .. 7` relative to the reserved base with
  `pack_reconfig_l1_acc(1)` before the single push (`sdpa_fw_compute_kernel.cpp:139-200`); the exp stage reserves the
  second chunk of the 16-tile score CB, two passes with the running row sum L1-accumulated into `cb_cur_exp_sum[0]`
  (`p + n > 0`), pops the raw scores and pushes the exp tiles (`sdpa_compute_utils.hpp:168-209`); PV contracts the
  eight V rows in DEST with `ct_dim = pv_block_size = 4` (`sdpa_compute_utils.hpp:225-267`); row max over eight tiles
  into one DEST register. L1 per core in chain mode ~0.55 MB.
- Wide w2 path (24): `skinny_matmul` reads only w's K columns of the wide x (`Kx` stride), its reduce writer stores h to
  the output and to tail page `r * tail_stride + tail_col + n`, then zero tiles up to `tail_tiles` from a cleared
  1-tile CB (`writer_skinny_matmul.cpp:38-65`); `swiglu_packed_fw(out_cols)` writes h with the wide row stride;
  `swiglu_packed_bw` reads the first I columns; `transpose_split(dx_cols)` returns the wide dx whose tail columns are
  the unscaled tail rows (never read); `dA` sliced from `[r, in + 64]`; the hook is installed only for a fused w2 with
  h unreduced on the rank (`fused_lora.py` `_install_wide_swiglu`, `transformer.py:141-142`); `SwigluPackedWide.backward`
  returns one gradient. In-place tail write and the base matmul are on one command queue.
- Fused transpose (25): per dx item (t, r < rows_t) `s_dh_t @ w_r` into DEST0 (K = one tile, fp32), the transposed
  input tile into DEST1, SFPU add, one pack; the s_dh tile recomputed per new t into a 1-tile bf16 CB exactly as the
  tail output (so bit-exact with the two-op path, which also rounds dx once since a bf16 transpose is exact); reader and
  compute track `cur_t` identically, `cb_w` one tile per fused item, the wide tail columns and tail items on the plain
  path; `w` validated as one tile row `[1, 1, tail_rows, rows]` (`transpose_split_kernel.cpp:174-236`,
  `reader_transpose_split.cpp:271-300`, `transpose_split_device_operation.cpp:371-382`).
- Review-9 P1 fix (4041c62353d): `kv_block = use_chain ? chain_chunk : 1` (`sdpa_bw_q_program_factory.cpp:441`); the
  list mode reproduces the pre-22 dQ bit for bit at six shapes (`O/review9_list_mode_check.log`, 483 us), the chain
  mode unchanged, attention tests pass; the "list 548" number struck from the docs.

## Scope Inspected

- Goal/skill paths: `skills/plugins/tt-train-bringup/skills/stage-review/SKILL.md`,
  `.../optimization-loop/SKILL.md`; `optimization/review_round9.md`.
- Artifact paths: `perf_analysis.md` (MFU history, ranked table, iterations 22-27, state after 20, Rejected, Ledger),
  `optimization/README.md`, `optimization/work_log.md` (iterations 22-27, commits table); every `O/iter23*`, `iter24*`,
  `iter24b*`, `iter25*`, `iter25b*`, `iter25c*`, `iter26*`, `iter26b*`, `iter26c*`, `iter27*`, `iter27b*` file read or
  grepped (chain logs, harness logs, TP gradient checks, MFU logs and JSON, tt-smi samples, `tp_analysis.txt`,
  `tp_ranked_ops.txt`, loss comparisons, host-loss run logs, build checks, SDPA checks, attribution logs and scripts,
  sweeps, patch scripts, `iter24_wide_check.{py,log}`, `iter25_fused_check.{py,log}`, `iter26_sdpa_time.py`,
  `iter26_sdpa_baseline.log`), `O/review9_*`, `O/iter22_loss_comparison.txt`, `O/iter22_tp_{analysis,ranked_ops}.txt`,
  `O/perf_analysis_details.md` (headings).
- Code paths (HEAD): `metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp`, `kernels/compute/sdpa_fw_compute_kernel.cpp`,
  `kernels/compute/sdpa_compute_utils.hpp` (26 diff and the helpers), `kernels/dataflow/sdpa_fw_{reader,writer}_kernel.cpp`;
  `metal/ops/sdpa_bw/device/sdpa_bw_{q,kv}_program_factory.cpp` (chain schedule, bounds, CBs),
  `kernels/compute/sdpa_bw_{q,kv}_compute_kernel.cpp`, `kernels/dataflow/sdpa_bw_{q,kv}_{reader,writer}_kernel.cpp`;
  `metal/common/program_utils.hpp`; `metal/ops/transpose_split/device/*` (factory, device op, three kernels);
  `metal/ops/skinny_matmul/device/*` (24 diff, writer, hashes); `metal/ops/swiglu_packed_{fw,bw}/device/*` (24 diff,
  hashes); `metal/ops/lowrank_update/device/*` (24 diff, hash); `nanobind/nb_ops.cpp` (diff);
  `ttml/modules/fused_lora.py` (24 / 25 diffs), `ttml/modules/lora.py` (`skinny_matmul`, `skinny_matmul_fits`),
  `ttml/models/llama/transformer.py` (diff), `tests/python/test_mistral.py` (diff, thresholds).
- Commands run: `git show --stat` and messages of the six commits, `git show <commit>:<path>` for the two files with
  worktree edits, `git diff --stat 912f9562ba6..46dca91d991`, `git ls-tree` for the UMD pointer, `git grep` for
  fidelity strings, attribution leftovers and `initial_load`; a python pass over the six `tp_ranked_ops.txt` files
  (chip-0 sums with and without collectives, per-row deltas); `grep` / `sed` / `awk` over the logs and docs. No device,
  build, test or profile was run.

## Residual Risk

- The credit protocol's correctness rests on NoC ordering of flushed unicast writes before the remote semaphore
  increment on the same NoC (the established tt-metal pattern) and on semaphore re-initialisation per launch; both hold
  empirically over every run since iteration 15, neither is asserted in the code.
- The SDPA gains of 26 and 27 (-0.12 / -0.18 ms per 4 layers) are at the 2-step profile's noise floor; the standalone
  walls (136 -> 115 forward, 393 -> 345 backward) are the evidence.
- The forward's non-chain modes at 8-row chunks and the backward's list mode are exercised by nothing in the model or
  the harness.
- The worktree carries iteration-28 edits to the KV compute kernel and factory plus a changed UMD pointer
  (`git status`); whichever lands next must keep `7eb549a29db` or record the bump, and rerun the six-shape check against
  `O/iter26_sdpa_baseline.npz` (the 27 baseline outputs were not saved as a new npz).
- The box's clock mix now moves the wall more than the kernels do (468.1 with chip 3 at 0.98 GHz effective in the
  profile; 472.5 with all chips at 1.16-1.18): the next iterations' result rows will keep carrying that noise until the
  collectives' overlap item (untried since iteration 12) is taken.
