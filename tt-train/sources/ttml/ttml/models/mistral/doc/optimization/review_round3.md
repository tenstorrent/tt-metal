# Stage review, round 3: optimisation loop resumed (iterations 4-8)

Independent review (fresh subagent, read-only, no device use) of commits dc6ccf3db8e (4), 7c19e385351 (5),
758a7300a8d (5b), 190f3aad6fd (6), d546567c67f (7, rejected), d03dfffa23c (TP sweeps), 4777ff7646a (8) against the
stage-review skill, the resumed loop's docs (`../perf_analysis.md`, `README.md`, `work_log.md`), the evidence
under `evidence/` and the code. Date 2026-10-29.

## Verdict: more-work-needed

## Findings (as returned)

1. **P2. Iterations 6 and 8 add a wasted backward chain for the frozen embedding output (every step, every chip);
   the write-up attributes it as inherent.** Evidence: `iter5b_tp_ranked_ops.txt` vs `iter6_tp_ranked_ops.txt`
   (chip 2, 4 layers): ReduceScatter `[1024,5120]` 17 -> 18, AllGather 17 -> 18, BinaryNg `[1024,5120]` 30 -> 31
   (-> 32 in iter8), qkv base dX matmul 3 -> 4, RMSNormBackward 8 -> 9 (+9 Partial); re-derived from the CSV: a new
   tail `qkv dX 195 us -> add 68 -> RS 271 -> AG 126 -> RMSNormBackwardPartial 88 -> RMSNormBackward 113` absent in
   5b. Cause: a Python `Function` always wraps its output `requires_grad=True` and registers a node
   (`autograd/function.py`), the C++ ops gate on `any_requires_grad`; `RMSNormFrozenGamma` and
   `FrozenVocabParallelEmbedding` therefore make layer 0's chain "require grad". ~0.9 ms per step per chip at TP;
   one of the 18 "inherent" all-reduces of the Loop-resumed table is wasted; "the rest of the compute rows
   unchanged" and "nothing left is a glue, layout or wasted item" are contradicted. Single chip has had the same
   chain since the composite phase (`FrozenEmbedding` was a Function). Fix: fall back to the C++ op / return the
   value without a node when nothing requires a gradient; re-profile, rerun the gradient check, correct the text.
2. **P2. The batched LoRA-gradient reduction (5 / 5b) is costed at "54 us" / "29 + 27 + 27 us" per step, but the
   profile shows ~0.6 ms per 4-layer step per chip of layout glue that scales with the layer count** (Slice,
   TilizeWithValPadding, UntilizeCodegen, UntilizeWithUnpadding, Concat around the two collectives): ~5-6 ms per
   40-layer step (~1 %), never stated; the removed `[1024,32]` reduce-scatters cost ~18 ms on the slowest chip, so
   the net is ~12 ms, not ~34. Fix: state the measured cost as a layout item; remove it by tile-aligned stacking or
   record it as deferred with its cost.
3. **P2. Stale loss-match artefact for the final path.** `final_loss_comparison.txt` / `final_full_tp4_real.log`
   are from iteration 3; iterations 4, 6 (kernels) and 8 (TP embedding) came after. Fix: rerun the 8-layer
   `--host-loss` comparison (`compare_loss.py --tol 0.10`) and the TP=4 real-weight 10-step run on HEAD.
4. **P2. Thermal attribution of the 5 -> 5b wall-time difference is contradicted by tt-smi**; the clock part is
   supported (effective clock 1.106-1.154 vs 1.119-1.259 GHz), the "hotter" part is not (52-76 C in both, same
   power, AICLK 1350 MHz reported throughout), and the effective clock swings +-5 % between runs at identical
   temperatures (iter6 1.135-1.193 vs iter8 1.217-1.239 GHz). Chip 2 is never the hottest chip. "The box is
   thermally bound" is a hypothesis. Fix: reword to "lower effective clock at equal temperature, cause not
   identified"; add a ledger entry for the swing.
5. **P3.** `perf_analysis.md` "best 7.35 ms" misstates the iteration-4 sweep: best overall 7.09 ms (one bf16 ulp
   from a different K-blocking, still HiFi4 / fp32-acc), 7.35 is the best bit-exact one; the implicit bit-exactness
   policy is not stated as the reason for passing over 3.5 %.
6. **P3.** "no config of the family fits L1 for the dX shapes" is not what the logs show: the narrow filters
   enumerated nothing (`0 configs tried`).
7. **P3. Doc hygiene.** (a) `[Q | K | V]` in a code span breaks the Loop-resumed table row; (b) work_log commit row
   of iteration 8 reads "(this commit)"; (c) README "14 at the final path" is 15 since iteration 6; (d) untracked
   `evidence/iter6_diff_vs_iter4_8layers.txt` is a traceback, the single-chip diff for iteration 6 was never
   produced; (e) "1.23-1.26 on the others": chip 3 was 1.164 GHz; (f) iteration-4 temps 52.4-76.3 not 54-76;
   (g) gate_up fwd "75 %" / dX "88 %" recompute to 74 % / 87 %; (h) `rmsnorm_bw_allcore.cpp` passes a hard-coded
   epsilon the kernels never read.

Verified clean by the reviewer: every MFU-history number of iterations 4-8 against its log; the per-chip tables and
collective counts; iteration 4's fit, bench and op; the iterations 5 / 5b math on both parallel paths and its
gradient checks; iteration 6's port (byte-identical to `umales/qwen38` after renames), frozen-gamma use and test;
iteration 8's op-for-op equivalence with `VocabParallelEmbedding.forward` (gradient check equal to iteration 6's
to every printed digit); HiFi4 + fp32 accumulation everywhere; the TP sweeps and the iteration-7 rejection; the
commit table through the TP sweeps; the other tables.

## Response (follow-up, same day)

- Finding 1: `frozen_embedding.py` returns the lookup outside the autograd graph (`ttml.autograd.create_tensor(...,
  requires_grad=False)`, both the plain and the vocab-parallel frozen path); `rmsnorm_allcore.py` routes a
  no-grad input through the C++ `ttml.ops.rmsnorm.rmsnorm` (which builds no node). Re-verified by the follow-up
  chain (`evidence/review3_followup.sh` -> `review3_followup.log`): harness, TP gradient check, 8-layer single-chip
  profile, cooled TP=4 measurement and 4-layer profile (collective and norm-backward counts), harness 15 passed, gradient check PASS unchanged, TP profile 17/17 collectives and 8+8 norm-backward ops per
  4-layer step (was 18/18 and 9+9), single-chip 16 norm-backward calls at 8 layers (was 17), TP=4 545.7 ms /
  29.73 % (545.9 before); numbers in `../perf_analysis.md`, "Review round 3 follow-up".
- Finding 2: cost stated in iteration 5b's section and the Loop-resumed table (layout row ~48); tile-aligned
  stacking measured on device (`evidence/review3_stack_bench.{py,log}`): no gain, the unpad back to the 16-row
  logical shape costs what the unaligned slice cost; recorded as a layout item inherent to rank-16 adapters.
- Finding 3: the 8-layer real-weight host-loss comparison and the TP=4 real-weight 10-step run rerun on HEAD +
  fixes in the follow-up chain: 8-layer host-loss max |diff| 0.027 vs the torch fp64 reference (tol 0.10, PASS,
  `evidence/review3_loss_comparison.txt`), TP=4 real weights 10 steps with the baseline's trajectory
  (`review3_full_tp4_real.log`).
- Finding 4: 5b's row and section and the ledger entry reworded ("lower effective clock at equal temperature,
  cause not identified"), the +-5 % swing recorded with the per-run clocks and temperatures, "thermally bound"
  removed from the Loop-resumed section.
- Finding 5: the sweep's fastest configs and the bit-exactness policy stated; the 7.09 ms config listed as deferred.
- Finding 6: reworded to what the logs show (no candidates under the narrow filter; wrong family for a tall dX).
- Finding 7: (a) escaped; (b) commit row filled; (c) 15; (d) traceback file deleted, the single-chip comparison of
  iteration 6 is the per-op table pair `iter4_tracy_8layers_ranked_ops.txt` / `iter6_tracy_8layers_ranked_ops.txt`
  (and the follow-up adds `review3_tracy_8layers_ranked_ops.txt`); (e) 1.164-1.261; (f) 52-76 C; (g) 74 % / 87 %;
  (h) commented as a host-side attribute the kernels never read.
