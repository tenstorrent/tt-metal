# Stage review, round 9: iterations 19-22 (2026-11-01)

Independent review (fresh subagent, read-only, no device use, no build) of commits 11f9d923524 (19, `add_rmsnorm_fw_rows`
+ w2 2D matmul config + round-8 answers), a3d3ec03947 (20, rejected: FPU norm math), c219767ba30 (state after 20),
fd17a21a00b (21, rejected: `sdpa_scheduler` forward) and 66b8a997d14 (22, blocked `sdpa_bw_q` kernel + single-barrier
chain injectors) against the optimization-loop skill (hard constraints, kernel boost checklist, rejection standard,
document format), the docs, the `O/` evidence and the kernels. Round-8 items re-checked. The worktree carries
uncommitted iteration-23 edits in the SDPA readers / writers / factories; every code statement below is about the
committed HEAD (`git show 66b8a997d14:...`), not the worktree. Paths relative to `doc/` (`O/` = `optimization/evidence/`);
kernel paths relative to `tt-train/sources/ttml/`.

## Verdict: more-work-needed

One P1 (a CB-alignment hazard in the non-chain modes of the blocked Q kernel, untested and reachable), five P2 (evidence
provenance, stale and contradicting numbers, a stale state conclusion, commit hygiene). The four landed numbers
(495.4 / 32.75 %, 494.5 / 32.81 %, 509.1, 504.9) and every attribution row re-derive from the logs; the rejections of 20
and 21 and of the multicast stream are earned by the standalone evidence.

## Findings

1. **P1. The blocked Q compute kernel reads K/V in 4-row windows from a CB that the non-chain readers fill one row at
   a time: the window straddles the CB end after the first partial block** (`metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.cpp:411-412`
   `kv_block = use_chain ? chain_chunk : std::min(St, 4)`, `kv_cb_rows = 2 * kv_block`;
   `kernels/compute/sdpa_bw_q_compute_kernel.cpp:168-175, 186, 296, 320-321`; `kernels/dataflow/sdpa_bw_q_reader_kernel.cpp:119-122, 288-301`).
   - Without `CHAIN` the compute waits `cols * qk_tiles` tiles and indexes K tile `(c0 + j) * qk_tiles + f` linearly from
     the read pointer, then pops `cols` rows; the work-list / balanced / standard readers push K and V one row
     (`qWt` tiles) at a time. The CB holds 8 rows. A causal row `q` processes `q + 1` rows in blocks of 4 with a tail of
     `(q + 1) % 4`, so the read pointer advances by `q + 1` rows per row and lands at every offset mod 8: for Ht = 32 the
     row q = 3 starts its 4-row block at offset 6 and its columns 2-3 are addressed past the CB end (tile indexing does
     not wrap; the producer wrapped its rows to the buffer start). Non-causal (standard mode) hits the same when
     `Ht % 4 != 0`. Before iteration 22 the kernel consumed one row per window, so this is new.
   - Reachable: `TTML_SDPA_BW_Q_STREAM=list` (the attribution path: `O/iter22_attrib3.log` "full list 548 us", quoted in
     `perf_analysis.md:269` and `work_log.md:162`, was measured on it with no `--ref`, so that number is void), any causal
     call with `B x kv_heads > 55` streams, and every non-causal call. The three SDPA harness cases
     (`tests/python/test_mistral.py:573-575`: 2 / 2 / 1 streams) and the six `iter11_sdpa_check.py` cases all take the
     chain path, so nothing on the branch exercises the broken modes. Same class as the Ledger's "latent CB wrap hazard"
     of iteration 9 and the `CB pushes must not straddle` rule.
   - Required: `kv_block = 1` when `!use_chain` (or a block-granular non-chain reader that pushes padded `kv_block`-row
     blocks and a compute pop of `kv_block` rows), one standalone or harness case that forces the list mode
     (`TTML_SDPA_BW_Q_STREAM=list` with `--ref`, or a `B x kv_heads > 55` shape), and the "list 548" attribution
     re-measured or struck. Hardware confirmation needed (this review ran nothing); the addressing argument is from the
     code alone.

2. **P2. The iteration-19 loss match did not exercise the w2 matmul config it was rerun for** (`perf_analysis.md:261`,
   `work_log.md:197`, `O/iter19.sh` comment). `PROGRAM_CONFIG_2D` is keyed `(1024, 8256, 5120, False)`
   (`ttml/modules/frozen_linear.py:121-126`), the TP=4 shard; the single-chip 8-layer host-loss run has K = 32832, so it
   took ttnn's default. `O/iter19_loss_comparison.txt` is byte-identical to `O/iter19a_loss_comparison.txt` (the chain
   that ran without the config, `work_log.md:192`). What covers the config is the TP=4 real-weight trajectory alone
   (`O/iter19_full_tp4_real.log` 0.4434 -> ... -> 0.4473 vs 0.4414 -> ... -> 0.4492 without it: one bf16 ulp at step 1),
   which has no torch reference. Required: say so in the Verification bullet (the 8-layer PASS is the norm op's evidence,
   the TP trajectory the config's), and list the torch-referenced check of the config as a hard-check gap (a TP=4
   host-loss comparison, or a single-chip run at the shard width).

3. **P2. The iteration-22 wall-vs-attribution explanation repeats round-8 finding 1: the quoted clocks belong to the
   profile runs, not the MFU runs** (`perf_analysis.md:268` "this run's chips clocked 1.03-1.09 GHz against 1.08-1.16 in
   the iteration-19 run", Ledger `:364`, `README.md:43` "at a 4 % lower clock"). 1.03-1.09 / 1.08-1.16 are the effective
   clocks of the 4-layer Tracy runs (`O/iter22_tp_analysis.txt` 1.056 / 1.068 / 1.092 / 1.031; `O/iter19_tp_analysis.txt`
   1.081 / 1.076 / 1.091 / 1.164); the MFU runs' tt-smi samples read 1350 MHz on all four chips in both
   (`O/iter19_tp_mfu_tp4_tt-smi.txt`, `O/iter22_tp_mfu_tp4_tt-smi.txt`, equal temperatures and power, chip 2 at 77-78 W in
   both), so no per-chip clock of the MFU runs exists. What the evidence supports: -4.2 ms per step cycle-time on the SDPA
   rows (`O/iter22_tp_ranked_ops.txt` vs `O/iter19_tp_ranked_ops.txt`), chip 0 kernels without collectives 41.1 -> 40.6 ms
   per profiled step, and a wall delta (-0.9 ms) inside the per-step spread of both runs (480-506 / 485-506). Required:
   replace the clock sentence with the noise statement in the three places.

4. **P2. Stale and mutually inconsistent standalone numbers in the iteration-19 write-up** (`perf_analysis.md:260`).
   - "Standalone: norm forward 166 -> 158 us per call (bit-exact), below half the expected gain": 158 is the resident
     version (`O/iter19_build_check2.log`); the committed streaming version is 138 (`O/iter19_norm_check.log`,
     `O/iter19_build_check5.log`; `README.md:68`, `work_log.md:193`, commit message agree). Against the expected 120-130
     band (`work_log.md:189`) 138 is 78 % of the expected gain, so "below half" is wrong for the landed kernel.
   - "w2 forward 1174 -> 1043 us (interleaved A/B medians, -11 %)": 1043 is the 11x10 config of `O/iter19_ab_check.log`;
     the committed config is 11x8 (`frozen_linear.py:123-125`, `O/iter19_config_lookup.log`), 1058 (-10 %); the code
     comment carries a third pair (981 -> 882, the sweep). Nothing says why 11x8 over 11x10 (the two sweeps rank them
     882 / 886 and 873 / 882 us: equal within noise).
   - Required: 138 / 1058 (-10 %), drop "below half", one sentence on the 11x8 choice.

5. **P2. The "State after iteration 20" conclusion is contradicted by iteration 22 and the section is out of order**
   (`perf_analysis.md:274-288`, `README.md:79`). `:288` "took the owned kernels to their exact-fp32 ceilings ... not the
   remaining ttml kernels (~10 ms in total at their ceilings)", yet iteration 22 then removed 4.2 ms/step from the SDPA
   backward with no numerics change and names a further ~100 us per backward call (~8 ms/step) as the next lever
   (`:269`); the SDPA row of the table was patched to mention 22 while the summary paragraph and the README line were not.
   The section also sits between the 22 and 21 sections. Required: restate the summary (owned-kernel remainder with the
   SDPA lever included, or date it "before 22"), and either move the section after 21 or keep the sections chronological.

6. **P2. Commit hygiene: iteration 19 bundles four changes, and the "(this commit)" placeholder is back**
   (`git show --stat 11f9d923524`: the `add_rmsnorm_fw_rows` op, the `PROGRAM_CONFIG_2D` / `forward_matmul_kwargs(k=)` /
   `dx_matmul_kwargs` Python change, the review-round-8 answers in three docs, two `skinny_contract` harness cases;
   `README.md:74` "one change per commit"; the skill's bisect rule). The matmul config is a Python-only change with its own
   A/B (`O/iter19_ab_check.log`) and its own loss-match question (finding 2); it does not bisect from the op.
   `work_log.md:281` reads `| 22 | (this commit) |` (round-8 finding 4 was the same line for 18). Required: record the
   bundling in the commits table row (the history is local and unpushed, so a split is possible but not required), fill
   the hash in the next commit.

## Other concerns (P3)

- `README.md:76` "Reviews" stops at round 7 "re-check pending" (round 8 was answered in 11f9d923524); `README.md:71`
  "State (2026-10-30)" carries the 11-01 entries.
- Iteration-22 hang evidence: `O/iter22d_sdpa_check.log` completed all six cases (every wall 1.5-4.7x slower:
  q6 kv2 S512 551 vs 117 us) before the rc=137 kill 53 s after start (not the 900 s timeout), so 22d documents a
  degraded chip and a manual kill, not the "two injectors on the shard shape" deadlock the Ledger describes (no
  `iter22f_*` log exists for it); `O/iter22e_sdpa_check.log` (three cases, then nothing at q4 kv1) is the single-stream
  hang. `O/iter22d_chain_sanity.log` ran concurrently with the 22d attention tests (08:53:41-08:54:06 vs
  08:53:37-08:54:16, which is why those tests took 38 s instead of 5), carries no `--ref`, and its walls (541 / 704 /
  558 us) are meaningless: it shows "completes", not "correct". No `tt-smi -r` output is kept. Ledger entry wanted
  (what hung, how it was killed, the reset log).
- `perf_analysis.md:272` lists `O/iter22_attention_tests.log` among the evidence without saying it is the failed
  first build (4 failed; the final tree's run is `O/iter22i_attention_tests.log`).
- `perf_analysis.md:269` "plain per-row DRAM reads 548" and `work_log.md:162` "list ... 548": void after finding 1.
- Iteration 20: the precision-preserving hybrid ran 110 us at T = 512, C = 2048 against 38-42 for every other variant
  (`O/iter20_norm_bw_hybrid.log`); unexplained, irrelevant to the rejection, worth a word.
- Iteration 21: the port's outputs at q4 kv1 and S = 2048 differ from the chain kernel in 50 % of dK / dV elements
  (max 1.56e-2 / 3.1e-2, `O/iter21_sdpa_check.log`); "one bf16 ulp" holds only if those gradients sit in [2, 8), which
  the log does not show. The rejection does not rest on it.
- `metal/ops/add_rmsnorm_fw_allcore/add_rmsnorm_fw_allcore.cpp` dispatch: the L1 check counts gamma + sum only
  (`slice_tiles * 2 * 2 KB <= 512 KB`); the other CBs are 36 KB, so it holds, but the bound should name them.
- `O/perf_analysis_details.md` still ends at iteration 12 (round-8 P3); the iteration-19 Change bullet is four lines.

## Hard-check gaps

- Finding 1 needs a device run (the list mode with `--ref`); finding 2 needs a torch-referenced loss comparison on a
  path that takes the 2D config.
- Rank-16 zero-pad invariant (Ledger): still not validated on device (carried from round 8).

## Anomaly ledger

- Observed anomaly: `O/iter22d_sdpa_check.log` killed (rc=137) after all six cases at 1.5-4.7x the normal walls.
  Evidence: the log, `O/iter22d_build_check.log` timestamps. Affected path: the multicast experiment only (removed).
  Control: `O/iter22i_sdpa_check.log` (final tree) at the normal walls after the reset. Likely subsystem: a chip left
  wedged by the previous multicast hang (`iter22e` is the hang with a log; the first hang has none). Investigation
  performed: `tt-smi -r`, a chain-mode run (walls invalid, see P3). Resolution: controlled for the landed code; the
  Ledger entry should carry the above (P3).
- Observed anomaly: `O/iter22_attrib2.log` chain = no-chain (518 / 516). Evidence: the log, the Ledger. Control:
  `O/iter22_attrib3.log` after the rebuild (517 / 548). Resolution: controlled (Ledger entry present; the skeleton rows
  of attrib2 are JIT-compiled kernels and stay valid).
- Observed anomaly: iteration-22 wall -0.9 ms vs -5 ms expected from the cycle-time rows. Evidence: the two MFU logs,
  the two profiles. Control: per-step spread 480-506 in both runs. Resolution: controlled once the clock sentence is
  replaced (finding 3).

## Re-derived clean

- UMD pointer `7eb549a29db` in the tree of all five commits; no `HiFi2` / `LoFi` in the new code; `fp32_dest_acc_en`
  in both new programs; `_compute_config()` HiFi4 + fp32 accumulate for the 2D config.
- MFU rows: 19 = 495.4 (487.9) / 32.75 % over steps 3-8 of `O/iter19_tp_mfu_tp4.log`; 22 = 494.5 (486.9) / 32.81 %;
  19a = 498.8 (492.7) / 32.53 %; real weights warm 509.1 (503.0) and 504.9 (498.5) from the JSON step lists; 8 steps,
  2 warm-up, 40 layers, random init, cool-down 600 s in every chain (`O/iter19.sh`, `O/iter22.sh`, `O/tp_measure.sh`);
  per-chip matmul rows x1.00 in all three `tp_analysis.txt`.
- Harness 47 / 47 / 47 passed (19a / 19 / 22); TP gradient check PASS in all three; loss comparisons 0.0298 / 0.0298 /
  0.0292 (tol 0.1) with the baseline trajectory; the iteration-22 single-chip 8-layer run does take the chain SDPA
  path (8 streams), so its loss match is valid for the kernel change.
- Attribution rows (chip 0, per 4 layers): 18b -> 19: w2 3.693 -> 3.249, norm 0.565 + 0.558 -> 1.010; 19 -> 22: KV
  1.093 -> 0.896, Q 0.784 -> 0.574, fwd 0.558 -> 0.545 (sum -0.42); kernels without collectives 41.5 -> 41.1 -> 40.6;
  483 ops on every chip in 19a / 19 / 22 with the same op set (nothing new; `SwigluPackedBw` 1.055 / 1.094 is the usual
  row noise).
- Standalone chain: backward 532 (`O/iter21_sdpa_baseline.log`) -> 517 (blocked kernel, `O/iter22b_*`) -> 483 (Q
  injector, `O/iter22g_*`) -> 427 (KV injector, `O/iter22i_*`); forward 149 -> 145; dQ 139 / 1048576 elements at one bf16
  ulp, forward / dK / dV bit-exact at all six shapes; attribution stage deltas 6-11 us (`O/iter22_attrib.log`), skeleton
  517 -> 506, injector without reads 472, Tracy per op Q 178 / 193 (`O/iter22_tracy_modes.log`); multicast 526 / 514.
- Iteration 20: forward 138 -> 134, backward 173 -> 145 at rel 4.09e-3 (variant A 149 / 4.09e-3, B 171 / 1.66e-3,
  hybrid 173 / 1.66e-3), reverted kernels at 138 / 173 and 1.66e-3, norm tests 8 passed before and after
  (`O/iter20_*`). Rejection earned: the only faster variant breaks the exact-fp32 elementwise rule, the exact one
  gains nothing.
- Iteration 21: default 186.0, best 165.7 (`q128 k128 h1 p0`) / 167.1 (`q64 k128 h2 p4`), main's chunking 239.8, 21
  configs over the two benchmark logs; the check script's wall for the port 185 vs 149-157 for the chain kernel (same
  script, same session); S = 2048 401 vs 444; reverted tree bit-exact at six shapes (`O/iter21_sdpa_reverted.log`).
  Rejection earned (measured at the model shape with the same harness, several configs, the reason stated).
- Rejected-table costs: 22 multicast 7.8 = 0.784 x 10, 21 forward 5.6 = 0.558 x 10, 20 norms ~23 = (1.318 + 1.010) x 10.
- Blocked Q kernel (chain path): scores in DST[0 .. n) with the mask then the lse tile in DST[n] (n <= 2, four fp32
  tiles per acquire); causal mask only on the diagonal column, which is always the last column of the last block;
  arbitrary mask: one `cb_pop_front` per column matching the standard reader's one mask tile per K row; dS pass uses
  DST[0 .. n) and DST[2 .. 2 + n); the dQ pass's `reconfig_data_format(cb_key, cb_grad_scores)` maps in1 -> SrcA (bf16)
  and in0 -> SrcB (fp32) as a matmul unpacks (the first build's inf dQ was the swapped call); the
  `pack_reconfig_l1_acc(true)` / pop / reserve / push sequence is the pre-22 `update_grad_query` pattern on the
  `qWt`-tile single-slot accumulator (`sdpa_bw_q_program_factory.cpp:452-454`); P / dP / dS CBs hold one `kv_block`
  each, reserve / push always `kv_block` with the tail slots of a partial block unused; chain pops whole chunks
  (rows past the diagonal dropped with the chunk), `0xFFFFFFFF` rounds and the post-diagonal loop pop `Ht / kv_block`
  chunks; `block_size = get_block_size(qWt, 4)`.
- Chain injectors (Q, KV, forward): `cb_reserve_back` of the chunk before `get_write_ptr`, every page of the chunk in
  flight, one `noc_async_read_barrier`, then the full-chunk push; the forward's `read_tile_block_transposed<false>` /
  `read_tiles_by_row<false>` reserve the same count again (idempotent) and skip barrier and push; the KV injector steps
  the u tiles by `get_tile_size(cb_u_scalar_row)`, the same stride `read_one_tile` used.
- `add_rmsnorm_fw_rows`: one (row, slice) item per core (`rows x S <= cores` asserted), residual / delta streamed in
  padded groups of two (reader pushes `group`, compute pops `group`, reads `n`), gamma and sum resident (`St` tiles
  each, 216 KB at the model shape), output in padded blocks of four with the writer storing `n`; partial exchange: write
  to slot `s` of every row core, barrier, S semaphore increments, reader waits S and resets; the writer's sum-slice
  writes are covered by the exchange barrier; slice 0 writes rms; no circular wait (partials leave before the rms wait).
  Dispatch: `slices = min(cores / rows, 8)`, `>= 2` slices, width a multiple of 32, `TTML_ADD_RMSNORM_TWO_PHASE` forces
  the old path; bit-exact vs the two-phase op at the 2-6-slice shapes, the one-slice shapes stay on the two-phase op
  (`O/iter19_norm_check.log`).
- `forward_matmul_kwargs(..., k=x_cat.shape[-1])`: the fused call now finds the 2D config (`O/iter19_config_lookup.log`);
  the 1D gate_up config and `dx_matmul_kwargs` fall back to the full grid as before.
- Round-8 fixes present: attribution restated collectives-excluded with all product rows, Ledger entries for the failed
  run-B profile and the iteration-13 docs, ranked table regenerated from run B (`O/iter18b_tp_ranked_ops_top80.txt`),
  `O/iter18_cmake.log` removed, two `skinny_contract` cases (N = 1536 routed, N = 7072 short last group), loss match
  rerun in the iteration-19 chain.

## Scope inspected

- Skills: `stage-review/SKILL.md`; `review_round8.md`.
- Docs: `perf_analysis.md` (MFU history, ranked table, iterations 19-22, state after 20, Rejected, Ledger),
  `optimization/README.md`, `optimization/work_log.md` (iterations 19-22, round 8, commits table),
  `O/perf_analysis_details.md` (headings).
- Evidence: every `O/iter19*`, `O/iter19a*`, `O/iter20*`, `O/iter21*`, `O/iter22*` file (logs, JSON, tt-smi, ranked
  tables, analysis, scripts), `O/iter18b_tp_{analysis,ranked_ops}.txt`, `O/iter14_*` loss files, `O/tp_measure.sh`,
  `O/iter11_sdpa_check.py`; MFU means and bests recomputed from the JSON step lists.
- Code (HEAD): `metal/ops/sdpa_bw/device/kernels/compute/sdpa_bw_q_compute_kernel.cpp`, `sdpa_bw_compute_utils.hpp`
  (mask / softmax helpers, old `update_grad_query`), `kernels/dataflow/sdpa_bw_q_reader_kernel.cpp` (all modes),
  `sdpa_bw_kv_reader_kernel.cpp` (chain section), `sdpa_bw_q_program_factory.cpp` (modes, CBs, defines),
  `metal/ops/sdpa_fw/device/kernels/dataflow/sdpa_fw_reader_kernel.cpp` (chain section),
  `metal/common/dataflow_utils.hpp` (helpers), `metal/ops/add_rmsnorm_fw_rows/` (all kernels, factory, device op),
  `metal/ops/add_rmsnorm_fw_allcore/add_rmsnorm_fw_allcore.cpp` (dispatch), `ttml/modules/frozen_linear.py`,
  `ttml/modules/fused_lora.py` (diff), `tests/python/test_mistral.py` (diff, SDPA cases).
- Git: `git show --stat` and messages of the five commits, `git ls-tree` for the UMD pointer, `git status` /
  `git diff HEAD --stat` (iteration-23 worktree edits excluded from the review), `git grep` for the env knob and
  fidelity strings.

## Residual risk

- The chain path's correctness after 22 rests on the standalone comparisons against the iteration-20/21 outputs
  (dQ one ulp, the rest bit-exact) and the harness; the non-chain modes have no comparison at all (finding 1).
- The result-metric gain of 19 + 22 (503.9 -> 494.5 ms, 1.9 %) is of the order of the run-to-run noise; the
  cycle-time rows (w2 -0.44, norm -0.11, SDPA -0.42 ms per 4 layers) are the evidence, from 2-step profiles with
  ~0.1 ms row noise.
- The iteration-23 edits in the worktree touch the same reader / writer / factory files; whichever of them lands next
  should re-run the six-shape standalone check against `O/iter22i_sdpa_check.log`'s reference.
