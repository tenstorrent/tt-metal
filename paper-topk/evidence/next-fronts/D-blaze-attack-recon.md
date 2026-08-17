# Front D-blaze-attack-recon (design/recon swarm, 2026-08-17)

## Verdict

Blaze's 24.4µs is NOT an apples target for our 34.3µs prefill anchor: measured per-core Tracy decomposition (surviving comp4 device log) shows it is a decode-geometry program doing SDPA+streaming for ~10µs and a 16-core local top-2048 over only 8192 valid / 32768 padded positions per device (CP-8 round-robin of the 65536 global context) for the remaining ~14.4µs. The one thing we hold that they lack is the branch-only SFPLOADMACRO unfused merge/rebuild (725748766a6, 427 diff lines vs main — the ONLY unpushed LLK delta); their fork already has the replay/MOP machinery and the INT32 fused-word denormal guard (they cite our upstream 814f1b46), so the realistic transplant win is topk-phase 14.4→~11-12µs (program ~21-22µs), more if a fused-path SFPLOADMACRO sort variant is built (new work). Swapping our op in as an op (path b) loses by construction (~35-40µs serialized vs 24.4 fused); the winning coordination move is #1971 LLK unification: upstream 725748766a6 to tt-llk, and fold blaze's three fork-only surfaces (add_lsb row_major, the two reinit-after-copy hooks, local_sort_generic early-exit/int32) into the canonical header so the fork can die.

## Plan

FRONT D RECON — blaze fused SDPA+localTopK (all static; no device runs)

== 1. THEIR STRUCTURE (indexer_local_topk + fork) ==

Topology (/home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/kernels/op.hpp:4-26, op.py:104-159):
- ONE binary tree over the topk cores (core_id = bank*tpb + slot), generalizing distributed_topk. Phase 0 per core: topk_xl_copy_tile with per-chunk valid count (−inf pads tail) + add_lsb_indices<K, core_id, row_major=true> stamps within-bank position at bits 0..13 + local sort of the 2048-window (op.hpp:342-404). Stages 0..max_fused_stage-1 are FUSED bank-local; a "split bank stamp" trick (op.py:110-159) rides sub-bank bits in add_lsb's spare core_id field so the first cross-bank stages stay FUSED; separate_indices at unfuse_stage stamps group|device and switches to UNFUSED u32 (op.hpp:559-565). Direction-alternation discipline: every stage rebuilds to the next stage's direction so operands always arrive oppositely monotone (op.hpp:22-26, 611-615).
- Transport: DM0 posted single-packet NOC write + posted semaphore inc, per-stage recv CB slots with advancing base (op.hpp:496-533). Compile-time-unrolled stages — senders' later stages DCE away (op.hpp:549-644).
- Streaming: SDPA raw-NOC-writes 2048-position chunks into the topk cores' L1 input CB and bumps stream_sem; DM1 waits sem then cb_push (op.hpp:205-240). Input never touches DRAM.
- Validity machinery we lack: cp_local_valid_count (CP round-robin ownership), bank_valid_count (32-position round-robin across 8 banks), invalid-partner/subtree pruning on BOTH compute and DM sides (op.hpp:52-83, 303-322, 475-482), pos≤K lockstep early-out (min_active_pos), bank_validity_remap for root-first core rotation.
- Placement (blaze/ops/distributed_indexer/config.py:31-34, 98-118): topk row is the 1×4 directly below each SDPA 1×4 group, disjoint by construction; index encodes within-bank[13:0] | bank[16:14] | device[19:17].

Fork status (measured today, not the stale 1304 figure):
- fork vs tt-metal MAIN canonical: 1072 diff lines. Fork HAS replay/MOP parity (load_replay_buf 52 vs 55 occurrences) and HAS the fused-word INT32 denormal guard, explicitly citing our commit 814f1b46 (fork ckernel_sfpu_topk_xl.h:356-368) — the "fork lacks denormal protection" line in our notes is now WRONG for the fused path.
- fork vs our branch: 1512 lines; main vs branch: 427 lines = exactly one commit, 725748766a6 "SFPLOADMACRO-scheduled compare-exchange on the unfused path (BH)" — configure_sequences/program_templates/ce_first15/ce_tail/ce_full/record_ce_full (our header :112-339). This is the ONLY performance-bearing thing we have that neither main nor the fork has.
- FORK-ONLY surfaces their kernel depends on (canonical lacks): (a) add_lsb_indices row_major template param (fork :2253, used op.hpp:354,387 — canonical add_lsb is <K,APPROX,core_id> only, api/topk_xl.h:214); (b) topk_xl_reinit_mop_after_copy + topk_xl_reinit_unfused_rebuild_after_copy (fork LLK :299, used op.hpp:605,625 — absent from canonical LLK and API); (c) _topk_xl_local_sort_generic_<K,APPROX,early_exit_K64,int32_mode> (fork :1194, for their sparse-K reader column-sort). CANONICAL-ONLY: the separate_indices_row_major family (:2861-2992) — different row-major mechanism; the fork solves it with add_lsb row_major + fixed 14-bit shift.

== 2. DECOMPOSING THE 24.4µs (measured, from surviving artifacts) ==

Cell identity: comp4 blaze cell = tt-blaze test_glm52_indexer_sdpa_streaming_local_topk[64k] (harness _canonical_topk_sweep.py:300-318); ns_median 24432 over 9 GenericOpDeviceOperation rows (comp4 results json). CRITICAL GEOMETRY (test :89-104 + comp4 CT-args dump): global_valid=65536 under CP round-robin over 8 devices → local_valid = 8192 valid positions on the one measured device; capacity positions_per_device = 32768 → the 32k geometry: 16 topk cores (2/bank), chunks_per_core=1, num_stages=4, max_fused_stage=1, unfuse_stage=3, min_active_pos=2048 (all confirmed in comp4 work log CT args). 129 "cores" = FusedProgram launched grid-wide; median per-core busy span is 0.18µs — ~99 cores are inactive passengers; real work on ~30 cores (32 SDPA + 16 topk, minus overlap of pruned slots).

Per-core phase timeline (my parse of generated/profiler/reports/2026_08_16_19_26_20/profile_log_device.csv, run 5120, 1350 MHz):
- 0 → ~9.5-10.6µs: SDPA compute + K DRAM read + streaming. All NCRISC (DM1 stream waits) end 9.4-10.6µs; 24 non-anchor SDPA cores' TRISC1 end 8.5-9.9µs. SDPA+stream phase ≈ 10µs (~41%).
- ~10 → 18.1-18.8µs: leaf phase on the 8 surviving slot-0 topk cores (slot-1 cores prune: l_b=1024 ≤ 2048, drain and exit ~10µs). copy+add_lsb+sort(2048)+rebuild+pack ≈ 8µs.
- tree tail, matching the binary tree EXACTLY: 4 stage-1 senders end 18.1-18.8; 2 stage-2 enders 20.5-20.6; stage-3 sender 23.3; root (1,2) 24.4 (TRISC1 busy 0.1→24.37). Unfused merge+rebuild+transport ≈ 2.0-2.8µs/stage; root final merge+pack ≈ 1.1µs.
- Total topk phase ≈ 14.4µs (~59%) for 32768 padded slots (8192 valid) on 16 cores.

Cross-anchors from comp4 competition_table.csv: our op k2048: W=8192→22.88µs@4c, W=32768→30.23µs@16c, W=65536→34.32µs@32c, W=2048→15.74µs (fixed op envelope ≈9-10µs inside device time). Equal-cores/equal-slots comparison: their 14.4µs@16c vs our 30.23µs@16c at 32768 slots — they are ~2.1x faster at the decode shape, explained by L1-streamed input (no DRAM read/untilize/launch envelope) and SDPA overlap, NOT by better sort/merge bodies (ours are equal-or-better: SFPLOADMACRO merge 1.438 vs their 2.844 cyc/vec).

== 3. ATTACK PATHS ==

(a) Transplant our LLK wins into their fork — the near-term win:
- Clean transplant: 725748766a6 (SFPLOADMACRO unfused CE). The fork's unfused merge/rebuild call the same entry points; the macro section is self-contained with a DISABLE_TOPK_XL_SFPLOADMACRO opt-out (:279-282). Their 3 unfused stages + root final (≈7µs of the chain) carry merge/rebuild bodies at 2.844 cyc/vec → 1.438; expected save ~1.5-3µs. Better route than cherry-pick: reconcile fork→canonical-main first (fork is only 1072 lines off), then the macro applies as main+427.
- NOT covered: their leaf phase (8µs) is the FUSED sort — our SFPLOADMACRO targets unfused CE only. A fused-path macro variant is new engineering (SORTING.md B4: SFPLOADMACRO expresses a MAP at 1.000 cyc/vec; sort compare-exchange passes qualify); if it matches the unfused gains, leaf 8→~5-6µs.
- Expected program: 24.4 → ~21-22µs (transplant only) → ~18-19µs (with fused-path macro). Topk phase 14.4 → ~11-12 → ~8-9µs. SDPA phase (10µs) untouched — that half is theirs to optimize.
(b) Replace their topk cores with our op post-Front-C — NOT a real path for decode. Gaps (confirmed, callsite-map §E + this recon): explicit core list, L1-streamed sem-paced input with per-bank validity, bank/device bit-stamping, FusedProgram co-residency. Even with Front C placement+L1: as a separate op it serializes after SDPA and pays its own envelope → ≈10 (SDPA program) + ≥14-23 (our op) ≈ 30-40µs vs 24.4. Parity requires living INSIDE their FusedProgram — which collapses path (b) into path (a) + adopting their streaming/validity plumbing. Front C's placement+L1 matters for ttnn callers, not for beating the blaze decode number.
(c) Beat 24.4 with our op alone — wrong target, quantified: 24.4 = SDPA(10) + topk-over-8192-valid(14.4) at decode geometry. Our 34.3µs anchor processes 8x the topk positions with no SDPA. Honest claims to publish instead: (i) prefill (the GLM callsite our op owns by name) — blaze doesn't play there; 34.3µs stands alone; (ii) per-slot topk-only at equal cores: they 2.1x us at W=32768/16c today, entirely envelope+residency, not kernel bodies — the Front C target is closing exactly that (L1-resident input + envelope cuts → est. floor 14-18µs at that shape); (iii) launch envelope: their whole fused program pays ONE dispatch for SDPA+topk; our op's ~9-10µs internal envelope is 30-45% of device time at decode shapes. The "24.4 vs 34.3 = 1.4x" framing in the ledger should carry the new caveat: theirs also does 8x less topk work.

== 4. COORDINATION STATE (#1971) ==

- Upstream tt-metal main already has: canonical topk_xl + tests (#51777, df31fd4a847), the row_major separate_indices family, the INT32 denormal guard, replay/MOP. The fork already consumed the denormal fix (cites 814f1b46) — drift hazard is real but smaller than documented (1072 lines, not 1304).
- OUR offer for their rebase: exactly one tt-llk commit — 725748766a6 (SFPLOADMACRO unfused CE, 427 lines, opt-out define built in). Everything else measured on the branch (replay default-on, macro-trick rebuild, unpack-floor) is already content-identical on main.
- What canonical must absorb for the fork to die (the concrete #1971 §2B promotion list): (1) add_lsb_indices row_major template param; (2) topk_xl_reinit_mop_after_copy + topk_xl_reinit_unfused_rebuild_after_copy API+LLK; (3) _topk_xl_local_sort_generic_ early_exit_K64/int32_mode variants. All three are load-bearing in blaze/ops/indexer_local_topk/kernels/op.hpp and/or their sparse-K reader. Their bit-exact-vs-torch test oracle (test_indexer_local_topk.py, test_indexer_sdpa_local_topk.py check) is the acceptance gate any transplant must pass — tie-break order carries core_id/sub-bank bits, so byte-identical index layout is contractual.
- Also correct in our docs: the blaze cell's semantics (valid=8192/device, 16 topk cores, capacity 32768) should be recorded next to the 24.4µs ledger cell; and G8 ("no committed breakdown") is now closed by the comp4 profile_log_device.csv decomposition above.

## Evidence

- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/kernels/op.hpp:4-26 (tree topology, fused->unfused at bank boundary)
- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/kernels/op.hpp:205-240,342-404,496-533,549-644 (streaming DM1, phase-0 sort, DM0 transport, unrolled stages)
- /home/nachiket/tt-blaze/blaze/ops/indexer_local_topk/op.py:18-27,104-159 (K=2048, split bank stamp, unfuse_stage)
- /home/nachiket/tt-blaze/blaze/ops/distributed_indexer/config.py:17-34,98-118 (index bit layout, topk-below-SDPA placement, STREAM_CHUNK=2048)
- /home/nachiket/tt-blaze/blaze/kernels/kernel_includes/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h:356-368 (fork HAS INT32 denormal guard, cites commit 814f1b46), :1194 (local_sort_generic early_exit_K64/int32_mode), :299 (reinit_unfused_rebuild_after_copy)
- /home/nachiket/tt-metal/tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_topk_xl.h:112-339 (SFPLOADMACRO section, branch-only), :2861-2992 (row_major family, on main)
- diff counts (this session): fork-vs-main 1072 lines, main-vs-branch 427 lines (= commit 725748766a6 only), fork-vs-branch 1512 lines
- /home/nachiket/tt-blaze/tests/blaze/micro_ops/dsa/test_indexer_sdpa_local_topk.py:89-104 (CP math: 64k global -> local_valid 8192, ppd 32768), :417-468 (bench = whole-program DEVICE KERNEL DURATION, no per-phase markers), :520-543 (param ids incl [64k])
- /home/nachiket/tt-metal/tests/ttnn/unit_tests/operations/reduction/_canonical_topk_sweep.py:300-318 (blaze cell = [64k] test, Tracy median-of-9 datum)
- /home/nachiket/tt-metal/generated/canonical_sweep/comp4/results/comp_blaze_k2048_w65536.blaze.t0.json (ns_median 24432, 129 cores)
- /home/nachiket/tt-metal/generated/canonical_sweep/comp4/work/comp_blaze_k2048_w65536.blaze.t0.log:345+ (CT args: chunks_per_core=1, num_stages=4, max_fused_stage=1, unfuse_stage=3, scores_positions_per_device=32768, cp 8x256, min_active_pos=2048)
- /home/nachiket/tt-metal/generated/profiler/reports/2026_08_16_19_26_20/profile_log_device.csv (per-core decomposition: NCRISC stream done 9.4-10.6us; stage enders 18.1-18.8 x4, 20.5-20.6 x2, 23.3, root 24.4; median passenger span 0.18us)
- /home/nachiket/tt-metal/generated/canonical_sweep/comp4/competition_table.csv rows 2048,{2048:15.74us,8192:22.88us@4c,32768:30.23us@16c,65536:34.32us@32c}
- /home/nachiket/tt-metal/SORTING.md:77 (merge 2.844->1.438 cyc/vec, SFPLOADMACRO), :549 (B4 map-at-1.000 rule)
- /home/nachiket/tt-metal/tt_metal/hw/inc/api/compute/experimental/topk_xl.h:214 (canonical add_lsb lacks row_major param); API-name diff fork-vs-ours (fork-only: local_sort_generic, reinit_mop_after_copy, reinit_unfused_rebuild_after_copy; ours-only: separate_indices_row_major family)
- git log (tt-llk): 725748766a6 branch-only on the header; main has df31fd4a847 (#51777 tests) + 5a64046dbf7

## Risks

- Phase attribution uses KERNEL-zone start/end times (which include cb/sem waits), not explicit per-phase markers — the 10/8.4/6 us SDPA/leaf/tree split is inferred from end-time populations; +/-1-2us uncertainty, though the 4/2/1/root ender pattern matching the binary tree exactly gives high confidence.
- Transplant gain estimates (24.4 -> ~21-22us, ~18-19 with a fused-path macro) extrapolate SFPLOADMACRO cyc/vec ratios measured in our harness to their fp32-DEST fused-program context (DST_ACCUM juggling, SyncHalf flips at op.hpp:330-336) — untested there; the fused-path macro is unbuilt new work, not a measured artifact.
- Their correctness oracle is bit-exact winner sets with core_id/sub-bank bits riding in tie-break positions — any LLK transplant that perturbs compare-exchange tie order breaks their tests even when top-K sets are value-correct.
- Single Tracy capture (one process, 9 runs, 2026-08-16 build) underlies the decomposition; no cross-build replication.
- The 1072-line fork-vs-main figure was measured against the local tt-blaze checkout (main @ d39ab22, files dated Aug 16) — blaze HEAD may have moved.
- Per-unit cost calibration for our op (~3.5us/window + ~9-10us envelope) is derived arithmetic from comp4 cells, not a stage-marked profile.
