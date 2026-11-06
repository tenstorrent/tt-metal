Byte-identity gates for the mainline-cleanup commits (2026-11-05; tools in fabric_exp/port_tune/cov: cov.py, gate_wh.py,
covcmp_multiset.py, covcmp_cache.py, covcmp_rebuild_subst.py). 52 mock cases (BH P150, Galaxy, multi-mesh, WH T3K).
COV dirs: ~/.tt-metal-gate-{3efd,tip,80118}-cov. Old hashes (pre-rebase chain, branch ...-prerebase).

G3 b88a5c069f3 vs 3efd636e8df (own host builds + mocks): PASS. Router image multisets per case identical (g3.txt: 47/50
   via logged-command rebuilds; bh_trim_gtest re-mocked (first 3efd mock stopped at 11/66 gtests) 436/436, bh_vc1_vc2 96/96,
   glx2_mesh 232/232, wh_t3k_sanity_* 28/28 through the JIT's own ELFs (multi-rank logs drop commands)).
G4 2922cd8eb61 vs b88a5c069f3: PASS, 5148/5148 (4946 router + 202 other) identical.
G5 cc50e4d74d4 vs 2922cd8eb61: PASS, 5148/5148 identical, every kernel. No Quasar target here.
G6 80118e29656 vs cc50e4d74d4 (host-only; own mocks, JIT ELFs vs the tip run, every kernel): PASS (g6_cmp.txt).
   46 cases identical outright; run-to-run artifacts settled by rebuild: bh_2d_streaming / bh_2d_1erisc_streaming
   (streaming-profiler TT_PROFILER_TU_ID handed out first-come, jit_build/build.cpp:76; remapped: identical),
   bh_2d_udm_gtest (time seed CT arg, test_basic_1d_fabric.cpp:246; substituted: identical), glxz_mesh / glx3_passthrough
   (one sender the tip mock never linked = the tip's rebuild of that command), glx3_torus_yx (tip re-mocked: 211/211).
G7 539cb2a8434 vs 80118e29656 (router rebuild on the 80118 captures): PASS, 4942/4942 identical (g7.txt).
G8 de8a2ac17a8 vs 539cb2a8434 (full rebuild on the 80118 captures): PASS as expected (g8_cmp.txt, g8_classify.txt):
   BH two-ERISC routers 4122/4122 identical, WH routers 120/120 identical, non-router 202/202 identical,
   BH one-ERISC routers 700/700 changed. One-ERISC code shrinks 36-580 B (largest bh_vc1_1erisc 17504 -> 16924 of 26624),
   stack +0..+32 B, least free stack 3712 B.
Flush validation of 3efd636e8df: FAIL to run. The unpatched arm (PR base) hung on the first data-checked test on LB
   (lb2 180530, FlushA2ARing 1024 atomic_inc unicast; fabric_exp/results/fdaa_unp_val_ring_hangcap) and Galaxy (180570,
   same test at 4096; fabric_exp/results/glx180570). Patched arm never ran. Cause open (hypotheses: test-infra limit of
   flow-controlled all-to-all on Ring; base-router ring/atomic bug); isolating A/B = the same yaml on main's router.

G9 6a4a961aeab (receiver floor, host-only) vs de8a2ac17a8: PASS. Router named-arg maps per case from the JIT caches
   (g9_namedargs.txt): 51/52 identical sets, glx3_mesh's 6a4a set a strict subset (mock reached fewer configs), no map only
   on 6a4a. Images (g9_cmp.txt, g9_global.txt): every one-sided image is a run artifact (streaming TU ids, udm time seed)
   or multi-rank coverage.
G10 b923cac1c52 (trid read-back, device-only) vs 6a4a961aeab, routers rebuilt on the 6a4a captures (g10_cmp.txt):
   BH/Galaxy 4718/4718 identical (no captured BH config takes a changed per-ID flush path); WH 120/120 changed, largest
   text per case: line 8316 -> 8604, mesh 9980 -> 10260, ring 8760 -> 9072, ubench_2x2 7596 -> 7744 B (+148..312 B).
Rebase onto main b078540a262, c158abca268 (flush tip), own worktree: host+test build PASS (umd 5da9fe5, new sfpi);
   allocator gtests CPU-only 12/12 (c158_alloc_gtests.log); 52-case mocks with no JIT build error, router rebuild
   4935/4939 (4 garbled multi-rank log lines in glxz_mesh_novc; the JIT built all 108 there); a09u32 broker 882/883:
   val_ring_p150 19/19, AllToAll2DTorus_short 16/16, test_fabric_mesh_sanity_common 96/96 (results/flushval_c158_*.log).
Device flush validation on the tip b923cac1c52 (a09u32 broker): val_ring_p150 19/19 (861), torus stability 16/16 (875),
   p150_it_ring benchmark no-hang 16/16 (876). Test-pattern failures (main fails identically; TODO.md test infra):
   FlushRing1HopEW 4096 fused no-flush (851 base, 852 main), Ring1HopEW 3072 write data-checked (861 tip, 874 main),
   FlushA2ARing atomic_inc flow-controlled all-to-all hang on Galaxy (180570 PR base, 180726 main).
Gate trees and COV dirs removed 2026-11-06 (coordinator): gate-3efd, gate-bisect, gate-6a4a, gate-80118 (+ covs) and
   the trees that linked their builds (gate-de8a, gate-de8a-base, gate-b923). gate-c158 + its cov stay until the trid
   read-back lands.
