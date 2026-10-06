# t152: conv3d vol2col_rm CB sizing fix (off-device)

Branch ttp/t152-fix-conv3d-vol2col-rm-cb-sizing-for-unal, based on ttp/t149 @aa4ade43a28.

## Change
- `conv3d_program_factory.cpp`: vol2col_rm = min(n, 32) pages when n = T*H*W is a multiple of 32
  (unchanged), else exactly n pages (was min(n, 64)). Every block then ends at fifo_limit, so no
  32-page chunk straddles the CB end on push (reader) or pop (compute tilize, pops min(32, left)).
- For every n the old guard accepted (n <= 64 or n % 32 == 0) the size is the same as before:
  no L1, perf or numerics change for any existing blocking.
- Guard kept: the TT_FATAL still rejects unaligned n > 64 unless TT_CONV3D_ALLOW_UNALIGNED_VOL2COL=1.
  The sweep filter (vol2col_chunks_fit in run_sweep) is skipped under the same env var.
- Cost when lifted: (n - 64) extra pages. (64,128,5,4,4) at C_in 512, k=3: 64 -> 80 pages of 3456 B
  (+55 KB); prefetch_shard_fits() still True for all four hung blockings (CPU mirror).
- Tail tilize reads 32 rows from the tail start, i.e. (32 - tail) rows past the CB end. Read only;
  same as the existing n <= 64 unaligned case.

## Evidence (CPU only, nothing compiled)
- `models/tt_dit/tests/models/ltx/test_conv3d_sweep_halo_cpu.py`: 46 passed. New tests: old sizing
  straddles for exactly the 4 hung blockings of jobs 273-285 and the new sizing for none; n = 1..2048
  never straddle; sizes unchanged for every n the old guard accepted. With the old formula put back,
  5 of the new tests fail.
- `python tt-project/t149/sim_vol2col_cb.py`: old sizing straddles for 1922 of n = 1..2048 (all
  unaligned n > 64), new sizing for 0.
- The C++ is not compiled here (no ttnn build on g15blx02, no new build dirs allowed). clang-format passes.

## Device check (one broker job on blx03, not run)
- Build: the t152 C++ (be371d08a9f on top of aa4ade43a28) in an existing blx03 build tree, incremental
  (one .cpp). No new build dir.
- Job: full 4x8 mesh, then create_submesh(2,4) (the test does that):
  `TT_CONV3D_ALLOW_UNALIGNED_VOL2COL=1 SWEEP_ONLY_BLOCKINGS="64,128,5,4,4" SWEEP_MAX_SECONDS=300
   pytest models/tt_dit/tests/models/ltx/bruteforce_conv3d_sweep_ltx.py::test_bruteforce_sweep_ltx25_544p_145f_halo -k exact_s2_res`
  Broker timeout 15 min. Only after the health check passes and with 30 min with no tray-2 incident.
- Pass: "launching (64,128,5,4,4)" then "1 ok, 0 failed"; "Output check best vs table" PCC >= 0.9999
  (same C_in_block 64, so md5 identical or max_abs_diff ~1 bf16 ulp expected); post-job health OK,
  no eth-heartbeat freeze.
- Fail: no output for 300 s (old hang signature) or a FAIL line. On a hang, the broker recovers; do not
  rerun the same config more than twice.
- Second job after a pass: same with "64,128,7,4,4" (n = 112, 4 chunks per block).

## Removal plan (after both device checks pass)
1. Factory: delete the TT_FATAL, the allow_unaligned_vol2col lambda and the <cstdlib>/<string_view> includes.
2. Sweep: delete vol2col_chunks_fit, the straddle filter in run_sweep and the env check.
3. Tests: delete test_vol2col_chunks_fit_splits_job_484_bisect, test_table_blockings_fit_vol2col_chunks
   and test_vol2col_rm_pages_unchanged_for_guarded_blockings; keep the straddle and never-straddle tests.
4. Optional: re-sweep LTX blockings with unaligned n > 64 now open (e.g. exact_s2_res T=5|7).
