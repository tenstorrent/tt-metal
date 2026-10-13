# t478 notes (bit-identical upstream perf bundle for FastH3 Turbo)

Branch ttp/t478-turbo-bit-identical-upstream-perf-bundle, base ae9534f5fb2 (t473 head).
Code commits (base..56df249a1f6):
- bdfa96d9ef4 #58343 (clean), ac01664c9da #56023 (1 conflict: our noc_init still has the inline loop body;
  added the two MID-clear lines upstream put in noc_init_one), 2b265591829 #58517 (clean),
  e3a7314f3f5 #55792 (context-only conflict in get_optimal_dram_bank..., kept ours; the pick does not touch it),
  81e722fc598 #59214 (1 conflict: kept our first_tile addressing, which lacks upstream's block-cyclic origin
  pages, and routed it through the new concrete accessor), 5b1b5ac1cdb #59905 (clean),
  8c7ec888a48 #59325 part (3) only (writer deferral; metal2 dm kernels + descriptor hunks 3-4, applied clean).
- SKIPPED #59195 765cca59a0a: 8 conflicts in 4 ring-SDPA files (compute_streaming, ring_joint_sdpa.cpp,
  sliding_window_work_plan, program factory); needs the unported #58032/#58223 work-plan changes. Large rework,
  and t413 rates it small for H3 (~4768 rows/device is not a short slab).
- 56df249a1f6: test_sdpa_fidelity_timing_minimax_h3.py gains test_attention_layer_dump_10s (T478_DUMP=<file>,
  default fidelity, sha256 per device shard, run-to-run determinism check, timing) and
  test_attention_layer_dumps_bit_identical (host only; T478_BASE_DUMP / T478_BUNDLE_DUMP; logs
  "T478 layer/ring_joint_sdpa: bundle/base best" and "T478 bit-identical shards: N/32").

## blx01 state (2026-10-13 18:50 UTC)
- Nothing of t478 on blx01 yet. t466/b and t422/b are gone; t460/src is not a full tree.
- Submodule commits (tracy 9c3ca7e, tt-cluster-descriptors 9bcf698, umd 5380941) exist in the main checkout's
  submodule repos on g15blx02 (`git -C tt_metal/third_party/<n> archive <sha>` works).
- MiniMax-H3 SLA weights seen at /var/tmp/fasth3/models/lightx2v-h3-turbo-sla (gate 2 only).
- t466 build took ~3.5 min (build_metal.sh --build-type Release, ionice -c3 nice 19). Microbench process wall 60 s.

## Next step
1. On g15blx02: tarball = `git archive ae9534f5fb2` + test file from 56df249a1f6 + the three submodule archives
   (prefix tt_metal/third_party/<n>/); second tarball = `git archive 56df249a1f6` + same submodules.
   Stream each to blx01 /var/tmp/fasth3/t478/{base,bundle}; build each with build466.sh's recipe
   (ionice -c3 nice -n 19, HOME/TMPDIR/CPM under /var/tmp/fasth3, CCACHE_DISABLE=1), one after the other, detached.
2. Job script (pattern: /var/tmp/fasth3/t466/run466mb.sh, env466.yaml, python env t460/python_env): one broker job,
   arm base (TT_METAL_HOME=base, own TT_METAL_CACHE) then arm bundle (own cache), each
   `pytest ... -k 'test_attention_layer_dump_10s and 4x8sp1tp0nl2_ring_is_fsdp0 and not WH'` with T466_ROUNDS=10,
   then the host compare test. -t 600 (unmeasured; bundle kernels all recompile: #56023 changes NoC headers).
   device-lint.sh + health check first; submit via tt-project/t288/blx01-submit.sh.
3. Gate 2 only if >=1% faster and 32/32 shards identical.

## t481 progress (2026-10-13 12:05 PDT)
- Own-branch push confirmed: 1a9d5bda3e6 on origin/ttp/t478-turbo-bit-identical-upstream-perf-bundle.
- Tarballs built on g15blx02 (worktree tmp/t478tar, git-ignored): base = archive ae9534f5fb2 + test file from
  56df249a1f6; bundle = archive 56df249a1f6; both + tracy/tt-cluster-descriptors/umd archives. rsync'd
  (--bwlimit=100M) to blx01 /var/tmp/fasth3/t478; build478.sh (drv/) extracts and builds base then bundle,
  detached: log /var/tmp/fasth3/t478/detach/build478.log, rc /var/tmp/fasth3/t478/build.rc.
- drv/chain478.sh runs detached on g15blx02 (ttp detach, run 1803): waits for the build, submits spec478mb via
  t288/blx01-submit.sh (retries every 5 min while blx01 is busy/unhealthy, up to 4 h), waits for the job end.
  State: worktrees/t481/tmp/chain478/{job,marker}; log state/runs/1803/chain478.log.
- Job: run478mb.sh (on blx01 /var/tmp/fasth3/t478), -t 600 (unmeasured), arms 260 s each, compare 60 s.
  Output /var/tmp/fasth3/t478/out_mb/run.log; dumps base.pt/bundle.pt.

## Next step
Read out_mb/run.log: lines "T478 arm: ...", "T478 layer/ring_joint_sdpa: bundle/base best", "T478 bit-identical
shards: N/32". Gate 2 (10 s seed-0 e2e A/B, separate job) only if >=1% faster and 32/32 identical and
MiniMax-H3 weights are on blx01 (/var/tmp/fasth3/models/lightx2v-h3-turbo-sla). Then clean /var/tmp/fasth3/t478.

## t481 run 1805 (2026-10-13 12:05 PDT)
- First blx01 build failed (rc 1) at link step 1075/1445 (base tree): ld.lld "pthread_create has failed: Resource
  temporarily unavailable" linking tracy-capture-daemon. blx01 had rebooted ~18:37 UTC (19 min before the build);
  load ~21. ulimit/cgroup limits are high, so treated as transient. Old logs: /var/tmp/fasth3/t478/detach/old1/.
- build478.sh now skips extraction of an already-extracted tree and sets CMAKE_BUILD_PARALLEL_LEVEL=24.
  Restarted 19:01 UTC (ninja resumed at ~130/310). chain478 restarted (run 1805; log state/runs/1805/chain478.log,
  rc state/runs/1805/chain478.rc, state worktrees/t481/tmp/chain478/{job,marker}).
