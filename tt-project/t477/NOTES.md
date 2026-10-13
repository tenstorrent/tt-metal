# t477 notes (run 1796, split at the context line)

Branch ttp/t477-turbo-per-op-profile-of-one-10-s-block-t, base ae9534f5fb2 (t473 LoFi ring SDPA default).

## Done
- Code commit 7da5bf942d4: opt-in pack-thread SwiGLU in all_gather_minimal_matmul_async (port of upstream
  #59325 / d5461453b44 part 1). Env TT_AGMM_SWIGLU_PACK=1 adds compute define SWIGLU_PACK_THREAD when
  fuse_swiglu && no bias && subblock_w even. Kernel: matmul_blocks_swiglu in compute.cpp, op-local
  kernels/swiglu_sfpu.hpp (fp32 + bf16 sigmoid paths only). Not in the program hash: switch it only between
  processes or after mesh_device.clear_program_cache().
- Part (2) (block-float sigmoid) does not apply: MiniMax-H3 ff1 runs fp32_dest_acc_en=True
  (transformer_block_minimax_h3.py mm_compute_kernel_config), part (2) needs bfp8/bfp4 out with fp32 dest off.
- Exactness expectation (unmeasured): the last K block adds partials via the FPU (srcA/srcB, tf32 inputs on
  WH) instead of the packer's fp32 L1 accumulation, so new vs old is probably NOT bit-identical with
  fp32 dest. If the A/B confirms that, it stays opt-in (spec: bit-identical for (1)); report PCC and timing.
- Test: models/tt_dit/tests/models/minimax_h3/test_ff1_swiglu_pack_minimax_h3.py::test_ff1_swiglu_pack_block_10s
  One random-weight block at 10 s fl2va (sp=8 axis1, tp=4 axis0), arms T477_ARMS=old,new,old (default),
  T477_ROUNDS=3; logs "T477 arm ...", bit-identical/equal-frac/max|diff| for old-vs-old and new-vs-old
  (ff1 output and block output, device 0 shard), and "T477 summary: ff1 new/old, block new/old".
  Not compiled or run yet. Use -k '4x8sp1tp0nl2_ring_is_fsdp0 and not WH' as in t466 (check ids with
  --collect-only on blx01 first).

## Not done (next run)
1. Build on blx01: tarball of HEAD + submodules (tracy, tt-cluster-descriptors, umd at pinned SHAs from
   /home/smarton/fasth3/tt-metal/tt_metal/third_party/*) to /var/tmp/fasth3/t477/b; build like
   tt-project/t466/build466.sh (ionice -c3 nice -n 19 ./build_metal.sh --build-type Release, CCACHE_DISABLE=1,
   HOME/TMPDIR under /var/tmp/fasth3), detached with `ttp detach --remote g15blx01`.
   First compile errors are likely in the kernel (JIT, device side) — only the device job will show them.
2. Step 1 profile job: python -m tracy -r -p on test_transformer_minimax_h3.py::test_minimax_h3_transformer_block_perf
   [10s_768p, test_prompt_text_tokens, sp_sim1, 4x8 ring] (signposts start/stop), tt-perf-report; per-op table
   (ff1 AG+MM SwiGLU, ff2, qkv/out, norms/AdaLN, rope, K/V all-gather, ring SDPA) into this file.
3. A/B job (run script modelled on tt-project/t466/run466mb.sh, spec like spec_mb.txt, TIMEOUT 600 unmeasured,
   submitted via tt-project/t288/blx01-submit.sh after device-lint). After the run, grep the JIT cache for
   SWIGLU_PACK_THREAD to prove the new arm built the new kernel.
4. Clean /var/tmp/fasth3/t477 on blx01 when done.

## Drops seen
- 2026-10-13 18:28 UTC, blx01, broker job 183 (smarton ltx-rt live service, not this task): broker-kill
  "chips left PCIe"; chips 16-23 bridge reset failed, power-cycled; startup + fabric check OK 18:38 UTC.

## t480 (run 1801, 2026-10-13)
- Build: /var/tmp/fasth3/t477/b on blx01 from git-archive tarballs of 63329812d9f + tracy/tt-cluster-descriptors/umd
  at the pinned SHAs (t422/b and t466/b no longer exist). Driver build477.sh, started 18:52 UTC with
  `ttp detach --remote g15blx01 --dir /var/tmp/fasth3/t477/detach build477`; exit code in /var/tmp/fasth3/t477/build.rc.
- Job script run477.sh (MODE prof | ab), env477.yaml, specs spec_prof.txt / spec_ab.txt (TIMEOUT 600, unmeasured).
  Both tests use random weights: no MiniMax-H3 weights or h3hf cache are read.
- chain477.sh (g15blx02, ttp detach): waits for build.rc, collect-only check (2 ids), profile job (<= 3 tries),
  A/B job (<= 3 tries), each through t288/blx01-submit.sh (health + lint + no other smarton job on blx01).
  Marker chain477.done in the run dir (CHAIN_OUT), log chain477.log.
- Next: read out_prof/run.log + ops_perf_results CSV (tt-perf-report), out_ab/run.log (T477 lines + JIT grep),
  fill the tables below, rm -rf /var/tmp/fasth3/t477 on blx01, ttp push --own --detach.
- 18:55 UTC: build done (build.rc 0, ~3 min). First chain run stopped at collect (diffusers fork missing):
  job PYTHONPATH now adds /var/tmp/fasth3/t409/pydeps (MiniMax-H3 diffusers); collect then found 2/52.
- Chain restarted as chain477b (run 1801 dir: /home/smarton/fasth3/tt-metal/tt-project/state/runs/1801/chain477.log,
  marker chain477.done, rc chain477b.rc). Profile job = blx01 broker job 195 (submitted 18:57 UTC).
- 19:00 UTC job 195 (prof): exit 0 in 23 s but ran no test: tracy re-split the quoted -k ("file or directory
  not found: and"). Fixed: -k via PYTEST_ADDOPTS; run477.sh now sets rc 6 when there is no "1 passed" line.
- 19:01 UTC job 196 (A/B): failed after 75 s (cold JIT ok, 0 SWIGLU_PACK_THREAD files: the new arm never ran).
  First error: TT_FATAL device_operation.hpp:518 input_tensor.is_allocated() in ff2 forward_fused_addcmul
  (linear.py:748), from the timed round's run_block. Cause: the test deallocated the block output, which is a
  persistent CCL buffer reused by the next call. Fixed in code commit (no deallocs). Job 198 = chain's retry of the
  old test (expected to fail the same way); chain477b stopped by me after that.
- Fixed run477.sh + test file copied to blx01 (/var/tmp/fasth3/t477, b/models/...); chain restarted as chain477c.
- 19:03-19:07 UTC (chain477c, rc 0): profile = blx01 job 200 (exit 0, 1 passed), A/B = blx01 job 201 (exit 0, 1 passed).
  No drops. AICLK clamped at 900 MHz on all chips (UMD warnings): all numbers are relative only.

## Step 1 result: per-op profile, one 10 s block (job 200)
test_minimax_h3_transformer_block_perf [10s_768p, test_prompt_text_tokens, sp_sim1, 4x8 ring], random weights,
tracy signposts start..stop, tt-perf-report 49 ops, 32 devices merged. Device time 51.03 ms + op-to-op gaps 3.64 ms.
Labels for the 9184x32x1344 matmuls and the pre-block ops are inferred from shapes.

| op group | ops | device ms | share |
|---|---|---|---|
| ring SDPA (RingJointSDPA, K/V all-gather fused in it) | 1 | 31.01 | 60.8% |
| ff1 AG+MM SwiGLU (9184x1344x7168, HiFi2) | 1 | 6.23 | 12.2% |
| qkv AG+MM (9184x1344x5376, HiFi2) | 1 | 4.39 | 8.6% |
| out-proj AG+MM (9184x1792x1344, HiFi2) | 1 | 2.79 | 5.5% |
| ff2 MM+RS (9184x3584x5376, **HiFi4**, FLOP-bound) | 1 | 2.52 | 4.9% |
| norms (DitFusedDistributedRmsnorm) | 4 | 1.37 | 2.7% |
| 9184x32x1344 matmuls (AdaLN modulate/gate, inferred) | 6 | 1.14 | 2.2% |
| AdaLN projection (32x2688x24192) | 1 | 0.85 | 1.7% |
| create/concat heads | 2 | 0.42 | 0.8% |
| rope (embedding lookup + tilize) | 4 | 0.17 | 0.3% |
| misc pre-block tilize/untilize/slice/binary | 27 | 0.13 | 0.3% |

There is no separate K/V all-gather op: it runs inside ring SDPA. Next levers by size: ring SDPA (61%), then the
matmuls (ff1+qkv+out+ff2 = 31%). ff2 is the only HiFi4 matmul; HiFi2 there is a candidate (quality check needed).

## Step 3 result: pack-thread SwiGLU A/B (job 201, one job, arms old,new,old, 3 rounds each)
| arm | ff1 best / mean ms | block best / mean ms |
|---|---|---|
| old | 7.03 / 7.10 | 54.61 / 55.00 |
| new (TT_AGMM_SWIGLU_PACK=1) | 6.75 / 6.78 | 52.97 / 53.96 |
| old | 6.82 / 6.88 | 52.98 / 53.39 |
- Against the second (warm) old arm: ff1 new/old 0.990 (best), 0.985 (mean); block 1.000 (best), 1.011 (mean).
  Against the mean of both old arms: ff1 0.975 (best). The gain is about 1-2.5% of ff1 (0.07-0.17 ms per block),
  below block-level noise.
- Not bit-identical: new-vs-old ff1 equal-frac 0.757, max|diff| 0.0312; block equal-frac 0.945, max|diff| 0.0312.
  old-vs-old is bit-identical (ff1 and block), so the difference comes from the new kernel.
- JIT cache: 1 compute defines_generated.h names SWIGLU_PACK_THREAD, so the new arm built the new kernel.
  No kernel JIT compile errors.
- Verdict: keep TT_AGMM_SWIGLU_PACK opt-in (off by default). Not worth turning on: ~1% of ff1, not bit-identical.
- /var/tmp/fasth3/t477 on blx01 deleted after the runs (7.8 GB; / now 417 GB free).
