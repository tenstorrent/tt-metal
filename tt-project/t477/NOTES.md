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
