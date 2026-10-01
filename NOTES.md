# t82: BH SFPU constant hoists (PR 58179 @ba45c87952e, PR 58217 @49a0b6a0047) on LTX

Branch ttp/t82-cherry-pick-bh-sfpu-constant-hoists-5817, on t48 @a613d669eef. 4 cherry-picked commits,
kernel headers only (15 files under tt_metal/hw/ckernels/blackhole + tt-llk expm1_cw). No host .cpp change,
so no rebuild: blx03 runs both arms on the t48 build; the hoist arm points TT_METAL_RUNTIME_ROOT at
~/fasth3/t82rt (symlink farm of t48, the 15 headers real; `diff -rq` = exactly those 15). Separate JIT caches
/var/tmp/fasth3/t82/jit_{base,hoist}.

## Device run (blx03, launched 2026-10-01)
Driver: ~/fasth3/t48/tmp/t82/drive82.sh (detached), log /var/tmp/fasth3/t82/drive82.log, ends with
`DRIVE82_DONE <ok|fail_...>`. Job 1 = run82.sh vae (conv VAE 544x960/145f, base then hoist), job 2 = run82.sh
block (traced AV block 2x4 Linear 10x34x60), only if job 1 exits 0. Logs: /var/tmp/fasth3/t82/run82_{vae,block}.log.
Queued behind t81's job 040 at launch.

## Reading results
- `T82_CMP ... identical=True` per output and `T82_IDENTICAL=True` = bit-identical.
- VAE time: `AB arm=t1w1 ... min=` lines, first = base, second = hoist. Block: `AB_BLOCK arm=base|hoist ms=`.
- Check the hoist arm actually compiled the new headers: grep t82rt in /var/tmp/fasth3/t82/jit_hoist/**/*.d.

## Result (2026-10-01, blx03 jobs 044 vae + 045 block, full mesh then create_submesh(2,4), no drop)
- Bit-identical: VAE YUV out identical (max_abs_diff 0), block video+audio outputs identical.
- VAE conv decode 544x960/145f min: base 2.0472 s, hoist 2.0558 s (+8.6 ms, +0.4%; laps do not overlap,
  but hoist always ran second, so order may account for it).
- Traced AV block 2x4: base 29.931 ms, hoist 29.823 ms (-0.11 ms, -0.4%).
- Compiled code: of 411 compute ELFs, only 5 differ by instructions. ring_joint_sdpa and sdpa .text are
  byte-identical (the compiler already hoisted the exp constants there). The 4 fused RMSNORM+silu
  layernorm kernels shrink by 54 instructions; one sigmoid eltwise_sfpu kernel grows by 33.
- Decision: VAE is not faster, so per spec no merge into t48. Expected LTX gain is ~0 because the SDPA
  kernel code does not change. Dropped.
- Cleaned up: ~/fasth3/t82rt, ~/fasth3/t48/tmp/t82, /var/tmp/fasth3/t82/{jit_*,vae,block}. Logs and
  scripts kept in /var/tmp/fasth3/t82 (896K).
