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

## Next step
Both identical and hoist faster -> merge t82 into local ttp/t48-ltx25-integrated, push it (no PR).
Otherwise report the diff and stop. Then clean up ~/fasth3/t82rt, ~/fasth3/t48/tmp/t82, /var/tmp/fasth3/t82/jit_*.
