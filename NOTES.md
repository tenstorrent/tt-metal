# t235 — DIFFVAE_NA_KEY_PHASE default-on (5-seed check)

Code: branch ttp/t235-land @ ca360c17ccb (on t48 1a4d14830c9). Key phase is on by default, =0 turns it off.
A fit check (key_phase_applies) keeps it off where (2,4,4) does not tile every HxW shard alike (e.g. 720p).
An explicit DIFFVAE_NA_BRICK other than (2,4,4) now also turns key phase off.
Host pytest (blx01 ov tree): 15 passed; new tests fail on old code.

PCC/PSNR (from #232 kp1 host-noise decodes, vs ref in blx01 /var/tmp/fasth3/diffvae/ref):
seed0 0.999957/55.60  seed1 0.999958/55.16  seed2 0.999956/55.08  seed3 0.999957/55.77  seed4 0.999954/55.28 dB
Timing: kp1 4.494 s vs kp0 4.931 s (#232). Default env (job 868): 4.487 / 4.484 s, key_phase_applies=True.

In flight:
- blx01 driver /var/tmp/fasth3/t235/drv (job 868 done rc 0); marker drv/driver.marker (md5 def vs t232 kp1 seeds 0,1).
- g15 VBench: tmp/t235/local.sh detached as t235local (run dir 948); marker data/g15/t235/LOCAL.done, eval in data/g15/t235/eval.

Next: read the marker and the VBench BATCH line (vbench vs vbench_ref), look at eval/seed*_cmp_f*.png;
if it matches, `ttp push --detach` from ttp/t235-land; then remove blx01 /var/tmp/fasth3/t232 and t235 (ov, ov0, yuv, mp4).
