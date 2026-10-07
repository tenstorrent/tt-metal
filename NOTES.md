# t235 — DIFFVAE_NA_KEY_PHASE default-on (5-seed check)

Code: branch ttp/t235-land @ ca360c17ccb (on t48 1a4d14830c9). Key phase is on by default, =0 turns it off.
A fit check (key_phase_applies) keeps it off where (2,4,4) does not tile every HxW shard alike (e.g. 720p).
An explicit DIFFVAE_NA_BRICK other than (2,4,4) now also turns key phase off.
Host pytest (blx01 ov tree): 15 passed; new tests fail on old code.

PCC/PSNR (from #232 kp1 host-noise decodes, vs ref in blx01 /var/tmp/fasth3/diffvae/ref):
seed0 0.999957/55.60  seed1 0.999958/55.16  seed2 0.999956/55.08  seed3 0.999957/55.77  seed4 0.999954/55.28 dB
Timing: kp1 4.494 s vs kp0 4.931 s (#232). Default env (job 868): 4.487 / 4.484 s, key_phase_applies=True.

Result (2026-10-07):
- Default-env decode (blx01 job 868, rc 0): seeds 0,1 byte-identical (md5) to #232 kp1.
- VBench (5 seeds, x264 crf12 mp4, vs unoptimized ref): subject 0.8959/0.8959, background 0.9223/0.9222,
  imaging 0.5489/0.5489, aesthetic 0.6206/0.6202, motion 0.98594/0.98593. mp4-level PSNR 45.98 dB mean.
- Visuals: eval/seed*_cmp_f{000,072,144}.png, difference panels near black, no seams.
  Evidence: tt-project/data/g15/t235/{kp1,ref,eval}, eval.log BATCH OK line.
- Decision: key phase on by default. Landed on ttp/t48-ltx25-integrated via ttp push.
