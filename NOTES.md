# t503 notes (continue t498: LTX-2.5 Aug-17 Gemma-4 encoder A/B)

## State 2026-10-14 12:35 PDT (run 1897)
- Code: t498's 94825eb729e/df0b2742038 cherry-picked on origin/ltx-rt f6547442b30 (this branch). CPU tests 3 pass.
- Boxes 12:10 PDT: blx01 HELD degraded (until ~10-17), g15blx02 skipped (2 drops, jobs 773/775), blx03 healthy
  (back from a reboot 12:10 PDT after chips 8-15 left PCIe at 12:02 PDT, live LTX job 243 killed; not ours).
- blx03 had no LTX-2.5 weights. stage503 (run 1897, detached) copies MLPerf 28dac7ac DiT/upsampler/audio VAE/old encoder
  to blx03 /var/tmp/fasth3/t503/ltx25 (+2.5 conv VAE symlink to /var/tmp/fasth3/ltx25_vae), new encoder to
  /var/tmp/fasth3/t503/te0817, code+build (t313 build, same C++ as f6547442b30, overlaid with this branch) to
  blx03 ~/fasth3/t503/tt-metal. ~96 GB on blx03 /var/tmp: delete /var/tmp/fasth3/t503 when the task ends.
- drv503 (run 1897, detached) waits for staging, then queues runner specs t503-ab-r1 and t503-ab-r2 (same script,
  resumable: an arm with ARM_OK is skipped, weight caches persist in /var/tmp/fasth3/t503/cache; r2 exits 9 at once
  if r1 failed fast).
- tdiff503 (run 1897 tdiff503.log): per-tensor diff of old vs new file.

## Next on wake
1. ssh g14blx03 cat /var/tmp/fasth3/runner/done/t503-ab-r{1,2}.done; logs in /var/tmp/fasth3/t503/out_{old,new}/run.log.
2. Copy out_* (emb_*.pt, *_t3s.png, run.log) back, run cmp498.py-style PCC (tt-project/t498/cmp498.py <dir>).
3. Judge stills; if neutral or better: land on ltx-rt (standard e2e first), note sulphur.
