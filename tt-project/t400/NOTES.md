# t400 results notes (run 1546, 2026-10-11 18:30 PDT)

Drivers and specs live in the project root (fasth3-opt) under tt-project/t400; the rerun copies and all
logs are untracked under /home/smarton/fasth3/tt-metal/tt-project/state/t400 (chain.log, summary.md, mp4/).

## Chain run 1534 (jobs 763-815), log state/t400/chain-run1534.log
- Completed: off-s0 763, on-s0 767, on-s1 788, on-s2 796, off-s3 798.
- Failed: off-s1 (769, 780), off-s2 (790, 793), on-s3 (800, 803), off-s4 (806, 809), on-s4 (812, 815).
  All but 780 were our own guard, "[t286] warm deadline 300 s passed in Denoising", with 100% JIT cache hits.
  Construction warmup alone took 183-265 s (weight load varies), so a 300 s deadline was too tight. These are
  not device faults.
- Drops (we reset nothing):
  - 23:10:31 UTC, after job 769 (ours, off-s1): chip 22, UBB tray 4 off the bus; broker held, recovered itself.
  - 23:27:22 UTC, job 780 (ours, off-s1 retry) SIGKILLed by broker device recovery: 31/32 chips, then
    chips 5,13,16,17,19-28,30,31 off the bus; broker per-tray walk, recovered itself.
  - (run 1524) 22:14 UTC, job 751: see the project root's tt-project/t400/NOTES.md.
- Partial (analyze400.py): seed 0 on vs off -3.3% s/step, PSNR 25.3 dB avg / 20.1 min, PCC 0.908.
  On arm s/step is steady at 1.378-1.395 over 6 replays; off is 1.436 (s0) and 1.50-1.54 (s3).

## Chain run 1546 (chain400c): off-s1 off-s2 on-s3 off-s4 on-s4
- Specs: -t 500 (worst measured job 326 s, +50%), pytest 470, warm deadline 400.
- Detached as ttp detach chain400c; rc file tt-project/state/runs/1546/chain400c.rc; it ends with
  "T400_DONE <rc>" in state/t400/chain.log and runs analyze400.py.

## Next step
Read state/t400/chain.log and summary.md; look at every still (state/t400/mp4/s<N>_off_vs_on_*.png); recommend.
Then delete blx01 /var/tmp/fasth3/cache/dit-h3hf/minimax-h3/transformer_precomputed_adaln and /var/tmp/fasth3/t400
(git -C /var/tmp/fasth3/t48 worktree remove --force /var/tmp/fasth3/t400/b).
