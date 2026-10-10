# t288: AdaLN precomputed vs on device, FastH3 HyperFlow 8-step, 4x8 blx03

## 2026-10-10 ~06:10 UTC (run 1292)
- Code: ttp/t288-adaln-ab = a99af1c7dea + cherry-pick acebc7d39e2 (test import fix, cc3b6de2dc7)
  + f7266236d90 (MINIMAX_H3_ADALN_PRECOMPUTE=0 keeps AdaLN on device under HyperFlow; the turbo
  test skips the two-time embedder check under the table). Python only.
- `ttp push --own` refused: its push check runs LTX tests (test_vae_ltx_*_ref.py) that do not
  exist on this H3 branch. Branch not on origin yet. Applied on blx03 as patches instead:
  ~/fasth3/t286 detached at 03848fa4df (same tree as f7266236d90; build from #286 reused).
- HyperFlow adapter: HF videorebirth/hyperflow -> blx03 /var/tmp/fasth3/models/hyperflow
  (2.8 GB, sha256 matches hyperflow.json, READ_OK). Header has hyperflow sigmas (9 pts),
  gate 0.25, shifts 12/3.
- Driver /var/tmp/fasth3/t288/run288.sh <tag> <on|off> (copy here). fl2va, 5 s, 1344x768,
  NFE 8, seed 0, kf /var/tmp/fasth3/t209/kf_first.png, caches dit-h3hf + tt-metal-cache-h3hf
  (shared with #286). Off arm reuses transformer_resident_adaln cache; on arm writes a new
  transformer_precomputed_adaln cache (cold: expect on-a to stop at the warm deadline, on-b warm).
- Queued (FIFO): t288-off-a, t288-on-a, t288-on-b, t288-on-c, t288-off-b. ONCE guard per arm
  (out_on/PASS, out_off/PASS), so extra jobs exit at once after a pass. off-b re-runs only if
  off-a failed.
- AICLK on blx03 is clamped at 900 MHz: numbers are relative only.

## Next step
Wait on probe.sh t288-off-b. Then: done markers of all five; out_{on,off}/PASS; grep
"MINIMAX-H3 PERFORMANCE RESULTS" table from /var/tmp/fasth3/t288/out_{on,off}/run.log;
PSNR/PCC of out_on vs out_off mp4 (ffmpeg psnr + numpy on decoded frames); still frames;
copy small artifacts to tt-project/t288/. Cleanup: dit-h3hf/minimax-h3/transformer_precomputed_adaln
if not kept, hyperflow model if no follow-up needs it.
