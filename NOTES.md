# t41 notes: temporal-band S2 video self-attention (LTX-2.5, 1080p/145f, 4x8)

Branch ttp/t41-... = t20 tip (1eadde3ce6c) + 63c8860f08e. Pushed.

## Design
- S2 grid 19 latent frames x 34x60 = 2040 tokens/frame, N=38760 (38912 padded), SP=8 shard 4864 (~2.4 frames).
- Tokens are frame-major, so "attend only to keys within W latent frames" is one contiguous key range per
  query chunk. A ring-joint kernel can skip whole (q chunk, k chunk) pairs and stop the K/V ring after a few hops.
- Chunk-granular work left (tmp/t41/speed_model.py, q192/k512): W1 17.7%, W2 28.1%, W3 38.8%, W4 49.4%, W6 70.2%
  (max over devices; edge devices do less). Hops needed: W1 1, W2-3 2, W4-6 3 (of 7).
- S2 self-attn now 4.58 ms/op x 48 layers x 3 steps = 0.66 s of S2 2.50 s. W2 -> ~0.19 s + gather => ~0.4 s saved e2e
  if the kernel scales with work; W4 -> ~0.3 s saved.
- Quality check path (no kernel change): LTX_SELF_BAND=38760:2040:W = gathered K/V + masked SDPA (quality only,
  slower than dense). Q/K/V dump: LTX_DUMP_QKV=38760:48:8:0:2 (layers 0,8,..,40,47; steps 0-2; TP-rank-0 heads).
- CPU test: models/tt_dit/tests/models/ltx/test_self_band_reference.py (7 pass, torch only).

## Device job (blx03)
- blx03 worktree ~/fasth3/t41 (python-only, links ~/fasth3/tt-metal build_Release/runtime/_ttnn.so; no C++ diff vs its build).
- g15blx02 driver: tmp/t41/drive.sh 2 (log tmp/t41/drive.log, JOB[id] line, ends T41_DRIVE_DONE). Waits while another
  project job runs on blx03, then submits tmp/t41/job.sh 2: eager dense gen + dump, then eager band W=2 gen, seed 0.
- Outputs on blx03: ~/fasth3/out/t41/{dense,band2}/{run.log,*.mp4}; /var/tmp/fasth3/t41/{qkv,lat_dense*,lat_band2*}.

## Status 2026-09-30 20:25
- Driver ran at 20:24 but blx03 was down (ssh: no route to host, ping 100% loss). Nothing submitted, no job id.
- drive.sh now also retries on ssh rc 255. On wake: confirm blx03 is up and g14blx03-device is not paused,
  check ~/fasth3/t41 and /var/tmp/fasth3 survived, then relaunch:
  `setsid nohup tmp/t41/drive.sh 2 > tmp/t41/drive.log 2>&1 &` and wait on `grep -q T41_DRIVE_DONE tmp/t41/drive.log`.

## Status 2026-09-30 20:33
- blx03 back (rebooted ~20:26, broker power-cycled after a chip left PCIe, then recovered). ~/fasth3/t41 and
  /var/tmp/fasth3 intact; blx03 job.sh identical to branch. Project job 889 (other task) running.
- Driver relaunched: `tmp/t41/drive.sh 2` (setsid), log tmp/t41/drive.log; it retries every 2 min while 889
  (or any project job) is active, then submits. Old log: tmp/t41/drive.log.2024.

## Next
1. `ssh g14blx03 "cd ~/fasth3/t41 && ~/fasth3/tt-metal/python_env/bin/python tmp/t41/analyze.py /var/tmp/fasth3/t41/qkv"`
   -> attention mass by frame distance + per-W rel_l2 vs dense.
2. Latent PCC dense vs band2 (LTX_DUMP_LATENTS files), video PSNR (ffmpeg psnr filter), still at t=3s.
3. If W2 holds (no visible change): kernel work = band-aware chunk skip in ring_joint reader/compute (+ ring hop limit).
   If not: pick W from analyze.py and rerun job.sh <W>.
4. Cleanup when done: blx03 ~/fasth3/t41 worktree, /var/tmp/fasth3/t41, ~/fasth3/out/t41.
