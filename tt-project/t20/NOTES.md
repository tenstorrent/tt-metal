# t20 notes

Branch ttp/t20-... reset onto the t36 line (16ba9a383dc: t10/t26 LTX-2.5 port + blx03 setup), then 0533827a419.
Finding: conv was already the default in code (LTX25DistilledPipeline diffusion_decoder=False, test env LTX25_DIFFVAE
defaults "0", --diffvae is opt-in). Only tmp/blx03/run25.sh forced LTX25_DIFFVAE=1; 0533827a419 flips it to 0.
The conv path is the parent LTXDistilledPipeline decode (tuned 2.3 VAE, on-device YUV, traced).

Device run: blx03 broker job 877 (queued 19:11), label conv145_t20, 1080p/145f seed 0, warm+traced, run from
blx03:~/fasth3/tt-metal at 16ba9a383dc with LTX25_DIFFVAE=0 (python identical to 0533827a419; only scripts differ).
Output: blx03:~/fasth3/out/ltx25_1080p_6s/conv145_t20/{run.log,ltx_av_fast_*.mp4}
Check: ssh g14blx03 tt-device-mcp status -j 877
Next: grep -E "LTX_TIME|stage|decode|export|E2E" run.log for gen#1 timings; still via
ffmpeg -ss 3 -i <mp4> -frames:v 1 conv145_t20_t3s.png; copy mp4+png to g15blx02 tt-project/baselines/t20/.
Compare with dv145 job 874: S1 2.27, S2 2.50, DiffVAE 11.77, audio 0.39, compute 19.02, E2E_WALL_S 19.81.

## 19:23 update (after g15blx02 reboot; blx03 unaffected)
Job 877 FAILED on the 580s pytest timeout: warmup took 536s because conv VAE + audio vocoder kernels were JIT-cold
on blx03 (vocoder warmup alone 195s; video decode warmup 14s). It died at the S1 trace capture of gen#1. No timings.
Old log kept as conv145_t20/run_877_timeout.log. JIT cache is now warm (/var/tmp/fasth3/cache/tt-metal-cache).
Resubmitted as blx03 broker job 879: same config, PYTEST_TIMEOUT=1500, broker timeout 2400s.
Check: ssh g14blx03 tt-device-mcp status -j 879 ; then do the "Next" steps above on conv145_t20/run.log.

## 19:35 result (job 879, completed, exit 0)
Conv decoder, LTX-2.5, 1080p/145f, 4x8 ring, warm+traced, noise_seed=10000 (test default).
gen#1 (warm, new prompt "red paper boat"): E2E_WALL_S 8.761, compute total 7.93:
  encoder 1.91 | S1 2.28 | upsample 0.15 | S2 2.50 | VAE decode (conv) 0.72 | audio 0.37 | export 0.8
gen#0 (first traced run, trace capture inside): E2E 35.39 (S1 15.85, S2 15.13, VAE 1.81); not a timing number.
vs dv145 job 874 (DiffVAE): VAE 11.77 -> 0.72 s, E2E 19.81 -> 8.76 s.
Encoder 1.91 s is Gemma text encode of a new prompt (gen#0 got 0.26 s because warmup encoded the same prompt).
Denoise+decode+audio without encoder/export = 6.02 s; encoder and export are now the next-largest non-DiT costs.
Artifacts (g15blx02): tt-project/baselines/t20/{ltx_av_fast_1920x1088_{0,1}.mp4, conv145_t20_gen{0,1}_t3s.png, run.log}
Stills checked: both clean, no artifacts.
