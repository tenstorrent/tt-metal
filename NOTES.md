# t111 notes

Code (ttp/t111-denoise-host-overhead-cut-prompt-staging, pushed):
- 9eaa0aa2695: opt-in LTX_PROMPT_HOST_COPY=1 (tilize the new prompt on host, copy_host_to_device_tensor straight
  into the traced _prompt_v/_prompt_a buffers; falls back to the default path on the capture gen or a shape
  change), opt-in LTX_LATENT_STATS=0 (skip host latent fingerprints), LTX_PROMPT_STAGING_CHECK=1 (asserts the
  in-place write equals the default upload on every device), 2x4-submesh test param bh_4x8sub2x4sp1tp0,
  LTX_E2E_AB_ENV / LTX_E2E_AB_ENV_ONCE A/B hooks. CPU: test_ltx_stage_prompts.py 8 pass.
- ea85208b939: tmp/t111/run111.sh + driver.sh.

- 7030b0671f3: fix (pre-existing on t48): LTX-2.5's Gemma4TokenizerEncoderPair has no embedding_cache_identity,
  so every dynamic_load generate (BH 2x4 default) died in _device_embed_cache_path. Now encodes uncached
  (warning), and encode_prompts treats a None cache path as no cache. CPU: test_ltx_embedding_cache_identity.py 7 pass.

Device run: blx03, attempt 4 = launched 07:50 UTC with 7030b0671f3 staged in src (job ID in driver.log).
Attempt 3 = job 465: warmup OK (304 s), then gen0 died on the AttributeError above; no drop; logs in attempt3/.
Note: 2x4 runs dynamic_load=True (Gemma reloads per gen, uncached); prompt/stats lines are still comparable.
(attempt 1 = job 459, collection error, fixed in e686d3c0cf0, logs in attempt1/;
attempt 2 = job 462, 544x960 fails generate()'s H,W %64 assert, now 576x1024 in 42b431c02b3, logs in attempt2/). Job ID in driver.log (driver /var/tmp/fasth3/t111/driver.sh, log /var/tmp/fasth3/t111/driver.log,
job log /var/tmp/fasth3/t111/run111.log). 576x1024/145f traced, fresh prompts, gen0 capture,
gens 1,3 = baseline, gens 2,4 = HOST_COPY + LATENT_STATS=0 (gen 2 also runs the bit-identity check).

Next step when T111_DRIVER_DONE appears:
  grep -E 'pure replay|denoise init|LTX_PROMPT_STAGING_CHECK|latent\[|Stage|T111_EXIT' /var/tmp/fasth3/t111/run111.log
Compare the S1 "denoise init ... prompt N ms" of gens 1,3 vs 4 (gen 2 includes the check), and the S1->S2 gap
(latent stats lines gone on 2,4). Bit identity: CHECK lines must say bit-identical=True.
If the driver stopped with a drop during our job: stop all device work, report.

## Result (attempt 4 = blx03 job 468, 576x1024/145f, 2x4 submesh of the full mesh, exit 0, no drop)
gen1/gen3 = baseline, gen4 = LTX_PROMPT_HOST_COPY=1 + LTX_LATENT_STATS=0, gen2 = same + staging check.
- Bit identity: LTX_PROMPT_STAGING_CHECK video and audio: 8 devices, bit-identical=True (gen 2).
- S1 denoise init: 43/38 ms (prompt 38/32) -> 11 ms (prompt 6). Host cut 32 ms.
- But S1 step 1 grows 388.5/399.3 -> 425.2 ms: the device is still busy when the host reaches step 1
  (2x4 runs dynamic_load, 11.3 s transformer reload right before S1), so the saving is hidden.
  S1 init+step1: 431/437 -> 436 ms. Stage 1 denoise 2.69/2.72 -> 2.72 s (gen 4 steps 2-7 were ~4.5 ms
  slower each, device-side noise).
- Latent stats: s1/audio read ~1 ms (S1 end -> S2 start 446 vs 446 ms). s2/video+audio stats ~11 ms
  (S2 end -> VAE prepare done 447 -> 436 ms). Only this one shows on the wall clock.
- Totals: 18.00 / 18.03 s baseline vs 18.04 s opt-in: no measurable e2e gain in this config.
Conclusion: on 2x4 dynamic_load the prompt staging is not on the critical path. It may matter in
serving with resident weights (4x8, device idle at S1 start); measure there once 4x8 is allowed.
Log: g14blx03:/var/tmp/fasth3/t111/run111.log (src copy removed).
