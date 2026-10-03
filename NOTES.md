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
