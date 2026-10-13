# t466 notes (ring SDPA LoFi matmuls for FastH3 Turbo)

Branch ttp/t466-turbo-ring-joint-sdpa-lofi-matmul-fideli, base dbd6d66048f (t460 head).
Code: 6dcde6d7dcc/9113b4b2cd4/6a740564a56 = cherry-picks -x of #57979, #58032, #58223 (clean);
f16f89d0f92 = opt-in MINIMAX_H3_SDPA_MM_FIDELITY (lofi|hifi2|hifi3|hifi4; unset = unchanged), ring path only,
CPU tests test_sdpa_mm_fidelity_minimax_h3.py (9 passed); e02052af0ee = device microbench
test_sdpa_fidelity_timing_minimax_h3.py (one layer, 10 s fl2va shape, random weights, HiFi2 vs LoFi interleaved,
logs "T466 summary: ring_joint_sdpa gain X%, layer gain Y%" and PCC between arms).

## blx01 state (2026-10-13 06:22 UTC)
- /var/tmp/fasth3/t466/b: t422/b tree + `git archive e02052af0ee` overlay, fresh build_Release
  (driver build466.sh, log build.log, exit code in build.rc). Python env: /var/tmp/fasth3/t460/python_env.
- No MiniMax-H3 weights on blx01 (deleted after #463). Microbench needs none.

## Next step
1. Build done (build.rc = 0): write microbench job script under /var/tmp/fasth3/t466 (env like
   tt-project/t460/run460.sh: HOME/TMPDIR/TT_METAL_CACHE under /var/tmp/fasth3, TT_METAL_HOME and
   PYTHONPATH = t466/b, `timeout --foreground -k 10 560 python -m pytest -sv -p no:cacheprovider
   models/tt_dit/tests/models/minimax_h3/test_sdpa_fidelity_timing_minimax_h3.py -k 4x8sp1tp0nl2_ring_is_fsdp0`),
   device-lint.sh it, health check, submit run-bg on blx01's broker (-t 600, unmeasured).
2. Gain < 10% on ring_joint_sdpa: stop (record, keep knob opt-in).
3. Gain >= 10%: e2e A/B needs ~360 GB on blx01 (weights 135 + dit cache 115 + metal cache ~59 + build) vs the
   150 GB cap (memory, after #463) -> ask the user to raise the cap before re-fetching weights.

## 2026-10-13 06:29 UTC (12-12 23:29 PDT): microbench submitted
- build.rc = 0 (build done 06:25 UTC). Job script run466mb.sh + env466.yaml copied to /var/tmp/fasth3/t466
  (JIT cache /var/tmp/fasth3/t466/jit, cold). Spec spec_mb.txt submitted via tt-project/t288/blx01-submit.sh:
  health ok, lint ok (cold-cache warning only), blx01 broker job 103, queued behind ltx-host job 102. -t 600 (unmeasured).
- Log: /var/tmp/fasth3/t466/out_mb/run.log (grep 'T466'); broker log /var/log/tt-device-broker/2026-10-13_062903_103.log.
- Next: read "T466 summary: ring_joint_sdpa gain X%". <10% -> stop (knob stays opt-in). >=10% -> e2e A/B needs weights
  (cap 150 GB on blx01, now 48 GB used) -> ask for cap raise.

## 2026-10-13 06:45 UTC (10-12 23:45 PDT): microbench result (blx01 job 103, e02052af0ee, 900 MHz clamp, relative only)
- One ring SDPA layer, 10 s fl2va shape, 4x8, 3 interleaved rounds (tt-project/t466/mb1_summary.log):
  hifi2: layer best 47.18 ms, ring_joint_sdpa best 39.54 ms; lofi: layer best 40.75 ms, ring_joint_sdpa best 33.37 ms.
  Gain: ring_joint_sdpa 15.6%, layer 13.6%. LoFi vs HiFi2 output PCC 99.90%, CCC 99.75%, RMSE/sigma 6.9%.
  Passes the 10% gate -> e2e A/B is next.
- No drops.

## e2e A/B is waiting on blx03
- Needs MiniMax-H3 (135 GB) + dit-h3hf cache (152 GB) + JIT cache (17 GB): ~305 GB. blx01 holds 49 GB of ours and its
  cap is 150 GB (memory, after #463), so it cannot fit there. The only known copy of the weights and cache is on
  blx03 /var/tmp/fasth3, and blx03 is out (no route since 2026-10-11 22:37Z; g14blx03-device paused, trays #364).
- Plan when blx03 returns: run the A/B on blx03 itself through its serial runner (data already there), arms
  base = t460 head dbd6d66048f build, lofi = same build + MINIMAX_H3_SDPA_MM_FIDELITY=lofi (needs this branch's
  cherry-picks built there). Order: 10 s seed 0 pair, 6 s seed 0 pair, then 6 s seeds 1-4 pairs, each one job.
  Alternative if the user raises blx01's cap to ~360 GB: copy from blx03 to blx01 under ionice/nice/--bwlimit=100M.
- blx01 leftovers of this task: /var/tmp/fasth3/t466 (build b + jit). Keep the build until the A/B host is decided.

## 2026-10-13 12:40 UTC (05:40 PDT): reroute to blx01 sized, over the 150 GB cap -> blocked
- Weights: weka copy found, read-only, no copy needed:
  /mnt/MLPerf/tt-shield/persistent-volume/volume_id_tt_transformers-MiniMax-H3-v0.22.0/weights/MiniMax-H3
  (/home/sulphur/models/minimax-h3 is a dir of symlinks into it). transformer/ = 62 GB.
- dit-h3hf cache estimate: the tt-shield fl2va cache for the same model on weka
  (tt_dit_cache_robust_fl2va/minimax-h3) is 115 GB; the generic one is 90 GB. Ours (fl2va, Turbo) ~115 GB,
  plus ~17 GB JIT cache (t460 runs on blx03).
- blx01 /var/tmp/fasth3 now 49 GB (home 10, models 7, .cpmcache 7, t466 4, t460 4, t409 4, t422 3, t420 3,
  t284 3, t456/t457/t408 2 each, ~15 dirs of 1 GB). 49 + 115 + 17 = ~181 GB > 150 GB cap. Not written.
- Even deleting every finished-task leftover (~30 GB, not ours to judge) leaves ~150 GB, no headroom.
- Not used: the weka tt-shield caches read-only. They are built by v0.22.0 service code without the Turbo
  adapter, and a cache miss would write to /mnt (forbidden).
- t460's run460.sh refuses weights under /mnt; t466's job script must drop that guard (coordinator update allows
  the weka path).
- Next (after a decision): (a) temporary blx01 cap ~200 GB for this task (cache deleted at the end), or
  (b) wait for blx03 (data already there; g14blx03-device paused until 2026-10-17T07:01Z at the latest).
  Then: run466.sh = run460.sh pattern, one build t466/b for both arms (knob unset vs lofi) in ONE broker job,
  10 s seed 0 first; cold first job builds the cache, so give it its own warm-up job.

## 2026-10-13 ~12:55 UTC (05:55 PDT): coordinator go-ahead on blx01; stopped at the device lint -> blocked
- Step 0: branch pushed (ttp push --own) at 75abbc8379f.
- Step 1: deleted finished-task build dirs on blx01 (all FastH3-created, branches kept as bundles next to them):
  /var/tmp/fasth3/{t409,t422,t420,t284,t408}/b (~3 GB each, ~13 GB). Total now 36 GB. Kept t466/b, t460 (venv,
  run460.sh), t286 (pytest plugin), t408 (keyframe), models/ (adapter), home/, *_keep.
- Step 2: / has 536 GB free; after ~132 GB -> ~404 GB (>=300 ok); total 36+132 = 168 GB (<=200 ok).
- Step 3: tt-project/t466/run466.sh written and copied to blx01 (/var/tmp/fasth3/t466/run466.sh): one job runs
  arms in order (base = knob unset, lofi = MINIMAX_H3_SDPA_MM_FIDELITY=lofi), build t466/b, caches t466/jit and
  t466/cache/dit-h3hf, /mnt guard on every write path, space guards, 90 s watchdog, 2-hang stop, 400 s warm deadline.
  Planned cold job: run466.sh c10s0 10 0 base (-t 600, T466_PY_TIMEOUT 560 needed for cold).
- Step 4 refused by device-lint.sh (exit 2): "run466.sh reads /mnt ... device jobs never read from /mnt or other
  network fs" and, with a /var/tmp/fasth3 symlink, "resolves to /mnt/MLPerf/...: copy it to local first".
  The lint rule exists because job 086 hung in D state reading /mnt/MLPerf weights after opening the mesh; on
  blx01 that would also stall the live LTX service (ltx-host runs ~15 min jobs back to back). Not routed around.
  Symlink removed. No device job submitted, no drops.
- weka MiniMax-H3 sizes (apparent): text_encoder 63 GB, transformer 62 GB, vae 10 GB, rest ~2 GB
  (transformer_ref 62 GB not needed): a local copy is ~137 GB.
  Local copy totals: 36 + 137 + 115 (dit-h3hf) + 17 (JIT) = ~305 GB in /var/tmp/fasth3; / free after ~267 GB.
  Breaks both the 200 GB and the 300 GB-free limits of the exception.
- Options: (a) lint exception for this read-only weka path, accepting the D-state hang risk; (b) raise the
  exception to ~310 GB total / ~260 GB free floor and copy the 137 GB with ionice -c3 nice rsync --bwlimit=100M
  (~25 min), delete all of it at the end; (c) wait for blx03 (weights and cache already there, back by 10-17).

## 2026-10-13 12:50 UTC (05:50 PDT, #468): space gate for a local weights copy fails -> nothing copied, blocked
- Branch already pushed: origin ttp/t466-... = 92caf8f61ff.
- Loader needs (test_pipeline_turbo_minimax_h3.py:171 weights_dir("transformer","text_encoder","vae","audio_vae"),
  plus processor/tokenizer/scheduler/json). Measured on weka (du --apparent-size): text_encoder 63G, transformer 62G,
  vae 9.8G, audio_vae 0.58G, processor+tokenizer 22M -> ~135.4 GiB. transformer_ref (62G) not needed.
- Cache, measured from the weka fl2va cache of the same model: 115G (transformer_resident_adaln 63G, text_encoder 47G,
  vae decoder 4.6G, audio 0.5G, vae encoder 0.1G); JIT ~17G (t460 on blx03), 0.6G already in t466/jit.
- blx01 now: / 536G free, /var/tmp/fasth3 36G.
- Gate: free after = 536 - 135.4 - 115 - 16.4 = ~269 GiB < 300 floor (short ~31 GiB). Total 36 + 267 = ~303 GB <= 320 ok.
  Deleting every other FastH3 leftover on blx01 (~20 GB usable) still leaves < 300 free.
- Options for the coordinator: (a) lower the free floor to ~260 GB for #466 only (blx01 / would be ~70% used);
  (b) copy the weka text_encoder cache (47G, adapter-independent) plus the small text_encoder config/tokenizer files
  instead of its 63G shards: peak ~204G, ~332G free; needs a waiver of "do not reuse weka tt-shield caches", and a
  v0.22.0 cache-key mismatch would fail the cold job (no /mnt write: cache dir is local);
  (c) wait for blx03 (weights and cache already there; back by ~10-17).

## 2026-10-13 12:46 UTC (05:46 PDT, #468 attempt 2): floor lowered to 250 GiB -> gate passes, copy running
- Gate: / 536G free, /var/tmp/fasth3 36G. After weights 136 + h3hf 115 + JIT 16: ~269 GiB free (>= 250), total ~303 GB (<= 320).
- cp466.sh (on blx01 /var/tmp/fasth3/t466): nice -n19 ionice -c3 rsync -a --bwlimit=100M of MiniMax-H3 minus
  transformer_ref/.cache/README/LICENSE to /var/tmp/fasth3/models/MiniMax-H3; re-checks the projection every 20 s and
  deletes the copy if it would break the 250 GiB floor. Started via ttp detach --remote (pid 3972908),
  log /var/tmp/fasth3/t466/cp466.log, rc /var/tmp/fasth3/t466/detach/cp466.rc. ~65-90 MB/s, ~30 min.
- run466.sh now reads $F/models/MiniMax-H3 (refuses non-local), guards 250 + 131 cold / footprint 320 - 131 cold,
  and stops an arm if / drops under 250G mid-run. device-lint.sh exit 0 for both the cold and the A/B commands.
- chain468.sh (g15blx02, ttp detach): waits for cp466.rc, then spec_c10s0 (cold, <= 3 tries while cache builds),
  then spec_ab10s0 (<= 2 tries), each via tt-project/t288/blx01-submit.sh (health, no other smarton job, lint).
  Marker chain468.done (0 = both done), log chain468.log.
- blx01 drop seen (not ours): 2026-10-13 10:57 UTC, ltx-host job 130 broker-killed, chips 5,13,18,23,29 left PCIe;
  recovered by 11:02 UTC (pci-rescan, glx_reset, fabric-check, hold ended).
- Next: on chain468.done = 0, read out_ab10s0_{base,lofi}/run.log timing sections; then 6 s seed 0 A/B, seeds 1-4, VBench.
