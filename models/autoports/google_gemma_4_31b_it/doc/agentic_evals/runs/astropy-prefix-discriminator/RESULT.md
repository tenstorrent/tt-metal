# Astropy prefix discriminator result

The complete31001-token Astropy input produced HF→TT KL **0.3170207891** and JS **0.0412971337**, with the same top1 token. Release top20/top_p.95 filtered KL is infinite because sampling supports differ. These numbers quantify initial-distribution divergence; they establish neither precision collapse nor the cause of the discarded FormatError response.

Both arms used the identical reconstructed case and pinned revision842da3794eaa0b77d5f08bae87a17459d91ff475, all60 layers. HF's shortcache-control KL was0 with matchingtop1; all31001 prefix tokens then completed. TT used B8 and selectedhead_bfp4 policy. DirectTTgenerator does not recreate the originalserved scheduler/APC. Originaltokenarray, seed, discardedoutput and finishreason are unknown; nativeinput count/time matches originalserverprefill, not an originalwirecapture.

The earlier distinctDjango42823 prefix yielded KL.1351848168/JS.0337893580 with same top1 and filteredsupport mismatch. It is descriptive context, not a threshold or evidence of Astropy's later behavior. One terminalposition cannot establish continuationhealth, sampling-loop cause or score equivalence. No precision/sampler change was made or accepted.

HF CPU elapsed2347.235924s versus3625s forecast, peak88538.0078MiB, within7100s internal/7200s outer limits. TT retry driver elapsed171.649817s. Total devicequeuecharge **0.05005858278h** (180.210898s), including the initial3.693695s launcher failure, measured2.791999s healthwindow and173.725204s retryqueuewall, versus.12hforecast/.20hbox. These are diagnostic cost timings, not devicekernel or generation-throughput measurements.

AutoFix restored the omitted TT_METAL_CACHE launch environment after hostpermissionfailure beforemodel execution. Boundedhealth passed withoutreset; clean retry completed matchinginput/full60 comparison. Allfailed/health/retry artifacts remain preserved. Capped-call rootcause stays unresolved despite completed diagnostic.

Evidence: RESULT.json, cache-launch-AUTODEBUG.md, cache-launch-AUTOFIX.md and privateartifacts under bringup/artifacts/astropy-capped-prefix. No centraledits, modelchanges or hardware work by this result author.
