# Astropy full-prefix initial-distribution discriminator

Coordinator goal: quantify whether the distinct31001-token native input preceding Astropy's first missing FormatError response shows gross initial HF/TT divergence. Both agreement and divergence inform this bounded question. Neither establishes the original loop cause, reproduces the unknown stochastic trajectory, nor proves later continuation health. Validity unblocks capped-call cause; saving0 planning credit. Diagnosis only, no source branch, sampler change or precision change. Existing driver remains unchanged.

Input: bringup/artifacts/astropy-capped-prefix/case.json and request.json, prepared from native messages0..138 of Astropy14598 in run42210467509. Stockmini2.2.8 filtering and installed stockvLLM protocol/auto-openai template yield31001tokens, SHA2562cd074a353c38472a9c01b92fe84d7b393a8439233d910ec3409c7833fccfec9. Original server prefill endpoint31001 at13:07:05.921 follows tool138 at13:07:05.6998. Original tokenarray/seed/response/finishreason remain unknown; --allow-reconstructed explicitly acknowledges this. Preserve private raw content and use same case hash/tokens for both arms. Evidence ../astropy-capped-prefix-preparation/RESULT.json.

Coordinator confirmation: full60 pinnedHF revision842da3794eaa0b77d5f08bae87a17459d91ff475, CPUchunk512, shortcachecontrol KL<=.01 and top1agreement, then all31001 tokens with one terminal fullvocabulary logit vector. Full60 TT B8 current selected precision on same31001tokens with existing prefill_logits_continuation; no shortened prefix, teacherforcing-throughput claim or reduced-model substitute. Compare fullvocabulary KL(HF||TT), JS, top1 and top20/release-top_p.95 support; report support mismatch/infinite filtered KL honestly. Prior different Django42823 prefix KL.1351848/JS.0337894 is descriptive reference only, not pass threshold or evidence Astropy healthy. Verify the precise values from prior comparison artifact before final report.

CPU forecast3625s is a linear guess from prior5008s/42823; internalbudget7100s, outer7200s. Prior successful HF peak99583MiB and16threads; coordinator checks current memory headroom and avoids duplicateCPU reference processes. Chunk-control validates cache numerics, not case correctness. Existing per-chunk2xremaining projection may stop; preserve partial result and do not treat it as a completed fullprefix discriminator. No model load/inference launched by this subagent.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout --signal=TERM --kill-after=30s 7200s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py hf --case bringup/artifacts/astropy-capped-prefix/case.json --allow-reconstructed --chunk-size 512 --pilot-tokens 1024 --budget-seconds 7100 --output bringup/artifacts/astropy-capped-prefix/hf512
```

Separate device forecast0.12h(432s), box0.20h(720s) including load/cleanup, based on priorfull60B8terminal-logit332s. Coordinator owns queue, device-owner checks and launch after CPU result; this subagent never opens hardware. Direct TTgenerator and its initialcache state differ from original served scheduler/APC state; comparison cannot isolate a serving-only defect. No profiler or runtime sampling changes.

```bash
timeout --signal=TERM --kill-after=30s 720s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py tt --case bringup/artifacts/astropy-capped-prefix/case.json --allow-reconstructed --batch 8 --output bringup/artifacts/astropy-capped-prefix/tt-b8
python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py compare --hf bringup/artifacts/astropy-capped-prefix/hf512/logits.pt --tt bringup/artifacts/astropy-capped-prefix/tt-b8/logits.pt --output bringup/artifacts/astropy-capped-prefix/comparison
```

Coordinator hands off completed HF/TT/comparison artifacts and actual timing/device closure. Write one compact final result then, with input identity, gates, measured distributions, predictions-versus-results and unresolved scope. No ongoing progress-evidence maintenance and no changes to previous preparation results. Upcoming retained Astropy repeat is independent evidence, never substituted into this originalinput. No tool calls from native payload execute.
