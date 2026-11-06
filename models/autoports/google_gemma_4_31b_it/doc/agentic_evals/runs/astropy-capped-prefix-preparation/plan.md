# Astropy first missing capped/FormatError input preparation

Coordinator goal: determine whether native input before feedback139 can be reconstructed from saved messages0..138 even when model response was discarded. Host-only/device box0, no model inference, device opening, serving or dispatch. Follow pinned mini2.2.8 stock formatting and exact current vLLM offline tokenizer where feasible; compare prompt count with saved chronological server evidence. Keep raw content/IDs private, compact result metadata only. Inspect existing Django full-prefix HF/TT discriminator and resource timings, then recommend a separate minimal test only if useful. Output/seed unknown must remain unknown. This is a different original failed prefix; negative Django reproduction is not proof Astropy is healthy. Validity unblocks capped-call cause; savings0 planning credit, no runtime gain measured. Write one result after host preparation; upcoming repeated run remains distinct.


Prepared followup commands (proposal only; coordinator owns CPU launch and serializeddevice):

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout --signal=TERM --kill-after=30s 7200s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py hf --case bringup/artifacts/astropy-capped-prefix/case.json --allow-reconstructed --chunk-size 512 --pilot-tokens 1024 --budget-seconds 7100 --output bringup/artifacts/astropy-capped-prefix/hf512
```

Explicit --allow-reconstructed acknowledges originalwiretokenIDs absent despite stockoffline/count/time match; it must not be relabeled exactoriginalsampling. Existing driver cache-control64tokens is a numerical parity gate, not a shortened failing-prefix test. Complete31001prefix required; budget/projectionstop leaves no terminal comparison. Host reference predicted~3625s from prior5008s/42823, generous7100sbudget, prior~97.25GiB RSS, no CPU model inference launched by preparation.

```bash
timeout --signal=TERM --kill-after=30s 720s python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py tt --case bringup/artifacts/astropy-capped-prefix/case.json --allow-reconstructed --batch 8 --output bringup/artifacts/astropy-capped-prefix/tt-b8
python_env/bin/python bringup/artifacts/django-cap-loop/diagnose.py compare --hf bringup/artifacts/astropy-capped-prefix/hf512/logits.pt --tt bringup/artifacts/astropy-capped-prefix/tt-b8/logits.pt --output bringup/artifacts/astropy-capped-prefix/comparison
```

TT estimate0.12h/box0.20h is a separate proposeddeviceeffort, not spent or authorized here. Fullgenerator API prefill_logits_continuation accepts start,positions,prefill_prefix as existingdriveruses. No toolcommands execute; no sampler/modelchanges. First-token divergence is a bounded discriminator, not complete rootcause proof.
