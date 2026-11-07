# New doc160 period-three hypothesis

Coordinator goal: test a conditional full60 HF/currentTT numerical discrepancy around the newly observed doc160 period3 onset. Forecast0.08 device hours, box0.20; host box0.4h. Speculative release80 saving1h if a correctable cause exists, credited0. Normal80 run42323448662 proceeds independently. No original served-state or all-loop claim, no profiling, no hardware launch by preparer.

Original40 retained chain is uniquely anchored by doc160 prompt length312; physical decode chain has126975 samples and122336-token period3 suffix, onset physical index4639. Exact native onset would be index4640 if a separately verified first prefill output is recovered. Three or four checkpoints should bracket entry and cover all three interior phases (e.g. native4640,4655,4656,4657), with maximum input prefix about4969. No120K reference or full rollout.

Critical host finding: original40 artifacts retain physical decode IDs only. The first prefill output token is absent; saved visible response is empty, and CI INFO filtering removed the native JSON body. Existing logits/decode diagnostic drivers explicitly require a complete exact prefix including the first token. Do not set complete_native_ids/first_prefill_token_present flags from inference or splice a guessed token. Current host evidence is insufficient to construct the requested exact prefix.

Proposed prerequisite, requiring coordinator scope/budget admission: bounded native streaming capture of this one original prompt with exact release seed42/top_k20/top_p0.95/temperature1/thinking and output126976/timeout14400, return native token IDs, then close only the owned HTTP stream after~4700 generated IDs. Request cap remains126976; diagnostic cancellation is not a scored outcome. Reuse proven capture client/idle shutdown and current default B8 serving. Verify native prompt IDs equal pinned tokenizer312 and native[1:] exactly matches original retained physical prefix through all checkpoints. If mismatch, do not describe the new prefix as original or proceed to original-prefix HF/TT; report the distinct discrepancy and seek coordinator direction. This extra serving capture is not silently included in the original0.08h estimate.

Only after complete-prefix proof, adapt existing gpqa-loop-logits-discriminator case preparation with the new period3 checkpoints and reuse its diagnose.py unchanged. Validate exact case SHA/next-token indexing. Proposed bounded TT command (coordinator executes after owner/health/queue checks):

```sh
TT_METAL_CACHE="$PWD/bringup/artifacts/cache-env-validation/tt-metal-jit" XDG_CACHE_HOME="$PWD/bringup/artifacts/xdg-cache" timeout --signal=TERM --kill-after=15s 690s python_env/bin/python bringup/artifacts/gpqa-loop-logits-discriminator/diagnose.py tt --case bringup/artifacts/gpqa40-new-loop-diagnosis/case.json --output bringup/artifacts/gpqa40-new-loop-diagnosis/tt
```

This command is HELD: case does not exist yet and total capture+diagnostic cost must fit a revised coordinator allocation. Prefer safe natural completion inside the queue box; never intentionally interrupt outstanding TT work simply to force fit.

Host reference, after complete-prefix validation and memory check, uses pinned full60 HF offline with existing driver hf mode, max~5K prefix, internal900s/outer960s. Compare raw full-vocabulary top1/KL/JS and loop-token probabilities plus explicitly labeled CPU top20/top_p.95 transform. HF on a TT-generated exact prefix is conditional evidence only. TT direct-prefill logits do not reconstruct original scheduler/page mappings, asynchronous sampler RNG or decode state. Incremental decode is conditional follow-up only if a discrepancy motivates it and remaining budget permits; not an automatic second device run.

Stop once the conditional question is answered or exact-prefix prerequisite cannot be recovered within authorization. No numerical/config fix is accepted from these checks alone; appropriate native/scored paired confirmation remains required. Write final evidence once after admitted execution or precise exhausted blocker; no central-document edits.
