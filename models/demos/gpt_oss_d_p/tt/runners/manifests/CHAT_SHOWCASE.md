# GPT-OSS chat prefill manifests

`gpt_oss_120b_chat_2l.json` is the two-layer wiring checkpoint.
`gpt_oss_120b_chat_36l.json` selects all 36 layers.

Both use one Galaxy at SP=4/TP=8, two allocated slots, 1,024-token compute chunks and 32,768-token cache capacity. Quantized source weights and BF8 device KV follow the existing GPT-OSS runner. Sliding attention keeps the model's 128-token window while physical KV storage keeps the full prefix for migration.

Set `PREFILL_MANIFEST` to the selected file, plus unique service/completion-ring names and fresh migration table/device-map output paths, then run:

```bash
python -m models.demos.common.prefill.runners.prefill_runner
```

The two-layer adapter narrows a copy of the Hugging Face layer configuration before model construction. Full-model configuration is unchanged. Sliding gather storage reserves two compact halos because a reused prefix can rotate Q across a ring-group boundary.

Validation: real two-layer chat requests at reused-prefix offsets 64 and 96 passed after the halo fix; before the fix the first reused prefix hit the RingJointSDPA two-halo assertion. Full-model readiness and output checks are tracked separately.
