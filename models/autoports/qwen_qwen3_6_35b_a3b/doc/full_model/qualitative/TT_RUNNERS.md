# TT qualitative and resumed-prefill runners

Status: source-only preparation. These commands have not been run by the helper agent. Source lane-env.sh, activate python_env, and serialize all hardware work on the assigned idle mesh before invoking them.

All40 layers, all six canonical shared prompts,128 generated tokens per prompt:

```bash
timeout 3600 python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_tt_controls.py
```

One paired autoregressive readiness prompt, also128 tokens:

```bash
timeout 3600 python models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/run_tt_controls.py \
  --mode autoregressive --prompt-id q02 \
  --output-dir models/autoports/qwen_qwen3_6_35b_a3b/doc/full_model/qualitative/tt_autoregressive_q02
```

The suite calls the installed `readiness_check.run_autoregressive.run_autoregressive` for every prompt. It reuses one fully loaded generator through a non-owning constructor adapter, and substitutes the existing HF controls only after checking model/revision, suite, prompt rendering/tokens, greedy parameters and token budget. No HF model is loaded while the TT mesh is open. The package strips prompt-file text before encoding; an exact-token adapter preserves the original chat-template trailing newline and saved token IDs. There are no package or generator edits.

Each prompt directory contains the package's `hf_completion.txt`, `tt_completion.txt`, and `autoregressive_meta.json`, plus the exact rendered prompt. `tt_greedy_outputs.json` records prompt metadata, controls/provenance, generated tokens/text and first divergence. `paired_review.md` collects the actual paired completions for reading. Successful generation leaves **GENERATED_REVIEW_REQUIRED**, never an automatic qualitative pass. Read all pairs for coherence, repetition, language drift, leakage and early divergence, and write a separate source-bound verdict.

Before pursuing an apparent qualitative failure, follow the existing HF matched-seed and blind-comparison workflow in `README.md`, `run_hf_controls.py`, and `blind_compare.py`: lock the failure rubric, use model-card settings and at least10 matched seeds on both sides, and compare blind. A missing final answer within128 tokens or no greedy EOS is not independently a failure; the existing HF controls use the default thinking template and frequently spend the entire budget reasoning. Do not silently drop unsupported sampling features to claim a matched control.

Full-model resumed-prefill, with the installed checker's unchanged thresholds and negative controls:

```bash
timeout 7200 python -m models.autoports.qwen_qwen3_6_35b_a3b.tests.run_full_model_resume_check --mode resume
```

This loads all40 layers with context2048 and evaluates natural non-repeating book tokens at lengths257/1025, blocks32/128,64-position windows plus the final token. Coverage includes aligned, nonaligned and last-token starts. Both `no_history` and `fresh_suffix` controls must fail the same gates as the candidate. The wrapper warms explicit host-logit readbacks before trace capture and reuses that prepared generator through the package's constructor hook. Add `--reduced` only for a clearly labeled layers0/3 diagnostic.

A pass writes `readiness/prefill_resume_check.json`; failures write `readiness/prefill_resume_check.failed.json`, preserving any prior passing artifact. This numerical continuation check does not erase the architecture's `recurrent_state` prefix-caching exclusion: a matching full convolution/recurrent snapshot is still required alongside KV history, and serving capability declarations must state that contract.
