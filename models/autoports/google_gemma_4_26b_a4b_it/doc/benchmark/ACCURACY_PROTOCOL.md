# Final benchmark accuracy protocol

Prepared before inference. No score or live-server result is established by these setup checks.

- Client: `/home/mvasiljevic/.venvs/gemma4-benchmark/bin/python`, upstream lm-evaluation-harness 0.4.13. The imported `benchmark_stage` must resolve inside installed tt-model-bringup 0.1.14 runtime; the transport check verifies that path.
- Questions: packaged ci-v1 documents, reproduced using `prepare --reuse-manifest`; no question selection based on scores. `setup-previous/preparation/subset/manifest.json` preserves every document hash, original index, full population hash and few-shot hash. Manifest SHA256 (canonical object) is `ef36dff6d479319adc5b0438e79acc4e139bd108300e209479ab6ee864c4c994`; parent manifest is `a76275b17f9ae457e9e15c93c339d0e9c9c5de6ac74ff42099183cded18d733b`.
- MMLU-Pro: 280 of 12,032 documents, 14 upstream subject tasks, five-shot, few-shot messages preserved. Upstream `custom-extract` regex `answer is \(?([ABCDEFGHIJ])\)?`, then first match; exact-match scoring with case/punctuation normalization. Retain every subject and upstream aggregate score.
- IFEval: 256 of 541 documents, zero-shot, all four upstream strict/loose instruction/prompt metrics. `punkt_tab` is available in the client; task loading and dataset preparation succeeded. Upstream settings are retained in `setup-previous/preparation/task_settings.json`.
- Shared accuracy pool: 32 concurrent requests. Both task overrides: `max_gen_toks=4096`, `temperature=1.0`, `top_p=0.95`, `top_k=64`, `do_sample=true`, `until=[]`, `chat_template_kwargs={"enable_thinking":false}`, `logprobs=true`, `top_logprobs=0`. Native nonthinking is explicit. Empty textual stops override MMLU-Pro's `Question:` stop; native EOS remains enabled. This is a declared recipe difference from upstream default greedy generation and output limits. Keep all token-limited answers in the denominator.
- Chat rendering: upstream `LocalChatCompletion.apply_chat_template` serializes the role/content list into `JsonChatStr`; its HTTP path reconstructs the identical structured message list. It does not render the HF native template. The server renders its pinned native template once, including the nonthinking argument. Native server tokenizer/template revision is `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`; check server rendering and imported tokenizer separately.
- Exact top-64 sampling uses the existing host fallback, selected by accuracy-only `logprobs=true, top_logprobs=0`, with `GEMMA4_AUTOPORT_ALLOW_HOST_SAMPLING=1`. Stage 10 device sampler caps top-k at 32. This keeps the model/precision fixed but changes accuracy sampling execution to host logits; greedy performance requests must omit logprobs and remain on the validated device path. Runtime route validation is still required; static routing evidence is in `AUTODEBUG_sampling_policy.md`.
- Reference comparison: published MMLU-Pro 82.6% for this checkpoint is contextual only. The frozen subset, nonthinking mode, output budget, prompt/scorer and aggregation may differ from publication; report the numerical delta without an accuracy verdict. The parent report must attach its verified primary reference link.

## Transport verification

`tools/check_benchmark_transport.py` performs one synthetic async request through the actual upstream backend to a localhost recording HTTP stub. It verifies structured messages, 4096 token cap, top-k/top-p/temperature, nonthinking kwargs and zero-logprob host-route selector reach the wire unchanged, parses the response, and checks native EOS has not been disabled. Sanitized captured payload: `setup-previous/preparation/transport.json`. This is not benchmark inference; its synthetic usage values must never be used as measurements. No client wrapper or monkey patch is necessary.

The evaluation client's transformers 4.57.6 cannot directly load this newer tokenizer because its tokenizer configuration uses list-valued `extra_special_tokens` where transformers 4 expects a dictionary. This does not affect tokenizer-free accuracy transport. The server and performance tokenizer must use a compatible environment; do not rewrite native tokenizer configuration to conceal the incompatibility.

Setup command (successful):

```bash
PYTHONPATH=/home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.14/runtime /home/mvasiljevic/.venvs/gemma4-benchmark/bin/python -m benchmark_stage prepare --tasks mmlu_pro,ifeval --counts 280,256 --reuse-manifest /home/mvasiljevic/.codex-personal-gemma4/plugins/cache/tenstorrent-skills/tt-model-bringup/0.1.14/runtime/benchmark_stage/profiles/ci-v1.json --output models/autoports/google_gemma_4_26b_a4b_it/doc/benchmark/setup-previous/preparation/subset
```

The timed runner must reverify the selected document populations and few-shot hashes. No benchmark/server/hardware command was run during this preparation.
