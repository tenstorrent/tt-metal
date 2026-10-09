# Gemma 4 26B A4B on QB2

TP4 implementation of `google/gemma-4-26B-A4B-it` on four Blackhole devices
with a `(1, 4)` mesh. It supports prefill, traced decode, device sampling, and
vLLM serving. The implementation is independent of the canonical
`models/demos/gemma4` path and does not replace its defaults.

## Capacity

- Maximum supported context: 262,144 tokens.
- Up to 32 concurrent users.
- Hybrid sliding-window and full-attention KV caches use 32-token pages.
- Prefix caching and chunked prefill are not supported.

## Validation

The Tier-3 Agentic Research job runs the pinned 10-question GPQA subset,
request-isolation checks, and fixed-length serving measurements on P300X2.
The accuracy gate is defined in `models/model_targets.yaml`; release evidence
and exact source revisions are linked from the publication pull request.

## vLLM serving

Set `EXTRA_MODELS_DIR` to `models/demos` and select architecture
`TTGemma4A4BForCausalLM`. The model entrypoint is
`models.demos.gemma4_26b_a4b_qb2.tt.generator_vllm:Gemma4ForCausalLM`.

The precision policy is fixed by `config/precision.json`. Experiment logs and
bring-up artifacts remain in the Agentic Research repository.
