# GPT-OSS 20B optimized decoder

> 🗄️ **OLD / SUPERSEDED EXPERIMENT (2026-07-25).** This is the earlier *single-device* autoport approach:
> it optimizes the decoder derived **from `tt/functional_decoder.py`**. It is **NOT** the current
> prettify→optimize work. The current optimize runs on the *prettified multichip* model on ttnn-models
> branch `mvasiljevic/gpt-oss-optimize` (driven by `.agents/prompts/alchemy_optimize/`). See the autoport
> top-level `../../README.md` for the distinction. Kept here for history only.

This stage optimizes the single-device decoder layer derived from
`tt/functional_decoder.py`. There was no optional fused-decoder artifact in
this checkout, so the functional decoder is the source baseline. The scope is
one 1x1 Blackhole mesh decoder layer; no multichip, full-model, generator, or
vLLM implementation is included.

Evidence and final selected configuration will be recorded here as the stage
progresses.

