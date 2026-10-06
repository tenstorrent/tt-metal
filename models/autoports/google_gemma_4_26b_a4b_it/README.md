# Gemma 4 26B A4B autoport

This directory contains the alternate TP4 Blackhole implementation of
`google/gemma-4-26B-A4B-it`. It is intentionally isolated from the canonical
`models/demos/gemma4` implementation and does not replace its defaults.

The vLLM entrypoint is
`tt.generator_vllm:AutoportGemma4ForCausalLM`. Select it through the
inference-server `gemma4-autoport` implementation profile, which registers the
entrypoint for that launch with `TT_MODEL_CLASS_OVERRIDES`. No vLLM plugin
source fork is required.

Runtime source is under `tt/`. The only checked-in generated policy is
`doc/datatype_sweep/selected_precision_config.json`; experiment logs and
bringup artifacts belong in agentic-research rather than tt-metal.
