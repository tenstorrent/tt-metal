# GPT-OSS 20B optimized decoder

This stage optimizes the single-device decoder layer derived from
`tt/functional_decoder.py`. There was no optional fused-decoder artifact in
this checkout, so the functional decoder is the source baseline. The scope is
one 1x1 Blackhole mesh decoder layer; no multichip, full-model, generator, or
vLLM implementation is included.

Evidence and final selected configuration will be recorded here as the stage
progresses.

