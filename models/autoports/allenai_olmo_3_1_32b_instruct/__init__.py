# Autoport of allenai/Olmo-3.1-32B-Instruct for Tenstorrent Blackhole. The model implementation lives in
# models/tt_transformers (post-norm decoder, sliding/full hybrid attention, full-width QK-norm, YaRN on the
# full-attention layers); this package carries the vLLM plugin bundle, the tt-model-manager manifest and the
# bring-up evidence. See README.md.
