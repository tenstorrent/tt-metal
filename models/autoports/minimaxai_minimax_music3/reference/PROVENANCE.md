# Provenance

`dit.py`, `depth_decoder.py`, `condition_encoder.py`, `vocoder.py` are ports of
`src/diffusers/models/{transformers/transformer_minimax_music3.py, transformers/minimax_music3_rvq_depth_decoder.py,
condition_embedders/condition_embedder_minimax_music3.py, autoencoders/minimax_music3_vocoder.py}` and
`prompt.py` / `sampling.py` / `pipeline.py` port `src/diffusers/modular_pipelines/minimax_music3/*.py` from
huggingface/diffusers commit `dafe3733fcfdbf3c48915fe77be3aef65b5d6a2d` (PR #14456, "MiniMax Music 3"), Copyright 2026
The MiniMax Team and The HuggingFace Team, Apache License 2.0. Changes: diffusers mixins/config plumbing replaced by
plain `torch.nn.Module`s that load the HF safetensors sub-folders directly (state-dict keys unchanged); the modular
pipeline blocks are flattened into `Music3Reference.generate()` and a `teacher_forced()` dump path. Numerics are
unchanged; stage 01 asserts the vendored pipeline reproduces the diffusers pipeline's codes for the same seed.
