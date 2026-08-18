# SPDX-FileCopyrightText: © 2025 The NVIDIA Team and The HuggingFace Team
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored Cosmos3 reference implementation from huggingface/diffusers main.

These files (`transformer_cosmos3.py`, `pipeline_cosmos3_omni.py`,
`autoencoder_cosmos3_audio.py`) are copies from diffusers' main branch with
local modifications: relative imports rewritten to absolute (so they work
against the diffusers 0.38.0 that tt-metal pins — the Cosmos3 modules first
ship in the v0.39.0 release), the optional `Cosmos3AVAEAudioTokenizer` import
made lazy (not needed for I2V), and the `TT_COSMOS3_VE_FULL_ENCODE` hook in
the pipeline.

Upstream sources:
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos3.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/autoencoders/autoencoder_cosmos3_audio.py

Vendored on 2026-06-10. Once the repo-wide diffusers pin reaches >= 0.39.0,
de-vendor: delete this directory, import from diffusers, and re-apply the
local modifications as wrappers.
"""
