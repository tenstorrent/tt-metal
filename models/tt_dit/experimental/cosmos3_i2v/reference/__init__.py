# SPDX-FileCopyrightText: © 2025 The NVIDIA Team and The HuggingFace Team
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Vendored Cosmos3 reference implementation from huggingface/diffusers main.

These files (`transformer_cosmos3.py`, `pipeline_cosmos3_omni.py`) are
copied verbatim from diffusers' main branch with only the relative
imports rewritten to absolute (so they work against the diffusers 0.35.1
that tt-metal pins) and the optional `Cosmos3AVAEAudioTokenizer` import
made lazy (the audio tokenizer doesn't exist in 0.35.1 and isn't needed
for I2V).

Upstream sources:
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/transformers/transformer_cosmos3.py
- https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/cosmos/pipeline_cosmos3_omni.py

Vendored on 2026-06-10. When diffusers releases stable Cosmos3 support,
delete this directory and switch back to the upstream import path.
"""
