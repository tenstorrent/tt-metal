# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `layer` for FLUX.2-klein-9B's text encoder.

`layer` and `decoder_layer` are two discovery passes' names for the SAME module
of this checkpoint: both `_captured/layer/manifest.json` and
`_captured/decoder_layer/manifest.json` record `submodule_path = model.layers.0`,
a `Qwen3DecoderLayer`. So this component is the same block, and giving it a
second, separately-maintained implementation would only create two ports that
can silently drift apart while both claim to model one module.

It therefore shares `TtDecoderLayer` — the native tensor-parallel port that
already graduated at TP=8. See `decoder_layer.py` for the scheme: replicated
residual stream, column-parallel q/k/v + row-parallel o_proj with an all_reduce,
column-parallel gate/up + row-parallel down with an all_reduce, norms replicated.
"""
from __future__ import annotations

from .decoder_layer import TtDecoderLayer


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def layer(device, torch_module=None):
    return TtDecoderLayer.build(device, torch_module)
