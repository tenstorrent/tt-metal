# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native TTNN port of `mlp` for FLUX.2-klein-9B's text encoder.

`mlp` and `m_l_p` are two discovery passes' names for the SAME module of this
checkpoint: both `_captured/mlp/manifest.json` and `_captured/m_l_p/manifest.json`
record `submodule_path = model.layers.0.mlp`, a `Qwen3MLP`. Giving it a second,
separately-maintained implementation would only create two ports that can
silently drift apart while both claim to model one module.

It therefore shares `TtMLP` — the native tensor-parallel port that already
graduated at TP=8. See `m_l_p.py` for the scheme: column-parallel gate/up over
the intermediate axis, row-parallel down with an all_reduce.
"""
from __future__ import annotations

from .m_l_p import TtMLP


# Module-level `build` — primary test entry point.
def build(device, torch_module=None):
    return TtMLP.build(device, torch_module)


# Module-level shim with the component's lowercase slug name, for legacy SMOKE/PCC tests.
def mlp(device, torch_module=None):
    return TtMLP.build(device, torch_module)
