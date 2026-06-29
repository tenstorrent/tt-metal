# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""DiffusionGemma model subclass.

The DiffusionGemma text backbone IS Gemma-4 26B-A4B (shared weights), so this
subclass keeps the full Gemma4 graph and adds only what the diffusion denoise
loop needs without touching the shared model:

- ``_get_rope_mats`` gains a bounds check. The denoise pass requests
  ``seq_len = prompt_len + canvas_len``; a silent under-length RoPE slice would
  corrupt the canvas positions, so an out-of-range request is rejected instead.

The bidirectional / read-only denoise attention itself lives in
:mod:`models.experimental.diffusion_gemma.tt.diffusion_attention`; prefill-write
and commit-append are stock Gemma4 prefill / decode.
"""

from __future__ import annotations

from models.demos.gemma4.tt.model import Gemma4Model


class DiffusionGemma4Model(Gemma4Model):
    """Gemma4 backbone with DiffusionGemma denoise-friendly RoPE bookkeeping."""

    def _get_rope_mats(self, layer_idx, seq_len=None, for_decode=False):
        # Self-contained (does not call super) so the bounds guard is unit-testable
        # without a constructed model and stays correct if the base slicing changes.
        layer_type = self.hf_config.layer_types[layer_idx]
        if for_decode:
            return self.rope_caches_2d[layer_type]
        cos, sin = self.rope_caches[layer_type]
        if seq_len is not None:
            if seq_len > cos.shape[-2]:
                raise ValueError(f"requested RoPE seq_len {seq_len} exceeds cache length {cos.shape[-2]}")
            cos = cos[:, :, :seq_len, :]
            sin = sin[:, :, :seq_len, :]
        return (cos, sin)
