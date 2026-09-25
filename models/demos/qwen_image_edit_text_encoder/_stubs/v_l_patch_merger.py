# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native, tensor-parallel TTNN port of `Qwen2_5_VLPatchMerger` (`model.visual.merger`):
RMSNorm(1280) -> merge 4 consecutive tokens (5120) -> Linear -> GELU -> Linear(3584).
fc1 column-parallel, fc2 row-parallel + all_reduce over TP; norm and fc2 bias replicated.
"""

from __future__ import annotations

import ttnn
from models.demos.qwen_image_edit_text_encoder._stubs.attention import pad_to_tile
from models.demos.qwen_image_edit_text_encoder._stubs.encoder_stack import TtPatchMerger


class TtVLPatchMerger:
    def __init__(self, device, torch_module):
        self.merger = TtPatchMerger(device, torch_module)
        self.unit = self.merger.hidden // torch_module.ln_q.weight.shape[0]

    def forward_padded(self, x):
        """Device path used by the vision tower: [N, 1, s_pad, C] -> [N, 1, s_pad // unit, out]."""
        return self.merger(x)

    def __call__(self, x, **kwargs):
        s, c = x.shape
        s_pad = pad_to_tile(s)
        x = ttnn.reshape(x, (1, 1, s, c))
        if s_pad != s:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, s_pad - s), (0, 0)], 0.0)
        out = self.merger(x)  # [1, 1, s_pad // unit, out]
        m = s // self.unit
        out = ttnn.slice(out, [0, 0, 0, 0], [1, 1, m, out.shape[-1]])
        return ttnn.reshape(out, (m, out.shape[-1]))


def build(device, torch_module=None):
    return TtVLPatchMerger(device, torch_module)


def v_l_patch_merger(device, torch_module=None):
    return build(device, torch_module)
