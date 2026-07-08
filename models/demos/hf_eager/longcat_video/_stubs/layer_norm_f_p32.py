# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
#
# SPDX-License-Identifier: Apache-2.0

"""Native ttnn port of `layer_norm_f_p32` (meituan-longcat/LongCat-Video's
`dit.blocks.*.mod_norm_attn`, class `LayerNorm_FP32` in the vendored
`longcat_video/modules/blocks.py`):

    class LayerNorm_FP32(nn.LayerNorm):
        def forward(self, inputs):
            return F.layer_norm(inputs.float(), self.normalized_shape,
                                 None if self.weight is None else self.weight.float(),
                                 None if self.bias is None else self.bias.float(),
                                 self.eps).to(inputs.dtype)

Every call site in this checkpoint constructs it with `elementwise_affine=False`
(`mod_norm_attn`/`mod_norm_ffn`/`norm_final` all pass no weight/bias), so this
is a plain non-affine LayerNorm -- no learned parameters to load.
"""

from __future__ import annotations

import ttnn


class TtLayerNormFP32:
    def __init__(self, device: ttnn.Device, torch_module) -> None:
        self.device = device
        self.eps = torch_module.eps
        assert torch_module.weight is None and torch_module.bias is None, (
            "this checkpoint only ever constructs LayerNorm_FP32 with "
            "elementwise_affine=False; a weighted instance needs the "
            "weight/bias uploaded and passed to ttnn.layer_norm."
        )

    def __call__(self, inputs: ttnn.Tensor) -> ttnn.Tensor:
        return ttnn.layer_norm(inputs, epsilon=self.eps)


def build(device: ttnn.Device, torch_module) -> TtLayerNormFP32:
    return TtLayerNormFP32(device, torch_module)
