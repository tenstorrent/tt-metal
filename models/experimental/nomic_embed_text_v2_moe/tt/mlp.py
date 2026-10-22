# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The dense FFN used on even-numbered layers, the TTNN form of reference.NomicBertMLP."""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import dense_linear
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup


class TtNomicBertMLP(LightweightModule):
    """H -> F -> H with a GELU between the two projections, tt_config.dense_gelu.

    The reference runs exact erf, nn.GELU(approximate="none"). The tanh form is within 4.7e-4 of
    it, a thirtieth of bfloat16's own rounding, and runs 18% faster (171 against 208 us at 8x512).
    fast_and_approximate_mode selects a LUT whose error (2.34e-2) exceeds the bfloat16 noise
    floor (1.58e-2), so it is not hidden by the dtype; the repo's BERT idiom
    fused_activation=(ttnn.UnaryOpType.GELU, True) picks that LUT. Above 32 tile rows of M the GELU
    is fused into fc1 (matmul_config.dense_linear), where the tanh form runs from the packer beside
    the matmul: 260 us for both at 8x512, against 132 + 171 unfused.

    Token-wise, so it runs on the (B, 1, S, H) block layout untouched: the weights are 2D and
    broadcast over both leading axes.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config

        def weight(name, group):
            return to_device(
                transpose_linear_weight(state_dict[f"{state_dict_prefix}{name}.weight"]),
                device,
                dtype=tt_config.matmul_weight_dtype(group),
            )

        def bias(name):
            return to_device(state_dict[f"{state_dict_prefix}{name}.bias"], device, dtype=tt_config.weight_dtype)

        self.fc1_weight, self.fc1_bias = weight("fc1", OpGroup.FC1), bias("fc1")
        self.fc2_weight, self.fc2_bias = weight("fc2", OpGroup.FC2), bias("fc2")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Widen, activate, project back.

        Args:
            x: (B, 1, S, H) activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H), via (B, 1, S, F) at the activation.
        """
        activated = dense_linear(
            x, self.fc1_weight, self.fc1_bias, OpGroup.FC1, self.tt_config, gelu=self.tt_config.dense_gelu
        )
        out = dense_linear(activated, self.fc2_weight, self.fc2_bias, OpGroup.FC2, self.tt_config)
        ttnn.deallocate(activated)
        return out
