# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The dense FFN used on even-numbered layers, the TTNN form of reference.NomicBertMLP."""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import to_device, transpose_linear_weight


class TtNomicBertMLP(LightweightModule):
    """H -> F -> H with an exact-erf GELU between the two projections.

    ttnn.gelu defaults to the accurate variant, which is what the reference's
    nn.GELU(approximate="none") needs. fast_and_approximate_mode selects a LUT whose error
    (2.34e-2) exceeds the bfloat16 noise floor (1.58e-2), so it is not hidden by the dtype;
    the repo's BERT idiom fused_activation=(ttnn.UnaryOpType.GELU, True) picks that LUT.

    Token-wise, so it runs on the (B, 1, S, H) block layout untouched: the weights are 2D and
    broadcast over both leading axes.
    """

    def __init__(self, device, config, tt_config, state_dict, state_dict_prefix):
        super().__init__()
        self.tt_config = tt_config

        def weight(name):
            return to_device(
                transpose_linear_weight(state_dict[f"{state_dict_prefix}{name}.weight"]),
                device,
                dtype=tt_config.weight_dtype,
            )

        def bias(name):
            return to_device(state_dict[f"{state_dict_prefix}{name}.bias"], device, dtype=tt_config.weight_dtype)

        self.fc1_weight, self.fc1_bias = weight("fc1"), bias("fc1")
        self.fc2_weight, self.fc2_bias = weight("fc2"), bias("fc2")

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Widen, activate, project back.

        Args:
            x: (B, 1, S, H) activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H), via (B, 1, S, F) at the activation.
        """
        hidden = ttnn.linear(
            x,
            self.fc1_weight,
            bias=self.fc1_bias,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        activated = ttnn.gelu(hidden)
        ttnn.deallocate(hidden)

        out = ttnn.linear(
            activated,
            self.fc2_weight,
            bias=self.fc2_bias,
            compute_kernel_config=self.tt_config.compute_kernel_config,
        )
        ttnn.deallocate(activated)
        return out
