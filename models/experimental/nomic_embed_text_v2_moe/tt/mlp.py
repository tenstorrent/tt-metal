# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The dense FFN used on even-numbered layers, the TTNN form of reference.NomicBertMLP."""

from __future__ import annotations

import ttnn

from models.common.lightweightmodule import LightweightModule
from models.experimental.nomic_embed_text_v2_moe.tt.common import (
    sharded_norm_layout,
    to_device,
    transpose_linear_weight,
)
from models.experimental.nomic_embed_text_v2_moe.tt.matmul_config import (
    SMALL_M_TILES,
    dense_linear,
    grouped_projection,
    prepare_grouped_projection,
)
from models.experimental.nomic_embed_text_v2_moe.tt.model_config import OpGroup


class TtNomicBertMLP(LightweightModule):
    """H -> F -> H with a GELU between the two projections, tt_config.dense_gelu.

    The reference runs exact erf, nn.GELU(approximate="none"). fc1 runs as grouped_matmul with one
    expert (fc1_grouped), which computes the GELU from the pack thread beside the matmul in the form
    tt_config.dense_gelu_form names, its polynomial within 1.3e-3 of erf. fast_and_approximate_mode
    selects a LUT whose error (2.34e-2) exceeds the bfloat16 noise floor (1.58e-2), so it is not
    hidden by the dtype; the repo's BERT idiom fused_activation=(ttnn.UnaryOpType.GELU, True) picks
    that LUT.

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
        # fc1 as one expert of grouped_matmul, beside the TILE weight dense_linear reads up to
        # tt_config.dense_fc1_grouped_min_tiles tile rows.
        self.fc1_chunked = prepare_grouped_projection(self.fc1_weight, self.fc1_bias, device)

    def fc1_grouped(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """gelu(x @ fc1 + bias) through grouped_matmul with one expert (grouped_projection).

        grouped_matmul computes the GELU from the pack thread in the form tt_config.dense_gelu_form
        names (2, its polynomial, |error| <= 1.3e-3) and writes the intermediate fc2 reads in
        tt_config.dense_intermediate_dtype.
        """
        return grouped_projection(
            x,
            self.fc1_chunked,
            self.fc1_weight,
            OpGroup.FC1,
            self.tt_config,
            activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU_TANH, float(self.tt_config.dense_gelu_form)),
            dtype=self.tt_config.dense_intermediate_dtype,
        )

    def forward(self, x: ttnn.Tensor) -> ttnn.Tensor:
        """Widen, activate, project back.

        Args:
            x: (B, 1, S, H) activations.

        Returns:
            ttnn.Tensor: (B, 1, S, H), via (B, 1, S, F) at the activation.
        """
        m_tiles = x.shape[0] * ttnn.core.divup(x.shape[-2], ttnn.TILE_SIZE)
        if m_tiles > self.tt_config.dense_fc1_grouped_min_tiles:
            activated = self.fc1_grouped(x)
        else:
            activated = dense_linear(
                x, self.fc1_weight, self.fc1_bias, OpGroup.FC1, self.tt_config, gelu=self.tt_config.dense_gelu
            )
        # Above SMALL_M_TILES fc2 (pirotski_matmul) writes norm2's split layout, bit for bit its interleaved output;
        # at or below it the small-M norm-layout program would change fc2's accumulation order, so the norm converts.
        norm_layout = None
        if m_tiles > SMALL_M_TILES:
            norm_layout = sharded_norm_layout(x.shape, self.tt_config.core_grid)
        out = dense_linear(
            activated, self.fc2_weight, self.fc2_bias, OpGroup.FC2, self.tt_config, live=(x,), norm_layout=norm_layout
        )
        ttnn.deallocate(activated)
        return out
