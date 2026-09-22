# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
# SPDX-License-Identifier: Apache-2.0

"""TTNN Chronos stubs."""

from models.experimental.chronos_forecast.tt.model import TtChronos
from models.experimental.chronos_forecast.tt.residual_block import (
    TtResidualBlock,
    TtResidualBlockWeights,
)
from models.experimental.chronos_forecast.tt.mha_core import TtMhaCore, TtMhaWeights
from models.experimental.chronos_forecast.tt.encoder_block import (
    TtEncoderBlock,
    TtEncoderBlockWeights,
)
from models.experimental.chronos_forecast.tt.group_attention import (
    TtGroupAttention,
    TtGroupAttentionWeights,
    build_group_mask,
)
from models.experimental.chronos_forecast.tt.time_attention import (
    TtTimeAttention,
    TtTimeAttentionWeights,
    build_rope_cache,
)
from models.experimental.chronos_forecast.tt.model_preprocessing import (
    Chronos2PackedInputs,
    Chronos2PatchedInputs,
    encode_categorical_covariate,
    instance_norm,
    instance_norm_inverse,
    normalize_chronos2_inputs,
    patch,
    patch_chronos2_inputs,
    prepare_chronos2_inputs,
    prepare_patched_context,
    prepare_patched_future,
    preprocess_model_parameters,
    target_encode,
)

__all__ = [
    "Chronos2PackedInputs",
    "Chronos2PatchedInputs",
    "TtChronos",
    "TtEncoderBlock",
    "TtEncoderBlockWeights",
    "TtGroupAttention",
    "TtGroupAttentionWeights",
    "TtMhaCore",
    "TtMhaWeights",
    "TtResidualBlock",
    "TtResidualBlockWeights",
    "TtTimeAttention",
    "TtTimeAttentionWeights",
    "build_group_mask",
    "build_rope_cache",
    "encode_categorical_covariate",
    "instance_norm",
    "instance_norm_inverse",
    "normalize_chronos2_inputs",
    "patch",
    "patch_chronos2_inputs",
    "prepare_chronos2_inputs",
    "prepare_patched_context",
    "prepare_patched_future",
    "preprocess_model_parameters",
    "target_encode",
]
