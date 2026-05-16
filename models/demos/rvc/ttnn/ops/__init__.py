# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""TTNN operator wrappers for RVC model components."""

from models.demos.rvc.ttnn.ops.conv1d import TTNNConv1d
from models.demos.rvc.ttnn.ops.conv_transpose1d import TTNNConvTranspose1d
from models.demos.rvc.ttnn.ops.linear import TTNNLinear
from models.demos.rvc.ttnn.ops.layer_norm import TTNNLayerNorm
