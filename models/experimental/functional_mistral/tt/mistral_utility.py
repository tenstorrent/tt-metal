# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
from models.experimental.mistral.reference.model import RMSNorm


def custom_preprocessor(model, name):
    parameters = {}
    if isinstance(model, RMSNorm):
        parameters["weight"] = ttnn.from_torch(model.weight, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    return parameters
