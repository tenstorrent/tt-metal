# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def rms_norm(config, input: ttnn.Tensor, *, parameters):
    return ttnn.rms_norm(input, parameters, epsilon=config.norm_eps)
