# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn


def rms_norm(config, input: ttnn.Tensor, *, weight=None, bias=None):
    return ttnn.rms_norm(input, weight=weight, bias=bias, epsilon=config.norm_eps)
