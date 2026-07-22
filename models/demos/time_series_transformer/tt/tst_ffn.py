# tt/tst_ffn.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.

"""Feed-forward block, shared by encoder and decoder layers."""

import ttnn


def tst_ffn(hidden_states, w):
    """hidden_states: ttnn tensor [B, T, padded_width]."""
    ffn = ttnn.linear(hidden_states, w["fc1_weight"], bias=w["fc1_bias"], activation="gelu")
    return ttnn.linear(ffn, w["fc2_weight"], bias=w["fc2_bias"])
