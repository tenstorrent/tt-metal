# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native gate arithmetic with an optional compact user-row layout."""


def from_packed(packed, a_neg, dt_bias, *, compact=False):
    import ttnn

    if type(compact) is not bool:
        raise ValueError("compact gates must be Boolean")
    if len(packed.shape) != 3 or packed.shape[0] != 1 or not 1 <= packed.shape[1] <= 32 or packed.shape[2] != 4160:
        raise ValueError("Compact GDN projection must have shape [1,B,4160], B=1..32")
    batch = packed.shape[1]
    gates = packed[:, :, 4096:4160]
    if not compact:
        gates = ttnn.reshape(gates, [batch, 1, 64])
    beta = ttnn.sigmoid(gates[:, :, :12])
    a = ttnn.typecast(gates[:, :, 32:44], ttnn.float32)
    log_decay = ttnn.mul(
        a_neg,
        ttnn.add(a, dt_bias),
        input_tensor_b_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.SOFTPLUS, 1.0, 20.0)],
    )
    return log_decay, beta
