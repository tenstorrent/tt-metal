# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
from models.experimental.functional_mistral.tt.ttnn_functional_rms_norm import rms_norm
from models.experimental.functional_mistral.tt.ttnn_functional_attention import attention
from models.experimental.functional_mistral.tt.ttnn_functional_feed_forward import feed_forward


def transformer_block(
    config,
    x,
    bcast_freq_xq: tt_lib.tensor.complex_tensor,
    bcast_freq_xk: tt_lib.tensor.complex_tensor,
    positions,
    mask,
    seqlen,
    parameter,
    device,
    memory_config,
):
    r = attention(
        config,
        rms_norm(config, input=x, parameters=parameter.attention_norm.weight),
        bcast_freq_xq,
        bcast_freq_xk,
        positions,
        mask,
        seqlen,
        parameter.attention,
        device,
        memory_config,
    )
    h = x + r
    r = feed_forward(
        config, x=rms_norm(config, input=x, parameters=parameter.ffn_norm.weight), parameters=parameter.feed_forward
    )

    return h + r
