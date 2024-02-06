# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import ttnn
import torch
import tt_lib
from models.experimental.functional_mistral.tt.mistral_helper_funcs import get_freqs_cis
from models.experimental.functional_mistral.tt.ttnn_functional_rms_norm import rms_norm
from models.experimental.functional_mistral.tt.ttnn_functional_transformer_block import transformer_block
from models.utility_functions import tt_to_torch_tensor
from typing import Optional


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def mistral_transformer(config, input_ids, positions, parameters, device, mem_config):
    seqlen = input_ids.shape[-1]
    bsz = input_ids.shape[0]
    input_ids = ttnn.to_device(input_ids, device)
    h = ttnn.embedding(input_ids, parameters.tok_embeddings.weight)
    h = ttnn.to_layout(h, layout=ttnn.TILE_LAYOUT)
    freq_cis = precompute_freqs_cis(config.head_dim, 128_000)
    freqs_cis = freq_cis[positions]
    query_shape = [bsz, seqlen, config.n_heads, config.head_dim // 2]
    key_shape = [bsz, seqlen, config.n_kv_heads, config.head_dim // 2]
    print("Query key shape :", query_shape, " ", key_shape)
    bcast_freq_xq, bcast_freq_xk = get_freqs_cis(freqs_cis, query_shape, key_shape, device, mem_config)

    mask: Optional[torch.Tensor] = None
    if input_ids.shape[-1] > 1:
        seqlen = input_ids.shape[-1]
        tensor = tt_lib.tensor.full((1, 1, seqlen, seqlen), fill_value=1.0)
        diagonal = 0

        mask = tt_lib.tensor.tril(tensor, diagonal)
        tensor.deallocate()

        diagonal = -config.sliding_window
        mask = tt_lib.tensor.triu(mask, diagonal)
        mask = tt_to_torch_tensor(mask).squeeze(0).squeeze(0)
        mask = ttnn.from_torch(mask, dtype=ttnn.bfloat16, device=device, layout=ttnn.TILE_LAYOUT)
        mask = ttnn.log(mask)
        mask = ttnn.to_layout(
            ttnn.pad(mask, ((0, 32 - mask.shape[-2] % 32), (0, 32 - mask.shape[-1] % 32)), value=-10000),
            layout=ttnn.TILE_LAYOUT,
        )

    positions = ttnn.from_torch(positions, dtype=ttnn.bfloat16)

    for params in parameters.layers:
        h = transformer_block(
            config, h, bcast_freq_xq, bcast_freq_xk, positions, mask, seqlen, params, device, mem_config
        )

    bcast_freq_xq.deallocate()
    bcast_freq_xk.deallocate()

    output_norm = rms_norm(config, input=h, parameters=parameters.norm.weight)
    output = output_norm @ parameters.output.weight
    return output
