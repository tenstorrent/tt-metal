# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import tt_lib
import torch
import torch.nn as nn
from models.experimental.mistral.tt.mistral_configuration import TtModelArgs
from models.experimental.mistral.tt.mistral_transformer_block import TtTransformerBlock
from models.experimental.mistral.tt.mistral_rms_norm import TtRMSNorm
from models.experimental.mistral.mistral_helper_funcs import (
    Linear as TtLinear,
    format_tensor,
    unpad_from_zero,
    get_freqs_cis,
)
from models.utility_functions import torch_to_tt_tensor_rm
from typing import Optional


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


class TtTransformer(nn.Module):
    def __init__(
        self,
        args: TtModelArgs,
        device=None,
        base_address=None,
        tt_cache_path=None,
    ):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.device = device
        self.base_address = base_address
        assert self.vocab_size > 0

        embedding_weights = torch.load(tt_cache_path + "tok_embeddings.weight.pt")
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim, _weight=embedding_weights)
        self.output_mem_config = tt_lib.tensor.MemoryConfig(
            tt_lib.tensor.TensorMemoryLayout.INTERLEAVED, tt_lib.tensor.BufferType.DRAM
        )
        self.layers = torch.nn.ModuleList(
            [
                TtTransformerBlock(
                    args=args,
                    base_address=f"layers.{i}.",
                    device=self.device,
                    tt_cache_path=tt_cache_path,
                    output_mem_config=self.output_mem_config,
                )
                for i in range(args.n_layers)
            ]
        )
        self.norm = TtRMSNorm(
            args.dim,
            base_address=f"norm.",
            eps=args.norm_eps,
            tt_cache_path=tt_cache_path,
            device=self.device,
            output_mem_config=self.output_mem_config,
        )

        self.output_weight = tt_lib.tensor.load_tensor(
            tt_cache_path + "output.weight" + str(self.args.WEIGHTS_DTYPE) + ".bin"
        )
        self.output = TtLinear(
            args.dim,
            args.vocab_size,
            self.output_weight,
        )
        self.freqs_cis = precompute_freqs_cis(self.args.head_dim, 128_000)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        seqlen = input_ids.shape[-1]
        bsz = input_ids.shape[0]
        h = self.tok_embeddings(input_ids)
        input_ids = torch_to_tt_tensor_rm(input_ids, self.device, put_on_device=False)
        freqs_cis = self.freqs_cis[positions]
        query_shape = [bsz, seqlen, self.args.n_heads, self.args.head_dim // 2]
        key_shape = [bsz, seqlen, self.args.n_kv_heads, self.args.head_dim // 2]
        bcast_freq_xq, bcast_freq_xk = get_freqs_cis(
            freqs_cis, query_shape, key_shape, self.device, self.output_mem_config
        )

        mask: Optional[torch.Tensor] = None
        if input_ids.get_legacy_shape()[-1] > 1:
            seqlen = input_ids.get_legacy_shape()[-1]
            tensor = tt_lib.tensor.full(
                (1, 1, seqlen, seqlen),
                fill_value=1.0,
            )
            diagonal = 0

            mask = tt_lib.tensor.tril(tensor, diagonal)
            tensor.deallocate()
            # make the mask banded to account for sliding window
            diagonal = -self.args.sliding_window
            mask = tt_lib.tensor.triu(mask, diagonal)
            mask = tt_lib.tensor.log(mask)
            mask = format_tensor(mask, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config, pad_value=-10000)

        positions = torch_to_tt_tensor_rm(positions, self.device, put_on_device=False)
        h = torch_to_tt_tensor_rm(h, self.device, put_on_device=False)
        h = format_tensor(h, tt_lib.tensor.Layout.TILE, self.device, self.output_mem_config)
        for layer in self.layers:
            h = layer(h, bcast_freq_xq, bcast_freq_xk, positions, mask, seqlen)

        bcast_freq_xq.deallocate()
        bcast_freq_xk.deallocate()
        output = self.output(self.norm(h))
        desired_output_shape = list(output.get_legacy_shape())
        desired_output_shape[2] = seqlen
        output = unpad_from_zero(output, desired_output_shape)
        output = torch_to_tt_tensor_rm(output, self.device, put_on_device=False)
        return output
