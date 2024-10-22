# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

#################################################################################################################
# Link: https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
# NOTE: changed the device from CUDA to CPU and dtype to float32
#################################################################################################################

# coding=utf-8
# Copyright 2023 Mistral AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Optional, Tuple


@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    norm_eps: float
    vocab_size: int

    max_batch_size: int = 0


def repeat_kv(keys: torch.Tensor, values: torch.Tensor, repeats: int):
    keys = torch.repeat_interleave(keys, repeats=repeats, dim=2)
    values = torch.repeat_interleave(values, repeats=repeats, dim=2)
    return keys, values


def _reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    freqs_cis: complex - (seq_len, head_dim / 2)
    x: complex - (bsz, seq_len, head_dim / 2)
    """
    ndim = x.ndim
    assert 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        freqs_cis.shape,
        (x.shape[1], x.shape[-1]),
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    The algorithm sources from Qwen2 codes in the Transformer library: https://github.com/huggingface/transformers/blob/1bd604d11c405dfb8b78bda4062d88fc75c17de0/src/transformers/models/qwen2/modeling_qwen2.py#L182.
    """
    xq_ = torch.complex(xq.float()[..., : xq.shape[-1] // 2], xq.float()[..., xq.shape[-1] // 2 :])
    xk_ = torch.complex(xk.float()[..., : xk.shape[-1] // 2], xk.float()[..., xk.shape[-1] // 2 :])
    freqs_cis = _reshape_for_broadcast(freqs_cis, xq_)
    xq_ = xq_ * freqs_cis
    xk_ = xk_ * freqs_cis
    xq_out = torch.cat((torch.real(xq_), torch.imag(xq_)), dim=-1)
    xk_out = torch.cat((torch.real(xk_), torch.imag(xk_)), dim=-1)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class Emb(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_id):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, hidden_size, pad_id)

    def forward(self, x):
        return self.emb(x)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.n_heads: int = args.n_heads
        self.n_kv_heads: int = args.n_kv_heads

        self.repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = self.args.sliding_window

        self.scale = self.args.head_dim**-0.5

        self.q_proj = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
        self.k_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.v_proj = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
        self.o_proj = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)
        self.cache_k = torch.empty(
            args.max_batch_size,
            args.sliding_window,
            self.n_kv_heads,
            self.args.head_dim,
            dtype=torch.bfloat16,
        )
        self.cache_v = torch.empty(
            args.max_batch_size,
            args.sliding_window,
            self.n_kv_heads,
            self.args.head_dim,
            dtype=torch.bfloat16,
        )

        self.to(torch.bfloat16)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x.to(torch.bfloat16)

        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seqlen, self.n_heads, self.args.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.args.head_dim)
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
        # The cache is a rotating buffer
        scatter_pos = (positions[-self.sliding_window :] % self.sliding_window)[None, :, None, None]
        scatter_pos = scatter_pos.repeat(bsz, 1, self.n_kv_heads, self.args.head_dim)
        self.cache_k[:bsz].scatter_(dim=1, index=scatter_pos, src=xk[:, -self.sliding_window :])
        self.cache_v[:bsz].scatter_(dim=1, index=scatter_pos, src=xv[:, -self.sliding_window :])

        # positions = [0].. 1D [0,1]
        # src = [32,1,8,64]
        # index = [32,1,8,64] -> [None, -1, :, :] [32,1,8,64] + [32,1,8,64]
        # self = [32,4096,8,64]

        if positions.shape[0] > 1:
            # prefill
            key, value = repeat_kv(xk, xv, self.repeats)
        else:
            cur_pos = positions[-1].item() + 1
            key, value = repeat_kv(self.cache_k[:bsz, :cur_pos, ...], self.cache_v[:bsz, :cur_pos, ...], self.repeats)

        query = xq.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        # scores : [bsz, n_heads, seqlen | 1, seqlen]
        scores = torch.matmul(query, key.transpose(2, 3)) * self.scale

        if mask is not None:
            scores += mask[None, None, ...]

        scores = scores.float()
        scores = nn.functional.softmax(scores, dim=-1).type_as(query)
        output = torch.matmul(scores, value)  # (bs, n_local_heads, slen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.gate_proj = nn.Linear(args.dim, args.hidden_dim, bias=False)
        self.down_proj = nn.Linear(args.hidden_dim, args.dim, bias=False)
        self.up_proj = nn.Linear(args.dim, args.hidden_dim, bias=False)

        self.to(torch.bfloat16)

    def forward(self, x) -> torch.Tensor:
        x = x.to(torch.bfloat16)

        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

        self.to(torch.bfloat16)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = x.to(torch.bfloat16)

        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.self_attn = Attention(args)  # attention
        self.mlp = FeedForward(args=args)  # feed_forward
        self.input_layernorm = RMSNorm(args.dim, eps=args.norm_eps)  # attention_norm
        self.post_attention_layernorm = RMSNorm(args.dim, eps=args.norm_eps)  # ffn_norm
        self.args = args

        self.to(torch.bfloat16)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, positions: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        x = x.to(torch.bfloat16)

        r = self.self_attn.forward(self.input_layernorm(x), freqs_cis, positions, mask)
        h = x + r
        r = self.mlp.forward(self.post_attention_layernorm(h))
        out = h + r
        return out


class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        assert self.vocab_size > 0

        self.layers = torch.nn.ModuleList([TransformerBlock(args=args) for _ in range(args.n_layers)])

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)

        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # Convert the data type to bfloat16 to shrink the bit width, allowing this reference model to fit within the CPU's memory limit. Without this conversion, the Qwen2-7B model will consume up to 28GB memory for its float32 weights.
        # Additionally, the official Qwen2-7B model is expected to use bfloat16 for inference. Refer to https://huggingface.co/Qwen/Qwen2-7B/blob/main/config.json for the `torch_dtype` setting.
        self.to(torch.bfloat16)

    def forward(self, input_ids: torch.Tensor, freqs_cis_i, positions: torch.Tensor, mode="decode"):
        h = input_ids.to(torch.bfloat16)  # self.tok_embeddings(input_ids)
        freqs_cis = freqs_cis_i  # [positions]

        mask: Optional[torch.Tensor] = None
        if input_ids.shape[1] > 1:
            seqlen = input_ids.shape[1]
            tensor = torch.full(
                (seqlen, seqlen),
                dtype=h.dtype,
                fill_value=1,
                device=h.device,
            )
            mask = torch.tril(tensor, diagonal=0).to(h.dtype)
            # make the mask banded to account for sliding window
            mask = torch.triu(mask, diagonal=-self.args.sliding_window)
            mask = torch.log(mask)
        for layer in self.layers:
            h = layer(h, freqs_cis, positions, mask)
        if mode == "prefill":
            return h.float()
        return self.lm_head(self.norm(h)).float()

    # FIXME(cthsieh): Don't know why needs this.
    """
    def from_folder(
        folder: Path,
        n_layers: int = 1,
        max_batch_size: int = 1,
        device="cpu",
        dtype=torch.float32,
        is_whole_model: bool = True,
    ):
        with open(folder / "params.json", "r") as f:
            model_args = ModelArgs(**json.loads(f.read()))
        model_args.max_batch_size = max_batch_size
        model_args.n_layers = n_layers
        state_dict = torch.load(folder / "consolidated.00.pth")
        if not is_whole_model:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if (
                    k.startswith("layers.0")
                    or k.startswith("tok_embeddings")
                    or k.startswith("norm.weight")
                    or k.startswith("output.weight")
                )
            }
        model = Transformer(model_args).to(device=device, dtype=dtype)
        model.load_state_dict(state_dict)

        return model
    """
