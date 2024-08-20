# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import ttnn
from models.helper_funcs import Linear
import tt_lib.fallback_ops as fallback_ops

import ttnn

import models.experimental.nanogpt.tt.nanogpt_block as nanogpt_block
from models.experimental.nanogpt.nanogpt_utils import unpad_from_zero

from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


class TtGPT(nn.Module):
    def __init__(self, config, device, tt_cache_path, dtype):
        super().__init__()

        assert config.vocab_size is not None

        self.config = config
        self.config.block_size = 1024
        base_address = f"transformer"
        self.device = device

        self.beta = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.bias" + str(dtype) + ".bin")

        self.gamma = ttnn.load_tensor(tt_cache_path + base_address + ".ln_f.weight" + str(dtype) + ".bin")

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, config.n_embd)

        self.wte.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wte.weight.pt"))

        self.wpe.weight = torch.nn.Parameter(torch.load(tt_cache_path + "transformer.wpe.weight.pt"))

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, f"{base_address}.h.{i}", self.device, tt_cache_path, dtype)
            blocks.append(block)

        self.h = nn.ModuleList(blocks)

        self.ln_f = ttnn.layer_norm

        tt_lm_weight = ttnn.load_tensor(tt_cache_path + "lm_head.weight" + str(dtype) + ".bin")

        weight = unpad_from_zero(tt_lm_weight, (1, 1, self.config.vocab_size, self.config.n_embd))
        weight_torch = weight
        weight = torch_to_tt_tensor_rm(weight, device=self.device)

        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, weight)

        self.wte.weight = nn.Parameter(weight_torch.squeeze())  # https://paperswithcode.com/method/weight-tying

    def forward(self, idx: torch.Tensor) -> ttnn.Tensor:
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = ttnn.arange(0, t, 1)
        pos = tt_to_torch_tensor(pos)
        pos = pos.squeeze(0).squeeze(0)
        pos = pos.to(dtype=torch.int64)

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)

        tt_tok_emb = torch_to_tt_tensor_rm(tok_emb, self.device)
        tt_pos_emb = torch_to_tt_tensor_rm(pos_emb, self.device)

        tt_tok_emb = ttnn.permute(tt_tok_emb, (0, 2, 1, 3))
        tt_pos_emb = ttnn.permute(tt_pos_emb, (0, 2, 1, 3))

        tt_x = ttnn.add(tt_tok_emb, tt_pos_emb)
        tt_tok_emb.deallocate()
        tt_pos_emb.deallocate()
        tt_x = ttnn.permute(tt_x, (0, 2, 1, 3))

        for block in self.h:
            tt_x = block.forward(tt_x)

        tt_x = self.ln_f(tt_x, epsilon=1e-5, weight=self.gamma, bias=self.beta)
        logits = self.lm_head(tt_x)

        return logits

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 20,
        temperature: int = 1.0,
        top_k=None,
    ) -> torch.Tensor:
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            tt_logits = self.forward(idx_cond)

            logits_shapes = tt_logits.get_legacy_shape()

            slice_list = [
                slice(None),
                slice(None),
                slice(logits_shapes[2] - 1, logits_shapes[2]),
                slice(None),
            ]
            tt_logits = fallback_ops.tensor_slice(tt_logits, slice_list)

            tt_temperature = fallback_ops.full(tt_logits.get_legacy_shape(), temperature)

            tt_temperature = ttnn.reciprocal(tt_temperature)
            tt_logits = ttnn.multiply(tt_logits, tt_temperature)

            logits = tt_to_torch_tensor(tt_logits)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # apply softmax to convert logits to (normalized) probabilities
            tt_logits = torch_to_tt_tensor_rm(logits, self.device, put_on_device=False)
            tt_probs = fallback_ops.softmax(tt_logits, dim=-1)
            probs = tt_to_torch_tensor(tt_probs)
            probs = probs.squeeze(0)
            probs = probs.squeeze(0)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
