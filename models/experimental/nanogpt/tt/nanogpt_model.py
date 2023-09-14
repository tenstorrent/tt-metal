# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
import tt_lib
from models.helper_funcs import Linear

import models.experimental.nanogpt.tt.nanogpt_block as nanogpt_block
import tt_lib.fallback_ops as fallback_ops
from models.utility_functions import (
    torch_to_tt_tensor_rm,
    tt_to_torch_tensor,
)


class TtGPT(nn.Module):
    def __init__(self, config, state_dict, device):
        super().__init__()

        assert config.vocab_size is not None

        self.config = config
        self.config.block_size = 1024
        base_address = f"transformer"
        self.device = device
        self.beta = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ln_f.bias"], self.device)

        self.gamma = torch_to_tt_tensor_rm(state_dict[f"{base_address}.ln_f.weight"], self.device)

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(self.config.block_size, config.n_embd)

        self.wte.weight = torch.nn.Parameter(state_dict[f"{base_address}.wte.weight"])

        self.wpe.weight = torch.nn.Parameter(state_dict[f"{base_address}.wpe.weight"])

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, state_dict, f"{base_address}.h.{i}", self.device)
            blocks.append(block)

        self.h = nn.ModuleList(blocks)

        self.ln_f = tt_lib.tensor.layernorm

        # Push weights to Tt device
        tt_lm_weight = torch_to_tt_tensor_rm(state_dict["lm_head.weight"], self.device)

        self.lm_head = Linear(self.config.n_embd, self.config.vocab_size, tt_lm_weight)

        self.wte.weight = nn.Parameter(state_dict["lm_head.weight"])  # https://paperswithcode.com/method/weight-tying

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

            logits_shapes = tt_logits.shape()

            slice_list = [
                slice(None),
                slice(None),
                slice(logits_shapes[2] - 1, logits_shapes[2]),
                slice(None),
            ]
            tt_logits = fallback_ops.tensor_slice(tt_logits, slice_list)

            tt_temperature = fallback_ops.full(tt_logits.shape(), temperature)

            tt_temperature = tt_lib.tensor.recip(tt_temperature)
            tt_logits = tt_lib.tensor.mul(tt_logits, tt_temperature)

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

    def forward(self, idx: torch.Tensor) -> tt_lib.tensor.Tensor:
        b, t = idx.shape
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0)  # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos)  # position embeddings of shape (1, t, n_embd)

        tt_tok_emb = torch_to_tt_tensor_rm(tok_emb, self.device)
        tt_pos_emb = torch_to_tt_tensor_rm(pos_emb, self.device)

        tt_x = tt_lib.tensor.add(tt_tok_emb, tt_pos_emb)

        for block in self.h:
            tt_x = block.forward(tt_x)

        tt_x = self.ln_f(tt_x, eps=1e-5, gamma=self.gamma, beta=self.beta)
        logits = self.lm_head(tt_x)

        return logits
