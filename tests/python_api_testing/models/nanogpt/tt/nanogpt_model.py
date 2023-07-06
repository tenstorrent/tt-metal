import torch
from torch.nn import functional as F
import torch.nn as nn
import tt_lib
import python_api_testing.models.nanogpt.helper_funcs as nanogpt_utils
import python_api_testing.models.nanogpt.tt.nanogpt_mlp as nanogpt_mlp
import python_api_testing.models.nanogpt.tt.nanogpt_attention as nanogpt_attention
from python_api_testing.models.nanogpt.tt.nanogpt_config import GPTConfig


import python_api_testing.models.nanogpt.tt.nanogpt_block as nanogpt_block
from tt_lib.fallback_ops import fallback_ops

import numpy as np

from dataclasses import dataclass
import math

from transformers import GPT2LMHeadModel

from utility_functions_new import (
    torch2tt_tensor,
    tt2torch_tensor,
    torch_to_tt_tensor_rm,
)

class TtGPT(nn.Module):
    def __init__(self, config: GPTConfig(), state_dict, device):
        super().__init__()

        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        base_address = f"transformer"

        self.beta = torch2tt_tensor(
            state_dict[f"{base_address}.ln_f.bias"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )
        self.gamma = torch2tt_tensor(
            state_dict[f"{base_address}.ln_f.weight"], device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)


        self.wte.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.wte.weight"]
        )

        self.wpe.weight = torch.nn.Parameter(
            state_dict[f"{base_address}.wpe.weight"]
        )

        self.device = device

        blocks = []

        for i in range(config.n_layer):
            block = nanogpt_block.TtBlock(self.config, state_dict, f"{base_address}.h.{i}", device)
            blocks.append(block)

        self.h = torch.nn.ModuleList(blocks)


        self.ln_f = fallback_ops.LayerNorm(
            self.gamma,
            self.beta,
            eps=1e-5,
            normalized_shape=config.n_embd,
        )

        self.lm_weight = state_dict["lm_head.weight"]

        # Push weights to Tt device
        self.tt_weight_lm_head = torch2tt_tensor(
            self.lm_weight, device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR
        )

        self.tt_weight_lm_head = tt_lib.tensor.transpose(self.tt_weight_lm_head)

        self.wte.weight = nn.Parameter(self.lm_weight) # https://paperswithcode.com/method/weight-tying


    def forward(self, idx):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        tok_emb = self.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)

        tt_tok_emb = torch_to_tt_tensor_rm(tok_emb, self.device, put_on_device=False)
        tt_pos_emb = torch_to_tt_tensor_rm(pos_emb, self.device, put_on_device=False)

        tt_sum = tt_lib.tensor.add(tt_tok_emb, tt_pos_emb)

        sum = tt2torch_tensor(tt_sum)

        x = self.drop(sum)

        tt_x = torch_to_tt_tensor_rm(x, self.device, put_on_device=True)

        for block in self.h:
            tt_x = block.forward(tt_x)

        tt_x = self.ln_f(tt_x)
        x = tt2torch_tensor(tt_x)
        x = x[:, [-1], :]
        tt_x = torch2tt_tensor(x, self.device, tt_layout=tt_lib.tensor.Layout.ROW_MAJOR)

        logits = nanogpt_utils.tt_linear(tt_x, weight=self.tt_weight_lm_head, bias=None)
        loss = None

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
            """
            Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
            the sequence max_new_tokens times, feeding the predictions back into the model each time.
            Most likely you'll want to make sure to be in model.eval() mode of operation for this.
            """
            for _ in range(max_new_tokens):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
                # forward the model to get the logits for the index in the sequence

                tt_logits, _ = self.forward(idx_cond)
                # pluck the logits at the final step and scale by desired temperature
                logits = tt2torch_tensor(tt_logits)

                #logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                #tt_logits = nanogpt_utils.torch2tt_tensor(logits, device)
            # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, :, -1, :] / temperature
            # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')

            # apply softmax to convert logits to (normalized) probabilities
                tt_logits = torch_to_tt_tensor_rm(logits, self.device, put_on_device=False)
                tt_probs = fallback_ops.softmax(tt_logits, dim=-1)
                probs = tt2torch_tensor(tt_probs)
                probs = probs.squeeze(0)
                probs = probs.squeeze(0)

                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)

            return idx
