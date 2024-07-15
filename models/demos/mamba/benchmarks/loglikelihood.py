# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from typing import Union

from models.demos.mamba.reference.decode_model import MambaDecode
from models.demos.mamba.tt.full_model import MambaTT


def compute_loglikelihood(probs, labels) -> float:
    assert len(probs.shape) == 3, "Expected rank 3"
    assert labels.shape[-1] == 1, "Label should be 1 in final dim"
    assert len(labels.shape) == 3, "Expected rank 3"
    assert probs.shape[-1] > 1, "Probs should be >1 in final dim"
    assert labels.shape[1] == probs.shape[1], "Length dimension should match"
    return torch.gather(probs, -1, labels).sum().detach().cpu().item()


def compute_loglikelihood_given_prompt_and_target(
    context_ids, target_ids, model: Union[MambaDecode, MambaTT], vocab_size: int
):
    # Reset the model hidden/conv states before decoding
    model.reset_states()

    # We want logits for each target token so slice the last one off
    input_ids = torch.cat([context_ids, target_ids], dim=-1)[:, :-1]  # B x L

    num_target_tokens = target_ids.shape[1]
    last_token = context_ids[:, -1].unsqueeze(1)  # Model expects (Bx1)

    def fwd(inputs):
        with torch.no_grad():
            if isinstance(model, MambaDecode):
                return model(inputs[:, idx].unsqueeze(1))  # (B x 1) => (B x 1 x VOCAB)
            elif isinstance(model, MambaTT):
                # Replicate inputs to match the model's expected input shape
                inputs = input_ids[:, idx].unsqueeze(1).repeat(32, 1)
                with torch.no_grad():
                    out = model(inputs)  # (B x 1) => (B x 1 x VOCAB)
                    return out[0].unsqueeze(0)

    logits = []
    is_greedy = True
    for idx in range((input_ids.shape[-1])):
        out = fwd(input_ids)
        probs = torch.nn.functional.log_softmax(out, dim=-1)
        logits.append(probs)

        last_token = torch.argmax(out, dim=-1)
        target_token = input_ids[:, idx].unsqueeze(0)
        assert (
            last_token.shape == target_token.shape
        ), f"Expected actual and target token to be same shape ({last_token.shape} vs. {target_token.shape})"

        if last_token.item() != target_token.item():
            is_greedy = False

    # Compute loglikelihood using the recorded logits
    logits = torch.cat(logits, dim=1)[:, -num_target_tokens:, :vocab_size]  # (B x L x VOCAB )
    labels = target_ids.unsqueeze(-1)  # (B x L x 1)
    loglikelihood = compute_loglikelihood(logits, labels)

    return loglikelihood, is_greedy
