# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

import torch

from models.demos.wormhole.mamba.reference.decode_model import MambaDecode


def compute_loglikelihood(logits, labels) -> float:
    assert len(logits.shape) == 3, "Expected rank 3"
    assert labels.shape[-1] == 1, "Label should be 1 in final dim"
    assert len(labels.shape) == 3, "Expected rank 3"
    assert logits.shape[-1] > 1, "Logits should be >1 in final dim"
    assert labels.shape[1] == logits.shape[1], "Length dimension should match"
    logits = torch.nn.functional.log_softmax(logits, dim=-1)  # (B x L x VOCAB)
    return torch.gather(logits, -1, labels).sum().detach().cpu().item()


def compute_loglikelihood_given_prompt_and_target(context_ids, target_ids, model: MambaDecode, vocab_size: int):
    # Reset the model hidden/conv states before decoding
    model.initialize_states()

    # We want logits for each target token so slice the last one off
    input_ids = torch.cat([context_ids, target_ids], dim=-1)[:, :-1]  # B x L

    num_target_tokens = target_ids.shape[1]
    last_token = context_ids[:, -1].unsqueeze(1)  # Model expects (Bx1)

    logits = []
    is_greedy = True
    for idx in range((input_ids.shape[-1])):
        out = model(input_ids[:, idx].unsqueeze(1))  # (B x 1) => (B x 1 x VOCAB)
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
