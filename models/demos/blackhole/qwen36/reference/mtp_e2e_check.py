# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end MTP reference check against a REAL target forward pass.

mtp_torch.py's __main__ only proves build_mtp_reference() runs (synthetic
random hidden state, meaningless predictions). This script instead runs the
real ~27B backbone (Qwen3_5ForCausalLM, CPU, torch) on an actual prompt,
extracts its real post-final-norm hidden state and real next token, feeds
that into the MTP drafter (the same real-weight reference), and checks the
drafter's speculative predictions against what the target itself would
generate next -- i.e. a real speculative-decoding accept/reject check, not
just a shape/finite sanity check.

Drafting more than one token is CHAINING, not a config knob on
MTPTorchReference: each step after the first feeds the drafter's OWN
predicted token and OWN output hidden back in as the next step's input, with
position += 1 (see mtp_torch.py's module docstring / MTPTorchReference.step).
--k controls how many chained draft tokens to produce; the real target is
walked forward the same number of steps to get ground truth to compare
against.

    HF_HUB_OFFLINE=1 python models/demos/blackhole/qwen36/reference/mtp_e2e_check.py \
        --prompt "The capital of France is" --k 4

Loads the full backbone in bf16 (~54GB for the 27B) -- slow and memory-heavy
by design; this is a correctness check, not something to run per-commit.
"""

import argparse

import torch

from models.demos.blackhole.qwen36.reference.mtp_torch import build_mtp_reference
from models.demos.blackhole.qwen36.tt.model_config import Qwen36ModelArgs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--k", type=int, default=4, help="number of chained draft tokens to speculate")
    args_cli = parser.parse_args()

    # Same HF class/config recipe Qwen36ModelArgs.load_state_dict() uses (see its
    # docstring): naming the classes directly sidesteps vLLM's AutoConfig
    # registration of its own Qwen3_5Config for model_type "qwen3_5".
    from transformers import AutoTokenizer
    from transformers.models.qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    model_args = Qwen36ModelArgs(mesh_device=None)
    ckpt_dir = model_args.CKPT_DIR

    print(f"Loading backbone from {ckpt_dir} (bf16, ~54GB for the 27B) ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    text_config = Qwen3_5TextConfig.from_pretrained(ckpt_dir)
    model = Qwen3_5ForCausalLM.from_pretrained(ckpt_dir, config=text_config, dtype="auto")
    model.eval()

    input_ids = tokenizer(args_cli.prompt, return_tensors="pt").input_ids
    pos = input_ids.shape[1] - 1  # last prompt token's position

    with torch.no_grad():
        backbone_out = model.model(input_ids)
        hidden_states = backbone_out.last_hidden_state  # [1, T, dim], POST-final-norm
        logits = model.lm_head(hidden_states)  # [1, T, vocab]

        target_hidden = hidden_states[0, pos]
        real_next_token = int(logits[0, pos].argmax())  # y1: real, not speculative

        # Ground truth for the K tokens AFTER that (y2..y_{K+1}): walk the real
        # target forward K more real greedy steps.
        real_future = []
        walk_ids = torch.cat([input_ids, torch.tensor([[real_next_token]])], dim=1)
        for _ in range(args_cli.k):
            step_logits = model(walk_ids).logits
            nxt = int(step_logits[0, -1].argmax())
            real_future.append(nxt)
            walk_ids = torch.cat([walk_ids, torch.tensor([[nxt]])], dim=1)

    print("Building MTP reference (real mtp.* + real embed/lm_head weights) ...")
    ref = build_mtp_reference()

    # Seed the drafter's KV over the prompt: position i needs (target_hidden[i],
    # token[i+1]) filled before position `pos` can be drafted (its own KV cache
    # is position-addressed, same as the target's -- see docs/mtp_design.md's
    # "Prefill" section on prompt seeding).
    for i in range(pos):
        ref.step(token_id=int(input_ids[0, i + 1]), hidden_row=hidden_states[0, i], position=i)

    # Chained drafting: step 0 consumes the REAL (token=y1, hidden=target_hidden)
    # pair at drafter position `pos` and predicts a candidate for y2. Every step
    # after that feeds the drafter's OWN predicted token + OWN output hidden
    # back in (NOT the target's), with position += 1 -- this is what makes it
    # "drafting" rather than single-step verification.
    draft_tokens, draft_topk = [], []
    step_token, step_hidden = real_next_token, target_hidden
    for j in range(args_cli.k):
        mtp_logits, step_hidden = ref.step(token_id=step_token, hidden_row=step_hidden, position=pos + j)
        topk = torch.topk(mtp_logits, args_cli.topk).indices.tolist()
        draft_topk.append(topk)
        step_token = topk[0]  # drafter's own greedy pick feeds the next chain step
        draft_tokens.append(step_token)

    accept_len = 0
    while accept_len < args_cli.k and draft_tokens[accept_len] == real_future[accept_len]:
        accept_len += 1

    print()
    print(f"Prompt:                   {args_cli.prompt!r}")
    print(f"Target's real next token (y1): {tokenizer.decode([real_next_token])!r}")
    print()
    print(f"{'pos':>4}  {'target (ground truth)':<28} {'MTP draft':<20} match")
    for j in range(args_cli.k):
        tgt = tokenizer.decode([real_future[j]])
        drf = tokenizer.decode([draft_tokens[j]])
        match = "accept" if draft_tokens[j] == real_future[j] else "reject"
        print(f"y{j + 2:<3} {tgt!r:<28} {drf!r:<20} {match}")
    print()
    print(f"Longest accepted draft prefix: {accept_len}/{args_cli.k}")
    print(f"MTP top-{args_cli.topk} at first draft step: {[tokenizer.decode([t]) for t in draft_topk[0]]}")


if __name__ == "__main__":
    main()
