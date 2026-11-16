# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""WS5 proof: a 2-layer model (layers 0 sliding and 5 full) decoding batch 1 through the fused path (fused_decode/:
the layer boundaries, the router, the merged MoE programs, HeadsB1) vs the HuggingFace reference of the same two
layers stacked, over 8 decode steps after a 66-token prefill. Each step feeds the reference's greedy token
(teacher forcing) and compares the final logits (after the softcap, from the standard LM head: the logits request
takes the sampler path), on the decoder fixture tokens and three chat prompts (32 steps). Bar: the median step PCC
>= 0.995 and every step >= 0.90 (near-tie expert flips: see the assertion; GEMMA4_26B_WS5_FUSED=0 runs the same check
on the unfused path as the control).
"""

import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm, Gemma4TextRotaryEmbedding

import ttnn
from models.common.utility_functions import comp_pcc
from models.demos.gemma4_26b_a4b_qb2.tests.test_device_numerics import reference_layer
from models.demos.gemma4_26b_a4b_qb2.tt.generator import Gemma4Generator
from models.demos.gemma4_26b_a4b_qb2.tt.model import MODEL_ID, REVISION, Checkpoint

LAYERS = (0, 5)
STEPS = 8


class Reference:
    """HF layers 0 and 5 stacked, with the model's embedding, final norm, tied LM head and softcap (FP32)."""

    def __init__(self):
        config = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION).text_config
        config._attn_implementation = "eager"
        self.config = config
        self.layers = [reference_layer(config, index) for index in LAYERS]
        self.embedding = Checkpoint().load("model.language_model.embed_tokens.")["weight"].float()
        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm.load_state_dict(
            {k: v.float() for k, v in Checkpoint().load("model.language_model.norm.").items()}, strict=False
        )
        self.rotary = Gemma4TextRotaryEmbedding(config)

    @torch.inference_mode()
    def logits(self, tokens):
        """The last position's logits for the token sequence (full recompute)."""
        n = len(tokens)
        hidden = self.embedding[torch.tensor(tokens)][None] * torch.tensor(self.config.hidden_size**0.5)
        hidden = hidden.bfloat16().float()
        mask = torch.zeros(n, n).masked_fill(torch.triu(torch.ones(n, n, dtype=torch.bool), 1), -torch.inf)[None, None]
        for index, layer in zip(LAYERS, self.layers):
            rope = self.rotary(hidden, torch.arange(n)[None], layer_type=self.config.layer_types[index])
            hidden = layer(hidden, position_embeddings=rope, attention_mask=mask)
        out = self.norm(hidden[:, -1]) @ self.embedding.T
        cap = self.config.final_logit_softcapping
        return (torch.tanh(out / cap) * cap)[0] if cap else out[0]


def device_logits(t):
    return torch.cat([ttnn.to_torch(x).float()[0, 0, 0] for x in ttnn.get_device_tensors(t)], -1)


CHAT_PROMPTS = (
    "Explain, step by step, why the sky appears blue during the day and red at sunset.",
    "Write a short Python function that returns the n-th Fibonacci number iteratively, then explain it.",
    "What is the difference between a process and a thread in an operating system?",
)


def prompts():
    from transformers import AutoTokenizer

    fixture = json.loads(Path(__file__).with_name("decoder_input_tokens.json").read_text())
    out = {"fixture": list(fixture["token_ids"])}
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    for i, text in enumerate(CHAT_PROMPTS):
        chat = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False)
        out[f"chat{i}"] = tok.encode(chat, add_special_tokens=False)
    return out


def run_prompt(gen, ref, name, prompt):
    caches, table = gen.model.allocate_cache(slots=1, context=256)
    gen.prefill_forward(torch.tensor([prompt]), page_table=table, kv_cache=caches, prompt_lens=[len(prompt)], slots=[0])
    tokens = list(prompt)
    next_token = int(ref.logits(tokens).argmax())
    pccs, top1 = [], 0
    for step in range(STEPS):
        tokens.append(next_token)
        expected = ref.logits(tokens)
        got = gen.decode_forward(
            torch.tensor([next_token]),
            torch.tensor([len(tokens) - 1]),
            page_table=table,
            kv_cache=caches,
            return_logits=True,
        )
        got = device_logits(got)[: expected.numel()]
        _, pcc = comp_pcc(expected, got, 0.995)
        pccs.append(pcc)
        top1 += int(got.argmax()) == int(expected.argmax())
        print(
            f"FUSED_DECODE {name} step={step} pcc={pcc:.6f} top1 hf={int(expected.argmax())} tt={int(got.argmax())}",
            flush=True,
        )
        next_token = int(expected.argmax())
    print(f"FUSED_DECODE {name} min_pcc={min(pccs):.6f} top1={top1}/{STEPS}", flush=True)
    return pccs


def test_fused_decode_vs_hf(qb2_mesh):
    torch.set_num_threads(16)
    ref = Reference()
    gen = Gemma4Generator(qb2_mesh, max_seq_len=2048, layer_indices=LAYERS)
    assert gen.model.fused_b1 is not None or os.environ.get("GEMMA4_26B_WS5_FUSED") == "0", "fused path inactive"
    results = {name: run_prompt(gen, ref, name, prompt) for name, prompt in prompts().items()}
    steps = sorted(p for v in results.values() for p in v)
    median = steps[len(steps) // 2]
    print(
        "FUSED_DECODE_SUMMARY",
        {k: round(min(v), 6) for k, v in results.items()},
        f"median={median:.6f} at_or_above_0.995={sum(p >= 0.995 for p in steps)}/{len(steps)}",
        flush=True,
    )
    # Real weights: a step whose top-8 expert choice sits on a near tie flips against HF on any device path (the
    # unfused integration path drops to ~0.955 on the same prompts), so the bar is the median step plus a floor that
    # a broken path cannot reach.
    assert median >= 0.995, f"median step PCC {median} < 0.995"
    assert steps[0] >= 0.90, f"step PCC {steps[0]} < 0.90"
