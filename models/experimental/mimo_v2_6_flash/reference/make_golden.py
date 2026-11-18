# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Run the fp32 torch reference over all layers for a few prompts and save golden activations.

Layers are streamed one at a time (all prompts go through a layer before the next one is loaded),
so host memory stays at one layer of dequantized experts. Saves, per prompt: token ids, the
hidden state after the embedding and after every decoder layer, and the final logits.

    python -m models.experimental.mimo_v2_6_flash.reference.make_golden --out <file.pt>
"""

import argparse
import time

import torch
from loguru import logger
from transformers import AutoTokenizer

from models.experimental.mimo_v2_6_flash.reference import model as ref
from models.experimental.mimo_v2_6_flash.reference.checkpoint import MiMoCheckpoint, default_model_path
from models.experimental.mimo_v2_6_flash.reference.config import MiMoConfig

DEFAULT_PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Water boils at a temperature of 100 degrees",
    "1, 2, 3, 4, 5, 6, 7, 8,",
    "The quick brown fox jumps over the lazy",
    "In 1969, the first humans to walk on the Moon were Neil Armstrong and",
    "Translate to French: 'Good morning' ->",
    "The largest planet in the solar system is",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=default_model_path())
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-layers", type=int, default=None)
    args = parser.parse_args()

    torch.set_num_threads(min(64, torch.get_num_threads()))
    cfg = MiMoConfig.from_model_path(args.model_path)
    ckpt = MiMoCheckpoint(args.model_path, cfg)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    num_layers = args.num_layers or cfg.num_hidden_layers

    token_ids = [torch.tensor(tok(p)["input_ids"], dtype=torch.long) for p in DEFAULT_PROMPTS]
    embedding = ckpt.embedding()
    hidden = [embedding[t].float() for t in token_ids]
    del embedding
    per_layer = [[h.clone()] for h in hidden]

    for layer_idx in range(num_layers):
        t0 = time.time()
        layer = ref.ReferenceDecoderLayer(cfg, layer_idx, ckpt=ckpt)
        t1 = time.time()
        with torch.no_grad():
            for i, t in enumerate(token_ids):
                hidden[i] = layer(hidden[i], torch.arange(t.shape[0]))
                per_layer[i].append(hidden[i].clone())
        del layer
        logger.info(f"layer {layer_idx}: load {t1 - t0:.1f}s, compute {time.time() - t1:.1f}s")

    norm = ckpt.final_norm()
    lm_head = ckpt.lm_head().float()
    logits = []
    with torch.no_grad():
        for i, h in enumerate(hidden):
            lg = ref.rms_norm(h, norm, cfg.layernorm_epsilon) @ lm_head.T
            logits.append(lg)
            nxt = tok.decode([int(lg[-1].argmax())])
            logger.info(f"prompt {i}: {DEFAULT_PROMPTS[i]!r} -> next token {nxt!r}")

    torch.save(
        {
            "prompts": DEFAULT_PROMPTS,
            "token_ids": token_ids,
            "hidden": [torch.stack(p) for p in per_layer],  # [num_layers + 1, S, hidden]
            "logits": logits,  # [S, vocab]
            "num_layers": num_layers,
        },
        args.out,
    )
    logger.info(f"saved {args.out}")


if __name__ == "__main__":
    main()
