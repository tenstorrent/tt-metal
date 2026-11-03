# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-derived layer inputs for precision decisions (CPU oracle only)."""

import json

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from .fused_reference import reference
from .run_fused_decoder import MODEL, REVISION, ROOT, SNAPSHOT

TEXT = "Explain how a computer stores information and how attention uses earlier words to interpret a sentence. A small experiment compares precise calculations with approximations. The engineer checks every result and records the time. "

TEXTS = [
    TEXT,
    "def fibonacci(n):\n    a, b = 0, 1\n    for i in range(n):\n        a, b = b, a + b\n    return a\n# Verify the recurrence and explain its time complexity. ",
    "The train travels 180 kilometers in two hours. Calculate its average speed. Let x and y be positive integers satisfying x + y = 17 and xy = 60. Solve the equations and check the result. ",
    "请解释计算机如何存储和处理信息。比较精确计算和近似计算的优点，并给出一个简单的例子。今天的天气很好，我们计划去公园散步。 ",
]


def recorded_inputs(layer, batch, length, variant=0):
    path = ROOT / "doc/optimized_decoder/activation_cache" / f"layer{layer}_b{batch}_s{length}_v{variant}.pt"
    if path.exists():
        return torch.load(path, weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=True, local_files_only=True)
    tokens = [
        tokenizer.encode(TEXTS[(variant + user) % len(TEXTS)] * (length // 16 + 2), add_special_tokens=False)[:length]
        for user in range(batch)
    ]
    assert all(len(row) == length for row in tokens)
    ids = torch.tensor(tokens)
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    name = "model.embed_tokens.weight"
    with safe_open(SNAPSHOT / index[name], framework="pt") as f:
        x = f.get_tensor(name)[ids].bfloat16()
    mask = torch.full((length, length), float("-inf")).triu(1)[None, None]
    with torch.no_grad():
        for i in range(layer):
            _, hf, rope, _ = reference(i)
            cs = rope(x, torch.arange(length).unsqueeze(0).expand(batch, -1))
            x = hf(x, position_embeddings=cs, attention_mask=mask).detach()
            del hf
    path.parent.mkdir(exist_ok=True)
    torch.save(x, path)
    path.with_suffix(".json").write_text(
        json.dumps(
            dict(
                model=MODEL,
                revision=REVISION,
                layer=layer,
                token_ids=tokens,
                batch=batch,
                shape=list(x.shape),
                source="checkpoint embedding then original HF preceding layers",
                text=TEXTS[variant],
                variant=variant,
            ),
            indent=2,
        )
    )
    return x
