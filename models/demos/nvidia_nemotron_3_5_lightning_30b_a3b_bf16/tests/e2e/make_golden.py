# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Build the BATCH=32 input (Source A) for the text-generation gate.

32 DISTINCT prompts, tokenized with the model's own tokenizer and LEFT-padded
to a common length so every row's last position is that row's real last prompt
token. The identical tensor is fed to the TT pipeline and to `model.generate()`,
with `attention_mask = ones` on both sides so the two see the same computation
(all-ones makes NemotronHModel drop the mamba mask and use a plain causal mask,
which is exactly what the graduated attention stub applies).
"""
from __future__ import annotations

from pathlib import Path

import torch

from models.demos.nvidia_nemotron_3_5_lightning_30b_a3b_bf16.tt import _hf_ref

OUT = Path(__file__).resolve().parents[2] / "_captured" / "_e2e_golden"

PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "The largest planet in our solar system is",
    "Photosynthesis converts sunlight into",
    "The Pacific Ocean is the world's",
    "A prime number is defined as",
    "The Great Wall of China was built to",
    "Electricity flows most easily through",
    "The human heart pumps blood through",
    "Mount Everest is located on the border of",
    "The speed of light in a vacuum is",
    "Shakespeare wrote the tragedy titled",
    "A triangle with three equal sides is called",
    "The chemical symbol for gold is",
    "Bees produce honey by collecting",
    "The Amazon rainforest is mostly located in",
    "Gravity causes objects to fall toward",
    "The longest river in Africa is",
    "A computer's memory stores",
    "The moon orbits around the",
    "Antibiotics are used to treat infections caused by",
    "The Industrial Revolution began in",
    "Sound travels faster through water than through",
    "A democracy is a system of government where",
    "The seasons on Earth are caused by",
    "Volcanoes form when molten rock rises through",
    "The first person to walk on the moon was",
    "Renewable energy sources include wind and",
    "The stomach digests food using",
    "A leap year occurs every",
    "The Sahara is the largest hot desert in",
    "Vaccines protect people by training the",
]


def build_input_ids(batch: int = 32, cache: bool = True) -> torch.Tensor:
    """(batch, T0) int64 -- the exact tensor both sides consume."""
    p = OUT / "input_ids.pt"
    if cache and p.exists():
        ids = torch.load(p)
        if ids.shape[0] >= batch:
            return ids[:batch]

    tok = _hf_ref.get_tokenizer()
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    prompts = (PROMPTS * ((batch // len(PROMPTS)) + 1))[:batch]
    enc = tok(prompts, return_tensors="pt", padding=True)
    ids = enc["input_ids"].to(torch.int64)
    if cache:
        OUT.mkdir(parents=True, exist_ok=True)
        torch.save(ids, p)
        (OUT / "prompts.txt").write_text("\n".join(prompts))
    return ids


def prompts(batch: int = 32) -> list[str]:
    return (PROMPTS * ((batch // len(PROMPTS)) + 1))[:batch]


if __name__ == "__main__":
    ids = build_input_ids(32)
    print(f"[golden] input_ids {tuple(ids.shape)} distinct_rows={len(set(map(tuple, ids.tolist())))}")
