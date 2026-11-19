# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only real token embedding fixture for diverse decoder routing probes.

These are actual first-layer inputs, not full-stack activations for later layers.
No full model is constructed. Stored tensors remain outside the repository.
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

PROMPTS = [
    "Explain how sunlight powers photosynthesis",
    "Describe the structure of a volcano",
    "Write a recipe using fresh tomatoes",
    "Explain the purpose of a compiler",
    "How do birds navigate during migration",
    "Describe the history of jazz",
    "Explain a proof by induction",
    "Give advice for growing orchids",
    "How does a battery store energy",
    "Describe the shape of a galaxy",
    "Explain the rules of chess",
    "Compare the climates of deserts",
    "How does a camera focus light",
    "Describe the lifecycle of butterflies",
    "Explain the mathematics of probability",
    "Write about ancient architecture",
    "How do submarines control buoyancy",
    "Explain the role of enzymes",
    "Describe a traditional weaving technique",
    "How does gravity affect tides",
    "Explain the difference between poetry and prose",
    "Describe an effective exercise routine",
    "How do computers represent integers",
    "Explain the physics of sound",
    "Describe the ecology of coral reefs",
    "How do artists create perspective",
    "Explain the purpose of encryption",
    "Describe the process of fermentation",
    "How does irrigation support agriculture",
    "Explain the formation of glaciers",
    "Describe the design of suspension bridges",
    "How do astronomers detect exoplanets",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    encoded = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in PROMPTS]
    token_ids = [row[-1] for row in encoded]
    assert len(set(token_ids)) >= 28
    name = "model.language_model.embed_tokens.weight"
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    all_ids = [token for row in encoded for token in row]
    all_ids = (all_ids * 8)[:1027]
    with safe_open(args.snapshot / index[name], framework="pt") as source:
        table = source.get_slice(name)

        def hidden(ids):
            return (
                torch.cat([table[token : token + 1] for token in ids])
                .repeat(1, 4)
                .reshape(1, 1, len(ids), 10240)
                .bfloat16()
            )

        inputs = [hidden([token]) for token in token_ids]
        sequence = hidden(all_ids)
    provenance = dict(
        snapshot=str(args.snapshot),
        weight=name,
        prompts=PROMPTS,
        token_ids=token_ids,
        unique_token_ids=len(set(token_ids)),
        semantics="Checkpoint embedding, repeated four times exactly as HF first-layer hyperconnection input; later-layer rows are shape-faithful routing stress, not claimed full-stack activations",
    )
    torch.save(
        dict(inputs=inputs, token_ids=token_ids, sequence=sequence, sequence_token_ids=all_ids, provenance=provenance),
        args.output,
    )
    args.manifest.write_text(json.dumps(dict(tensor_file=str(args.output), **provenance), indent=2) + "\n")
    print(json.dumps(dict(rows=len(inputs), unique_tokens=len(set(token_ids)), sequence_rows=len(all_ids))))


if __name__ == "__main__":
    main()
