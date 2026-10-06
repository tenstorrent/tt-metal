# SPDX-License-Identifier: Apache-2.0
"""Record actual checkpoint activations for decoder precision experiments (CPU)."""
import gc
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from .reference import REVISION, SNAPSHOT, ReferenceDecoder, load_weights

ROOT = Path(__file__).resolve().parents[1]
TEXT = """The library opened early on Monday. A student was studying how rivers change
course over time, using maps, measurements, and historical accounts. She compared the
records carefully and wrote a short explanation of the evidence. In another room,
an engineer calculated the energy needed to heat a tank of water. The calculation
used the mass, the temperature difference, and the specific heat capacity. A teacher
asked the class to explain each step and check the units. Die Forscher vergleichen
die Ergebnisse und beschreiben ihre Beobachtungen. Eine genaue Messung hilft dabei,
Fehler zu erkennen. Voici une courte description de la ville et de son histoire.
The next question concerns a computer program: how can a list of numbers be sorted
without losing repeated values? The answer depends on the algorithm and its rules.
"""


def main():
    torch.set_num_threads(8)
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True)
    ids = tokenizer.encode(TEXT * 8, add_special_tokens=False)[:1025]
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(SNAPSHOT / index[key], framework="pt") as f:
        x = f.get_tensor(key)[torch.tensor(ids)][None].clone()
    out = ROOT / "doc/optimized_decoder/recorded_inputs"
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(revision=REVISION, text=TEXT, token_ids=ids, tensors={})
    for layer in range(5):
        if layer in (0, 4):
            path = out / f"layer_{layer}.pt"
            torch.save(x, path)
            meta["tensors"][str(layer)] = dict(
                shape=list(x.shape), sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
        if layer == 4:
            break
        print(f"CPU reference layer {layer}", flush=True)
        weights = load_weights(layer)
        reference = ReferenceDecoder(weights, layer)
        with torch.no_grad():
            x = reference(x)
        del reference, weights
        gc.collect()
    (out / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    print("RECORDED_REAL_ACTIVATIONS", flush=True)


if __name__ == "__main__":
    main()
