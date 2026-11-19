# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only HF front-four-layer activations for decoder precision decisions.

No TTNN import, device execution, full-model construction or text generation.
Teacher-forced text provides evolving real inputs and independent HF outputs
for the two representative decoder kinds.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

ROOT = Path(__file__).resolve().parents[1]
REVISION = "489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
TEXTS = [
    "A researcher compares two ways to store a sequence of observations. Each observation has a timestamp, a location, and a measured temperature. Explain how to compute a daily average while handling missing values. Then write a Python function that groups observations by day and returns the number of valid samples alongside each average. Discuss why changing the order of independent observations should preserve the result.\n",
    "A train leaves the first station at nine o'clock and travels sixty kilometers per hour. Another train leaves a different station twenty minutes later. We need enough information about the distances and directions before deciding whether they meet. Separate the known facts from the assumptions, express the distances as functions of time, and explain how units help detect mistakes. A short proof should make each inference explicit.\n",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--lengths", default="32,33,128,1024,1025")
    ap.add_argument("--steps", type=int, default=16)
    args = ap.parse_args()
    torch.set_num_threads(8)
    snapshot = Path(os.environ["HF_HOME"]) / f"hub/models--ornith-ai--Ornith-1.5-9B/snapshots/{REVISION}"
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    cfg = Qwen3_5TextConfig(**json.loads((ROOT / "tests/target_config.json").read_text())["text_config"])
    cfg._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    embed_name = "model.language_model.embed_tokens.weight"
    with safe_open(snapshot / index[embed_name], framework="pt", device="cpu") as source:
        embedding = source.get_tensor(embed_name)
    layers = []
    for i in range(4):
        prefix = f"model.language_model.layers.{i}."
        state = {}
        for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
            with safe_open(snapshot / shard, framework="pt", device="cpu") as source:
                for key in source.keys():
                    if key.startswith(prefix):
                        state[key[len(prefix):]] = source.get_tensor(key)
        layer = Qwen3_5DecoderLayer(cfg, i).bfloat16().eval()
        layer.load_state_dict(state, strict=True)
        layers.append(layer)
        del state
    rotary = Qwen3_5TextRotaryEmbedding(cfg)
    lengths = [int(x) for x in args.lengths.split(",")]
    cases = []
    with torch.inference_mode():
        for corpus, text in enumerate(TEXTS):
            repetitions = 32
            while True:
                ids = tokenizer(text * repetitions, add_special_tokens=False, return_tensors="pt")["input_ids"]
                if ids.shape[1] >= max(lengths) + args.steps:
                    break
                repetitions *= 2
            for length in lengths:
                cache = DynamicCache(config=cfg)
                x = torch.nn.functional.embedding(ids[:, :length], embedding).bfloat16()
                position = torch.arange(length)[None]
                mask = torch.full((length, length), float("-inf")).triu(1)[None, None]
                records = {i: {"decode_input": [], "decode_output": []} for i in (0, 3)}
                for i, layer in enumerate(layers):
                    if i in records:
                        records[i]["prefill_input"] = x.clone()
                    x = layer(x, position_embeddings=rotary(x, position),
                              attention_mask=mask if cfg.layer_types[i] == "full_attention" else None,
                              past_key_values=cache)
                    if i in records:
                        records[i]["prefill_output"] = x.clone()
                for step in range(args.steps):
                    x = torch.nn.functional.embedding(ids[:, length + step:length + step + 1], embedding).bfloat16()
                    position = torch.tensor([[length + step]])
                    for i, layer in enumerate(layers):
                        if i in records:
                            records[i]["decode_input"].append(x.clone())
                        x = layer(x, position_embeddings=rotary(x, position), past_key_values=cache)
                        if i in records:
                            records[i]["decode_output"].append(x.clone())
                for i, record in records.items():
                    cases.append(dict(corpus=corpus, layer=i, length=length, steps=args.steps,
                                      token_ids=ids[:, :length + args.steps].clone(), **record))
                print("RECORDED", corpus, length, args.steps, flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model="ornith-ai/Ornith-1.5-9B", revision=REVISION, texts=TEXTS, cases=cases), output)
    sidecar = dict(model="ornith-ai/Ornith-1.5-9B", revision=REVISION, texts=TEXTS, lengths=lengths,
                   steps=args.steps, layers=[0, 3], reference_prefix_layers=[0, 1, 2, 3],
                   tensor_file=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   semantics="CPU HF partial-stack, teacher-forced text, BF16; no TTNN/full-model generation")
    output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print("RECORDED_INPUTS_PASS", len(cases))


if __name__ == "__main__":
    main()
