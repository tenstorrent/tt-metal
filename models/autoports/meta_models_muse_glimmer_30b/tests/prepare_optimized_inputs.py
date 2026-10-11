# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Record real checkpoint activations with the official CPU reference, no TT device."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoTokenizer
from transformers.models.muse_glimmer.modeling_muse_glimmer import (
    MuseGlimmerRMSNorm,
    MuseGlimmerTextDecoderLayer,
    MuseGlimmerTextRotaryEmbedding,
)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = (
    Path(os.environ.get("HF_HOME", "/mnt/models/huggingface"))
    / "hub/models--meta-models--Muse-Glimmer-30B/snapshots/a4e59da52a7bc87ae7251dd5545c0dd437c44b68"
)


def load_state(layer):
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer}."
    state = {}
    for file in sorted({index[k] for k in index if k.startswith(prefix)}):
        with safe_open(SNAPSHOT / file, framework="pt") as f:
            for key in index:
                if key.startswith(prefix) and index[key] == file:
                    state[key[len(prefix) :]] = f.get_tensor(key)
    return state


@torch.no_grad()
def main(args):
    torch.set_num_threads(16)
    cfg = AutoConfig.from_pretrained(ROOT / "tests/assets").text_config
    cfg._attn_implementation = "sdpa"
    tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = (
        "The river winds through a quiet valley. Engineers measured its flow each morning and recorded the results. "
        "Explain how a solar panel converts sunlight into electricity. Write a Python function that sorts a list. "
        "A farmer has seven baskets of apples, with twelve apples in each basket. How many apples are there? "
        "The library opened in spring, welcoming readers of science, history, poetry, and mathematics.\n"
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    ids = torch.tensor(([tokenizer.bos_token_id] + tokens * ((args.length // len(tokens)) + 1))[: args.length])
    index = json.loads((SNAPSHOT / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.language_model.embed_tokens.weight"
    with safe_open(SNAPSHOT / index[key], framework="pt") as f:
        table = f.get_slice(key)
        unique = {i: table[i : i + 1] for i in ids.unique().tolist()}
        x = torch.cat([unique[i] for i in ids.tolist()])[None]
    x = MuseGlimmerRMSNorm(eps=cfg.rms_norm_eps, with_scale=False)(x)
    rotary = MuseGlimmerTextRotaryEmbedding(cfg)
    manifest = dict(snapshot=str(SNAPSHOT), text=text, token_ids=ids.tolist(), length=args.length, layers={})
    args.output.mkdir(parents=True, exist_ok=True)
    for layer_idx in range(4):
        with torch.device("meta"):
            layer = MuseGlimmerTextDecoderLayer(cfg, layer_idx)
        layer.load_state_dict(load_state(layer_idx), assign=True)
        layer.eval()
        if layer_idx in (0, 3):
            path = args.output / f"layer_{layer_idx}.pt"
            torch.save(x, path)
            manifest["layers"][layer_idx] = dict(
                path=str(path),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                shape=list(x.shape),
                mean=x.float().mean().item(),
                std=x.float().std().item(),
            )
        if layer_idx == 3:
            break

        # Chunk the official HF layer with its exact sliding causal mask.
        class Cache:
            k = v = None

            def update(self, k, v, layer_idx):
                self.k = k if self.k is None else torch.cat((self.k, k), 2)
                self.v = v if self.v is None else torch.cat((self.v, v), 2)
                return self.k, self.v

        cache = Cache()
        output = []
        for start in range(0, args.length, 512):
            chunk = x[:, start : start + 512]
            pos = torch.arange(start, start + chunk.shape[1])
            keys = torch.arange(start + chunk.shape[1])
            visible = (keys[None] <= pos[:, None]) & (keys[None] > pos[:, None] - cfg.sliding_window)
            mask = torch.zeros(visible.shape, dtype=x.dtype).masked_fill_(~visible, float("-inf"))[None, None]
            output.append(
                layer(chunk, position_embeddings=rotary(chunk, pos[None]), attention_mask=mask, past_key_values=cache)
            )
        x = torch.cat(output, 1)
        del layer, output, cache
        gc.collect()
        print(f"HF_INPUT_LAYER_{layer_idx+1}_READY", flush=True)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--length", type=int, default=4098)
    p.add_argument("--output", type=Path, default=ROOT / "doc/optimized_decoder/inputs")
    main(p.parse_args())
