# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stream the HF prefix needed to capture genuine Qwen3 layer-32 inputs.

This is an offline test-artifact generator, not a TTNN full-model path. It keeps
only one CPU decoder layer resident at a time and stops before layer 32.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer, Qwen3RotaryEmbedding


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tensors(snapshot: Path, index: dict, prefix: str) -> tuple[dict[str, torch.Tensor], set[Path]]:
    weight_map = index["weight_map"]
    keys = [key for key in weight_map if key.startswith(prefix)]
    shards = {snapshot / weight_map[key] for key in keys}
    tensors = {}
    for shard in sorted(shards):
        with safe_open(shard, framework="pt", device="cpu") as file:
            for key in keys:
                if weight_map[key] == shard.name:
                    tensors[key.removeprefix(prefix)] = file.get_tensor(key)
    return tensors, shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-tokens", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=21)
    args = parser.parse_args()

    index_path = args.snapshot / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    config = Qwen3Config.from_json_file(args.config)
    config._attn_implementation = "eager"
    tokens_artifact = torch.load(args.reference_tokens, map_location="cpu", weights_only=True)
    token_key = "reference_tokens" if "reference_tokens" in tokens_artifact else "token_ids"
    reference_tokens = tokens_artifact[token_key].to(torch.long)
    repetitions = (args.tokens + reference_tokens.shape[1] - 1) // reference_tokens.shape[1]
    token_ids = reference_tokens.repeat(1, repetitions)[:, : args.tokens].contiguous()

    embed_key = "model.embed_tokens.weight"
    embed_path = args.snapshot / index["weight_map"][embed_key]
    with safe_open(embed_path, framework="pt", device="cpu") as file:
        embedding = file.get_tensor(embed_key)
    hidden = F.embedding(token_ids, embedding).to(torch.bfloat16)
    del embedding

    position_ids = torch.arange(args.tokens, dtype=torch.long).unsqueeze(0)
    rotary = Qwen3RotaryEmbedding(config)
    position_embeddings = rotary(hidden, position_ids)
    causal_mask = torch.full((1, 1, args.tokens, args.tokens), float("-inf"), dtype=torch.float32)
    causal_mask = torch.triu(causal_mask, diagonal=1).to(torch.bfloat16)

    used_shards = {embed_path}
    with torch.no_grad():
        for layer_idx in range(32):
            prefix = f"model.layers.{layer_idx}."
            state, shards = _load_tensors(args.snapshot, index, prefix)
            used_shards.update(shards)
            with torch.device("meta"):
                layer = Qwen3DecoderLayer(config, layer_idx)
            layer.load_state_dict(state, assign=True)
            layer.eval()
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                use_cache=False,
            ).to(torch.bfloat16)
            print(f"captured prefix layer {layer_idx}: finite={bool(torch.isfinite(hidden).all())}")
            del layer, state

    payload = {
        "hidden_states": hidden.contiguous(),
        "token_ids": token_ids.contiguous(),
        "metadata": {
            "model": "Qwen/Qwen3-32B",
            "checkpoint_revision": args.snapshot.name,
            "captured_boundary": "input_to_model.layers.32",
            "shape": list(hidden.shape),
            "dtype": str(hidden.dtype),
            "reference_tokens_sha256": _sha256(args.reference_tokens),
            "reference_tokens_key": token_key,
            "token_policy": "repeat reference token sequence only when requested length exceeds its length",
            "config_sha256": _sha256(args.config),
            "index_sha256": _sha256(index_path),
            "used_shards": {path.name: _sha256(path.resolve()) for path in sorted(used_shards)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(f"saved {args.output}: sha256={_sha256(args.output)}")


if __name__ == "__main__":
    main()
