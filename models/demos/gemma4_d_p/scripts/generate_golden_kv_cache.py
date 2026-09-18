# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Capture HF Gemma4 KV before sliding-window truncation, without TT hardware."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Gemma4ForConditionalGeneration
from transformers.cache_utils import Cache, DynamicLayer, DynamicSlidingWindowLayer

from models.demos.gemma4_d_p.tt.model_config import validate_31b_config
from models.demos.gemma4_d_p.tt.runners.adapter import Gemma4PrefillAdapter, Gemma4ServiceConfig


class RecordingCache(Cache):
    def __init__(self, config, directory):
        super().__init__(
            layers=[
                (
                    DynamicSlidingWindowLayer(sliding_window=config.sliding_window)
                    if layer_type == "sliding_attention"
                    else DynamicLayer()
                )
                for layer_type in config.layer_types
            ]
        )
        self.directory = directory
        self.part = 0

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        save_file(
            {"key": key_states.detach().cpu().contiguous(), "value": value_states.detach().cpu().contiguous()},
            str(self.directory / f"layer_{layer_idx}_part_{self.part}.safetensors"),
        )
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=16384)
    args = parser.parse_args()
    if not 1 <= args.tokens <= Gemma4ServiceConfig.MAX_SEQ_LEN:
        parser.error("--tokens must be between 1 and 262144")
    if args.out.exists():
        parser.error("--out must not already exist")
    hf_model_id = Gemma4PrefillAdapter().hf_model_id
    tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
    tokens = tokenizer.encode(args.text.read_text(), add_special_tokens=True)
    if len(tokens) < args.tokens:
        parser.error(f"text has only {len(tokens)} tokens")
    tokens = tokens[: args.tokens]
    model = (
        Gemma4ForConditionalGeneration.from_pretrained(hf_model_id, dtype=torch.bfloat16, attn_implementation="sdpa")
        .eval()
        .model.language_model
    )
    validate_31b_config(model.config)
    args.out.mkdir(parents=True)
    kv_dir = args.out / "kv_cache"
    kv_dir.mkdir()
    with TemporaryDirectory(dir=args.out) as temporary:
        cache = RecordingCache(model.config, Path(temporary))
        with torch.inference_mode():
            for part, start in enumerate(range(0, len(tokens), Gemma4ServiceConfig.CHUNK_SIZE)):
                cache.part = part
                input_ids = torch.tensor([tokens[start : start + Gemma4ServiceConfig.CHUNK_SIZE]])
                model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        for layer in range(Gemma4ServiceConfig.NUM_LAYERS):
            parts = [
                load_file(str(cache.directory / f"layer_{layer}_part_{part}.safetensors"))
                for part in range(cache.part + 1)
            ]
            save_file(
                {
                    f"key_cache_layer_{layer}": torch.cat([part["key"] for part in parts], dim=2),
                    f"value_cache_layer_{layer}": torch.cat([part["value"] for part in parts], dim=2),
                },
                str(kv_dir / f"layer_{layer}.safetensors"),
            )
    (args.out / "metadata.json").write_text(json.dumps({"hf_model_id": hf_model_id, "token_ids": tokens}) + "\n")


if __name__ == "__main__":
    main()
