# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Capture representative HF inputs to Falcon3 decoder layer 20.

This is an offline evidence helper, not part of the TTNN runtime. It runs the
HF embedding and layers 0..19 for 17-, non-aligned 31-, and 128-token recorded
prompts. Decode inputs are recorded only where the 128-slot fixture cache
has remaining capacity.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoTokenizer, DynamicCache

from models.autoports.tiiuae_falcon3_10b_base.tests.test_functional_decoder import (
    HF_MODEL,
    _config,
    _hf_decode,
    _hf_layer,
    _hf_prefill,
    _real_layer_state_dict,
)
from models.autoports.tiiuae_falcon3_10b_base.tt.functional_decoder import (
    EMITTED_PREFILL_SEQUENCE,
    IR_REPRESENTATIVE_LAYER,
)

PROMPTS = {
    17: "The history of efficient language models begins with careful measurement of every computational bottleneck.",
    31: (
        "Efficient language model inference depends on measuring memory traffic, arithmetic intensity, kernel "
        "launch overhead, cache behavior, and numerical accuracy across realistic prompts before selecting an "
        "implementation for production deployment."
    ),
    128: " ".join(
        [
            (
                "A disciplined accelerator study records tensor shapes, data types, memory layouts, numerical "
                "agreement, warmed latency, and repeated execution behavior before changing the next kernel."
            )
        ]
        * 8
    ),
}


def _embedding_rows(token_ids: torch.Tensor) -> torch.Tensor:
    index_path = Path(hf_hub_download(HF_MODEL, "model.safetensors.index.json"))
    weight_map = json.loads(index_path.read_text())["weight_map"]
    shard_path = hf_hub_download(HF_MODEL, weight_map["model.embed_tokens.weight"])
    with safe_open(shard_path, framework="pt", device="cpu") as tensors:
        embedding = tensors.get_tensor("model.embed_tokens.weight")
        return embedding[token_ids].to(torch.bfloat16).contiguous()


@torch.no_grad()
def _capture_path(config, encoded: torch.Tensor, seq_len: int, decode_count: int):
    required_tokens = seq_len + decode_count
    if encoded.numel() < required_tokens:
        raise ValueError(f"seq_len={seq_len} capture requires {required_tokens} real tokens, got {encoded.numel()}")
    encoded = encoded[:required_tokens]
    prefill_ids = encoded[:seq_len]
    hidden = _embedding_rows(prefill_ids).unsqueeze(0)
    cache = DynamicCache(config=config)
    for layer_idx in range(IR_REPRESENTATIVE_LAYER):
        state_dict = _real_layer_state_dict(layer_idx)
        layer = _hf_layer(config, state_dict, layer_idx)
        hidden = _hf_prefill(layer, config, hidden, cache=cache)
        del layer, state_dict
        gc.collect()

    layer20_prefill = hidden.to(torch.bfloat16).contiguous()
    layer20_decode = []
    for decode_offset in range(decode_count):
        decode_id = encoded[seq_len + decode_offset : seq_len + decode_offset + 1]
        decode_hidden = _embedding_rows(decode_id).view(1, 1, config.hidden_size)
        for layer_idx in range(IR_REPRESENTATIVE_LAYER):
            state_dict = _real_layer_state_dict(layer_idx)
            layer = _hf_layer(config, state_dict, layer_idx)
            decode_hidden = _hf_decode(layer, config, decode_hidden, cache, seq_len + decode_offset)
            del layer, state_dict
            gc.collect()
        layer20_decode.append(decode_hidden.to(torch.bfloat16).contiguous())
    return layer20_prefill, layer20_decode, prefill_ids, encoded[seq_len:required_tokens]


@torch.no_grad()
def main() -> None:
    config = _config()
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    encoded_17 = tokenizer(PROMPTS[17], add_special_tokens=True, return_tensors="pt").input_ids[0]
    if encoded_17.numel() < EMITTED_PREFILL_SEQUENCE + 1:
        encoded_17 = torch.cat(
            [encoded_17, torch.full((EMITTED_PREFILL_SEQUENCE + 1 - encoded_17.numel(),), config.eos_token_id)]
        )
    capture_17 = _capture_path(config, encoded_17, EMITTED_PREFILL_SEQUENCE, 1)

    encoded_31 = tokenizer(PROMPTS[31], add_special_tokens=True, return_tensors="pt").input_ids[0]
    capture_31 = _capture_path(config, encoded_31, 31, 2)
    encoded_128 = tokenizer(PROMPTS[128], add_special_tokens=True, return_tensors="pt").input_ids[0]
    capture_128 = _capture_path(config, encoded_128, 128, 0)
    layer20_prefill, layer20_decode, prefill_ids, decode_ids = capture_17
    layer20_prefill_31, layer20_decode_31, prefill_ids_31, decode_ids_31 = capture_31
    layer20_prefill_128, layer20_decode_128, prefill_ids_128, decode_ids_128 = capture_128

    output_dir = Path(__file__).with_name("activations")
    output_dir.mkdir(parents=True, exist_ok=True)
    activation_path = output_dir / "layer20_inputs.safetensors"
    save_file(
        {
            "prefill": layer20_prefill,
            "decode": layer20_decode[0],
            "prefill_31": layer20_prefill_31,
            "decode_31": layer20_decode_31[0],
            "decode_32": layer20_decode_31[1],
            "prefill_128": layer20_prefill_128,
        },
        activation_path,
    )
    metadata = {
        "model": HF_MODEL,
        "model_revision": Path(hf_hub_download(HF_MODEL, "config.json")).parent.name,
        "prompt": PROMPTS[EMITTED_PREFILL_SEQUENCE],
        "prefill_token_ids": prefill_ids.tolist(),
        "decode_token_id": int(decode_ids[0].item()),
        "prompt_31": PROMPTS[31],
        "prefill_31_token_ids": prefill_ids_31.tolist(),
        "decode_31_token_ids": decode_ids_31.tolist(),
        "prompt_128": PROMPTS[128],
        "prefill_128_token_ids": prefill_ids_128.tolist(),
        "decode_128_token_ids": decode_ids_128.tolist(),
        "producer": "HF embedding plus decoder layers 0..19, eager BF16 CPU",
        "prefill_shape": list(layer20_prefill.shape),
        "decode_shape": list(layer20_decode[0].shape),
        "prefill_31_shape": list(layer20_prefill_31.shape),
        "decode_31_shape": list(layer20_decode_31[0].shape),
        "decode_32_shape": list(layer20_decode_31[1].shape),
        "prefill_128_shape": list(layer20_prefill_128.shape),
    }
    (output_dir / "layer20_inputs.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(activation_path)


if __name__ == "__main__":
    main()
