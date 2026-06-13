# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Extract a standard Qwen2.5-3B HF checkpoint from the LocateAnything-3B weights.

tt_transformers expects a vanilla HF model dir (config.json model_type=qwen2 +
model.* / lm_head.* weights + tokenizer). We strip the `language_model.` prefix
and synthesize a Qwen2 config from LocateAnything's nested text_config.

Output dir is reusable across runs; skips re-extraction if already present.
"""
import json
import os
import shutil
import sys

from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(__file__))
import la_inputs  # noqa: E402

OUT_DIR = os.environ.get(
    "LA_LLM_DIR",
    os.path.expanduser("~/.cache/locate_anything/LA-Qwen2.5-3B"),
)
TOKENIZER_FILES = [
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "generation_config.json",
    "chat_template.json",
]


def main():
    mp = la_inputs.find_model_path()
    print(f"[extract] source: {mp}")
    print(f"[extract] dest:   {OUT_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg_full = json.load(open(os.path.join(mp, "config.json")))
    tc = cfg_full["text_config"]
    qwen_cfg = {
        "architectures": ["Qwen2ForCausalLM"],
        "model_type": "qwen2",
        "hidden_size": tc["hidden_size"],
        "intermediate_size": tc["intermediate_size"],
        "num_hidden_layers": tc["num_hidden_layers"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "head_dim": tc["hidden_size"] // tc["num_attention_heads"],
        "max_position_embeddings": tc["max_position_embeddings"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "rope_theta": tc["rope_theta"],
        "vocab_size": tc["vocab_size"],
        "tie_word_embeddings": tc.get("tie_word_embeddings", True),
        "hidden_act": tc.get("hidden_act", "silu"),
        "bos_token_id": tc.get("bos_token_id", 151643),
        "eos_token_id": tc.get("eos_token_id", 151645),
        "torch_dtype": "bfloat16",
        "use_sliding_window": tc.get("use_sliding_window", False),
        "sliding_window": tc.get("sliding_window", None),
        "attention_dropout": 0.0,
        "initializer_range": tc.get("initializer_range", 0.02),
        "transformers_version": "4.53.0",
    }
    json.dump(qwen_cfg, open(os.path.join(OUT_DIR, "config.json"), "w"), indent=2)
    print(f"[extract] wrote config.json (vocab={qwen_cfg['vocab_size']}, layers={qwen_cfg['num_hidden_layers']})")

    # copy tokenizer files
    for f in TOKENIZER_FILES:
        src = os.path.join(mp, f)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_DIR, f))

    out_weights = os.path.join(OUT_DIR, "model.safetensors")
    if os.path.exists(out_weights) and os.environ.get("LA_FORCE_EXTRACT") != "1":
        print(f"[extract] weights already present, skipping: {out_weights}")
        return OUT_DIR

    # gather language_model.* tensors from all shards
    idx = json.load(open(os.path.join(mp, "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    tensors = {}
    for shard in shards:
        path = os.path.join(mp, shard)
        with safe_open(path, framework="pt") as f:
            for k in f.keys():
                if k.startswith("language_model."):
                    new_k = k[len("language_model.") :]  # -> model.* / lm_head.*
                    tensors[new_k] = f.get_tensor(k).contiguous()
    has_lm_head = "lm_head.weight" in tensors
    if not has_lm_head and qwen_cfg["tie_word_embeddings"]:
        tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"].contiguous()
    print(
        f"[extract] {len(tensors)} LLM tensors (lm_head={'tied/explicit' if 'lm_head.weight' in tensors else 'MISSING'})"
    )
    save_file(tensors, out_weights, metadata={"format": "pt"})
    print(f"[extract] saved {out_weights} ({os.path.getsize(out_weights)/1e9:.2f} GB)")
    return OUT_DIR


if __name__ == "__main__":
    main()
