# SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Collect real-weight tensor stats and layer-input activation stats for
GLM-4.7-Flash functional-decoder tests. Run once on CPU:

    python models/autoports/zai_org_glm_4_7_flash/tests/collect_stats.py

Writes tests/weight_stats.json: per-tensor {shape, dtype, mean, std} for the
dense (layer 0) and moe (layer 1) layer kinds, plus activation stats for each
layer kind's input (embedding output for layer 0; layer-0 output for layer 1),
measured on a real tokenized text sample.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3].parent))
from models.autoports.zai_org_glm_4_7_flash.tests.utils import (  # noqa: E402
    SNAPSHOT,
    STATS_PATH,
    build_hf_layer,
    hf_config,
    hf_forward,
    layer_weight_keys,
    load_real_layer_state_dict,
)


def tensor_stats(sd):
    out = {}
    for k, t in sd.items():
        out[k] = {
            "shape": list(t.shape),
            "dtype": "bfloat16",  # checkpoint storage dtype
            "mean": float(t.float().mean()),
            "std": float(t.float().std()),
        }
    return out


def main():
    cfg = hf_config()
    stats = {"model": "zai-org/GLM-4.7-Flash", "layers": {}, "activations": {}}

    sd0 = load_real_layer_state_dict(cfg, 0)
    sd1 = load_real_layer_state_dict(cfg, 1)
    stats["layers"]["dense"] = {"layer_idx": 0, "tensors": tensor_stats(sd0)}
    stats["layers"]["moe"] = {"layer_idx": 1, "tensors": tensor_stats(sd1)}
    print(f"dense tensors: {len(sd0)}, moe tensors: {len(sd1)}")

    # Real activation scale: embed a real text sample, run layer 0.
    from safetensors import safe_open
    from transformers import AutoTokenizer

    with open(SNAPSHOT / "model.safetensors.index.json") as f:
        index = json.load(f)["weight_map"]
    emb_key = "model.embed_tokens.weight"
    with safe_open(str(SNAPSHOT / index[emb_key]), framework="pt") as f:
        emb = f.get_tensor(emb_key).to(torch.float32)

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    text = (SNAPSHOT / "README.md").read_text()[:8000]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024]
    x0 = emb[ids[0]].unsqueeze(0)  # [1, S, H]
    stats["activations"]["dense"] = {
        "source": "embed_tokens output on 1024 real README tokens",
        "mean": float(x0.mean()),
        "std": float(x0.std()),
    }

    layer0 = build_hf_layer(cfg, 0, sd0)
    x1 = hf_forward(cfg, layer0, x0)
    stats["activations"]["moe"] = {
        "source": "HF layer-0 fp32 output on the same tokens",
        "mean": float(x1.mean()),
        "std": float(x1.std()),
    }
    print("activation stats:", stats["activations"])

    STATS_PATH.write_text(json.dumps(stats, indent=1))
    print(f"wrote {STATS_PATH}")


if __name__ == "__main__":
    main()
