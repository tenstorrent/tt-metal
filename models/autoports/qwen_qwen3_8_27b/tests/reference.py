# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint-only setup helpers; never called from a TTNN forward pass."""

import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

MODEL_ID = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
LAYER_INDICES = {"linear_attention": 0, "full_attention": 3}


def load_config(snapshot):
    return AutoConfig.from_pretrained(snapshot, local_files_only=True).text_config


def load_layer_weights(snapshot, layer_idx):
    """Read only the tensors for one text decoder, without loading the full LM."""
    snapshot = Path(snapshot)
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"model.language_model.layers.{layer_idx}."
    selected = {key: shard for key, shard in index.items() if key.startswith(prefix)}
    if not selected:
        raise ValueError(f"No checkpoint tensors for {prefix}")
    state_dict = {}
    for shard in sorted(set(selected.values())):
        with safe_open(snapshot / shard, framework="pt", device="cpu") as handle:
            for key, filename in selected.items():
                if filename == shard:
                    state_dict[key.removeprefix(prefix)] = handle.get_tensor(key)
    return state_dict


def make_reference(config, layer_idx, state_dict):
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        layer = Qwen3_5DecoderLayer(config, layer_idx)
    layer.load_state_dict(state_dict, strict=True, assign=True)
    return layer.eval()


def synthetic_layer_weights(stats_path, kind, seed=42):
    """Deterministic real-shape tensors from the recorded target statistics."""
    info = json.loads(Path(stats_path).read_text())["layers"][kind]
    generator = torch.Generator().manual_seed(seed)
    return {
        tensor["name"]: (torch.randn(tensor["shape"], generator=generator) * tensor["std"] + tensor["mean"]).to(
            getattr(torch, tensor["dtype"].removeprefix("torch."))
        )
        for tensor in info["tensors"]
    }


def record_checkpoint_contract(snapshot, output_dir):
    """Record real tensor statistics and validate HF's exact key/shape contract."""
    snapshot, output_dir = Path(snapshot), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(snapshot)
    stats = {}
    for kind, layer_idx in LAYER_INDICES.items():
        state_dict = load_layer_weights(snapshot, layer_idx)
        reference = make_reference(config, layer_idx, state_dict)
        assert reference.layer_type == kind
        stats[kind] = {
            "layer_idx": layer_idx,
            "strict_hf_load": "passed",
            "tensors": [
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "mean": tensor.float().mean().item(),
                    "std": tensor.float().std().item(),
                    "bytes": tensor.numel() * tensor.element_size(),
                }
                for name, tensor in state_dict.items()
            ],
        }
        del reference, state_dict
    report = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
        "hf_advertised_context": config.max_position_embeddings,
        "layers": stats,
        "validation_scope": "Host-only checkpoint load and statistics; no TTNN correctness claim.",
    }
    (output_dir / "weight_stats.json").write_text(json.dumps(report, indent=2) + "\n")
    (output_dir / "hf_config.json").write_bytes((snapshot / "config.json").read_bytes())
    print(json.dumps({kind: info["strict_hf_load"] for kind, info in stats.items()}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    record_checkpoint_contract(args.snapshot, args.output_dir)
