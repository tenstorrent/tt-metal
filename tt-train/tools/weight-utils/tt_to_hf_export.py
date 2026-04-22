#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Export a TTML pipeline-parallel LLaMA checkpoint to HuggingFace format.

Typical usage (PP4, TP1):
    python tt_to_hf_export.py \\
        --checkpoint-dir ./checkpoints/step_5000 \\
        --model-config configs/model_configs/llama8b.yaml \\
        --output-dir ./exported_hf_model \\
        --tokenizer-source /path/to/original/hf/model

Per-rank pkl files must follow the naming convention:
    step_{step}_rank_{rank}.pkl
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml


# TP sharding maps — used when multiple TP shards exist per key
_COLUMN_PARALLEL_SUFFIXES = {
    "attention/q_linear/weight",
    "attention/kv_linear/weight",
    "mlp/w1/weight",
    "mlp/w3/weight",
}
_ROW_PARALLEL_SUFFIXES = {
    "attention/out_linear/weight",
    "mlp/w2/weight",
}


def _load_rank_pkl(path: Path) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "model_state" in data:
        return data["model_state"]
    # Fallback: assume the file is a raw state dict
    return data


def _discover_rank_pkls(checkpoint_dir: Path) -> list[Path]:
    """Find step_*_rank_*.pkl files sorted by rank number."""
    files = sorted(checkpoint_dir.glob("step_*_rank_*.pkl"), key=lambda p: int(p.stem.split("_rank_")[1]))
    if not files:
        raise FileNotFoundError(f"No step_*_rank_*.pkl files found in {checkpoint_dir}")
    return files


def _merge_rank_state_dicts(rank_dicts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Merge PP rank state dicts.  PP ranks own disjoint keys — no conflict handling needed."""
    merged: dict[str, np.ndarray] = {}
    for rd in rank_dicts:
        merged.update(rd)
    return merged


def _gather_tp_shards(merged: dict[str, np.ndarray], tp_size: int) -> dict[str, np.ndarray]:
    """Reconstruct full tensors from TP shards stored as key/tp_{rank} sub-keys.

    For the initial PP4/TP1 case this is a no-op (tp_size == 1).
    """
    if tp_size <= 1:
        return merged

    # Group keys by base name and TP rank
    base_to_shards: dict[str, dict[int, np.ndarray]] = {}
    passthrough: dict[str, np.ndarray] = {}

    for key, arr in merged.items():
        if "/tp_" in key:
            base, rank_str = key.rsplit("/tp_", 1)
            base_to_shards.setdefault(base, {})[int(rank_str)] = arr
        else:
            passthrough[key] = arr

    gathered: dict[str, np.ndarray] = dict(passthrough)
    for base, shards in base_to_shards.items():
        ordered = [shards[r] for r in range(len(shards))]
        suffix = base.split("/", 1)[1] if "/" in base else base
        is_col = any(suffix.endswith(s) for s in _COLUMN_PARALLEL_SUFFIXES)
        is_row = any(suffix.endswith(s) for s in _ROW_PARALLEL_SUFFIXES)
        if is_col:
            gathered[base] = np.concatenate(ordered, axis=0)
        elif is_row:
            gathered[base] = np.concatenate(ordered, axis=1)
        else:
            # Replicated (norms, embeddings) — take shard 0
            gathered[base] = ordered[0]

    return gathered


def _parse_yaml_config(yaml_path: Path) -> SimpleNamespace:
    """Parse the transformer_config section of a model YAML into a namespace."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    tc = data.get("transformer_config", data)

    # Map CppLlamaConfig field names to a plain namespace understood by _normalize_config
    ns = SimpleNamespace(
        num_heads=tc["num_heads"],
        num_groups=tc["num_groups"],
        embedding_dim=tc["embedding_dim"],
        intermediate_dim=tc.get("intermediate_dim"),
        num_blocks=tc["num_blocks"],
        vocab_size=tc["vocab_size"],
        max_sequence_length=tc["max_sequence_length"],
        theta=tc.get("theta", 10000.0),
        scaling_factor=tc.get("scaling_factor", 0.0),
        original_context_length=tc.get("original_context_length", 0),
        high_freq_factor=tc.get("high_freq_factor", 4.0),
        low_freq_factor=tc.get("low_freq_factor", 1.0),
    )
    return ns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a TTML pipeline-parallel LLaMA checkpoint to HuggingFace format."
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="Directory containing step_*_rank_*.pkl files.",
    )
    src_group.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        dest="checkpoints",
        metavar="PATH",
        help="Explicit per-rank pkl file (repeatable, ordered by rank).",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        required=True,
        help="Path to model YAML config (e.g. configs/model_configs/llama8b.yaml).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for HF model files.",
    )
    parser.add_argument(
        "--tokenizer-source",
        type=Path,
        default=None,
        help="Path to a HF model directory to copy tokenizer files from.",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor-parallel size (default 1 = no TP gathering).",
    )
    args = parser.parse_args()

    # Resolve pkl paths
    if args.checkpoint_dir:
        rank_paths = _discover_rank_pkls(args.checkpoint_dir)
    else:
        rank_paths = args.checkpoints

    print(f"Loading {len(rank_paths)} rank checkpoint(s)...")
    rank_dicts = [_load_rank_pkl(p) for p in rank_paths]
    for i, (p, rd) in enumerate(zip(rank_paths, rank_dicts)):
        print(f"  rank {i}: {p.name} ({len(rd)} tensors)")

    merged = _merge_rank_state_dicts(rank_dicts)

    if args.tp_size > 1:
        print(f"Gathering TP shards (tp_size={args.tp_size})...")
        merged = _gather_tp_shards(merged, args.tp_size)

    print(f"Merged state dict: {len(merged)} tensors")

    config = _parse_yaml_config(args.model_config)

    # Import here so the script can be used without installing the ttml package
    # as long as the sources tree is on PYTHONPATH
    try:
        from ttml.models.llama.hf_exporter import save_to_hf_format
    except ImportError:
        # Allow running from the repo root without installing
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "hf_exporter",
            Path(__file__).resolve().parent.parent.parent / "sources/ttml/ttml/models/llama/hf_exporter.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        save_to_hf_format = mod.save_to_hf_format

    print(f"Exporting to {args.output_dir} ...")
    save_to_hf_format(merged, args.output_dir, config, tokenizer_source=args.tokenizer_source)
    print("Done.")
    print(f"  {args.output_dir}/model.safetensors")
    print(f"  {args.output_dir}/config.json")
    print(f"  {args.output_dir}/generation_config.json")
    if args.tokenizer_source:
        print(f"  tokenizer files copied from {args.tokenizer_source}")


if __name__ == "__main__":
    main()
