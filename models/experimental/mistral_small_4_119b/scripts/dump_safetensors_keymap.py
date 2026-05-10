#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Phase-A bring-up: dump Hugging Face ``text_config`` (and related fields) plus a
**safetensors key → shape → dtype** inventory for Mistral Small 4 (or any HF
repo using ``model.safetensors*``).

By default only the **text stack** is enumerated: keys under ``language_model.``
(Mistral4 LM + ``lm_head``), matching the on-disk safetensors layout for
``mistralai/Mistral-Small-4-119B-2603``. Use ``--full-checkpoint`` to include
``vision_tower.*`` and ``multi_modal_projector.*``.

Uses ``safe_open`` metadata only (no full weight tensors loaded into RAM).

Examples
────────
  export HF_MODEL=mistralai/Mistral-Small-4-119B-2603
  python3 models/experimental/mistral_small_4_119b/scripts/dump_safetensors_keymap.py

  python3 .../dump_safetensors_keymap.py --full-checkpoint --out-dir ./full_keymap

  python3 .../dump_safetensors_keymap.py --prefix language_model.model.layers.0. \\
      --out-dir ./layer0_only

  python3 .../dump_safetensors_keymap.py --summary-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from safetensors import safe_open

# Top-level HF state_dict prefix for the language model in Small 4 multimodal checkpoints.
TEXT_STACK_KEY_PREFIX = "language_model."


def _default_ckpt() -> str:
    return os.environ.get("HF_MODEL") or _constants_model_id()


def _constants_model_id() -> str:
    from models.experimental.mistral_small_4_119b.constants import HF_MODEL_ID

    return HF_MODEL_ID


def _resolve_index_and_dir(ckpt: str, local_files_only: bool) -> Tuple[Path, Optional[Dict[str, str]]]:
    """Return (resolved_root_path, weight_map or None for single-file)."""
    ckpt_path = Path(ckpt).expanduser()
    if ckpt_path.is_dir():
        root = ckpt_path.resolve()
        index_path = root / "model.safetensors.index.json"
        if index_path.is_file():
            with open(index_path, encoding="utf-8") as f:
                data = json.load(f)
            return root, data["weight_map"]
        single = root / "model.safetensors"
        if single.is_file():
            return root, None
        raise FileNotFoundError(f"No model.safetensors.index.json or model.safetensors under {root}")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError

    try:
        index_fn = hf_hub_download(
            repo_id=ckpt,
            filename="model.safetensors.index.json",
            local_files_only=local_files_only,
        )
    except (EntryNotFoundError, LocalEntryNotFoundError):
        st_fn = hf_hub_download(
            repo_id=ckpt,
            filename="model.safetensors",
            local_files_only=local_files_only,
        )
        return Path(st_fn).parent, None

    root = Path(index_fn).parent
    with open(root / "model.safetensors.index.json", encoding="utf-8") as f:
        data = json.load(f)
    return root, data["weight_map"]


def _shard_path(root: Path, shard_file: str, ckpt: str, local_files_only: bool) -> Path:
    p = root / shard_file
    if p.is_file():
        return p.resolve()
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=ckpt,
            filename=shard_file,
            local_files_only=local_files_only,
        )
    ).resolve()


def _slice_dtype_str(f, key: str) -> str:
    sl = f.get_slice(key)
    try:
        return str(sl.get_dtype())
    except Exception:
        return "unknown"


def iter_tensor_entries(
    ckpt: str,
    root: Path,
    weight_map: Optional[Dict[str, str]],
    *,
    prefix_filter: Optional[str],
    local_files_only: bool,
) -> Iterable[Tuple[str, List[int], str, str]]:
    """Yield (key, shape, dtype, shard_filename)."""
    if weight_map is None:
        path = _shard_path(root, "model.safetensors", ckpt, local_files_only)
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                if prefix_filter and not key.startswith(prefix_filter):
                    continue
                sl = f.get_slice(key)
                yield key, list(sl.get_shape()), _slice_dtype_str(f, key), path.name
        return

    keys_by_shard: Dict[str, List[str]] = defaultdict(list)
    for key, shard in weight_map.items():
        if prefix_filter and not key.startswith(prefix_filter):
            continue
        keys_by_shard[shard].append(key)

    for shard in sorted(keys_by_shard.keys()):
        path = _shard_path(root, shard, ckpt, local_files_only)
        wanted = set(keys_by_shard[shard])
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                if key not in wanted:
                    continue
                sl = f.get_slice(key)
                yield key, list(sl.get_shape()), _slice_dtype_str(f, key), shard


def _top_prefix(key: str, n: int = 1) -> str:
    parts = key.split(".")
    return ".".join(parts[:n]) if parts else key


def _language_layer_index(key: str) -> Optional[int]:
    m = re.match(r"language_model\.model\.layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _decoder_submodule(key: str) -> Optional[str]:
    """Block name inside a decoder layer: input_layernorm, self_attn, post_attention_layernorm, mlp."""
    m = re.match(r"language_model\.model\.layers\.\d+\.([^.]+)", key)
    return m.group(1) if m else None


def build_summary(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [e["key"] for e in entries]
    top1 = Counter(_top_prefix(k, 1) for k in keys)
    top2 = Counter(_top_prefix(k, 2) for k in keys)
    top3 = Counter(_top_prefix(k, 3) for k in keys)
    layer_counts: Dict[str, int] = defaultdict(int)
    submodule_counts = Counter()
    for k in keys:
        li = _language_layer_index(k)
        if li is not None:
            layer_counts[str(li)] += 1
        sub = _decoder_submodule(k)
        if sub is not None:
            submodule_counts[sub] += 1
    return {
        "num_tensors": len(keys),
        "prefix_1": dict(top1.most_common()),
        "prefix_2": dict(top2.most_common(40)),
        "prefix_3": dict(top3.most_common(50)),
        "decoder_layer_block_counts": dict(submodule_counts.most_common()),
        "language_model_layers_key_count": dict(
            sorted(((int(a), b) for a, b in layer_counts.items()), key=lambda x: x[0])
        ),
    }


def serialize_config(ckpt: str, local_files_only: bool) -> Dict[str, Any]:
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(ckpt, trust_remote_code=True, local_files_only=local_files_only)
    dtype_val = getattr(cfg, "dtype", None) or getattr(cfg, "torch_dtype", None)
    out: Dict[str, Any] = {
        "model_type": getattr(cfg, "model_type", None),
        "architectures": getattr(cfg, "architectures", None),
        "dtype": str(dtype_val) if dtype_val is not None else None,
    }
    if hasattr(cfg, "to_dict"):
        try:
            out["config"] = cfg.to_dict()
        except Exception as exc:
            out["config_error"] = str(exc)
    if getattr(cfg, "text_config", None) is not None and hasattr(cfg.text_config, "to_dict"):
        out["text_config"] = cfg.text_config.to_dict()
    if getattr(cfg, "vision_config", None) is not None and hasattr(cfg.vision_config, "to_dict"):
        out["vision_config"] = cfg.vision_config.to_dict()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", default=_default_ckpt(), help="HF repo id or local checkpoint directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("mistral_small_4_text_keymap_dump"),
        help="Output directory (created if missing)",
    )
    parser.add_argument(
        "--full-checkpoint",
        action="store_true",
        help="Do not filter by prefix (include vision_tower + multi_modal_projector tensors)",
    )
    parser.add_argument(
        "--prefix",
        default=TEXT_STACK_KEY_PREFIX,
        help=f"Only tensors whose key starts with this string (default: {TEXT_STACK_KEY_PREFIX!r}). Ignored if --full-checkpoint.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Pass through to HF hub / config (same meaning as transformers)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only write config.json + summary.json (still scans tensor metadata for summary)",
    )
    args = parser.parse_args()

    prefix_filter: Optional[str] = None if args.full_checkpoint else args.prefix

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg_blob = serialize_config(args.ckpt, local_files_only=args.local_files_only)
    (out_dir / "config.json").write_text(json.dumps(cfg_blob, indent=2, default=str), encoding="utf-8")

    root, weight_map = _resolve_index_and_dir(args.ckpt, args.local_files_only)

    entries: List[Dict[str, Any]] = []
    for key, shape, dtype, shard in iter_tensor_entries(
        args.ckpt,
        root,
        weight_map,
        prefix_filter=prefix_filter,
        local_files_only=args.local_files_only,
    ):
        entries.append({"key": key, "shape": shape, "dtype": dtype, "shard": shard})

    summary = build_summary(entries)
    summary["checkpoint_root"] = str(root)
    summary["ckpt_arg"] = args.ckpt
    summary["prefix_filter"] = prefix_filter
    summary["full_checkpoint"] = args.full_checkpoint
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    if not args.summary_only:
        keys_path = out_dir / "tensors.jsonl"
        with open(keys_path, "w", encoding="utf-8") as f:
            for row in entries:
                f.write(json.dumps(row, default=str) + "\n")

    print(f"Wrote {out_dir / 'config.json'}")
    print(f"Wrote {out_dir / 'summary.json'} ({summary['num_tensors']} tensors)")
    if not args.summary_only:
        print(f"Wrote {out_dir / 'tensors.jsonl'}")


if __name__ == "__main__":
    main()
