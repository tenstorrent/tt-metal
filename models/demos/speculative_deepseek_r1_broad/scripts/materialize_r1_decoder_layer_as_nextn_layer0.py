#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Copy one **main-model** decoder layer from ``DeepSeek-R1-0528`` (safetensors index) into ``model.layers.0.*``.

The single-layer NextN Hugging Face draft uses the same ``DeepseekV3DecoderLayer`` graph as the base
model. This script remaps::

  model.layers.<K>.*  →  model.layers.0.*

so you can pass the result to ``NextNFullHuggingfaceDraftAdapter`` /
``NextNSglangStructureDraftAdapter`` via ``decoder_layer0_override_safetensors`` (see
``nextn_full_layer_draft._apply_decoder_layer0_override_from_safetensors``).

**Layer index (important)**

- ``config.num_hidden_layers == 61`` ⇒ decoder blocks are ``model.layers.0`` … ``model.layers.60``.
- Keys under ``model.layers.61.*`` in some checkpoints are **MTP / shared_head** style tensors, **not**
  a full attention+MoE layer. For a full decoder+MoE block, use **``--layer-index 60``** (last
  transformer layer), not 61.

**CPU / RAM / disk**

- One R1 MoE layer is **large** (many experts). The output file can be **tens of GB**; peak RAM is
  similar while building the dict before ``save_file``.
- Hub shards are downloaded as needed (same layout as ``materialize_nextn_embed_head_aux_from_r1_shards.py``).

Example::

  python models/demos/speculative_deepseek_r1_broad/scripts/materialize_r1_decoder_layer_as_nextn_layer0.py \\
    --layer-index 60 \\
    --out /proj_sw/user_dev/dchrysostomou/hf_cache/r1_layer60_as_nextn_layer0.safetensors

Then (case study)::

  python models/demos/speculative_deepseek_r1_broad/scripts/case_study_nextn_full_layer_draft_from_record_cpu.py \\
    --decoder-layer0-override-safetensors /proj_sw/user_dev/dchrysostomou/hf_cache/r1_layer60_as_nextn_layer0.safetensors \\
    ...
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from models.demos.speculative_deepseek_r1_broad.default_paths import DEFAULT_HF_HOME
from models.demos.speculative_deepseek_r1_broad.hf_cache import set_hf_home


def _load_weight_map(*, repo_id: str | None, local_snapshot: Path | None, hub_cache: str) -> tuple[dict[str, str], Path]:
    if local_snapshot is not None:
        root = local_snapshot.expanduser().resolve()
        index_path = root / "model.safetensors.index.json"
        if not index_path.is_file():
            raise FileNotFoundError(f"Missing {index_path}")
        with open(index_path, encoding="utf-8") as f:
            idx = json.load(f)
        return idx["weight_map"], root
    if not repo_id:
        raise ValueError("Provide --repo-id or --local-snapshot")
    index_path = Path(
        hf_hub_download(
            repo_id,
            "model.safetensors.index.json",
            cache_dir=hub_cache,
        )
    )
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    return idx["weight_map"], index_path.parent


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract R1 decoder layer K as model.layers.0.* for NextN single-layer draft override.",
    )
    ap.add_argument(
        "--repo-id",
        default="deepseek-ai/DeepSeek-R1-0528",
        help="HF repo with model.safetensors.index.json (ignored if --local-snapshot is set)",
    )
    ap.add_argument(
        "--local-snapshot",
        type=Path,
        default=None,
        help="Directory containing model.safetensors.index.json + shards (no Hub download for tensors)",
    )
    ap.add_argument(
        "--layer-index",
        type=int,
        default=60,
        help="Decoder layer index in the main model (default 60 = last of 61 layers 0..60).",
    )
    ap.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"HF root when using Hub. Default: {DEFAULT_HF_HOME}",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output .safetensors path",
    )
    ap.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Stored dtype (override loader casts to draft torch_dtype). Default: bfloat16",
    )
    args = ap.parse_args()

    if args.layer_index == 61:
        print(
            "WARNING: layer index 61 is often MTP/shared_head only, not a full decoder+MoE block. "
            "Prefer --layer-index 60 for the last transformer layer.",
            file=sys.stderr,
        )

    home = set_hf_home(args.hf_home)
    hub_cache = str(home / "hub")
    weight_map, root_or_hub_parent = _load_weight_map(
        repo_id=args.repo_id,
        local_snapshot=args.local_snapshot,
        hub_cache=hub_cache,
    )

    prefix = f"model.layers.{args.layer_index}."
    layer_keys = sorted(
        k
        for k in weight_map
        if k.startswith(prefix) and ".shared_head." not in k
    )
    if len(layer_keys) < 8:
        raise SystemExit(
            f"Too few keys for {prefix!r} (got {len(layer_keys)}). "
            f"Wrong layer index or checkpoint layout?"
        )

    by_shard: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for old_k in layer_keys:
        shard_file = weight_map[old_k]
        new_k = "model.layers.0." + old_k[len(prefix) :]
        by_shard[shard_file].append((old_k, new_k))

    tgt_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    tensors: dict[str, torch.Tensor] = {}
    use_local = args.local_snapshot is not None

    for shard_rel, pairs in sorted(by_shard.items(), key=lambda x: x[0]):
        if use_local:
            shard_path = Path(root_or_hub_parent) / shard_rel
            if not shard_path.is_file():
                raise FileNotFoundError(f"Shard not found: {shard_path}")
        else:
            shard_path = Path(
                hf_hub_download(
                    args.repo_id,
                    shard_rel,
                    cache_dir=hub_cache,
                )
            )
        with safe_open(str(shard_path), framework="pt", device="cpu") as sf:
            for old_k, new_k in pairs:
                if old_k not in sf.keys():
                    raise KeyError(f"{old_k} missing in {shard_path}")
                tensors[new_k] = sf.get_tensor(old_k).to(tgt_dtype).contiguous()

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    print(
        f"Wrote {out_path} ({len(tensors)} tensors, layer_index={args.layer_index}, dtype={args.dtype})"
    )


if __name__ == "__main__":
    main()
