#!/usr/bin/env python3
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Build ``embed_head_aux.safetensors`` for NextN record runs without a full local R1 snapshot.

``lmsys/DeepSeek-R1-NextN`` / ``nextn_layer_parameters.safetensors`` often omits token embeddings
and ``shared_head.head`` (they duplicate the base checkpoint). This script downloads **only**
``model.safetensors.index.json`` plus the **two** weight shards that hold:

- ``model.embed_tokens.weight``
- the last layer's ``*.shared_head.head.weight`` (e.g. ``model.layers.61.*`` on R1-0528)

and writes a single small file with keys ``embed_tokens.weight`` and ``shared_head.head.weight``
(compatible with :func:`specfr.local_hf_snapshot.load_nextn_mtp_auxiliary_safetensors`).

**Disk:** the two shards are ~5–7 GB each (~12 GB total); the output is ~4 GB in bfloat16.

Run from anywhere; imports ``specfr`` from the parent ``speculative_from_record_deepseek_r1`` directory.

Example::

  cd models/demos/speculative_from_record_deepseek_r1
  python scripts/materialize_nextn_embed_head_aux_from_r1_shards.py \\
    --out /path/to/hf_cache/embed_head_aux_deepseek_r1_0528.safetensors

Then::

  python run_mtp_from_record_cpu.py \\
    --embed-head-aux-safetensors /path/to/hf_cache/embed_head_aux_deepseek_r1_0528.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_BUNDLE_ROOT = Path(__file__).resolve().parent.parent
if str(_BUNDLE_ROOT) not in sys.path:
    sys.path.insert(0, str(_BUNDLE_ROOT))

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file
import torch

from specfr.default_paths import DEFAULT_EMBED_HEAD_AUX_PATH, DEFAULT_HF_HOME
from specfr.hf_cache import set_hf_home


def _pick_shared_head_key(weight_map: dict[str, str]) -> str:
    candidates = [k for k in weight_map if k.endswith("shared_head.head.weight")]
    if not candidates:
        raise KeyError(
            "No shared_head.head.weight in weight_map; is this a DeepSeek-class index.json?"
        )

    def layer_index(key: str) -> int:
        parts = key.split(".")
        if len(parts) >= 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit():
            return int(parts[2])
        return -1

    return max(candidates, key=layer_index)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download two R1-class safetensors shards and write embed+head aux for NextN runs.",
    )
    ap.add_argument(
        "--repo-id",
        default="deepseek-ai/DeepSeek-R1-0528",
        help="HF repo with model.safetensors.index.json + shards (default: R1-0528)",
    )
    ap.add_argument(
        "--hf-home",
        type=Path,
        default=DEFAULT_HF_HOME,
        help=f"HF root; shard downloads use (hf-home)/hub. Default: {DEFAULT_HF_HOME}",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_EMBED_HEAD_AUX_PATH,
        help=f"Output .safetensors path. Default: {DEFAULT_EMBED_HEAD_AUX_PATH}",
    )
    ap.add_argument(
        "--dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
        help="Stored dtype (loader converts to float32 for the draft adapter). Default: bfloat16",
    )
    args = ap.parse_args()

    home = set_hf_home(args.hf_home)
    hub_cache = str(home / "hub")

    index_path = Path(
        hf_hub_download(
            args.repo_id,
            "model.safetensors.index.json",
            cache_dir=hub_cache,
        )
    )
    with open(index_path, encoding="utf-8") as f:
        idx = json.load(f)
    weight_map: dict[str, str] = idx["weight_map"]

    embed_key = "model.embed_tokens.weight"
    if embed_key not in weight_map:
        raise KeyError(f"{embed_key!r} not in {index_path}")
    head_key = _pick_shared_head_key(weight_map)

    shard_embed = hf_hub_download(args.repo_id, weight_map[embed_key], cache_dir=hub_cache)
    shard_head = hf_hub_download(args.repo_id, weight_map[head_key], cache_dir=hub_cache)

    with safe_open(shard_embed, framework="pt", device="cpu") as sf:
        if embed_key not in sf.keys():
            raise KeyError(f"{embed_key} missing in shard {shard_embed}")
        embed = sf.get_tensor(embed_key).float().cpu().contiguous()
    with safe_open(shard_head, framework="pt", device="cpu") as sf:
        if head_key not in sf.keys():
            raise KeyError(f"{head_key} missing in shard {shard_head}")
        head = sf.get_tensor(head_key).float().cpu().contiguous()

    tgt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "embed_tokens.weight": embed.to(tgt),
            "shared_head.head.weight": head.to(tgt),
        },
        str(out_path),
    )
    print(f"Wrote {out_path} (embed {tuple(embed.shape)}, head {tuple(head.shape)}, dtype={args.dtype})")


if __name__ == "__main__":
    main()
