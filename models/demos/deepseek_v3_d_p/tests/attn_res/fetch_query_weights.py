# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pull Kimi K3's AttnRes query weights out of the published checkpoint, and nothing else.

    python models/demos/deepseek_v3_d_p/tests/attn_res/fetch_query_weights.py $PWD/k3_subset
    export KIMI_K3_CKPT=$PWD/k3_subset

Run by hand, never from a test — the suites read the checkpoint if that variable names one
and generate random queries if it does not, so CI never reaches the network.

What lands is an indexed checkpoint subset rather than a tensor dump, so it is the same
thing every other K3 module reads and shards written by other fetches merge into the same
index instead of colliding with it.

The 374 tensors this wants are 14 kB each and scattered one or two per shard across a
1.5 TB checkpoint, so downloading the shards that hold them costs ~200 GB to keep 5 MB.
Safetensors puts a JSON header at the front of each shard giving every tensor's byte
range, which makes the payload reachable with an HTTP range read instead.
"""

import argparse
import collections
import json
import struct
from pathlib import Path

import torch
from huggingface_hub import HfFileSystem, hf_hub_download
from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.kimi_k3.attn_res.weights import query_weight_names

REPO = "moonshotai/Kimi-K3"
SHARD_NAME = "attn_res.safetensors"

# Safetensors names its dtypes itself; only the ones K3 stores these vectors in are listed.
DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32}


def fetch(names):
    """Range-read `names` from the shards that hold them, one open file per shard."""
    index = json.load(open(hf_hub_download(REPO, "model.safetensors.index.json")))["weight_map"]
    by_shard = collections.defaultdict(list)
    for name in names:
        by_shard[index[name]].append(name)

    filesystem, out = HfFileSystem(), {}
    for shard, shard_names in sorted(by_shard.items()):
        with filesystem.open(f"{REPO}/{shard}", "rb") as handle:
            header_length = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(header_length))
            for name in shard_names:
                meta = header[name]
                start, end = meta["data_offsets"]
                handle.seek(8 + header_length + start)
                raw = bytearray(handle.read(end - start))
                # `frombuffer` aliases `raw` rather than copying it; clone so what is saved
                # owns its storage instead of a view into a mutable buffer.
                out[name] = torch.frombuffer(raw, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"]).clone()
        print(f"{shard}: {len(shard_names)} tensors", flush=True)
    return out


def stage(weights, checkpoint_dir):
    """Write one shard and fold it into the directory's index, keeping any other shards."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(weights, checkpoint_dir / SHARD_NAME)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    index = json.loads(index_path.read_text()) if index_path.is_file() else {}
    weight_map = index.get("weight_map", {})
    weight_map.update({name: SHARD_NAME for name in weights})
    index["weight_map"] = weight_map
    # `total_size` is the whole subset's, not this shard's, so a merge cannot shrink it.
    index["metadata"] = {"total_size": sum(path.stat().st_size for path in checkpoint_dir.glob("*.safetensors"))}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="checkpoint directory to stage the shard and index in")
    args = parser.parse_args()

    weights = fetch(query_weight_names())
    stage(weights, args.out)
    total = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"staged {len(weights)} tensors, {total / 1024:.1f} kB -> {args.out}/{SHARD_NAME}")


if __name__ == "__main__":
    main()
