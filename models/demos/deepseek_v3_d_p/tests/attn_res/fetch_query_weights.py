# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Pull Kimi K3's AttnRes query weights out of the published checkpoint, and nothing else.

    python models/demos/deepseek_v3_d_p/tests/attn_res/fetch_query_weights.py k3_attn_res.pt
    export TT_K3_ATTN_RES_WEIGHTS=$PWD/k3_attn_res.pt

Run by hand, never from a test — `tests/attn_res/model/test_attn_res.py` reads the file if
that variable names one and generates random queries if it does not, so CI never reaches
the network.

The 374 tensors this wants are 14 kB each and scattered one or two per shard across a
1.5 TB checkpoint, so downloading the shards that hold them costs ~200 GB to keep 5 MB.
Safetensors puts a JSON header at the front of each shard giving every tensor's byte
range, which makes the payload reachable with an HTTP range read instead.
"""

import argparse
import collections
import json
import struct

import torch
from huggingface_hub import HfFileSystem, hf_hub_download

from models.demos.deepseek_v3_d_p.reference.attn_res.weights import query_weight_names

REPO = "moonshotai/Kimi-K3"

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", help="path to write the tensor dict to")
    args = parser.parse_args()

    weights = fetch(query_weight_names())
    torch.save(weights, args.out)
    total = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"saved {len(weights)} tensors, {total / 1024:.1f} kB -> {args.out}")


if __name__ == "__main__":
    main()
