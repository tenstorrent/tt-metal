# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Llama2-70B weights are saved as 8 sharded checkpoints. Loading weights for a
single layer is slow since we load all 80 layers into memory to construct the
model. This script repacks the weights into checkpoints chunked by layers to
speed up development.
"""
import math
from pathlib import Path
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse


def layer_num(key):
    if "layers" in key:
        return int(key.split("layers.")[1].split(".")[0])
    return 0


def chunk_key(key, chunk_size):
    """
    Return the chunk number that a key should go into
    """
    chunk_id = layer_num(key) // chunk_size
    print(f"Key: {key} -> chunk_id: {chunk_id}")
    return chunk_id


def repack(in_dir, out_dir, chunk_size):
    """
    Repack llama2-70b weights into checkpoints chunked by layers.
    Non-layer weights are saved in the first checkpoint.
    """
    N_LAYERS = 80
    num_chunks = math.ceil(N_LAYERS / chunk_size)
    print(f"Repacking {N_LAYERS} layers into {num_chunks} chunks of size {chunk_size}")
    checkpoints = sorted(Path(in_dir).glob("*.pth"))
    merged_checkpoints = defaultdict(list)
    assert len(checkpoints) > 0, f"no checkpoint files found in {in_dir}"
    print(f"Loading {len(checkpoints)} checkpoint files")
    for ckpt in tqdm(checkpoints):
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for key, value in loaded_ckpt.items():
            merged_checkpoints[key].append(value)

    # concat checkpoint values
    chunks = [dict() for _ in range(num_chunks)]
    for key, value in merged_checkpoints.items():
        if len(value) == 1 or "norm" in key:
            val = value[0]
        else:
            # cat_dim is index of the smallest dimension in value[0].shape
            cat_dim = torch.argmin(torch.tensor(value[0].shape))
            val = torch.cat(value, dim=cat_dim)

        chunk_id = chunk_key(key, chunk_size)
        chunks[chunk_id][key] = val

    # save chunks
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, chunk in enumerate(chunks):
        # each chunk file name should tell which layers are in it
        start_layer = i * chunk_size
        end_layer = (i + 1) * chunk_size - 1
        end_layer = min(end_layer, N_LAYERS - 1)
        out_file = out_dir / f"layers_{start_layer}-{end_layer}.pth"
        torch.save(chunk, out_file)
        print(f"Saved {out_file}")


if __name__ == "__main__":
    # Take in command line arguments
    parser = argparse.ArgumentParser(description="Repack llama2-70b weights")
    parser.add_argument("in_dir", type=str, help="input directory")
    parser.add_argument("out_dir", type=str, help="output directory")
    parser.add_argument("chunk_size", type=int, default=5, help="number of layers per chunk")
    args = parser.parse_args()
    repack(args.in_dir, args.out_dir, args.chunk_size)
