# SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.

# SPDX-License-Identifier: Apache-2.0

"""
Llama3.2-90B weights are saved as 8 sharded checkpoints. Loading weights for a
single layer is slow since we load all layers into memory to construct the
model. This script repacks the weights into checkpoints chunked by layers to
speed up development.
"""
import argparse
import asyncio
import json
import math
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm

from models.tt_transformers.tt.load_checkpoints import is_param_replicated_across_shards


def layer_num(key):
    if "layers" in key:
        return int(key.split("layers.")[1].split(".")[0])
    return -1


def chunk_key(key, chunk_size):
    """
    Return the chunk number that a key should go into
    """
    layer_id = layer_num(key)
    assert layer_id >= 0, f"Unexpected key {key}"
    chunk_id = layer_id // chunk_size
    print(f"Key: {key} -> chunk_id: {chunk_id}")
    return chunk_id


def get_unified_tensor(key, value, hidden_size):
    res = None
    if len(value) == 1 or is_param_replicated_across_shards(key):
        res = value[0]
    else:
        if key.endswith("tok_embeddings.weight") or key.endswith("output.weight"):
            assert value[0].shape[1] == hidden_size
            res = torch.cat(value, dim=0)
        else:
            cat_dim = torch.argmin(torch.tensor(value[0].shape))
            res = torch.cat(value, dim=cat_dim)

    assert res is not None, f"Failed to unify tensor for key {key}"
    return res


def copy_file_if_no_exist(src_path: Path, dst_path: Path, file_name: str) -> None:
    src_file = src_path / file_name
    if src_file.exists() and not (dst_path / file_name).exists():
        shutil.copy(src_file, dst_path)
        print(f"Copied {file_name} to {dst_path}")


async def torch_save_async(chunk, file_full_path):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, torch.save, chunk, file_full_path)


async def repack(in_dir, out_dir, chunk_size, stop_after: int = None):
    """
    Repack llama3.2-90b weights into checkpoints chunked by layers.
    Non-layer weights are saved in the first checkpoint.

    Args:
        in_dir: input directory containing llama3.2-90b weights from Meta
        out_dir: output directory to save the chunked checkpoints
        chunk_size: number of layers per chunk
        stop_at: stop repacking at this many chunks
    """
    assert stop_after is None or stop_after > 0, f"Invalid stop_at value: {stop_after}"

    # load model params
    params_file = Path(in_dir) / "params.json"
    assert params_file.exists(), f"params.json not found in {in_dir}"
    with open(params_file, "r") as f:
        params = json.load(f)
    num_layers = params["n_layers"]
    hidden_size = params["dim"]

    # chunk the vision_model and the first FIVE decoder layers into the first checkpoint
    # the rest of the decoder layers are chunked based on chunk_size

    # first load the Meta checkpoints
    checkpoints = sorted(Path(in_dir).glob("*.pth"))
    merged_checkpoints = defaultdict(list)
    assert len(checkpoints) > 0, f"no checkpoint files found in {in_dir}"
    print(f"Loading {len(checkpoints)} checkpoint files:")
    for ckpt in tqdm(checkpoints, leave=True):
        tqdm.write(f"Checkpoint file: {ckpt}")
        loaded_ckpt = torch.load(ckpt, map_location="cpu")
        for key, value in loaded_ckpt.items():
            merged_checkpoints[key].append(value)

    # next we iterate over the merged checkpoints and get all the vision model tensors,
    # the first decoder layer tensors, and all the non-layer tensors
    num_decoder_layers_in_first_chunk = 1
    chunk = {}
    for key in list(merged_checkpoints.keys()):
        if (
            key.startswith("vision_model")
            or layer_num(key) in range(num_decoder_layers_in_first_chunk)
            or "layers." not in key
        ):
            chunk[key] = get_unified_tensor(key, merged_checkpoints[key], hidden_size)
            del merged_checkpoints[key]

    save_tasks = []
    # save the first chunk
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_file_if_no_exist(Path(in_dir), out_dir, "params.json")
    copy_file_if_no_exist(Path(in_dir), out_dir, "tokenizer.model")
    out_file = out_dir / f"vision-model-and-layers_{0}-{num_decoder_layers_in_first_chunk - 1}.pth"
    save_tasks.append(asyncio.create_task(torch_save_async(chunk, out_file)))
    print(f"Saved the following layers in {out_file}:")
    for key in chunk.keys():
        print("\t" + key)
    del chunk

    if stop_after is not None and stop_after == 1:
        await wait_with_progress(save_tasks, desc="Writing chunked checkpoints to files")
        return  # early return to stop at the first chunk

    # save the rest of the merged checkpoints into chunks
    num_chunks = math.ceil((num_layers - num_decoder_layers_in_first_chunk) / chunk_size)
    # set stop_after to num_chunks if it is None, which means repacking all layers
    stop_after = num_chunks if stop_after is None else stop_after - 1  # [INFO] -1 because already saved the 1st chunk

    chunks = [list() for _ in range(num_chunks)]
    for key in merged_checkpoints.keys():
        assert key.startswith("text_model"), f"Unexpected key: {key}"
        layer_id = layer_num(key)
        assert layer_id != -1, f"Unexpected key: {key}"
        chunk_id = (layer_id - num_decoder_layers_in_first_chunk) // chunk_size  # the first few layers is already saved
        chunks[chunk_id].append(key)

    print(f"Repacking {num_layers} layers into {num_chunks} chunks of size {chunk_size}")
    for chunk_id in tqdm(range(num_chunks)):
        if chunk_id >= stop_after:
            break

        chunk = {}
        for key in chunks[chunk_id]:
            chunk[key] = get_unified_tensor(key, merged_checkpoints[key], hidden_size)
            del merged_checkpoints[key]

        # save the chunk
        start_layer = chunk_id * chunk_size + num_decoder_layers_in_first_chunk
        end_layer = (chunk_id + 1) * chunk_size + num_decoder_layers_in_first_chunk - 1
        end_layer = min(end_layer, num_layers - 1)
        out_file = out_dir / f"layers_{start_layer}-{end_layer}.pth"
        save_tasks.append(asyncio.create_task(torch_save_async(chunk, out_file)))
        print(f"Saving the following layers in {out_file}:")
        for key in chunk.keys():
            print("\t" + key)
        del chunk

    await wait_with_progress(save_tasks, desc="Writing chunked checkpoints to files")


async def wait_with_progress(tasks, desc):
    """Wait for tasks to finish, updating a progress bar as each completes."""
    total = len(tasks)
    with tqdm(total=total, desc=desc, leave=True) as pbar:
        pending = set(tasks)
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            pbar.update(len(done))


if __name__ == "__main__":
    # Take in command line arguments
    parser = argparse.ArgumentParser(description="Repack llama3.2-90b weights")
    parser.add_argument("in_dir", type=str, help="input directory")
    parser.add_argument("out_dir", type=str, help="output directory")
    parser.add_argument("chunk_size", type=int, default=10, help="number of layers per chunk")
    parser.add_argument(
        "--stop_after", type=int, default=None, help="stop repacking after this many chunks are saved (default to all)"
    )
    args = parser.parse_args()

    asyncio.run(repack(args.in_dir, args.out_dir, args.chunk_size, args.stop_after))
