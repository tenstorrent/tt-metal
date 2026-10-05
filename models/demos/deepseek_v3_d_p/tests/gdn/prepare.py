# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fetch one GDN layer of a pinned Qwen checkpoint by HTTP range reads; CPU only, opens no device.

    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --fetch-layer qwen38_27b [--layer 0]
    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --delete-layer qwen38_27b [--layer 0]

Writes a partial checkpoint to ``gdn_checkpoint_dir(model, layer)``: the pinned ``config.json``, a
``model.safetensors.index.json`` listing only that layer's ``linear_attn.*`` keys, ``layer<i>.safetensors`` holding
them (checkpoint keys and dtypes unchanged) and ``manifest.json`` (source shard per key, shapes, dtypes, bytes read,
file and layer digests). Only the listed tensors are read, never whole shards. Real weights stay local: delete the
directory after use (``--delete-layer``).
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

from safetensors.torch import save_file

from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import (
    QWEN_FIRST_GDN_LAYER,
    QWEN_GDN_MODELS,
    qwen_gdn_config,
    qwen_model_config,
)
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import (
    gdn_checkpoint_dir,
    gdn_layer_prefix,
    gdn_model_root,
    gdn_state_dict_sha256,
    load_gdn_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.gdn.hub_shard import HubSafetensorsShard, log


def _sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_layer(model: str, layer_idx: int) -> None:
    from huggingface_hub import hf_hub_download

    source = QWEN_GDN_MODELS[model]
    directory = gdn_checkpoint_dir(model, layer_idx)
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model_config = qwen_model_config(model)
    index_path = hf_hub_download(source.repo, "model.safetensors.index.json", revision=source.revision)
    weight_map = json.loads(open(index_path, encoding="utf-8").read())["weight_map"]
    prefix = gdn_layer_prefix(layer_idx, gdn_model_root(model_config))
    keys = sorted(key for key in weight_map if key.startswith(prefix))
    if not keys:
        raise ValueError(f"{model}: no {prefix}* keys in the hub index")
    shards: dict[str, HubSafetensorsShard] = {}
    for key in keys:  # open shards serially: concurrent creation would race on the dict
        if weight_map[key] not in shards:
            shards[weight_map[key]] = HubSafetensorsShard(source.repo, source.revision, weight_map[key])
    with ThreadPoolExecutor(8) as pool:
        tensors = dict(zip(keys, pool.map(lambda key: shards[weight_map[key]].tensor(key).contiguous(), keys)))
    shard_name = f"layer{layer_idx}.safetensors"
    save_file(tensors, directory / shard_name)
    (directory / "config.json").write_text(json.dumps(model_config, indent=2) + "\n", encoding="utf-8")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {key: shard_name for key in keys}}, indent=2) + "\n", encoding="utf-8"
    )
    state_dict = load_gdn_layer_state_dict(directory, layer_idx, qwen_gdn_config(model))
    manifest = {
        "repo": source.repo,
        "revision": source.revision,
        "layer": layer_idx,
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "HTTP range reads of the listed tensors only (no full shard download)",
        "source_tensors": {
            key: {"shard": weight_map[key], "shape": list(t.shape), "dtype": str(t.dtype)} for key, t in tensors.items()
        },
        "bytes_range_read": sum(shard.bytes_read for shard in shards.values()),
        "layer_state_dict_sha256": gdn_state_dict_sha256(state_dict),
        "files": {
            path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in sorted(directory.iterdir())
            if path.name != "manifest.json"
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(
        f"{model} layer {layer_idx}: {len(keys)} tensors from {len(shards)} shard(s), "
        f"{manifest['bytes_range_read'] / 2**20:.1f} MiB range-read in {time.perf_counter() - start:.1f} s -> {directory}"
    )
    log(f"  layer state_dict sha256 {manifest['layer_state_dict_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-layer", action="append", default=[], choices=list(QWEN_GDN_MODELS))
    parser.add_argument("--delete-layer", action="append", default=[], choices=list(QWEN_GDN_MODELS))
    parser.add_argument("--layer", type=int, default=QWEN_FIRST_GDN_LAYER)
    arguments = parser.parse_args()
    for model in arguments.fetch_layer:
        fetch_layer(model, arguments.layer)
    for model in arguments.delete_layer:
        directory = gdn_checkpoint_dir(model, arguments.layer)
        if directory.exists():
            shutil.rmtree(directory)
            log(f"{model} layer {arguments.layer}: deleted {directory}")


if __name__ == "__main__":
    main()
