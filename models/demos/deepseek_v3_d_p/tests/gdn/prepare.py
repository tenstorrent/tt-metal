# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fetch one GDN layer of a pinned Qwen checkpoint by HTTP range reads; CPU only, opens no device.

    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --fetch-layer qwen38_27b [--layer 0]
    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --delete-layer qwen38_27b [--layer 0]

Writes a partial checkpoint to ``gdn_checkpoint_dir(model, layer)``: the pinned ``config.json``, a
``model.safetensors.index.json`` listing only that layer's ``linear_attn.*`` keys, ``layer<i>.safetensors`` holding
them (checkpoint keys and dtypes unchanged) and ``manifest.json`` (source shard per key, shapes, dtypes, bytes read,
file and layer digests). Only the listed tensors are read, never whole shards. Real weights stay local: delete them
after use (``--delete-layer`` keeps only ``manifest.json``, stamped with the deletion time, as the download record).

Registered device cases (``tests/gdn/cases.py::GDN_CASES``) are prepared here too, outside the device lock: their
weight caches go to the checkout's ttnn model cache, their chained CPU references to the shared CPU oracle cache
(``TT_LINEAR_LAYERS_SHARED_CACHE``). Device tests then load them and fail fast on a miss:

    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --case toy-synthetic-mesh2x4-tpaxis1-T1280
    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --all --weights synthetic [--model qwen38_27b]
    python -m models.demos.deepseek_v3_d_p.tests.gdn.prepare --list

Host tilization initializes the TT-Metal runtime, so the case preparation runs against a mock cluster
(``TT_METAL_MOCK_CLUSTER_DESC_PATH``, defaulting to the LoudBox 8xP150 descriptor) and verifies on exit that it
holds no device handle.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[5]
_DEFAULT_MOCK_CLUSTER = (
    _REPOSITORY_ROOT
    / "tt_metal/third_party/tt-cluster-descriptors/blackhole/blackhole_8xP150_cluster_desc/blackhole_8xP150.yaml"
)
# Must precede the first TT-Metal runtime initialization (any TILE-layout host conversion).
os.environ.setdefault("TT_METAL_MOCK_CLUSTER_DESC_PATH", str(_DEFAULT_MOCK_CLUSTER))

import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

from safetensors.torch import save_file  # noqa: E402

from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import (  # noqa: E402
    QWEN_FIRST_GDN_LAYER,
    QWEN_GDN_MODELS,
    qwen_gdn_config,
    qwen_model_config,
)
from models.demos.deepseek_v3_d_p.tests.gdn.checkpoint_utils import (  # noqa: E402
    gdn_checkpoint_dir,
    gdn_layer_prefix,
    gdn_model_root,
    gdn_state_dict_sha256,
    load_gdn_layer_state_dict,
)
from models.demos.deepseek_v3_d_p.tests.gdn.hub_shard import HubSafetensorsShard, log  # noqa: E402


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


def _device_handles() -> list[str]:
    handles = []
    for descriptor in os.listdir("/proc/self/fd"):
        try:
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError:
            continue
        if "tenstorrent" in target:
            handles.append(target)
    return handles


def prepare_case(spec) -> None:
    """Write the case's weight cache for its mesh placement and every chained CPU reference."""
    from models.demos.deepseek_v3_d_p.tests.gdn.cases import (
        build_gdn_case,
        gdn_weight_cache_dir,
        gdn_weight_cache_prefix,
    )
    from models.demos.deepseek_v3_d_p.tests.gdn.reference_cache import prepare_cpu_references
    from models.demos.deepseek_v3_d_p.tt.gdn.weights import GDNWeights

    case = build_gdn_case(spec)
    cache_dir = gdn_weight_cache_dir(case.weights, spec.mesh_shape, spec.tensor_parallel_axis)
    prefix = gdn_weight_cache_prefix(case.weights)
    start = time.perf_counter()
    if GDNWeights.check_cache_complete(
        cache_dir, prefix, case.config, spec.mesh_shape, tensor_parallel_axis=spec.tensor_parallel_axis
    ):
        log(f"GDN prepare {spec.name}: weight cache hit {cache_dir}")
    else:
        log(f"GDN prepare {spec.name}: weight cache miss, writing {cache_dir}")
        GDNWeights.build_ttnn_cache(
            case.weights.load_state_dict(),
            cache_dir,
            prefix,
            case.config,
            spec.mesh_shape,
            tensor_parallel_axis=spec.tensor_parallel_axis,
        )
        log(f"GDN prepare {spec.name}: weight cache written in {time.perf_counter() - start:.1f} s")
    references = prepare_cpu_references(case)
    hits = sum(reference.cache_hit for reference in references)
    log(
        f"GDN prepare {spec.name}: CPU references {hits}/{len(references)} hits, "
        f"{sum(reference.seconds for reference in references):.1f} s"
    )


def _prepare_cases(arguments: argparse.Namespace) -> None:
    from models.demos.deepseek_v3_d_p.tests.gdn.cases import GDN_CASES
    from models.demos.deepseek_v3_d_p.utils.oracle_cache import oracle_cache_root

    if arguments.list:
        print("\n".join(GDN_CASES))
        return
    if arguments.all:
        specs = list(GDN_CASES.values())
    else:
        unknown = [name for name in arguments.case if name not in GDN_CASES]
        if unknown:
            raise SystemExit(f"unknown GDN case(s) {unknown}; see --list")
        specs = [GDN_CASES[name] for name in arguments.case]
    if arguments.weights is not None:
        specs = [spec for spec in specs if spec.weights == arguments.weights]
    if arguments.model:
        specs = [spec for spec in specs if spec.model in arguments.model]
    log(
        f"GDN prepare: mock cluster {os.environ['TT_METAL_MOCK_CLUSTER_DESC_PATH']}, CPU oracle cache {oracle_cache_root()}"
    )
    for index, spec in enumerate(specs, start=1):
        log(f"GDN prepare [{index}/{len(specs)}] {spec.name} start")
        prepare_case(spec)
        log(f"GDN prepare [{index}/{len(specs)}] {spec.name} done")
    handles = _device_handles()
    if handles:
        raise SystemExit(f"GDN prepare opened device handles {handles}; it must run without a device")
    log("GDN prepare: done; no device handle was opened")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-layer", action="append", default=[], choices=list(QWEN_GDN_MODELS))
    parser.add_argument("--delete-layer", action="append", default=[], choices=list(QWEN_GDN_MODELS))
    parser.add_argument("--layer", type=int, default=QWEN_FIRST_GDN_LAYER)
    parser.add_argument("--case", action="append", default=[], help="registered device case name (repeatable)")
    parser.add_argument("--all", action="store_true", help="every registered device case")
    parser.add_argument("--list", action="store_true", help="print registered device case names and exit")
    parser.add_argument("--weights", choices=("synthetic", "real"), help="restrict cases to one weight source")
    parser.add_argument("--model", action="append", default=[], help="restrict cases to model(s) (repeatable)")
    arguments = parser.parse_args()
    for model in arguments.fetch_layer:
        fetch_layer(model, arguments.layer)
    if arguments.case or arguments.all or arguments.list:
        _prepare_cases(arguments)
    for model in arguments.delete_layer:
        delete_layer(model, arguments.layer)


def delete_layer(model: str, layer_idx: int) -> None:
    """Remove everything but the manifest, which records what was downloaded and when it was deleted."""
    directory = gdn_checkpoint_dir(model, layer_idx)
    if not directory.exists():
        return
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for path in directory.iterdir():
        if path.name != "manifest.json":
            path.unlink() if path.is_file() else shutil.rmtree(path)
    if manifest:
        manifest["deleted"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    log(f"{model} layer {layer_idx}: deleted the weights in {directory} (manifest kept)")


if __name__ == "__main__":
    main()
