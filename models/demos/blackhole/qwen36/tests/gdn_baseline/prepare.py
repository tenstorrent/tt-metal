# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only preparation for the GDN baseline device tests; opens no device.

Run from the tt-metal checkout root with the checkout's python_env (and the same ttnn model cache as the tests):

    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --fetch-config qwen38_27b
    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --fetch-weights qwen38_27b   # real, local only
    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --case <name> [--case ...]
    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --model qwen38_27b --weights synthetic
    python -m models.demos.blackhole.qwen36.tests.gdn_baseline.prepare --list

``--fetch-weights`` reads only the first GDN layer's tensors, ``input_layernorm.weight`` and the embedding rows of
the text tokens, by HTTP range reads of the pinned safetensors shards (no full shard download), and writes a
``manifest.json`` with source shards, keys, shapes, dtypes and file hashes. Case preparation fills the CPU-reference
cache (``cases.py``). Host tilization is never needed here, but ttnn is imported for the model cache path, so the
process runs against a mock cluster (``TT_METAL_MOCK_CLUSTER_DESC_PATH``) and verifies on exit that it holds no
device handle.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[6]
os.environ.setdefault(
    "TT_METAL_MOCK_CLUSTER_DESC_PATH",
    str(
        _REPOSITORY_ROOT
        / "tt_metal/third_party/tt-cluster-descriptors/blackhole/blackhole_8xP150_cluster_desc/blackhole_8xP150.yaml"
    ),
)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ThreadPoolExecutor  # noqa: E402

import torch  # noqa: E402

from models.demos.blackhole.qwen36.tests.gdn_baseline import cases as gc  # noqa: E402
from models.demos.deepseek_v3_d_p.reference.gdn.qwen_models import qwen_model_config  # noqa: E402
from models.demos.deepseek_v3_d_p.reference.gdn.weights import GDN_WEIGHT_NAMES  # noqa: E402
from models.demos.deepseek_v3_d_p.tests.gdn.hub_shard import HubSafetensorsShard  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_config(model: str) -> None:
    from huggingface_hub import hf_hub_download

    source = gc.MODELS[model]
    directory = gc.model_dir(model)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        hf_hub_download(source.repo, filename, revision=source.revision, local_dir=directory)
    if json.loads((directory / "config.json").read_text()) != qwen_model_config(model):
        raise ValueError(f"{model}: hub config.json differs from the pinned in-tree copy")
    log(f"{model}: config and tokenizer in {directory}")


def fetch_corpus() -> None:
    from huggingface_hub import get_session

    path = gc.corpus_path()
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = get_session().get(gc.TEXT_URL, follow_redirects=True)
    response.raise_for_status()
    path.write_bytes(response.content)
    log(f"corpus {gc.TEXT_URL} -> {path} sha256={_sha256(path)}")


def fetch_weights(model: str) -> None:
    """First GDN layer + input norm + the text's embedding rows, by range reads; writes manifest.json."""
    from huggingface_hub import hf_hub_download
    from safetensors.torch import save_file

    fetch_config(model)
    fetch_corpus()
    source = gc.MODELS[model]
    directory = gc.model_dir(model)
    index_path = hf_hub_download(
        source.repo, "model.safetensors.index.json", revision=source.revision, local_dir=directory
    )
    weight_map = json.loads(Path(index_path).read_text())["weight_map"]
    config = json.loads((directory / "config.json").read_text())
    prefix = "model.language_model." if "text_config" in config else "model."
    wanted = {gc.PREFIX + name: f"{prefix}layers.{gc.GDN_LAYER}.linear_attn.{name}" for name in GDN_WEIGHT_NAMES}
    wanted["input_layernorm.weight"] = f"{prefix}layers.{gc.GDN_LAYER}.input_layernorm.weight"
    scaled = [k for k in wanted.values() if f"{k}_scale_inv" in weight_map]
    if scaled:
        raise ValueError(f"block-quantized checkpoint tensors are not handled here: {scaled}")
    shards: dict[str, HubSafetensorsShard] = {}

    def shard(key: str) -> HubSafetensorsShard:
        filename = weight_map[key]
        if filename not in shards:
            shards[filename] = HubSafetensorsShard(source.repo, source.revision, filename)
        return shards[filename]

    for key in wanted.values():  # open shards serially: concurrent creation would race on the dict
        shard(key)
    tensors = {}
    with ThreadPoolExecutor(8) as pool:
        for local, value in zip(wanted, pool.map(lambda k: shard(k).tensor(k), wanted.values())):
            tensors[local] = value.contiguous()
    save_file(tensors, directory / "layer0.safetensors")

    token_count = gc.CHAINED_CHUNKS * max(gc.TOKENS)
    ids = sorted(set(gc._text_token_ids(model, token_count)))
    embed_key = f"{prefix}embed_tokens.weight"
    rows = shard(embed_key).rows(embed_key, ids)
    save_file({"rows": rows.contiguous(), "ids": torch.tensor(ids)}, directory / "embed_rows.safetensors")

    manifest = {
        "repo": source.repo,
        "revision": source.revision,
        "fetched": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "method": "HTTP range reads of the listed tensors only (no full shard download)",
        "source_tensors": {local: {"key": key, "shard": weight_map[key]} for local, key in wanted.items()},
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype)} for k, v in tensors.items()},
        "embedding": {
            "key": embed_key,
            "shard": weight_map[embed_key],
            "rows": len(ids),
            "text": gc.TEXT_URL,
            "text_tokens": token_count,
        },
        "bytes_range_read": sum(s.bytes_read for s in shards.values()),
        "files": {
            name: {"sha256": _sha256(directory / name), "bytes": (directory / name).stat().st_size}
            for name in (
                "layer0.safetensors",
                "embed_rows.safetensors",
                "config.json",
                "tokenizer.json",
                "model.safetensors.index.json",
            )
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    log(f"{model}: {len(tensors)} tensors + {len(ids)} embedding rows from {len(shards)} shards -> {directory}")
    log(f"  bytes range-read {manifest['bytes_range_read'] / 2**20:.1f} MiB; dtypes " f"{manifest['tensors']}")


def prepare_case(case: gc.GdnCase) -> None:
    start = time.perf_counter()
    state_dict = gc.load_layer_weights(case.model, case.weights)
    identity = gc.case_identity(case, gc.weights_fingerprint(state_dict))
    path = gc.reference_cache_path(case, identity)
    if path.is_file():
        log(f"{case.name}: reference cache hit {path}")
        return
    log(f"{case.name}: reference cache miss; computing (weights {time.perf_counter() - start:.1f} s)")
    x = gc.build_inputs(case)
    reference = gc.compute_reference(case, state_dict, x)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"identity": identity, "inputs": x, **reference}, path)
    log(f"{case.name}: reference written in {time.perf_counter() - start:.1f} s -> {path}")


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch-config", action="append", default=[], choices=list(gc.MODELS))
    parser.add_argument("--fetch-weights", action="append", default=[], choices=list(gc.MODELS))
    parser.add_argument("--case", action="append", default=[], help="registered case name (repeatable)")
    parser.add_argument("--model", choices=list(gc.MODELS), help="all cases of one model")
    parser.add_argument("--weights", choices=("synthetic", "real"), help="restrict --model to one weight source")
    parser.add_argument("--list", action="store_true")
    arguments = parser.parse_args()
    if arguments.list:
        print("\n".join(gc.CASES))
        return
    # Wide thread counts thrash on the per-token [Nv, K, V] recurrence (measured 23 ms/token at 32 vs 0.16 at 8).
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    for model in arguments.fetch_config:
        fetch_config(model)
    for model in arguments.fetch_weights:
        fetch_weights(model)
    unknown = [name for name in arguments.case if name not in gc.CASES]
    if unknown:
        raise SystemExit(f"unknown case(s) {unknown}; see --list")
    selected = [gc.CASES[name] for name in arguments.case]
    if arguments.model:
        selected += [
            c
            for c in gc.CASES.values()
            if c.model == arguments.model and (arguments.weights is None or c.weights == arguments.weights)
        ]
    for index, case in enumerate(selected, start=1):
        log(f"[{index}/{len(selected)}] {case.name} start")
        prepare_case(case)
    handles = _device_handles()
    if handles:
        raise SystemExit(f"prepare opened device handles {handles}; it must run without a device")
    log("prepare: done; no device handle was opened")


if __name__ == "__main__":
    main()
