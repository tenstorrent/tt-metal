# SPDX-License-Identifier: Apache-2.0
"""Opt-in setup-only cache of immutable TP4 routed-expert BF4 weights.

Source shard contents are hashed when sealing a setup identity. Subsequent
loads require identical file stats and safetensors headers, and verify every
cached payload checksum. This is a local trusted-filesystem cache, not a
replacement for checkpoint distribution authenticity. Never call during a
forward pass: cold fills deliberately read back immutable weight tensors.
"""

import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
MODEL = Path(__file__).resolve().parents[1]
REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
_CACHES = {}


def digest_file(path):
    sha = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(block)
    return sha.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest_object(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(canonical({"sha256": digest_object(payload), "payload": payload}) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def checked_json(path):
    wrapped = json.loads(Path(path).read_text())
    if set(wrapped) != {"sha256", "payload"} or wrapped["sha256"] != digest_object(wrapped["payload"]):
        raise ValueError(f"Expert cache JSON checksum mismatch: {path}")
    return wrapped["payload"]


def source_fingerprint(path):
    path = Path(path)
    stat = path.stat()
    with path.open("rb") as stream:
        size = int.from_bytes(stream.read(8), "little")
        if not 0 < size < min(stat.st_size, 64 * 1024 * 1024):
            raise ValueError(f"Invalid safetensors header: {path}")
        header = stream.read(size)
        if len(header) != size:
            raise ValueError(f"Truncated safetensors header: {path}")
        json.loads(header)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }


def seal_sources(checkpoint, cache_root, mode):
    """Hash source contents once; every invocation validates source fingerprints."""
    checkpoint = Path(checkpoint).resolve()
    if checkpoint.name != REVISION:
        raise ValueError("Expert cache requires the pinned checkpoint snapshot directory")
    index_path = checkpoint / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    names = sorted(
        {
            value
            for key, value in weight_map.items()
            if re.fullmatch(r"model\.language_model\.layers\.\d+\.mlp\.experts\.(gate_up_proj|down_proj)", key)
        }
    )
    if not names:
        raise ValueError("Checkpoint index has no routed-expert source tensors")
    fingerprints = {name: source_fingerprint(checkpoint / name) for name in names}
    base = {
        "revision": REVISION,
        "checkpoint": str(checkpoint),
        "index_sha256": digest_file(index_path),
        "config_sha256": digest_file(checkpoint / "config.json"),
        "files": fingerprints,
    }
    path = Path(cache_root) / "source_seals" / (digest_object(base) + ".json")
    if path.exists():
        sealed = checked_json(path)
        if sealed["fingerprints"] != base or set(sealed["content_sha256"]) != set(names):
            raise ValueError("Expert source seal does not match current checkpoint")
        return sealed
    if mode == "read":
        raise FileNotFoundError(f"Required expert checkpoint source seal is absent: {path}")
    contents = {}
    for name in names:
        contents[name] = digest_file(checkpoint / name)
        if source_fingerprint(checkpoint / name) != fingerprints[name]:
            raise RuntimeError(f"Checkpoint changed while hashing {name}")
    sealed = {"fingerprints": base, "content_sha256": contents}
    atomic_json(path, sealed)
    return sealed


def locate_state(state):
    """Resolve PrefixViews without requesting any checkpoint tensor payload."""
    prefix = ""
    while hasattr(state, "state") and hasattr(state, "prefix"):
        prefix = state.prefix + prefix
        state = state.state
    from .checkpoint import CheckpointState

    if not isinstance(state, CheckpointState):
        raise TypeError("Expert cache requires CheckpointState-backed immutable weights")
    prefix = state.prefix + prefix
    match = re.fullmatch(r"model\.language_model\.layers\.(\d+)\.", prefix)
    if match is None or not 0 <= int(match[1]) < 48:
        raise ValueError(f"Cannot identify exact expert layer from prefix {prefix!r}")
    return state.path.resolve(), int(match[1])


class ExpertWeightCache:
    def __init__(self, checkpoint, root, mode):
        begin = time.perf_counter()
        if mode not in ("fill", "read"):
            raise ValueError("Expert cache mode must be fill or read")
        self.root, self.mode = Path(root).resolve(), mode
        if self.root == ROOT or ROOT in self.root.parents:
            raise ValueError("Expert tensor cache must reside outside the git checkout")
        self.root.mkdir(parents=True, exist_ok=True)
        sources = seal_sources(checkpoint, self.root, mode)
        native = [ROOT / "build_Release/lib/_ttnncpp.so", ROOT / "ttnn/ttnn/_ttnn.so"]
        transform = [Path(__file__), MODEL / "tt/multichip_decoder.py", MODEL / "tt/precision.py"]
        self.identity = {
            "schema": 1,
            "source_seal": sources,
            "source_seal_sha256": digest_object(sources),
            "transform_sha256": {str(p.resolve()): digest_file(p) for p in transform},
            "native_sha256": {str(p.resolve()): digest_file(p) for p in native},
            "precision_sha256": digest_file(MODEL / "doc/datatype_sweep/selected_precision_config.json"),
            "mesh": [1, 4],
            "arch": "BLACKHOLE",
            "dtype": "bfloat4_b",
            "layout": "TILE_LAYOUT",
            "memory": "DRAM_INTERLEAVED",
            "shard_order": "rank r owns global expert r*128+local_expert; shard tensor dim0",
            "conversion": "exact existing ops.shards from checkpoint BF16; no approximate BF4 conversion",
        }
        self.identity_hash = digest_object(self.identity)
        self.directory = self.root / "identities" / self.identity_hash
        self.index_path = self.directory / "index.json"
        # One setup writer per identity; concurrent fill must fail explicitly
        # instead of dropping another writer's entries during atomic publication.
        self._fill_lock = None
        if mode == "fill":
            self.directory.mkdir(parents=True, exist_ok=True)
            self._fill_lock = (self.directory / "fill.lock").open("a")
            fcntl.flock(self._fill_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if self.index_path.exists():
            index = checked_json(self.index_path)
            if index["identity"] != self.identity:
                raise ValueError("Expert cache identity mismatch")
            self.entries = index["entries"]
        elif mode == "read":
            raise FileNotFoundError(f"Required expert weight index is absent: {self.index_path}")
        else:
            self.entries = {}
        self.stats = {
            "root": str(self.root),
            "mode": mode,
            "identity_sha256": self.identity_hash,
            "source_seal_sha256": self.identity["source_seal_sha256"],
            "source_validation": "setup full SHA256; this launch exact stat/header/index/config match",
            "hits": 0,
            "misses": 0,
            "loaded_bytes": 0,
            "written_bytes": 0,
            "initialization_seconds": time.perf_counter() - begin,
            "layer_seconds": {},
            "loaded_layers": [],
        }

    @staticmethod
    def metadata(tensor, ops):
        import ttnn

        if tuple(ops.mesh.shape) != (1, 4) or ops.mesh.get_num_devices() != 4:
            raise ValueError("Expert cache is qualified only for a1x4 mesh")
        if ops.mesh.arch() != ttnn.device.Arch.BLACKHOLE:
            raise ValueError("Expert cache is qualified only for Blackhole")
        if (
            tensor.dtype != ttnn.bfloat4_b
            or tensor.layout != ttnn.TILE_LAYOUT
            or tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG
        ):
            raise ValueError("Expert cache tensor must be tiled BF4 interleaved DRAM")
        shards = ttnn.get_device_tensors(tensor)
        if len(shards) != 4:
            raise ValueError("Expert cache must restore four device shards")
        return {
            "shape": list(tensor.shape),
            "padded_shape": list(tensor.padded_shape),
            "dtype": str(tensor.dtype),
            "layout": str(tensor.layout),
            "memory_config": str(tensor.memory_config()),
            "shard_shapes": [list(t.shape) for t in shards],
        }

    def get(self, key, expected_shape, producer, ops):
        import ttnn

        entry = self.entries.get(key)
        if entry is not None:
            digest = entry["sha256"]
            if re.fullmatch("[0-9a-f]{64}", digest) is None:
                raise ValueError("Malformed cache blob digest")
            path = self.root / "blobs" / f"{digest}.tensorbin"
            if path.stat().st_size != entry["bytes"] or digest_file(path) != digest:
                raise ValueError(f"Expert cache payload checksum mismatch: {key}")
            value = ttnn.load_tensor(path, device=ops.mesh)
            if self.metadata(value, ops) != entry["metadata"] or tuple(value.shape) != expected_shape:
                raise ValueError(f"Expert cache tensor specification mismatch: {key}")
            self.stats["hits"] += 1
            self.stats["loaded_bytes"] += entry["bytes"]
            return value
        if self.mode == "read":
            raise FileNotFoundError(f"Required expert weight cache entry is absent: {key}")
        value = producer()
        metadata = self.metadata(value, ops)
        if tuple(value.shape) != expected_shape:
            raise ValueError(f"Unexpected expert weight shape: {key}")
        blobs = self.root / "blobs"
        blobs.mkdir(exist_ok=True)
        temporary = blobs / f".{os.getpid()}.{uuid.uuid4().hex}.tensorbin"
        try:
            # This setup-only readback preserves existing exact packed weights.
            ttnn.dump_tensor(temporary, value)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            digest = digest_file(temporary)
            size = temporary.stat().st_size
            path = blobs / f"{digest}.tensorbin"
            if path.exists() and (path.stat().st_size != size or digest_file(path) != digest):
                raise ValueError(f"Existing content-addressed blob is corrupt: {path}")
            os.replace(temporary, path)
            self.entries[key] = {"sha256": digest, "bytes": size, "metadata": metadata}
            self.stats["misses"] += 1
            self.stats["written_bytes"] += size
        finally:
            temporary.unlink(missing_ok=True)
        return value

    def load_layer(self, layer, state, ops):
        import ttnn

        begin = time.perf_counter()
        checkpoint_weights = {}

        def source(name):
            if name not in checkpoint_weights:
                checkpoint_weights[name] = state[name]
            return checkpoint_weights[name]

        result = [[], [], []]
        for expert in range(128):
            for target, role in zip(result, ("gate", "up", "down")):

                def produce(role=role, expert=expert):
                    if role == "down":
                        weights = source("mlp.experts.down_proj")
                        shards = [weights[r * 128 + expert].T for r in range(4)]
                    else:
                        weights = source("mlp.experts.gate_up_proj")
                        region = slice(0, 640) if role == "gate" else slice(640, None)
                        shards = [weights[r * 128 + expert, region].T for r in range(4)]
                    return ops.shards(shards, ttnn.bfloat4_b)

                shape = (640, 2560) if role == "down" else (2560, 640)
                target.append(self.get(f"layer{layer}/{expert}/{role}", shape, produce, ops))
        if self.mode == "fill":
            atomic_json(self.index_path, {"identity": self.identity, "entries": self.entries})
        self.stats["layer_seconds"][str(layer)] = time.perf_counter() - begin
        self.stats["loaded_layers"].append(layer)
        self.stats["index_sha256"] = digest_file(self.index_path)
        return result


def cached_experts(state, ops):
    path = os.environ.get("QWEN_FLASH_EXPERT_CACHE")
    if not path:
        return None
    mode = os.environ.get("QWEN_FLASH_EXPERT_CACHE_MODE", "read")
    checkpoint, layer = locate_state(state)
    key = (str(checkpoint), str(Path(path).resolve()), mode)
    if key not in _CACHES:
        _CACHES[key] = ExpertWeightCache(checkpoint, path, mode)
    return _CACHES[key].load_layer(layer, state, ops)


def runtime_cache_summary():
    return [dict(cache.stats) for cache in _CACHES.values()]
