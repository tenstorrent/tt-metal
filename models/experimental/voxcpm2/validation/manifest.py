# SPDX-License-Identifier: Apache-2.0
"""Versioned on-disk tensor records, with integrity checks."""
import hashlib
import json
from pathlib import Path
import torch

SCHEMA_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class TensorRecorder:
    """Snapshot nested inputs immediately, before upstream can mutate KV buffers."""
    def __init__(self, directory):
        self.directory = Path(directory)
        (self.directory / "tensors").mkdir(parents=True, exist_ok=True)
        self.tensors = {}
        self.events = []

    def snapshot(self, value, key):
        if isinstance(value, torch.Tensor):
            filename = f"tensors/{len(self.tensors):08d}.pt"
            cpu = value.detach().cpu().clone()
            torch.save(cpu, self.directory / filename)
            self.tensors[key] = {
                "file": filename, "shape": list(value.shape), "dtype": str(value.dtype),
                "source_device": str(value.device), "sha256": sha256_file(self.directory / filename),
            }
            return {"tensor": key}
        if isinstance(value, dict):
            return {str(k): self.snapshot(v, f"{key}/{k}") for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return {"sequence_type": type(value).__name__, "items": [self.snapshot(v, f"{key}/{i}") for i, v in enumerate(value)]}
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Cannot capture {key}: unsupported {type(value).__name__}")

    def begin(self, component, args, kwargs):
        key = f"{component}/{sum(e['component'] == component for e in self.events):06d}"
        event = {"component": component, "key": key,
                 "args": self.snapshot(args, key + "/args"),
                 "kwargs": self.snapshot(kwargs, key + "/kwargs")}
        self.events.append(event)
        return event

    def end(self, event, result):
        event["output"] = self.snapshot(result, event["key"] + "/output")


def load_manifest(directory, *, require_cuda=True):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported capture schema")
    if manifest.get("complete") is not True:
        raise ValueError("Incomplete capture")
    if require_cuda:
        if manifest.get("oracle") != "official_voxcpm2_cuda" or not manifest.get("cuda", {}).get("device_name"):
            raise ValueError("Reference must be an official CUDA capture")
        if not manifest.get("source", {}).get("revision") or not manifest.get("checkpoint", {}).get("revision"):
            raise ValueError("Reference revisions are missing")
    for name, record in manifest["tensors"].items():
        path = directory / record["file"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError(f"Tensor path escapes capture: {name}")
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"Tensor checksum mismatch: {name}")
    return manifest


def load_tensor(directory, manifest, name):
    record = manifest["tensors"][name]
    value = torch.load(Path(directory) / record["file"], map_location="cpu", weights_only=True)
    if list(value.shape) != record["shape"] or str(value.dtype) != record["dtype"]:
        raise ValueError(f"Tensor metadata mismatch: {name}")
    return value


def materialize_snapshot(directory, manifest, value):
    """Restore a captured nested value on the host for component replay."""
    if isinstance(value, dict):
        if set(value) == {"tensor"}:
            return load_tensor(directory, manifest, value["tensor"])
        if set(value) == {"sequence_type", "items"}:
            items = [materialize_snapshot(directory, manifest, item) for item in value["items"]]
            if value["sequence_type"] == "tuple":
                return tuple(items)
            if value["sequence_type"] == "list":
                return items
            raise ValueError("Unsupported captured sequence type")
        return {key: materialize_snapshot(directory, manifest, item) for key, item in value.items()}
    return value
