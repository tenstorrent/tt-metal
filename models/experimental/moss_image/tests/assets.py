# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fail-closed, read-only access to pinned CUDA qualification assets."""

import hashlib
import json
from pathlib import Path

import torch


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contained(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError(f"Asset path escapes its root: {relative}")
    return path


class Capture:
    def __init__(self, directory, manifest_sha256, variant):
        self.directory = Path(directory)
        path = self.directory / "manifest.json"
        if digest(path) != manifest_sha256:
            raise ValueError("Capture manifest digest mismatch")
        self.manifest = json.loads(path.read_text())
        from models.experimental.moss_image.tt.checkpoint import PIPELINE_SHA256

        if self.manifest["pipeline_sha256"] != PIPELINE_SHA256:
            raise ValueError("Capture pipeline identity mismatch")
        arguments = self.manifest["arguments"]
        expected = {
            "variant": variant,
            "height": 1024,
            "width": 1024,
            "steps": 50,
            "seed": 42,
            "prompt": "the quick brown fox jumps over the lazy dog",
        }
        if any(arguments.get(key) != value for key, value in expected.items()):
            raise ValueError("Capture workload configuration mismatch")
        if (
            self.manifest["status"] != "complete"
            or self.manifest["guidance_scale"] != 3.0
            or self.manifest["negative_skip_layers"] != [22]
            or self.manifest["checkpoint"]["declared_variant"] != variant
        ):
            raise ValueError("Capture checkpoint or guidance configuration mismatch")

    def load(self, key):
        record = self.manifest["artifacts"][key]
        path = contained(self.directory, record["path"])
        if digest(path) != record["sha256"]:
            raise ValueError(f"Capture tensor digest mismatch: {key}")
        return torch.load(path, map_location="cpu", weights_only=True)


def metrics(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError(f"Tensor shape mismatch: {actual.shape} != {reference.shape}")
    actual, reference = actual.double().flatten(), reference.double().flatten()
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise ValueError("Nonfinite qualification tensor")
    left, right = actual - actual.mean(), reference - reference.mean()
    denominator = left.norm() * right.norm()
    pcc = float(left.dot(right) / denominator) if denominator else float(torch.equal(actual, reference))
    return {
        "pcc": pcc,
        "relative_rms": float((actual - reference).norm() / reference.norm().clamp_min(1e-30)),
        "max_absolute_error": float((actual - reference).abs().max()),
    }


def check(actual, reference, record_property, *, min_pcc=0.99, max_relative_rms=0.03):
    values = metrics(actual, reference)
    for key, value in values.items():
        record_property(key, value)
    assert values["pcc"] >= min_pcc, values
    assert values["relative_rms"] <= max_relative_rms, values
    return values
