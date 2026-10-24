# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Pinned official checkpoint resolution; no remote Python execution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

CHECKPOINTS = {
    "Base": ("OpenMOSS-Team/MOSS-Image-Base", "1ceede80badddd6a7812069619edaa5633587192"),
    "SFT": ("OpenMOSS-Team/MOSS-Image-SFT", "fc5de0d56fc910cbca6f6625a70f5f582b26deb1"),
}
VENDORED_PIPELINE_SHA256 = "edd48ba53f0c11fd6382f08b80cbeeea5189015bccb234587570a0d282ee9324"
PIPELINE_SHA256 = "1dac1aca9feb1bfdfc727a401e4b67d61a505d3190706d1b7d2d784f88678b31"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_checkpoint(variant="Base", checkpoint=None, *, local_files_only=False) -> Path:
    """Resolve a local checkpoint or the fixed Hub revision, then check completeness.

    A user-supplied directory is explicitly identified as local in capture metadata;
    its actual file hashes, rather than an assumed Hub identity, identify its weights.
    """
    if variant not in CHECKPOINTS:
        raise ValueError(f"Unknown checkpoint variant: {variant!r}; expected Base or SFT")
    if checkpoint is None:
        from huggingface_hub import snapshot_download

        repo_id, revision = CHECKPOINTS[variant]
        checkpoint = snapshot_download(
            repo_id,
            revision=revision,
            local_files_only=local_files_only,
            allow_patterns=[
                "transformer/*",
                "text_encoder/*",
                "tokenizer/*",
                "vae/*",
                "scheduler/*.json",
                "model_index.json",
            ],
        )
    path = Path(checkpoint).expanduser().resolve()
    required = [
        "transformer/config.json",
        "text_encoder/config.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "scheduler/scheduler_config.json",
    ]
    for relative in required:
        if not (path / relative).is_file():
            raise FileNotFoundError(f"Incomplete MOSS-Image checkpoint: missing {path / relative}")
    for folder, index_name in (
        ("transformer", "diffusion_pytorch_model.safetensors.index.json"),
        ("text_encoder", "model.safetensors.index.json"),
    ):
        index = path / folder / index_name
        if index.is_file():
            required += [f"{folder}/{index_name}"]
            required += [
                f"{folder}/{name}" for name in sorted(set(json.loads(index.read_text())["weight_map"].values()))
            ]
        else:
            required += [f"{folder}/{'diffusion_pytorch_model' if folder == 'transformer' else 'model'}.safetensors"]
    for relative in required:
        if not (path / relative).is_file():
            raise FileNotFoundError(f"Incomplete MOSS-Image checkpoint: missing {path / relative}")
    return path


def checkpoint_manifest(path: Path, variant: str, *, components=None) -> dict:
    """Hash all consumed configs, tokenizer files and tensor shards (including symlinks)."""
    files = {}
    path = Path(path)
    selected = set()
    components = set(components or ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"))
    if components - {"transformer", "text_encoder", "tokenizer", "vae", "scheduler"}:
        raise ValueError("Unknown checkpoint component")
    for component in sorted(components & {"transformer", "text_encoder", "vae"}):
        weights = SafeWeights(path / component)
        selected.update(weights.files)
        if weights.index_path is not None:
            selected.add(weights.index_path)
        selected.add(path / component / "config.json")
    for component in sorted(components & {"tokenizer", "scheduler"}):
        selected.update(
            file
            for file in (path / component).glob("*")
            if file.is_file() and file.suffix in {".json", ".jinja", ".txt", ".model"}
        )
    for file in sorted(selected):
        files[str(file.relative_to(path))] = {"sha256": sha256(file), "bytes": file.stat().st_size}
    repo_id, revision = CHECKPOINTS[variant]
    return {
        "declared_variant": variant,
        "expected_hub_repo": repo_id,
        "expected_hub_revision": revision,
        "actual_checkpoint_path": str(path),
        "files": files,
    }


class SafeWeights:
    """Select only index-declared shards, rejecting unsafe paths and duplicate keys.

    The index is authoritative: adding a safetensors file to a checkpoint cannot
    override its weights. A non-sharded checkpoint uses its exact canonical name.
    Tensor shapes are checked by each consuming architecture before execution.
    """

    def __init__(self, root):
        from safetensors import safe_open

        self.root = Path(root)
        prefix = "model" if self.root.name == "text_encoder" else "diffusion_pytorch_model"
        self.index_path = self.root / (prefix + ".safetensors.index.json")
        declared = None
        if self.index_path.is_file():
            declared = json.loads(self.index_path.read_text())["weight_map"]
            if not isinstance(declared, dict) or not declared:
                raise ValueError("Checkpoint index must contain a nonempty weight_map")
            names = sorted(set(declared.values()))
        else:
            self.index_path = None
            names = [prefix + ".safetensors"]
        self.files = []
        self.mapping = {}
        self.shapes = {}
        for name in names:
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
                raise ValueError(f"Invalid checkpoint shard path: {name!r}")
            file = self.root / name
            if not file.is_file():
                raise FileNotFoundError(file)
            self.files.append(file)
            with safe_open(file, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key in self.mapping:
                        raise ValueError(f"Duplicate checkpoint key: {key}")
                    if declared is not None and declared.get(key) != name:
                        raise ValueError(f"Shard/index disagreement for {key}")
                    self.mapping[key] = file
                    self.shapes[key] = tuple(handle.get_slice(key).get_shape())
        if declared is not None and set(declared) != set(self.mapping):
            raise ValueError("Checkpoint index declares missing tensor keys")

    def get(self, key):
        from safetensors import safe_open

        with safe_open(self.mapping[key], framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def require_shape(self, key, shape):
        if self.shapes.get(key) != tuple(shape):
            raise ValueError(f"Unsupported shape for {key}: {self.shapes.get(key)}; expected {tuple(shape)}")


def verify_checkpoint_manifest(checkpoint, manifest, components=None):
    """Verify every consumed file against a capture, including its index/config.

    Refuse missing manifest records instead of treating the capture's file list
    as an allowlist that could omit an actual runtime dependency.
    """
    root = Path(checkpoint)
    record = manifest.get("checkpoint", manifest)
    variant = record["declared_variant"]
    selected = set(components or ("transformer", "text_encoder", "tokenizer", "vae", "scheduler"))
    current = checkpoint_manifest(root, variant, components=selected)
    actual = {name: item for name, item in current["files"].items() if name.split("/")[0] in selected}
    expected = {name: item for name, item in record["files"].items() if name.split("/")[0] in selected}
    if actual != expected:
        differing = sorted(name for name in set(actual) | set(expected) if actual.get(name) != expected.get(name))
        raise ValueError(f"Checkpoint differs from the verified capture: {differing}")
    return actual


def validate_tensor_shapes(weights, component, prefix=""):
    """Validate the complete learned component against the pinned architecture."""
    schema = json.loads(Path(__file__).with_name("tensor_shapes.json").read_text())[component]
    for name, shape in schema.items():
        if component == "text_encoder":
            name = prefix + name.removeprefix("language_model.")
        weights.require_shape(name, shape)
