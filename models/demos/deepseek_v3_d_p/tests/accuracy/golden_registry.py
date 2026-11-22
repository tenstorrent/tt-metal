# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context golden registry: where FP64 goldens live, what every golden's manifest says, how a test finds one.

A golden is a directory ``<root>/<golden_id>/`` written segment by segment by a CPU producer (for example
``tests/kda/long_context_golden.py``). Tests name a golden only by its id; ``open_golden`` finds it under the
registry root and refuses one that is missing, incomplete or malformed, so a deleted or half-written golden fails
the test instead of skipping it. The registry checks identity, completeness and the product schema; reading the
products stays with each producer, whose file layouts differ.

``manifest.json`` fields every golden carries (a producer may add its own):

* ``golden_id``: the directory name (a copied or renamed directory no longer matches);
* ``producer``, ``producer_version``: the module that wrote it and the integer version of its numerics;
* ``identity``: what the products depend on, at least ``model``, ``token_source`` (provenance of the prompt) and
  ``weights_sha256``; a consumer compares the digests with what it loads itself;
* ``rows``: prompt rows covered; ``token_ids_sha256``: digest of the prompt's int64 token ids, which the golden
  holds in ``token_ids.pt``;
* ``segment`` and ``segments``: the segment size and the complete segments, ``[{start, end, ...}]`` with segment
  ``i`` covering ``[i * segment, min((i + 1) * segment, rows))``;
* ``products``: ``{name: {kind, dtype, shape}}``, ``kind`` one of ``PRODUCT_KINDS`` (``row``: one value per prompt
  row; ``segment_end``: one value after each segment, such as a carried state; ``sampled_row``: rows the producer
  lists), ``shape`` without the row axis.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT_ENV = "TT_ACCURACY_GOLDEN_ROOT"
DEFAULT_ROOT = Path("/mnt/weka/model-cache/scratch/accuracy-goldens")
MANIFEST = "manifest.json"
TOKEN_IDS = "token_ids.pt"
PRODUCT_KINDS = ("row", "segment_end", "sampled_row")


class GoldenError(Exception):
    """A golden cannot serve the request."""


class GoldenNotFound(GoldenError):
    pass


class GoldenInvalid(GoldenError):
    pass


class GoldenMismatch(GoldenError):
    pass


def golden_root() -> Path:
    """The registry root: ``$TT_ACCURACY_GOLDEN_ROOT``, else ``DEFAULT_ROOT``."""
    return Path(os.environ.get(ROOT_ENV) or DEFAULT_ROOT)


def token_ids_sha256(token_ids: Sequence[int] | torch.Tensor) -> str:
    return hashlib.sha256(torch.as_tensor(token_ids, dtype=torch.int64).numpy().tobytes()).hexdigest()


def write_manifest(directory: Path, manifest: dict) -> None:
    """Write ``manifest.json`` atomically (a reader sees the old or the new manifest, never a partial one)."""
    temporary = Path(directory) / f"{MANIFEST}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    temporary.replace(Path(directory) / MANIFEST)


def write_token_ids(directory: Path, token_ids: Sequence[int] | torch.Tensor) -> str:
    """Store the prompt's token ids in the golden atomically and return their digest."""
    ids = torch.as_tensor(token_ids, dtype=torch.int64).clone()
    path = Path(directory) / TOKEN_IDS
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    try:
        torch.save(ids, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return token_ids_sha256(ids)


def product_schema(kind: str, tensor: torch.Tensor) -> dict:
    """The ``products`` entry of a tensor whose leading axis is rows (``row``, ``sampled_row``) or that is one
    value per segment end (``segment_end``)."""
    if kind not in PRODUCT_KINDS:
        raise ValueError(f"product kind {kind!r} is not one of {PRODUCT_KINDS}")
    shape = list(tensor.shape[1:] if kind != "segment_end" else tensor.shape)
    return {"kind": kind, "dtype": str(tensor.dtype).removeprefix("torch."), "shape": shape}


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def manifest_problems(golden_id: str, manifest: dict) -> list[str]:
    """Every way ``manifest`` breaks the registry contract for a complete golden named ``golden_id``."""
    problems = []
    if manifest.get("golden_id") != golden_id:
        problems.append(f"golden_id {manifest.get('golden_id')!r} is not the directory name {golden_id!r}")
    if not isinstance(manifest.get("producer"), str) or not manifest["producer"]:
        problems.append("producer must be a non-empty string")
    if not _is_int(manifest.get("producer_version")):
        problems.append("producer_version must be an integer")
    identity = manifest.get("identity")
    if not isinstance(identity, dict):
        problems.append("identity must be an object")
    else:
        problems += [
            f"identity lacks {key}" for key in ("model", "token_source", "weights_sha256") if key not in identity
        ]
    digest = manifest.get("token_ids_sha256")
    if not (isinstance(digest, str) and len(digest) == 64):
        problems.append("token_ids_sha256 must be a sha256 hex digest")
    rows, segment, segments = manifest.get("rows"), manifest.get("segment"), manifest.get("segments")
    sizes_valid = _is_int(rows) and rows > 0 and _is_int(segment) and segment > 0
    if not (_is_int(rows) and rows > 0):
        problems.append("rows must be a positive integer")
    if not (_is_int(segment) and segment > 0):
        problems.append("segment must be a positive integer")
    if not isinstance(segments, list) or not segments:
        problems.append("segments must be a non-empty list")
    elif sizes_valid:
        for index, entry in enumerate(segments):
            want = (index * segment, min((index + 1) * segment, rows))
            got = (entry.get("start"), entry.get("end")) if isinstance(entry, dict) else None
            if got != want:
                problems.append(f"segment {index} covers {got}, expected [{want[0]}, {want[1]})")
                break
        else:
            if segments[-1]["end"] != rows:
                problems.append(f"the segments end at row {segments[-1]['end']}, not at rows {rows} (incomplete)")
    products = manifest.get("products")
    if not isinstance(products, dict) or not products:
        problems.append("products must be a non-empty object")
    else:
        for name, schema in products.items():
            if not (
                isinstance(schema, dict)
                and schema.get("kind") in PRODUCT_KINDS
                and isinstance(schema.get("dtype"), str)
                and isinstance(schema.get("shape"), list)
                and all(_is_int(n) and n >= 0 for n in schema["shape"])
            ):
                problems.append(f"product {name!r} needs kind in {PRODUCT_KINDS}, a dtype string and an integer shape")
    return problems


@dataclass(frozen=True)
class Golden:
    """One registered, complete golden: its directory and validated manifest."""

    path: Path
    manifest: dict

    @property
    def golden_id(self) -> str:
        return self.manifest["golden_id"]

    @property
    def rows(self) -> int:
        return self.manifest["rows"]

    @property
    def identity(self) -> dict:
        return self.manifest["identity"]

    def require(
        self, *, producer: str, producer_version: int, identity: dict | None = None, products: Iterable[str] = ()
    ) -> None:
        """Raise ``GoldenMismatch`` naming every field that differs from what the consumer was calibrated on."""
        mismatched = []
        for key, want in (("producer", producer), ("producer_version", producer_version)):
            if self.manifest[key] != want:
                mismatched.append(f"{key} is {self.manifest[key]!r}, expected {want!r}")
        for key, want in (identity or {}).items():
            if self.identity.get(key) != want:
                mismatched.append(f"identity.{key} is {self.identity.get(key)!r}, expected {want!r}")
        mismatched += [f"no product {name!r}" for name in products if name not in self.manifest["products"]]
        if mismatched:
            raise GoldenMismatch(f"golden {self.golden_id} at {self.path}: " + "; ".join(mismatched))

    def token_ids(self) -> torch.Tensor:
        """The prompt's ``rows`` token ids (int64, 1-D), verified against ``token_ids_sha256``."""
        path = self.path / TOKEN_IDS
        if not path.is_file():
            raise GoldenInvalid(f"golden {self.golden_id} has no {TOKEN_IDS}")
        ids = torch.load(path, weights_only=True)
        if not (torch.is_tensor(ids) and ids.dtype == torch.int64 and ids.shape == (self.rows,)):
            raise GoldenInvalid(f"{path} must hold {self.rows} int64 token ids")
        if token_ids_sha256(ids) != self.manifest["token_ids_sha256"]:
            raise GoldenInvalid(f"{path} does not match the manifest's token_ids_sha256")
        return ids


def open_golden(golden_id: str, root: Path | None = None) -> Golden:
    """The complete golden ``golden_id`` under ``root`` (default ``golden_root()``).

    Raises ``GoldenNotFound`` without a directory or manifest and ``GoldenInvalid`` when the manifest breaks the
    contract (including an incomplete golden).
    """
    path = Path(root or golden_root()) / golden_id
    manifest_path = path / MANIFEST
    if not manifest_path.is_file():
        raise GoldenNotFound(f"no golden {golden_id!r}: {manifest_path} does not exist")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise GoldenInvalid(f"{manifest_path} is not JSON: {error}") from error
    problems = manifest_problems(golden_id, manifest) if isinstance(manifest, dict) else ["not a JSON object"]
    if problems:
        raise GoldenInvalid(f"golden {golden_id!r} at {path}: " + "; ".join(problems))
    return Golden(path, manifest)
