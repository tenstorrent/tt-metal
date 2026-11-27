# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Long-context golden registry: where FP64 goldens live, what every golden's manifest says, how a test finds one.

A golden is a directory ``<root>/<golden_id>/`` written segment by segment by a CPU producer (one per model folder,
``deepseek_v3_d_p/tests/long_context_accuracy/<model>/golden_producer.py``). Tests name a golden only by its id;
``open_golden`` finds it under the registry root and refuses one that is missing, incomplete or malformed, so a deleted
or half-written golden fails the test instead of skipping it. The registry checks identity, completeness and the product
schema. Producers' file layouts differ, so reading the products is one small reader per golden layout: a subclass of
``Fp64Golden`` that names its producer and implements ``_read_rows`` (and ``_segment_end`` / ``read_sampled`` where a
consumer needs them); ``open_golden(id, reader=...)`` returns it. Consumers read through ``Fp64Golden.read_rows`` only.

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

Goldens are shared: CI users other than the producer read them. Every file in a golden is mode ``GOLDEN_FILE_MODE``
(0664) and every directory the writers create (the golden, the registry root and any missing ancestor) is
``GOLDEN_DIR_MODE`` (2775), whatever the producer's umask; ``make_golden_dir`` and ``publish`` set them, and
producers write through them. An existing directory keeps its mode.

Producers start or resume a golden with ``start_or_resume``: a resume continues only the same run, and each run
session's provenance (``session_provenance``) is recorded with the segment it started from.

Readers copy row slices out of ``torch.save`` files with ``read_slices``: the file is opened with ``MappedFile``
(``torch.load(mmap=True)``, which reads only the metadata), the reader slices the mapped tensors as usual, and
``read_slices`` reads exactly those bytes with ``READ_STREAMS`` parallel ``preadv`` streams instead of faulting the
mapped pages in one at a time.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import torch

ROOT_ENV = "TT_ACCURACY_GOLDEN_ROOT"
DEFAULT_ROOT = Path("/mnt/weka/model-cache/scratch/accuracy-goldens")
MANIFEST = "manifest.json"
TOKEN_IDS = "token_ids.pt"
PRODUCT_KINDS = ("row", "segment_end", "sampled_row")
GOLDEN_FILE_MODE = 0o664
GOLDEN_DIR_MODE = 0o2775
# Parallel read streams of ``read_slices``. Cold reads of one chunk's products from Weka on the dev host (m06n.19,
# page cache evicted, median of 3): KDA FP64 5120 rows (306 MB) 0.87 s through mapped pages, 0.27 / 0.19 / 0.13 / 0.11 s
# with 1 / 2 / 4 / 8 streams; K3.1 MoE and DSA 2560 rows (~145 MB) 0.53 / 0.90 s mapped, 0.05 s with 8 streams.
READ_STREAMS = 8


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


def make_golden_dir(directory: Path) -> None:
    """Create the golden directory and every missing ancestor (such as a new registry root), each with
    ``GOLDEN_DIR_MODE``; existing directories keep their mode."""
    directory = Path(directory)
    for path in reversed((directory, *directory.parents)):
        if not path.is_dir():
            path.mkdir(exist_ok=True)
            path.chmod(GOLDEN_DIR_MODE)


def publish(write: Callable[[Path], None], path: Path) -> None:
    """Write ``path`` atomically with ``GOLDEN_FILE_MODE``: ``write`` fills a temporary file, which then replaces
    ``path`` (a reader sees the old or the new file, never a partial or unreadable one)."""
    path = Path(path)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        temporary.chmod(GOLDEN_FILE_MODE)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest(directory: Path, manifest: dict) -> None:
    """Write ``manifest.json`` atomically and readable by every registry user."""
    publish(lambda path: path.write_text(json.dumps(manifest, indent=1), encoding="utf-8"), Path(directory) / MANIFEST)


def write_token_ids(directory: Path, token_ids: Sequence[int] | torch.Tensor) -> str:
    """Store the prompt's token ids in the golden atomically and return their digest."""
    ids = torch.as_tensor(token_ids, dtype=torch.int64).clone()
    publish(lambda path: torch.save(ids, path), Path(directory) / TOKEN_IDS)
    return token_ids_sha256(ids)


def session_provenance() -> dict:
    """What determines a golden's last bits besides its identity: the code revision, torch and the CPU backend."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {
        "code_revision": revision,
        "torch": torch.__version__,
        "cpu_capability": torch.backends.cpu.get_cpu_capability(),
        "machine": platform.machine(),
    }


def start_or_resume(directory: Path, expected: dict, token_ids: Sequence[int] | torch.Tensor, segments: int) -> dict:
    """The manifest of the golden run in ``directory``: a new one (``expected`` plus empty ``sessions`` and
    ``segments``, the token ids stored) or the stored one, which must hold every field of ``expected`` unchanged
    (``ValueError`` naming the mismatched fields otherwise: its segments would not continue this run). When fewer
    than ``segments`` segments are complete, a session starting at the next one is recorded and the manifest written.

    ``expected`` holds what fixes the run's bytes: identity, prompt digest, segment size, dtype and thread count (the
    thread count sets the FP64 summation order, so a resumed golden stays bit-identical to an uninterrupted one)."""
    directory = Path(directory)
    make_golden_dir(directory)
    path = directory / MANIFEST
    if path.is_file():
        manifest = json.loads(path.read_text(encoding="utf-8"))
        mismatched = sorted(key for key, value in expected.items() if manifest.get(key) != value)
        if mismatched:
            raise ValueError(f"{directory} holds a different run; mismatched fields: {mismatched}")
    else:
        write_token_ids(directory, token_ids)
        manifest = {**expected, "sessions": [], "segments": []}
    done = len(manifest["segments"])
    if done < segments:
        manifest["sessions"].append({"first_segment": done, **session_provenance()})
        write_manifest(directory, manifest)
    return manifest


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


class MappedFile:
    """A ``torch.save`` file loaded with ``mmap=True`` (``contents``: no tensor data is read), whose tensors' views
    ``read_slices`` copies out of the file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.contents = torch.load(self.path, weights_only=True, mmap=True)
        self.size = self.path.stat().st_size
        # torch maps the whole file once and every storage is a slice of that mapping at its record's file offset:
        # matched in address order, the storages give the mapping's address of file offset 0.
        reader = torch._C.PyTorchFileReader(str(self.path))
        offsets = sorted(reader.get_record_offset(r) for r in reader.get_all_records() if r.startswith("data/"))
        addresses = sorted({t.untyped_storage().data_ptr() for t in _tensors(self.contents)})
        bases = {address - offset for address, offset in zip(addresses, offsets)}
        if len(addresses) != len(offsets) or len(bases) != 1:
            raise GoldenInvalid(f"{self.path}: cannot locate its {len(offsets)} tensor records in the mapped file")
        self._base = bases.pop()

    def span(self, view: torch.Tensor) -> tuple[int, int]:
        """The file byte range ``[start, end)`` covering ``view``, a view of a tensor in ``contents``."""
        last = sum((n - 1) * s for n, s in zip(view.shape, view.stride()))
        start = view.data_ptr() - self._base
        end = start + (last + 1) * view.element_size()
        if not 0 <= start < end <= self.size:
            raise ValueError(f"the view is not inside {self.path}")
        return start, end


def _tensors(tree: object) -> Iterable[torch.Tensor]:
    if torch.is_tensor(tree):
        yield tree
    elif isinstance(tree, dict):
        for value in tree.values():
            yield from _tensors(value)
    elif isinstance(tree, (list, tuple)):
        for value in tree:
            yield from _tensors(value)


def read_slices(parts: Mapping[str, Sequence[tuple[MappedFile, torch.Tensor]]]) -> dict[str, torch.Tensor]:
    """Per name, the concatenation along rows of non-empty views ``(file, view)`` of mapped tensors, as a new
        contiguous tensor, equal to ``torch.cat([view, ...])`` (``GoldenInvalid`` when a view's dtype or trailing shape
    differs from the first's: ``torch.cat`` would promote or refuse). All the bytes are read with ``READ_STREAMS`` parallel
        ``preadv`` streams of equal size; a non-contiguous view is read whole over its span, then copied."""
    out: dict[str, torch.Tensor] = {}
    reads: list[tuple[Path, int, memoryview]] = []  # (file, offset, destination bytes)
    strided: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []  # (span bytes, view, destination rows)
    for name, views in parts.items():
        first = views[0][1]
        result = torch.empty((sum(v.shape[0] for _, v in views), *first.shape[1:]), dtype=first.dtype)
        row = 0
        for file, view in views:
            if view.dtype != first.dtype or view.shape[1:] != first.shape[1:]:
                raise GoldenInvalid(
                    f"{file.path}: {name} is {view.dtype}{list(view.shape)}, not {first.dtype}[*, {list(first.shape[1:])}]"
                )
            rows = result[row : row + view.shape[0]]
            row += view.shape[0]
            start, end = file.span(view)
            if view.is_contiguous():
                if end - start != rows.nbytes:
                    raise GoldenInvalid(f"{file.path}: {name} spans {end - start} bytes, not its {rows.nbytes}")
                reads.append((file.path, start, memoryview(rows.view(-1).view(torch.uint8).numpy())))
            else:
                raw = torch.empty(end - start, dtype=torch.uint8)
                reads.append((file.path, start, memoryview(raw.numpy())))
                strided.append((raw, view, rows))
        out[name] = result
    _pread(reads)
    for raw, view, rows in strided:
        rows.copy_(raw.view(view.dtype).as_strided(view.shape, view.stride()))
    return out


def _pread(reads: Sequence[tuple[Path, int, memoryview]]) -> None:
    """Fill each destination from its file offset, the total bytes split into ``READ_STREAMS`` equal streams."""
    share = -(-sum(len(m) for _, _, m in reads) // READ_STREAMS)
    streams: list[list[tuple[Path, int, memoryview]]] = [[]]
    room = share
    for path, offset, memory in reads:
        position = 0
        while position < len(memory):
            take = min(room, len(memory) - position)
            streams[-1].append((path, offset + position, memory[position : position + take]))
            position += take
            room -= take
            if room == 0:
                streams.append([])
                room = share
    descriptors: dict[Path, int] = {}  # one per file, however many views it serves; all closed on return

    def stream(pieces: list[tuple[Path, int, memoryview]]) -> None:
        for path, offset, memory in pieces:
            done = 0
            while done < len(memory):
                count = os.preadv(descriptors[path], [memory[done:]], offset + done)
                if count <= 0:
                    raise GoldenInvalid(f"{path} ends before byte {offset + done}")
                done += count

    try:
        for path, _, _ in reads:
            if path not in descriptors:
                descriptors[path] = os.open(path, os.O_RDONLY)
        with ThreadPoolExecutor(READ_STREAMS, thread_name_prefix="golden-read") as pool:
            list(pool.map(stream, [s for s in streams if s]))
    finally:
        for descriptor in descriptors.values():
            os.close(descriptor)


@dataclass(frozen=True)
class Fp64Golden:
    """One registered, complete golden: its directory and validated manifest.

    A reader subclass for one golden layout sets ``producer`` / ``producer_version`` (what ``open_golden`` requires)
    and implements ``_read_rows``, and ``_segment_end`` / ``read_sampled`` when its consumers need them; the base
    class reads no products."""

    producer: ClassVar[str | None] = None
    producer_version: ClassVar[int | None] = None

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
        self,
        *,
        producer: str | None = None,
        producer_version: int | None = None,
        identity: dict | None = None,
        weights_sha256: dict[str, str] | None = None,
        products: Iterable[str] = (),
    ) -> None:
        """Raise ``GoldenMismatch`` naming every field that differs from what the consumer was calibrated on.
        ``weights_sha256``: digests of the weights the consumer loaded itself; each must be one the golden was built
        from (a subset check: a consumer loads only the layers it needs). ``producer`` / ``producer_version`` default
        to the reader's."""
        producer = producer if producer is not None else self.producer
        producer_version = producer_version if producer_version is not None else self.producer_version
        if producer is None or producer_version is None:
            raise ValueError(f"{type(self).__name__} names no producer: pass producer and producer_version")
        mismatched = []
        for key, want in (("producer", producer), ("producer_version", producer_version)):
            if self.manifest[key] != want:
                mismatched.append(f"{key} is {self.manifest[key]!r}, expected {want!r}")
        for key, want in (identity or {}).items():
            if self.identity.get(key) != want:
                mismatched.append(f"identity.{key} is {self.identity.get(key)!r}, expected {want!r}")
        built_from = self.identity.get("weights_sha256")
        built_from = built_from if isinstance(built_from, dict) else {}  # (per-tensor-group digests to check against)
        mismatched += [
            f"weights {key} are {digest[:12]}..., the golden's {str(built_from.get(key))[:12]}..."
            for key, digest in sorted((weights_sha256 or {}).items())
            if built_from.get(key) != digest
        ]
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

    def read_rows(self, names: Iterable[str], start: int, end: int) -> dict[str, torch.Tensor]:
        """The products ``names`` for rows ``[start, end)``: a ``row`` product as its ``[end - start, ...]`` rows, a
        ``segment_end`` product as its value after row ``end`` (``end`` must end a segment), each in its stored dtype.
        Raises ``ValueError`` for a product the golden does not hold, a ``sampled_row`` product (``read_sampled``) or
        rows outside the golden."""
        names = list(names)
        products = self.manifest["products"]
        missing = [n for n in names if n not in products]
        if missing:
            raise ValueError(f"golden {self.golden_id} has no products {missing}")
        kinds = {n: products[n]["kind"] for n in names}
        sampled = [n for n, kind in kinds.items() if kind == "sampled_row"]
        if sampled:
            raise ValueError(f"{sampled} of golden {self.golden_id} are sampled-row products: use read_sampled")
        if not 0 <= start < end <= self.rows:
            raise ValueError(f"rows [{start}, {end}) are not inside the {self.rows} rows of {self.golden_id}")
        row_names = [n for n in names if kinds[n] == "row"]
        state_names = [n for n in names if kinds[n] == "segment_end"]
        out = self._read_rows(row_names, start, end) if row_names else {}
        if state_names:
            ends = [segment["end"] for segment in self.manifest["segments"]]
            if end not in ends:
                raise ValueError(f"row {end} does not end a segment of {self.golden_id} (segment-end products)")
            out.update(self._segment_end(state_names, ends.index(end)))
        return {n: out[n] for n in names}

    def read_sampled(self, end: int | None = None) -> dict[str, torch.Tensor]:
        """The ``sampled_row`` products on the sampled rows below ``end`` (default: all rows)."""
        raise GoldenError(f"{type(self).__name__} reads no sampled-row products of golden {self.golden_id}")

    def _read_rows(self, names: list[str], start: int, end: int) -> dict[str, torch.Tensor]:
        """Rows ``[start, end)`` of the ``row`` products ``names`` (validated by ``read_rows``)."""
        raise GoldenError(f"{type(self).__name__} has no reader for the products of golden {self.golden_id}")

    def _segment_end(self, names: list[str], index: int) -> dict[str, torch.Tensor]:
        """The ``segment_end`` products ``names`` after segment ``index`` (validated by ``read_rows``)."""
        raise GoldenError(f"{type(self).__name__} reads no segment-end products of golden {self.golden_id}")


def open_golden(golden_id: str, root: Path | None = None, reader: type[Fp64Golden] = Fp64Golden) -> Fp64Golden:
    """The complete golden ``golden_id`` under ``root`` (default ``golden_root()``), as a ``reader`` (the reader of
    its layout; the base class reads no products).

    Raises ``GoldenNotFound`` without a directory or manifest, ``GoldenInvalid`` when the manifest breaks the
    contract (including an incomplete golden), and ``GoldenMismatch`` when a reader that names its producer gets
    another producer's or version's golden.
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
    golden = reader(path, manifest)
    if reader.producer is not None:
        golden.require()
    return golden
