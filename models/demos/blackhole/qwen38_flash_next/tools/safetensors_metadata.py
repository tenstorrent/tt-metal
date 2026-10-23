# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Fetch safetensors headers with bounded HTTP range requests.

This utility deliberately refuses an HTTP 200 response. A model host that
ignores ``Range`` must not turn a metadata probe into a multi-gigabyte weight
download.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

_CONTENT_RANGE_RE = re.compile(r"^bytes (?P<start>[0-9]+)-(?P<end>[0-9]+)/(?P<total>[0-9]+)$")
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}


class RangeResponseError(RuntimeError):
    """The remote response was not a bounded, exact byte range."""


def _default_opener(request: urllib.request.Request, timeout: float):
    return urllib.request.urlopen(request, timeout=timeout)


def read_exact_http_range(
    url: str,
    start: int,
    end: int,
    *,
    timeout: float = 30,
    opener: Callable[..., Any] | None = None,
) -> bytes:
    """Read exactly inclusive byte range ``start``--``end`` or fail closed."""

    if start < 0 or end < start:
        raise ValueError(f"invalid range {start}-{end}")
    expected_size = end - start + 1
    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "Range": f"bytes={start}-{end}",
            "User-Agent": "tt-metal-qwen38-flash-next-metadata/1",
        },
    )
    open_request = opener or _default_opener
    with open_request(request, timeout=timeout) as response:
        status = getattr(response, "status", None)
        if status != 206:
            raise RangeResponseError(f"expected HTTP 206 for bytes {start}-{end}, received {status}")
        content_range = response.headers.get("Content-Range")
        match = _CONTENT_RANGE_RE.fullmatch(content_range or "")
        if match is None:
            raise RangeResponseError(f"missing or malformed Content-Range: {content_range!r}")
        actual_start = int(match.group("start"))
        actual_end = int(match.group("end"))
        if (actual_start, actual_end) != (start, end):
            raise RangeResponseError(f"Content-Range returned {actual_start}-{actual_end}; expected {start}-{end}")
        payload = response.read(expected_size + 1)
        if len(payload) != expected_size:
            raise RangeResponseError(f"range returned {len(payload)} bytes; expected exactly {expected_size}")
        return payload


def _validate_tensor_entry(name: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RangeResponseError(f"tensor {name!r} metadata is not an object")
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if dtype not in _DTYPE_BYTES:
        raise RangeResponseError(f"tensor {name!r} has unsupported dtype {dtype!r}")
    if not isinstance(shape, list) or any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise RangeResponseError(f"tensor {name!r} has invalid shape {shape!r}")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or any(not isinstance(offset, int) or offset < 0 for offset in offsets)
        or offsets[1] < offsets[0]
    ):
        raise RangeResponseError(f"tensor {name!r} has invalid data offsets {offsets!r}")
    element_count = math.prod(shape)
    data_bytes = offsets[1] - offsets[0]
    expected_bytes = element_count * _DTYPE_BYTES[dtype]
    if data_bytes != expected_bytes:
        raise RangeResponseError(f"tensor {name!r} occupies {data_bytes} bytes; shape/dtype require {expected_bytes}")
    return {
        "dtype": dtype,
        "shape": shape,
        "data_offsets": offsets,
        "element_count": element_count,
        "data_bytes": data_bytes,
    }


def fetch_safetensors_header(
    url: str,
    *,
    timeout: float = 30,
    max_header_bytes: int = 16 * 1024 * 1024,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Fetch, decode, and validate one safetensors JSON header."""

    prefix = read_exact_http_range(url, 0, 7, timeout=timeout, opener=opener)
    header_length = int.from_bytes(prefix, byteorder="little", signed=False)
    if header_length <= 1 or header_length > max_header_bytes:
        raise RangeResponseError(f"safetensors header length {header_length} is outside 2..{max_header_bytes} bytes")
    raw_header = read_exact_http_range(
        url,
        8,
        7 + header_length,
        timeout=timeout,
        opener=opener,
    )
    try:
        decoded = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RangeResponseError(f"invalid safetensors JSON header: {error}") from error
    if not isinstance(decoded, dict):
        raise RangeResponseError("safetensors header root is not an object")
    metadata = decoded.pop("__metadata__", {})
    if not isinstance(metadata, dict):
        raise RangeResponseError("safetensors __metadata__ is not an object")
    tensors = {name: _validate_tensor_entry(name, entry) for name, entry in decoded.items()}
    ordered_offsets = sorted(
        (entry["data_offsets"][0], entry["data_offsets"][1], name) for name, entry in tensors.items()
    )
    previous_end = 0
    for start, end, name in ordered_offsets:
        if start < previous_end:
            raise RangeResponseError(f"tensor {name!r} overlaps a preceding tensor")
        previous_end = end
    return {
        "header_length": header_length,
        "metadata": metadata,
        "tensor_data_bytes": previous_end,
        "computed_file_bytes": 8 + header_length + previous_end,
        "tensors": tensors,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_with_retries(url: str, *, timeout: float, retries: int) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_safetensors_header(url, timeout=timeout)
        except (OSError, RangeResponseError) as error:
            last_error = error
            if attempt == retries:
                break
            time.sleep(2**attempt)
    raise RangeResponseError(f"failed after {retries + 1} attempts: {last_error}") from last_error


def build_repository_manifest(
    *,
    repo: str,
    revision: str,
    index_path: Path,
    workers: int = 4,
    timeout: float = 30,
    retries: int = 3,
) -> dict[str, Any]:
    """Fetch all shard headers named by a Transformers safetensors index."""

    with index_path.open("r", encoding="utf-8") as source:
        index = json.load(source)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("index has no non-empty weight_map object")
    shards = sorted(set(weight_map.values()))
    if any(not isinstance(shard, str) or not shard.endswith(".safetensors") for shard in shards):
        raise ValueError("weight_map contains an invalid shard name")

    def fetch(shard: str) -> tuple[str, dict[str, Any]]:
        encoded_repo = urllib.parse.quote(repo, safe="/")
        encoded_revision = urllib.parse.quote(revision, safe="")
        encoded_shard = urllib.parse.quote(shard, safe="/")
        url = f"https://huggingface.co/{encoded_repo}/resolve/{encoded_revision}/{encoded_shard}?download=true"
        return shard, _fetch_with_retries(url, timeout=timeout, retries=retries)

    fetched: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, shard): shard for shard in shards}
        for completed_count, future in enumerate(as_completed(futures), start=1):
            shard, header = future.result()
            fetched[shard] = header
            print(f"fetched {completed_count}/{len(shards)}: {shard}", file=sys.stderr, flush=True)

    expected_names_by_shard: dict[str, set[str]] = {shard: set() for shard in shards}
    for name, shard in weight_map.items():
        expected_names_by_shard[shard].add(name)
    for shard in shards:
        expected_names = expected_names_by_shard[shard]
        actual_names = set(fetched[shard]["tensors"])
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise RangeResponseError(f"{shard} disagrees with weight index: missing={missing}, extra={extra}")

    elements_by_dtype: dict[str, int] = {}
    data_bytes_by_dtype: dict[str, int] = {}
    for header in fetched.values():
        for tensor in header["tensors"].values():
            dtype = tensor["dtype"]
            elements_by_dtype[dtype] = elements_by_dtype.get(dtype, 0) + tensor["element_count"]
            data_bytes_by_dtype[dtype] = data_bytes_by_dtype.get(dtype, 0) + tensor["data_bytes"]
    tensor_data_bytes = sum(data_bytes_by_dtype.values())
    expected_total_size = index.get("metadata", {}).get("total_size")
    if expected_total_size is not None and tensor_data_bytes != expected_total_size:
        raise RangeResponseError(
            f"headers contain {tensor_data_bytes} tensor bytes; index declares {expected_total_size}"
        )
    return {
        "schema_version": 1,
        "repo": repo,
        "revision": revision,
        "index_path": str(index_path.resolve()),
        "index_sha256": _sha256(index_path),
        "index_metadata": index.get("metadata", {}),
        "summary": {
            "shard_count": len(shards),
            "tensor_count": len(weight_map),
            "tensor_data_bytes": tensor_data_bytes,
            "elements_by_dtype": dict(sorted(elements_by_dtype.items())),
            "data_bytes_by_dtype": dict(sorted(data_bytes_by_dtype.items())),
        },
        "shards": {shard: fetched[shard] for shard in shards},
    }


def _write_json_atomic(output_path: Path, value: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, prefix=f".{output_path.name}.", delete=False
    ) as temporary:
        json.dump(value, temporary, indent=2, sort_keys=True)
        temporary.write("\n")
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, output_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 16:
        parser.error("--workers must be in 1..16")
    if args.timeout <= 0 or args.retries < 0:
        parser.error("--timeout must be positive and --retries nonnegative")
    manifest = build_repository_manifest(
        repo=args.repo,
        revision=args.revision,
        index_path=args.index,
        workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
    )
    _write_json_atomic(args.output, manifest)
    print(json.dumps(manifest["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
