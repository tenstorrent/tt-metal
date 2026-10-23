# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Stage Qwen3.8 routed BF4 artifacts one host shard at a time, without devices.

This is an explicitly diagnostic, non-promoting staging path.  It reuses the
pinned checkpoint slicer, ``Qwen38MoEWeights`` placement, and the existing
``moe_compute`` host layout preparation functions.  A standalone packer writes
the BFLOAT4_B payload without invoking TTNN tensor construction (which lazily
initializes Metal even for host tensors).  The already-sealed layer-0
distributed tensorbins supply the exact topology/memory-config headers.

No production validator is changed or bypassed here.  Consumers must continue
to treat these files as explicit hash-pinned diagnostic artifacts until a
device load verifies their copied topology and coordinate-local shape.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import resource
import socket
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable

import torch
from ttnn.experimental.moe_compute_utils import (
    _shard_tiles,
    _w2_shard_tiles,
    prepare_w0_w1_tensor_for_moe_compute,
    prepare_w2_tensor_for_moe_compute,
)

from models.demos.blackhole.qwen38_flash_next.checkpoint import (
    CHECKPOINT_FILE_MANIFEST_SHA256,
    CHECKPOINT_TENSOR_MANIFEST_SHA256,
    INDEX_SHA256,
    PINNED_CHECKPOINT_REVISION,
    Qwen38Checkpoint,
)
from models.demos.blackhole.qwen38_flash_next.config import CONFIG_SHA256, Qwen38Placement
from models.demos.blackhole.qwen38_flash_next.tt.moe import Qwen38MoEWeights
from models.demos.blackhole.qwen38_flash_next.ttnn.bf4 import canonical_packed_shapes

MODE = "diagnostic_non_promoting_cpu_bf4_staging"
PHYSICAL_IDS = (1, 0, 2, 3)
MESH_COORDS = ((0, 0), (0, 1), (0, 2), (0, 3))
RING_SIZE = 8
EXPERT_RANGES = ((0, 128), (128, 256), (256, 384), (384, 512))
# The op's packed layout for the 8-bank ring (moe_compute's compact owned-column layout, 20-tile transactions), as
# the production cache derives it from the layout packer's geometry.
W01_GLOBAL_SHAPE, W2_GLOBAL_SHAPE = (canonical_packed_shapes(ring_size=RING_SIZE)[name] for name in ("w0_w1", "w2"))
W01_LOCAL_SHAPE = (*W01_GLOBAL_SHAPE[:2], 128, *W01_GLOBAL_SHAPE[3:])
W2_LOCAL_SHAPE = (*W2_GLOBAL_SHAPE[:2], 128, *W2_GLOBAL_SHAPE[3:])
BF4_TILE_BYTES = 576
TILE_ELEMENTS = 32 * 32
TENSORBIN_SUFFIX = "_dtype_BFLOAT4_B_layout_TILE.tensorbin"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        payload = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode()
        offset = 0
        while offset < len(payload):
            offset += os.write(descriptor, payload[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    if path.exists():
        raise RuntimeError(f"refusing to overwrite existing evidence file: {path}")
    os.replace(temporary, path)


def _payload_bytes(shape: tuple[int, ...]) -> int:
    elements = 1
    for value in shape:
        elements *= value
    if elements % TILE_ELEMENTS:
        raise RuntimeError(f"BF4 shape is not a whole number of tiles: {shape}")
    return elements // TILE_ELEMENTS * BF4_TILE_BYTES


def _checked_absolute(path: Path, *, label: str, regular: bool = False) -> Path:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {path}")
    resolved = path.resolve(strict=True)
    if regular and not resolved.is_file():
        raise ValueError(f"{label} must be a regular file: {resolved}")
    if not regular and not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _validate_template(path: Path, expected_sha256: str, expected_payload: int) -> bytes:
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"template digest differs for {path}: {actual} != {expected_sha256}")
    header_bytes = path.stat().st_size - expected_payload
    if header_bytes <= 0 or header_bytes > 16 << 20 or header_bytes % 8:
        raise RuntimeError(f"template header size is invalid for {path}: {header_bytes}")
    with path.open("rb") as stream:
        header = stream.read(header_bytes)
    if len(header) != header_bytes:
        raise RuntimeError(f"template header read was truncated for {path}")
    return header


def _write_all(stream: BinaryIO, view: memoryview) -> None:
    offset = 0
    while offset < len(view):
        count = stream.write(view[offset:])
        if count is None or count <= 0:
            raise RuntimeError("BF4 packer stdin stopped accepting bytes")
        offset += count


def _append_packed_tensor(
    packer: Path,
    tensor: torch.Tensor,
    destination: BinaryIO,
    *,
    expected_shape: tuple[int, ...],
) -> dict[str, Any]:
    if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected_shape:
        raise RuntimeError(
            f"prepared tensor is {tensor.dtype} {tuple(tensor.shape)}, expected torch.bfloat16 {expected_shape}"
        )
    tensor = tensor.contiguous()
    columns = expected_shape[-1]
    rows = tensor.numel() // columns
    expected_input_bytes = tensor.numel() * 2
    expected_output_bytes = _payload_bytes(expected_shape)
    destination.flush()
    start_offset = destination.tell()
    started = time.monotonic()
    process = subprocess.Popen(
        [str(packer), "--rows", str(rows), "--cols", str(columns)],
        stdin=subprocess.PIPE,
        stdout=destination,
        stderr=subprocess.PIPE,
        close_fds=True,
    )
    streamed = 0
    try:
        if process.stdin is None or process.stderr is None:
            raise RuntimeError("BF4 packer pipes were not created")
        raw = tensor.view(torch.uint16).reshape(-1).numpy()
        block_elements = 8 << 20
        for start in range(0, raw.size, block_elements):
            view = memoryview(raw[start : start + block_elements]).cast("B")
            _write_all(process.stdin, view)
            streamed += len(view)
        process.stdin.close()
        del raw, tensor
        gc.collect()
        stderr = process.stderr.read().decode(errors="replace")
        return_code = process.wait()
    except BaseException:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if return_code != 0:
        raise RuntimeError(f"BF4 host packer exited {return_code}: {stderr.strip()}")
    destination.flush()
    written = destination.tell() - start_offset
    if streamed != expected_input_bytes or written != expected_output_bytes:
        raise RuntimeError(
            f"BF4 host packer byte count differs: input={streamed}/{expected_input_bytes} "
            f"output={written}/{expected_output_bytes}"
        )
    return {
        "input_bf16_bytes": streamed,
        "packed_bf4_bytes": written,
        "elapsed_seconds": round(time.monotonic() - started, 6),
    }


def _slot_directory(root: Path, namespace: str, layer_index: int) -> Path:
    return root / namespace / f"layer-{layer_index:02d}"


def _final_paths(root: Path, namespace: str, layer_index: int) -> tuple[Path, Path]:
    directory = _slot_directory(root, namespace, layer_index)
    return directory / f"w0_w1{TENSORBIN_SUFFIX}", directory / f"w2{TENSORBIN_SUFFIX}"


def _layer_evidence_path(evidence_root: Path, namespace: str, layer_index: int) -> Path:
    return evidence_root / f"{namespace}-layer-{layer_index:02d}.json"


def _validate_completed_slot(
    evidence_root: Path,
    artifact_root: Path,
    namespace: str,
    layer_index: int,
) -> dict[str, Any] | None:
    w01, w2 = _final_paths(artifact_root, namespace, layer_index)
    evidence_path = _layer_evidence_path(evidence_root, namespace, layer_index)
    states = (w01.exists(), w2.exists(), evidence_path.exists())
    if states == (False, False, False):
        return None
    if states != (True, True, True):
        raise RuntimeError(
            f"refusing to adopt incomplete pre-existing slot {namespace}:{layer_index}: "
            f"w01={states[0]} w2={states[1]} evidence={states[2]}"
        )
    document = json.loads(evidence_path.read_text(encoding="utf-8"))
    if document.get("mode") != MODE or document.get("slot") != [namespace, layer_index]:
        raise RuntimeError(f"existing staging evidence identity differs: {evidence_path}")
    for name, path in (("w0_w1", w01), ("w2", w2)):
        recorded = document.get("artifacts", {}).get(name, {})
        actual = _sha256(path)
        if (
            recorded.get("path") != str(path)
            or recorded.get("sha256") != actual
            or recorded.get("bytes") != path.stat().st_size
        ):
            raise RuntimeError(f"existing staged artifact differs from its evidence: {path}")
    return document


def _prepare_maps() -> tuple[list[int], list[tuple[int, int]]]:
    intermediate_tiles = 640 // 32
    hidden_tiles = 2560 // 32
    w01 = [_shard_tiles(intermediate_tiles, core, RING_SIZE) for core in range(RING_SIZE)]
    max_w2_tiles = (hidden_tiles + RING_SIZE - 1) // RING_SIZE
    groups = (max_w2_tiles + 4 - 1) // 4
    w2 = []
    for core in range(RING_SIZE):
        tiles = _w2_shard_tiles(hidden_tiles, core, intermediate_tiles, RING_SIZE)
        last_group_tiles = tiles - (groups - 1) * 4
        w2.append((last_group_tiles, groups * 4 - tiles))
    return w01, w2


def _stage_slot(
    *,
    checkpoint: Qwen38Checkpoint,
    placement: Qwen38Placement,
    namespace: str,
    layer_index: int,
    artifact_root: Path,
    evidence_root: Path,
    packer: Path,
    packer_sha256: str,
    w01_header: bytes,
    w2_header: bytes,
    template_report: dict[str, Any],
    common_report: dict[str, Any],
) -> dict[str, Any]:
    existing = _validate_completed_slot(evidence_root, artifact_root, namespace, layer_index)
    if existing is not None:
        return {
            "slot": [namespace, layer_index],
            "status": "verified-existing",
            "evidence": str(_layer_evidence_path(evidence_root, namespace, layer_index)),
        }

    final_w01, final_w2 = _final_paths(artifact_root, namespace, layer_index)
    final_w01.parent.mkdir(parents=True, exist_ok=True)
    temp_w01 = final_w01.with_name(f".{final_w01.name}.staging.{os.getpid()}")
    temp_w2 = final_w2.with_name(f".{final_w2.name}.staging.{os.getpid()}")
    if temp_w01.exists() or temp_w2.exists():
        raise RuntimeError(f"task-owned staging temporary already exists: {temp_w01}, {temp_w2}")

    weights = Qwen38MoEWeights(
        checkpoint,
        placement,
        layer_index=layer_index if namespace == "backbone" else None,
        mtp_layer_index=layer_index if namespace == "mtp" else None,
    )
    if tuple(weights.expert_ranges) != EXPERT_RANGES:
        raise RuntimeError(f"expert ownership changed for {namespace}:{layer_index}: {weights.expert_ranges}")
    w01_map, w2_map = _prepare_maps()
    device_reports: list[dict[str, Any]] = []
    started = _utc_now()
    flags = "xb"
    with temp_w01.open(flags) as w01_stream, temp_w2.open(flags) as w2_stream:
        w01_stream.write(w01_header)
        w2_stream.write(w2_header)
        for device_index, expected_range in enumerate(EXPERT_RANGES):
            shard_started = time.monotonic()
            shard = weights.routed_device_shard(device_index)
            if tuple(shard.expert_range) != expected_range:
                raise RuntimeError(
                    f"checkpoint shard {device_index} owns {tuple(shard.expert_range)}, expected {expected_range}"
                )
            gate, up = torch.split(shard.gate_up, 640, dim=-1)
            prepared_w01 = prepare_w0_w1_tensor_for_moe_compute(
                gate,
                up,
                1,
                128,
                2560,
                640,
                w01_map,
            )
            w01_report = _append_packed_tensor(
                packer,
                prepared_w01,
                w01_stream,
                expected_shape=W01_LOCAL_SHAPE,
            )
            del prepared_w01, gate, up
            gc.collect()

            prepared_w2 = prepare_w2_tensor_for_moe_compute(
                shard.down,
                1,
                128,
                640,
                2560,
                w2_map,
                w01_map,
            )
            w2_report = _append_packed_tensor(
                packer,
                prepared_w2,
                w2_stream,
                expected_shape=W2_LOCAL_SHAPE,
            )
            del prepared_w2, shard
            gc.collect()
            device_reports.append(
                {
                    "device_index": device_index,
                    "mesh_coordinate": list(MESH_COORDS[device_index]),
                    "physical_id": PHYSICAL_IDS[device_index],
                    "expert_range": list(expected_range),
                    "w0_w1": w01_report,
                    "w2": w2_report,
                    "elapsed_seconds": round(time.monotonic() - shard_started, 6),
                }
            )
        w01_stream.flush()
        w2_stream.flush()
        os.fsync(w01_stream.fileno())
        os.fsync(w2_stream.fileno())

    expected_w01_bytes = len(w01_header) + 4 * _payload_bytes(W01_LOCAL_SHAPE)
    expected_w2_bytes = len(w2_header) + 4 * _payload_bytes(W2_LOCAL_SHAPE)
    if temp_w01.stat().st_size != expected_w01_bytes or temp_w2.stat().st_size != expected_w2_bytes:
        raise RuntimeError(
            f"staged tensorbin sizes differ for {namespace}:{layer_index}: "
            f"w01={temp_w01.stat().st_size}/{expected_w01_bytes} "
            f"w2={temp_w2.stat().st_size}/{expected_w2_bytes}"
        )
    if final_w01.exists() or final_w2.exists():
        raise RuntimeError(f"destination appeared during staging for {namespace}:{layer_index}")
    os.replace(temp_w01, final_w01)
    os.replace(temp_w2, final_w2)

    report = {
        **common_report,
        "mode": MODE,
        "production_qualification": False,
        "slot": [namespace, layer_index],
        "started_utc": started,
        "completed_utc": _utc_now(),
        "packer": {"path": str(packer), "sha256": packer_sha256},
        "template": template_report,
        "ownership": {
            "mesh_shape": [1, 4],
            "mesh_coordinates": [list(value) for value in MESH_COORDS],
            "physical_ids": list(PHYSICAL_IDS),
            "expert_ranges": [list(value) for value in EXPERT_RANGES],
            "ring_size": RING_SIZE,
        },
        "device_shards": device_reports,
        "artifacts": {
            "w0_w1": {
                "path": str(final_w01),
                "sha256": _sha256(final_w01),
                "bytes": final_w01.stat().st_size,
                "local_shape": list(W01_LOCAL_SHAPE),
                "global_shape": list(W01_GLOBAL_SHAPE),
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
            },
            "w2": {
                "path": str(final_w2),
                "sha256": _sha256(final_w2),
                "bytes": final_w2.stat().st_size,
                "local_shape": list(W2_LOCAL_SHAPE),
                "global_shape": list(W2_GLOBAL_SHAPE),
                "dtype": "BFLOAT4_B",
                "layout": "TILE",
            },
        },
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "device_locks_acquired": False,
        "device_opened": False,
        "next_required_check": "hash-pinned diagnostic device load must verify copied topology and local shapes",
    }
    evidence_path = _layer_evidence_path(evidence_root, namespace, layer_index)
    _atomic_json(evidence_path, report)
    return {
        "slot": [namespace, layer_index],
        "status": "staged",
        "evidence": str(evidence_path),
        "artifacts": report["artifacts"],
    }


def _parse_slots(values: Iterable[str]) -> tuple[tuple[str, int], ...]:
    values = tuple(values)
    if not values:
        return tuple(("backbone", index) for index in range(1, 48)) + (("mtp", 0),)
    slots = []
    for value in values:
        match = re.fullmatch(r"(backbone|mtp):(\d+)", value)
        if match is None:
            raise ValueError(f"invalid --slot {value!r}; use backbone:N or mtp:0")
        namespace, raw_index = match.groups()
        index = int(raw_index)
        if (namespace == "backbone" and not 0 <= index < 48) or (namespace == "mtp" and index != 0):
            raise ValueError(f"invalid Qwen3.8 BF4 slot: {value}")
        slots.append((namespace, index))
    if len(slots) != len(set(slots)):
        raise ValueError(f"duplicate BF4 slots are not allowed: {slots}")
    return tuple(slots)


def _loaded_device_descriptors() -> list[str]:
    result = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            target = os.readlink(entry)
        except OSError:
            continue
        if target.startswith("/dev/tenstorrent"):
            result.append(target)
    return sorted(result)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if os.environ.get("TT_VISIBLE_DEVICES") != "":
        raise RuntimeError("CPU BF4 staging requires TT_VISIBLE_DEVICES to be the exact empty string")
    if _loaded_device_descriptors():
        raise RuntimeError(f"CPU BF4 staging inherited Tenstorrent device descriptors: {_loaded_device_descriptors()}")
    checkpoint_root = _checked_absolute(args.checkpoint, label="checkpoint")
    packer = _checked_absolute(args.packer, label="BF4 packer", regular=True)
    template_w01 = _checked_absolute(args.template_w01, label="W0/W1 template", regular=True)
    template_w2 = _checked_absolute(args.template_w2, label="W2 template", regular=True)
    if not args.artifact_root.is_absolute() or not args.evidence_root.is_absolute():
        raise ValueError("artifact and evidence roots must be absolute paths")
    artifact_root = args.artifact_root.resolve()
    evidence_root = args.evidence_root.resolve()
    if artifact_root == evidence_root:
        raise ValueError("artifact and evidence roots must be distinct absolute paths")
    artifact_root.mkdir(parents=True, exist_ok=True)
    evidence_root.mkdir(parents=True, exist_ok=True)

    packer_sha256 = _sha256(packer)
    if packer_sha256 != args.packer_sha256:
        raise RuntimeError(f"BF4 packer digest differs: {packer_sha256} != {args.packer_sha256}")
    w01_header = _validate_template(template_w01, args.template_w01_sha256, 4 * _payload_bytes(W01_LOCAL_SHAPE))
    w2_header = _validate_template(template_w2, args.template_w2_sha256, 4 * _payload_bytes(W2_LOCAL_SHAPE))
    if len(w01_header) != len(w2_header):
        raise RuntimeError(f"template header sizes differ: {len(w01_header)} != {len(w2_header)}")

    checkpoint = Qwen38Checkpoint(checkpoint_root)
    placement = Qwen38Placement(checkpoint.config, mesh_shape=(1, 4), physical_ids=PHYSICAL_IDS)
    import ttnn

    extension = Path(ttnn._ttnn.__file__).resolve(strict=True)
    extension_sha256 = _sha256(extension)
    if extension != args.runtime_extension.resolve(strict=True) or extension_sha256 != args.runtime_sha256:
        raise RuntimeError(
            f"loaded TTNN extension differs: {extension} {extension_sha256}; "
            f"expected {args.runtime_extension} {args.runtime_sha256}"
        )
    common = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "source_head": args.source_head,
        "checkpoint": {
            "root": str(checkpoint.root),
            "revision": PINNED_CHECKPOINT_REVISION,
            "config_sha256": CONFIG_SHA256,
            "index_sha256": INDEX_SHA256,
            "file_manifest_sha256": CHECKPOINT_FILE_MANIFEST_SHA256,
            "tensor_manifest_sha256": CHECKPOINT_TENSOR_MANIFEST_SHA256,
        },
        "runtime": {"extension": str(extension), "sha256": extension_sha256},
    }
    template_report = {
        "w0_w1": {
            "path": str(template_w01),
            "sha256": args.template_w01_sha256,
            "header_bytes": len(w01_header),
        },
        "w2": {"path": str(template_w2), "sha256": args.template_w2_sha256, "header_bytes": len(w2_header)},
        "payload_replaced": True,
    }

    slots = _parse_slots(args.slot)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        slots = slots[: args.limit]
    results = []
    for namespace, layer_index in slots:
        result = _stage_slot(
            checkpoint=checkpoint,
            placement=placement,
            namespace=namespace,
            layer_index=layer_index,
            artifact_root=artifact_root,
            evidence_root=evidence_root,
            packer=packer,
            packer_sha256=packer_sha256,
            w01_header=w01_header,
            w2_header=w2_header,
            template_report=template_report,
            common_report=common,
        )
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)
    if _loaded_device_descriptors():
        raise RuntimeError(f"CPU BF4 staging opened Tenstorrent device descriptors: {_loaded_device_descriptors()}")
    return {
        **common,
        "mode": MODE,
        "production_qualification": False,
        "started_slot_count": len(slots),
        "results": results,
        "completed_utc": _utc_now(),
        "device_locks_acquired": False,
        "device_opened": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--packer-sha256", required=True)
    parser.add_argument("--template-w01", type=Path, required=True)
    parser.add_argument("--template-w2", type=Path, required=True)
    parser.add_argument("--template-w01-sha256", required=True)
    parser.add_argument("--template-w2-sha256", required=True)
    parser.add_argument("--runtime-extension", type=Path, required=True)
    parser.add_argument("--runtime-sha256", required=True)
    parser.add_argument("--source-head", required=True)
    parser.add_argument("--slot", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value) is None
        for value in (
            args.packer_sha256,
            args.template_w01_sha256,
            args.template_w2_sha256,
            args.runtime_sha256,
        )
    ):
        raise SystemExit("all SHA-256 arguments must be exact lowercase 64-hex")
    report: dict[str, Any]
    try:
        report = run(args)
        report["status"] = "pass"
    except BaseException as error:
        report = {
            "mode": MODE,
            "production_qualification": False,
            "status": "fail",
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "failed_utc": _utc_now(),
            "device_locks_acquired": False,
            "device_opened": False,
        }
        if not args.summary.exists():
            _atomic_json(args.summary, report)
        raise
    _atomic_json(args.summary, report)


if __name__ == "__main__":
    main()
