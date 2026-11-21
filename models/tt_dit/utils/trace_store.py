# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.

# SPDX-License-Identifier: Apache-2.0

"""Trace store for ``Tracer``: keep captured traces on disk and install them in a later process.

Opt in with ``TT_DIT_TRACE_STORE=1`` (off by default). A captured trace is a fixed command stream
with absolute buffer addresses, so a stored trace is reused only when everything that decides those
is the same: the key covers the build, the kernel cache, the program-changing environment, the mesh,
the traced function, its inputs and the exact list of live device buffers before the capture. On a
hit the restore installs the stored trace and places the output tensors at their captured addresses,
then checks the trace buffer and every live buffer against the stored manifest. Any mismatch or
error releases what it made, retires the entry and lets the caller capture as usual: all or nothing
per trace. ``check_gen()`` compares a seeded gen's output digest with the one recorded by a start
that captured everything, so a restored start is verified before it takes jobs.

Entries live in ``<TT_METAL_CACHE>/trace-store/<key>/`` beside the kernel cache.
"""

from __future__ import annotations

import hashlib
import json
import re
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from loguru import logger

import ttnn

_VERSION = 1
_context = ""
_build_id: str | None = None
# This process's entries: key -> "restored" | "captured".
_used: dict[str, str] = {}
_key_parts: dict[str, dict[str, str]] = {}
stats = {"restored": 0, "stored": 0, "captured": 0, "refused": {}, "restore_s": 0.0, "dump_s": 0.0}

# Environment variables that change nothing a trace holds (logging, this switch).
# Per-start paths and secrets: they change nothing a trace holds.
_ENV_IGNORED_WORDS = ("SCRATCH", "LOG", "TOKEN", "KEY", "SECRET")
_HEX_ID = re.compile(r"at 0x[0-9a-fA-F]+")
_ENV_IGNORED = ("TT_DIT_TRACE_STORE", "TT_LOGGER_", "TT_METAL_LOGGER_")


def enabled() -> bool:
    return os.environ.get("TT_DIT_TRACE_STORE", "0") == "1" and hasattr(ttnn, "load_trace")


def set_context(text: str) -> None:
    """Add caller facts to every key (model config, shape family); traces captured after it use it."""
    global _context
    _context = text


def _root() -> Path | None:
    cache = os.environ.get("TT_METAL_CACHE")
    return Path(cache) / "trace-store" if cache else None


def _refuse(reason: str) -> None:
    stats["refused"][reason] = stats["refused"].get(reason, 0) + 1
    logger.info(f"trace store: {reason}")


def _build() -> str:
    global _build_id
    if _build_id is None:
        parts = []
        pkg = Path(ttnn.__file__).resolve().parent
        for lib in sorted(pkg.glob("_ttnn*.so")) + sorted(pkg.glob("build/lib/libtt_metal*.so")):
            st = lib.stat()
            parts.append(f"{lib.name}:{st.st_size}:{st.st_mtime_ns}")
        try:
            head = subprocess.run(
                ["git", "-C", str(pkg), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            head = ""
        env = sorted(
            (k, v)
            for k, v in os.environ.items()
            if k.startswith(("TT_", "TTNN_", "LTX_", "ARCH_NAME"))
            and not k.startswith(_ENV_IGNORED)
            and not any(w in k for w in _ENV_IGNORED_WORDS)
        )
        _build_id = json.dumps({"head": head, "libs": parts, "env": env})
    return _build_id


def live_buffers(device: ttnn.MeshDevice) -> list[tuple[int, int, int, int]]:
    trace_type = int(ttnn.BufferType.TRACE)
    return [tuple(b) for b in ttnn.allocated_buffers(device) if b[0] != trace_type]


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _leaves(tree: Any, out: list[Any]) -> list[Any]:
    if isinstance(tree, (tuple, list)):
        for x in tree:
            _leaves(x, out)
    elif isinstance(tree, dict):
        for k in sorted(tree, key=str):
            _leaves(tree[k], out)
    else:
        out.append(tree)
    return out


def _input_desc(x: Any) -> Any:
    if isinstance(x, ttnn.Tensor):
        return ["t", _spec_desc(x), x.buffer_address() if x.is_allocated() else None]
    # Object reprs carry a per-process id ("at 0x7f..."); keep the rest.
    return ["v", _HEX_ID.sub("at 0x", repr(x))]


def _spec_desc(t: ttnn.Tensor) -> dict[str, Any]:
    spec = t.spec
    tile = spec.tile
    return {
        "shape": list(spec.shape),
        "dtype": spec.dtype.name,
        "layout": spec.layout.name,
        "tile": list(tile.tile_shape) if tile is not None else [32, 32],
        "memory_config": spec.memory_config.to_json(),
        "padded": list(t.padded_shape),
    }


def _topology_desc(t: ttnn.Tensor) -> dict[str, Any]:
    topo = t.tensor_topology()
    placements = []
    for p in topo.placements():
        placements.append(["s", p.dim] if isinstance(p, ttnn.PlacementShard) else ["r"])
    return {
        "shape": list(topo.distribution_shape()),
        "placements": placements,
        "coords": [list(c) for c in topo.mesh_coords()],
    }


def _allocate(d: dict[str, Any], device: ttnn.MeshDevice) -> ttnn.Tensor:
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape(d["shape"]),
        getattr(ttnn.DataType, d["dtype"]),
        getattr(ttnn.Layout, d["layout"]),
        device,
        ttnn.MemoryConfig.from_json(d["memory_config"]),
    )


def _make_topology(d: dict[str, Any]) -> ttnn.TensorTopology:
    placements = [ttnn.PlacementShard(p[1]) if p[0] == "s" else ttnn.PlacementReplicate() for p in d["placements"]]
    return ttnn.TensorTopology(
        ttnn.MeshShape(d["shape"]), placements, [ttnn.MeshCoordinate(c) for c in d["coords"]]
    )


def key_for(tracer: Any, device: ttnn.MeshDevice, cq_id: int, pre: list[tuple[int, int, int, int]]) -> str:
    fn = tracer._function
    name = f"{getattr(fn, '__module__', '')}.{getattr(fn, '__qualname__', repr(type(fn)))}"
    inputs = [_input_desc(x) for x in _leaves((tracer._args, tracer._kwargs), [])]
    parts = {
        "v": _VERSION,
        "build": _build(),
        "cache": os.environ.get("TT_METAL_CACHE"),
        "context": _context,
        "mesh": [list(device.shape), list(device.get_device_ids())],
        "fn": name,
        "cq": cq_id,
        "inputs": inputs,
        "pre": pre,
    }
    key = _digest(parts)[:40]
    # Digest of each part, stored with the entry: two starts that should share a key can be compared.
    _key_parts[key] = {k: _digest(v)[:16] for k, v in parts.items()} | {"fn_name": name}
    return key


def _input_addresses(tracer: Any) -> dict[int, int]:
    out = {}
    for i, x in enumerate(_leaves((tracer._args, tracer._kwargs), [])):
        if isinstance(x, ttnn.Tensor) and x.is_allocated() and x.storage_type() == ttnn.StorageType.DEVICE:
            out.setdefault(x.buffer_address(), i)
    return out


def dump(tracer: Any, device: ttnn.MeshDevice, trace_id: Any, key: str, pre: list, outputs: Any) -> None:
    """Store a fresh capture, if it can be restored exactly; otherwise say why and store nothing."""
    root = _root()
    entry = root / key
    if entry.exists():
        return
    t0 = time.monotonic()
    post = live_buffers(device)
    pre_set, post_set = set(pre), set(post)
    if pre_set - post_set:
        return _refuse("not stored: the capture freed buffers that lived before it")
    new = {b[1]: b[2] for b in post_set - pre_set}
    if len(new) != len(post_set - pre_set):
        return _refuse("not stored: two new buffers share an address")
    inputs = _input_addresses(tracer)
    leaves = []
    placed = set()
    for x in _leaves(outputs, []):
        if not isinstance(x, ttnn.Tensor):
            if x is not None and not isinstance(x, (bool, int, float, str)):
                return _refuse(f"not stored: output {type(x).__name__}")
            leaves.append({"value": x})
            continue
        addr = x.buffer_address()
        if addr in inputs and addr not in new:
            leaves.append({"input": inputs[addr]})
            continue
        if x.memory_config().buffer_type != ttnn.BufferType.DRAM:
            return _refuse("not stored: an output is not in DRAM")
        if addr not in new or addr in placed:
            return _refuse("not stored: an output buffer is not a capture allocation")
        size = new[addr]
        placed.add(addr)
        spec, topo = _spec_desc(x), _topology_desc(x)
        try:
            same = (
                spec["tile"] == [32, 32]
                and ttnn.MemoryConfig.from_json(spec["memory_config"]) == x.memory_config()
                and getattr(ttnn.DataType, spec["dtype"]) == x.dtype
                and getattr(ttnn.Layout, spec["layout"]) == x.layout
                and _make_topology(topo) == x.tensor_topology()
            )
        except Exception as e:  # noqa: BLE001 - any spec this code cannot rebuild is simply not stored
            return _refuse(f"not stored: output spec not rebuildable ({type(e).__name__})")
        if not same:
            return _refuse("not stored: output spec not rebuildable")
        leaves.append({"addr": addr, "size": size, "spec": spec, "topology": topo})
    if placed != set(new):
        return _refuse("not stored: the capture kept buffers that are not outputs")
    tmp = root / f".{key}.{os.getpid()}.tmp"
    try:
        tmp.mkdir(parents=True, exist_ok=True)
        ttnn.dump_trace(device, trace_id, str(tmp / "trace.bin"))
        manifest = {
            "version": _VERSION,
            "pre": pre,
            "post": sorted(post),
            "trace_buffer": list(ttnn.trace_buffer_range(device, trace_id)),
            "outputs": leaves,
            "structure": _structure(outputs),
        }
        (tmp / "manifest.json").write_text(json.dumps(manifest))
        (tmp / "key.json").write_text(json.dumps(_key_parts.get(key, {}), sort_keys=True))
        tmp.rename(entry)
    except Exception as e:  # noqa: BLE001 - a trace that cannot be stored is captured every start
        shutil.rmtree(tmp, ignore_errors=True)
        return _refuse(f"not stored: {type(e).__name__}: {e}")
    dt = time.monotonic() - t0
    stats["stored"] += 1
    stats["dump_s"] += dt
    logger.info(f"trace store: stored {key} ({len(leaves)} outputs) in {dt:.2f}s")


def _structure(tree: Any) -> Any:
    if isinstance(tree, (tuple, list)):
        return [type(tree).__name__, [_structure(x) for x in tree]]
    if isinstance(tree, dict):
        return ["dict", [[k, _structure(tree[k])] for k in sorted(tree, key=str)]]
    return None


def _rebuild(structure: Any, leaves: list[Any]) -> Any:
    if structure is None:
        return leaves.pop(0)
    kind, items = structure
    if kind == "dict":
        return {k: _rebuild(s, leaves) for k, s in items}
    seq = [_rebuild(s, leaves) for s in items]
    return tuple(seq) if kind == "tuple" else seq


def restore(tracer: Any, device: ttnn.MeshDevice, cq_id: int, key: str, pre: list) -> tuple[Any, Any] | None:
    """Install a stored trace: (trace_id, outputs), or None to capture (nothing left behind)."""
    entry = _root() / key
    manifest_path = entry / "manifest.json"
    if not manifest_path.exists() or (entry / "retired").exists():
        return None
    t0 = time.monotonic()
    trace_id = None
    made: list[ttnn.Tensor] = []
    try:
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("version") != _VERSION or [list(b) for b in pre] != [list(b) for b in manifest["pre"]]:
            raise RuntimeError("manifest does not match the live buffers")
        trace_id = ttnn.load_trace(device, str(entry / "trace.bin"), cq_id)
        if list(ttnn.trace_buffer_range(device, trace_id)) != manifest["trace_buffer"]:
            raise RuntimeError("trace buffer is not where it was captured")
        inputs = _leaves((tracer._args, tracer._kwargs), [])
        leaves = []
        for leaf in manifest["outputs"]:
            if "value" in leaf:
                leaves.append(leaf["value"])
            elif "input" in leaf:
                leaves.append(inputs[leaf["input"]])
            else:
                ttnn.set_next_allocation_address(leaf["addr"], leaf["size"])
                try:
                    t = _allocate(leaf["spec"], device)
                finally:
                    ttnn.set_next_allocation_address(0, 0)
                made.append(t)
                if t.buffer_address() != leaf["addr"] or list(t.padded_shape) != leaf["spec"]["padded"]:
                    raise RuntimeError("output not at its captured address or shape")
                t.update_tensor_topology(_make_topology(leaf["topology"]))
                leaves.append(t)
        if sorted(tuple(b) for b in live_buffers(device)) != sorted(tuple(b) for b in manifest["post"]):
            raise RuntimeError("live buffers after restore differ from the capture")
        outputs = _rebuild(manifest["structure"], leaves)
    except Exception as e:  # noqa: BLE001 - all or nothing: undo, retire, capture
        for t in made:
            ttnn.deallocate(t)
        if trace_id is not None:
            ttnn.release_trace(device, trace_id)
        try:
            (entry / "retired").write_text(f"{type(e).__name__}: {e}\n")
        except OSError:
            pass
        _refuse(f"restore failed, entry retired, capturing: {type(e).__name__}: {e}")
        if live_buffers(device) != pre:
            raise RuntimeError("trace store: live buffers changed by a failed restore") from e
        return None
    dt = time.monotonic() - t0
    _used[key] = "restored"
    stats["restored"] += 1
    stats["restore_s"] += dt
    logger.info(f"trace store: restored {key} in {dt:.2f}s")
    return trace_id, outputs


def note_captured(key: str) -> None:
    _used.setdefault(key, "captured")
    stats["captured"] += 1


def check_gen(name: str, digest: str) -> bool | None:
    """Check a seeded gen's output digest for this start's set of traces.

    A start that captured every trace records the digest; a start that restored any trace must
    reproduce it. Returns True (recorded or matched), False (mismatch: every entry this process
    restored is retired, so the next start captures) or None (restored, nothing recorded yet)."""
    root = _root()
    if root is None or not _used:
        return None
    ref = root / "checks" / f"{_digest([sorted(_used), name])[:40]}.digest"
    restored = [k for k, v in _used.items() if v == "restored"]
    if not restored:
        ref.parent.mkdir(parents=True, exist_ok=True)
        ref.write_text(digest)
        return True
    if not ref.exists():
        logger.warning(f"trace store: no recorded {name} digest for this set of traces; restore unverified")
        return None
    if ref.read_text().strip() == digest:
        logger.info(f"trace store: check gen {name} matches the captured start")
        return True
    logger.error(f"trace store: check gen {name} differs from the captured start; retiring {len(restored)} entries")
    for k in restored:
        try:
            (root / k / "retired").write_text(f"check gen {name} mismatch\n")
        except OSError:
            pass
    return False
