# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Per-owner DRAM accounting for memory-tight meshes. Enabled with ``MINIMAX_H3_DRAM_PROBE=1``.

`report(label)` walks the registered pipeline's object graph for device tensors, groups their
bytes by owner (module weights, module-held tensors, pipeline state, CCL ping-pong pairs) and
reconciles the sum against the allocator's own view of device 0, so whatever is left -- live
activations and other transients the Python graph does not reach -- is explicit, down to the
individual allocator blocks. Called at a few checkpoints and from allocation-failure handlers;
every call is a no-op unless the probe is enabled, and a failing probe never masks the error it
was called to explain.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

from loguru import logger

import ttnn

ENABLED = os.environ.get("MINIMAX_H3_DRAM_PROBE", "0") == "1"

_pipeline: Any = None
_SKIP_ATTRS = {
    "coresident_exclusions",
    "_coresident_peers",
    "ccl_manager",
    "encoder_ccl_manager",
    "audio_ccl_manager",
    "mesh_device",
    "device",
    "weight_loader",
    "_weight_loader",
}


def register(pipeline: Any) -> None:
    global _pipeline
    _pipeline = pipeline
    if ENABLED:
        logger.info("DRAM probe armed (MINIMAX_H3_DRAM_PROBE=1)")


def enabled() -> bool:
    return ENABLED and _pipeline is not None


def _gb(n: int) -> str:
    return f"{n / 1e9:6.3f} GB"


def _mb(n: int) -> str:
    return f"{n / 1e6:8.1f} MB"


def _shard0(t: ttnn.Tensor) -> ttnn.Tensor | None:
    try:
        shards = ttnn.get_device_tensors(t)
        return shards[0] if shards else None
    except Exception:
        return t


def _device_dram_tensor(t: Any) -> bool:
    if not isinstance(t, ttnn.Tensor):
        return False
    try:
        if t.device() is None:
            return False
        return t.memory_config().buffer_type == ttnn.BufferType.DRAM
    except Exception:
        return False


def _bytes_addr(t: ttnn.Tensor) -> tuple[int, int | None]:
    s = _shard0(t)
    if s is None:
        return 0, None
    try:
        return s.buffer_aligned_page_size() * s.buffer_num_pages(), s.buffer_address()
    except Exception:
        return 0, None


def _ours(obj: Any) -> bool:
    mod = type(obj).__module__ or ""
    return mod.startswith("models.")


def _walk(
    obj: Any, path: str, seen: set[int], out: list[tuple[str, ttnn.Tensor, bool]], via_param: bool, depth: int
) -> None:
    """Collect (path, tensor, is_weight) for every device tensor reachable from `obj`."""
    if depth > 40 or obj is None:
        return
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)
    if isinstance(obj, ttnn.Tensor):
        if _device_dram_tensor(obj):
            out.append((path, obj, via_param))
        return
    from ..layers.module import Parameter  # local import: utils must not import layers at module load

    if isinstance(obj, Parameter):
        _walk(obj._data, path, seen, out, True, depth + 1)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for i, v in enumerate(obj):
            _walk(v, f"{path}[{i}]", seen, out, via_param, depth + 1)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(v, f"{path}[{k!r}]", seen, out, via_param, depth + 1)
        return
    if _ours(obj) and hasattr(obj, "__dict__"):
        for k, v in vars(obj).items():
            if k in _SKIP_ATTRS:
                continue
            _walk(v, f"{path}.{k}", seen, out, via_param, depth + 1)


def _ccl_entries(pipeline: Any) -> list[tuple[str, ttnn.Tensor]]:
    managers = {}
    for name in ("encoder_ccl_manager", "ccl_manager", "audio_ccl_manager"):
        m = getattr(pipeline, name, None)
        if m is not None:
            managers.setdefault(id(m), (name, m))
    out: list[tuple[str, ttnn.Tensor]] = []

    def flatten(entry, path):
        if entry is None:
            return
        if isinstance(entry, (list, tuple)):
            for i, e in enumerate(entry):
                flatten(e, f"{path}[{i}]")
        elif isinstance(entry, ttnn.Tensor):
            out.append((path, entry))

    for name, m in managers.values():
        for key, entry in getattr(m, "_ping_pong_buffer_cache", {}).items():
            flatten(entry, f"{name}{list(key)}")
    return out


def report(label: str) -> None:
    """Log the DRAM breakdown of device 0 under `label`. Never raises."""
    if not enabled():
        return
    try:
        _report(label)
    except Exception as err:  # noqa: BLE001 - diagnostics must not mask the failure being diagnosed
        logger.warning(f"DRAM probe [{label}] failed: {err!r}")


def _report(label: str) -> None:
    p = _pipeline
    mesh = p.mesh_device
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = view.num_banks
    alloc_total = view.total_bytes_allocated_per_bank * banks
    free_total = view.total_bytes_free_per_bank * banks
    cap_total = view.total_bytes_per_bank * banks

    owners = [
        ("text_encoder", getattr(p, "_text_encoder", None)),
        ("vision_tower", getattr(p, "_vision_tower", None)),
        ("transformer", getattr(p, "_transformer", None)),
        ("vae", getattr(p, "_vae", None)),
        ("audio_encoder", getattr(p, "_audio_encoder", None)),
        ("audio_decoder", getattr(p, "_audio_decoder", None)),
    ]
    seen: set[int] = set()
    by_addr: dict[int, tuple[int, str, str]] = {}  # addr -> (bytes, group, path)
    groups: dict[str, int] = defaultdict(int)
    counts: dict[str, int] = defaultdict(int)

    def take(group: str, path: str, t: ttnn.Tensor) -> None:
        nbytes, addr = _bytes_addr(t)
        if not nbytes:
            return
        key = addr if addr is not None else id(t)
        if key in by_addr:
            return
        by_addr[key] = (nbytes, group, path)
        groups[group] += nbytes
        counts[group] += 1

    for name, module in owners:
        if module is None:
            continue
        found: list[tuple[str, ttnn.Tensor, bool]] = []
        _walk(module, name, seen, found, False, 0)
        for path, t, is_weight in found:
            take(f"{name}.{'weights' if is_weight else 'other'}", path, t)

    for path, t in _ccl_entries(p):
        take("ccl_ping_pong", path, t)

    # Everything else hanging off the pipeline object (bucket states, arena StateTensors, indices...).
    rest: list[tuple[str, ttnn.Tensor, bool]] = []
    for k, v in vars(p).items():
        if k in _SKIP_ATTRS or k in {
            "_text_encoder",
            "_vision_tower",
            "_transformer",
            "_vae",
            "_audio_encoder",
            "_audio_decoder",
        }:
            continue
        _walk(v, f"pipeline.{k}", seen, rest, False, 1)
    for path, t, _ in rest:
        take("pipeline_state", path, t)

    attributed = sum(groups.values())
    lines = [
        f"DRAM probe [{label}] device 0: allocated {_gb(alloc_total)} of {_gb(cap_total)} "
        f"({_mb(view.total_bytes_allocated_per_bank)}/bank x {banks}), free {_gb(free_total)}, "
        f"largest free block {_mb(view.largest_contiguous_bytes_free_per_bank)}/bank",
        f"  {'owner':<28}{'bytes':>12}  tensors",
    ]
    for group in sorted(groups, key=groups.get, reverse=True):
        lines.append(f"  {group:<28}{_gb(groups[group]):>12}  {counts[group]}")
    lines.append(f"  {'attributed total':<28}{_gb(attributed):>12}")
    lines.append(f"  {'unattributed (transients)':<28}{_gb(alloc_total - attributed):>12}")

    biggest = sorted(by_addr.items(), key=lambda kv: kv[1][0], reverse=True)[:12]
    lines.append("  largest attributed tensors:")
    for addr, (nbytes, group, path) in biggest:
        lines.append(f"    {_mb(nbytes)}  {group:<22} {path}  @{addr}")

    # Allocator blocks nobody above claims: live activations / op outputs at this instant. The
    # block table is bank-relative while `buffer_address()` is absolute, so the constant offset is
    # recovered from same-sized (tensor, block) pairs before matching.
    blocks = []
    for block in view.block_table:
        try:
            if str(block.get("allocated", "")).lower() not in ("yes", "true", "1"):
                continue
            blocks.append((int(block.get("address", "0")), int(block.get("size", "0")) * banks))
        except (TypeError, ValueError):
            continue
    by_size: dict[int, list[int]] = defaultdict(list)
    for addr, size in blocks:
        by_size[size // banks].append(addr)
    offsets: dict[int, int] = defaultdict(int)
    for addr, (nbytes, _g, _p) in by_addr.items():
        if not isinstance(addr, int):
            continue
        for baddr in by_size.get(nbytes // banks, ()):
            offsets[addr - baddr] += 1
    base = max(offsets, key=offsets.get) if offsets else 0
    attributed_addrs = {a - base for a in by_addr if isinstance(a, int)}
    orphans = [(size, addr) for addr, size in blocks if addr not in attributed_addrs]
    orphans.sort(reverse=True)
    lines.append(f"  allocator block table: {len(blocks)} allocated blocks, address offset {base}")
    lines.append(
        f"  unattributed allocator blocks: {len(orphans)} blocks, {_gb(sum(s for s, _ in orphans))} total; largest:"
    )
    for size, addr in orphans[:12]:
        lines.append(f"    {_mb(size)}  @{addr}")
    logger.info("\n".join(lines))
