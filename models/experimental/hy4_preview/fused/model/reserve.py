# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Tools for a process in which another stack's trace is replayed beside the fused step (PLAN.md sections 24, 26).

DramReserve: a trace's temporaries are freed when its capture ends and their addresses are handed out again, while
every replay still writes through them. Whatever of ours must survive a replay may therefore only be allocated where
no temporary of that capture was. The Stack makes its L1 state at construction and its DRAM weights afterwards, so
"our DRAM tensors before the other stack's capture, our L1 state after it" cannot be written as an order. The
reserve gives the same guarantee by address:

    r = DramReserve(mesh, bytes_per_bank)   # before the capture: a ballast holds the range ours will take
    ... the other stack's trace is captured (its temporaries land outside the ballast) ...
    r.plug()                                # every free DRAM block is taken (plugs): nothing free is left
    r.open()                                # the ballast is freed: its range is the only free DRAM
    ... every persistent tensor of ours is allocated (L1 where it likes; DRAM inside the range) ...
    r.close(buffers_before)                 # the plugs are freed; every new DRAM buffer is checked to lie in the range

device_tensors / snapshot / differences / write_back: every device tensor reachable from an object, its contents on
the 32 chips as bytes, what changed between two snapshots (per tensor: chips, cores, byte range within the core),
and the host's write of a snapshot back into the tensors that changed.
"""

from __future__ import annotations

import hashlib

import torch
from loguru import logger

import ttnn

PAGE_LARGE, PAGE_SMALL = 4096, 64


def dram_view(mesh):
    v = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        "banks": v.num_banks,
        "allocated": v.total_bytes_allocated_per_bank,
        "free": v.total_bytes_free_per_bank,
        "largest": v.largest_contiguous_bytes_free_per_bank,
    }


def buffers(mesh, kind=None, device_id=None):
    """{(buffer type name, address): bytes per bank} of every buffer the first chip's allocators hold."""
    all_ = ttnn._ttnn.reports.get_buffers(mesh)
    first = min(b.device_id for b in all_) if all_ and device_id is None else device_id
    out = {}
    for b in all_:
        name = str(b.buffer_type).split(".")[-1]
        if b.device_id == first and (kind is None or name == kind):
            out[(name, int(b.address))] = int(b.max_size_per_bank)
    return out


def _allocate(mesh, per_bank, page):
    banks = dram_view(mesh)["banks"]
    rows = banks * (per_bank // page)
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, 1, rows, page // 4]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh, ttnn.DRAM_MEMORY_CONFIG
    )


class DramReserve:
    def __init__(self, mesh, bytes_per_bank: int):
        self.mesh = mesh
        self.size = PAGE_LARGE * (int(bytes_per_bank) // PAGE_LARGE)
        self.ballast = _allocate(mesh, self.size, PAGE_LARGE)
        self.address = int(self.ballast.buffer_address())
        self.plugs = []
        self.left = None
        got = buffers(mesh, "DRAM").get(("DRAM", self.address))
        assert got is not None and got >= self.size, (self.address, got, self.size)
        self.size = got

    def plug(self, limit=4096):
        """Take every free DRAM block. What cannot be taken (blocks below the allocator's granule) is reported."""
        last = None
        while len(self.plugs) < limit:
            largest = dram_view(self.mesh)["largest"]
            page = PAGE_LARGE if largest >= 1 << 20 else PAGE_SMALL
            if largest < page or largest == last and page == PAGE_SMALL:
                break
            last = largest
            try:
                self.plugs.append(_allocate(self.mesh, largest, page))
            except RuntimeError as e:  # (out of memory: thrown by the allocator on the host, nothing was sent)
                assert "Out of Memory" in str(e), e
                if page == PAGE_SMALL:
                    break
                self.plugs.append(_allocate(self.mesh, largest // 2, page))
        self.left = dram_view(self.mesh)
        logger.info(
            f"HY4_MK11 reserve: ballast {self.size} B per bank at {self.address}; {len(self.plugs)} plugs; DRAM left free "
            f"{self.left['free']} B per bank, largest block {self.left['largest']} B"
        )
        return self.left

    def open(self):
        ttnn.deallocate(self.ballast)
        self.ballast = None

    def close(self, before: dict):
        """Free the plugs; every DRAM buffer that is not in ``before`` (buffers() taken before open()) and is not a
        plug must lie in the ballast's range. Returns (bytes per bank used of the range, buffers outside it)."""
        plug_addresses = {int(p.buffer_address()) for p in self.plugs}
        now = buffers(self.mesh, "DRAM")
        before = {k: v for k, v in before.items() if k[1] != self.address}  # (the ballast's own entry)
        new = {k: v for k, v in now.items() if k not in before and k[1] not in plug_addresses}
        lo, hi = self.address, self.address + self.size
        outside = {k: v for k, v in new.items() if not (lo <= k[1] and k[1] + v <= hi)}
        used = sum(v for k, v in new.items() if k not in outside)
        for p in self.plugs:
            ttnn.deallocate(p)
        self.plugs = []
        logger.info(
            f"HY4_MK11 reserve: {len(new)} DRAM buffers allocated while the range was the only free DRAM, {used} B per bank of "
            f"{self.size}; outside the range: {outside}"
        )
        return used, outside

    def release(self):
        for t in self.plugs + ([self.ballast] if self.ballast is not None else []):
            ttnn.deallocate(t)
        self.plugs, self.ballast = [], None


# ----------------------------------------------------------------------------------------------------------------
def device_tensors(root, name="st", depth=7) -> dict:
    """{path: tensor} of every allocated device tensor reachable from ``root`` through attributes, lists, tuples and
    dicts of this model's objects, one path per buffer (the first found)."""
    found, seen, by_buffer = {}, set(), set()

    def walk(obj, path, level):
        if obj is None or isinstance(obj, (int, float, str, bytes, bool, torch.Tensor)) or id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, ttnn.Tensor):
            try:
                if obj.storage_type() != ttnn.StorageType.DEVICE or not obj.is_allocated():
                    return
                key = (str(obj.memory_config().buffer_type), int(ttnn.get_device_tensors(obj)[0].buffer_address()))
            except Exception:
                return
            if key not in by_buffer:
                by_buffer.add(key)
                found[path] = obj
            return
        if level >= depth:
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}[{k!r}]", level + 1)
        elif isinstance(obj, (list, tuple)):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]", level + 1)
        elif type(obj).__module__.startswith("models.experimental.hy4_preview") and hasattr(obj, "__dict__"):
            for k, v in vars(obj).items():
                walk(v, f"{path}.{k}", level + 1)

    walk(root, name, 0)
    return found


def kind_of(tensor) -> str:
    return str(tensor.memory_config().buffer_type).split(".")[-1]


def chip_bytes(tensor, chip):
    """(torch tensor as ttnn.to_torch gives it, its bytes as uint8) of one chip's part."""
    value = ttnn.to_torch(ttnn.get_device_tensors(tensor)[chip]).contiguous()
    raw = value.view(torch.uint8) if value.dtype != torch.bool else value.to(torch.uint8)
    return value, raw.flatten()


def elements(tensor) -> int:
    n = 1
    for d in tuple(tensor.shape):
        n *= int(d)
    return n


def snapshot(mesh, tensors: dict, chips=None, keep=True, max_elements=None) -> dict:
    """{path: {"sha": [per chip], "raw": [uint8 per chip] or None, "chips": [...]}}. keep=False: checksums only.
    max_elements: a tensor with more elements per chip is read on three chips only (a weight stream)."""
    ttnn.synchronize_device(mesh)
    n = mesh.get_num_devices()
    out = {}
    for path, t in tensors.items():
        which = list(range(n)) if chips is None else list(chips)
        if max_elements is not None and elements(ttnn.get_device_tensors(t)[0]) > max_elements and chips is None:
            which = [0, n // 2 - 3, n - 1]
        raws = [chip_bytes(t, c)[1] for c in which]
        out[path] = {
            "chips": which,
            "sha": [hashlib.sha1(r.numpy().tobytes()).hexdigest() for r in raws],
            "raw": raws if keep else None,
        }
    return out


def cores_of(tensor):
    """The cores of an L1 tensor's shards, in shard order (None: not sharded)."""
    spec = tensor.memory_config().shard_spec
    if spec is None:
        return None
    return [
        (c.x, c.y)
        for c in ttnn.corerange_to_cores(spec.grid, None, spec.orientation == ttnn.ShardOrientation.ROW_MAJOR)
    ]


def differences(tensors: dict, before: dict, after: dict) -> dict:
    """{path: {"chips": [...], "bytes": changed bytes over all chips, "cores": {(x, y): (first, last changed byte
    within the core's shard, changed bytes)} (the union over the chips), "was_zero": the bytes that changed were all
    zero before}} of every tensor whose contents differ."""
    out = {}
    for path, a in before.items():
        b = after[path]
        chips = [c for c, x, y in zip(a["chips"], a["sha"], b["sha"]) if x != y]
        if not chips:
            continue
        entry = {"chips": chips, "bytes": None, "cores": None, "was_zero": None}
        if a["raw"] is not None and b["raw"] is not None:
            cores = cores_of(tensors[path]) if kind_of(tensors[path]) == "L1" else None
            total, per_core, zero = 0, {}, True
            for i, c in enumerate(a["chips"]):
                if c not in chips:
                    continue
                ra, rb = a["raw"][i], b["raw"][i]
                changed = (ra != rb).nonzero().flatten()
                total += int(changed.numel())
                zero = zero and not bool(ra[changed].any())
                if cores:
                    per = ra.numel() // len(cores)
                    shard = changed // per
                    for s in shard.unique().tolist():
                        offs = changed[shard == s] - s * per
                        lo, hi, cnt = int(offs.min()), int(offs.max()), int(offs.numel())
                        old = per_core.get(cores[s])
                        per_core[cores[s]] = (
                            (lo, hi, cnt) if old is None else (min(lo, old[0]), max(hi, old[1]), max(cnt, old[2]))
                        )
            entry.update(bytes=total, cores=per_core if cores else None, was_zero=zero)
        out[path] = entry
    return out


def describe(diff: dict) -> list:
    lines = []
    for path, d in diff.items():
        text = f"{path}: {len(d['chips'])} chips, {d['bytes']} bytes changed" + (
            ", all zero before" if d["was_zero"] else ""
        )
        if d["cores"]:
            xs, ys = [c[0] for c in d["cores"]], [c[1] for c in d["cores"]]
            lo, hi = min(v[0] for v in d["cores"].values()), max(v[1] for v in d["cores"].values())
            text += f", {len(d['cores'])} cores in x {min(xs)}..{max(xs)} y {min(ys)}..{max(ys)}, bytes {lo}..{hi} of the core's shard"
        lines.append(text)
    return lines


def write_back(mesh, tensors: dict, snap: dict, paths):
    """The host writes ``snap``'s contents into the tensors ``paths`` (all chips of the snapshot must be kept)."""
    n = mesh.get_num_devices()
    for path in paths:
        t, s = tensors[path], snap[path]
        assert s["raw"] is not None and s["chips"] == list(range(n)), path
        like, _ = chip_bytes(t, 0)
        shards = []
        for raw in s["raw"]:
            value = raw.clone().view(like.dtype).reshape(like.shape)
            if value.dtype == torch.bfloat16:
                value = value.float()
            shards.append(ttnn.from_torch(value, dtype=t.dtype, layout=t.layout))
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, mesh.shape), t)
    ttnn.synchronize_device(mesh)


def report(mesh, label, prev=None, log=True) -> dict:
    """The first chip's allocators as the buffer report gives them, {type: {address: bytes per bank}} for DRAM, L1,
    L1_SMALL and TRACE, logged with what was allocated and freed since ``prev`` (an earlier report): count, bytes and
    address range per type, and the addresses themselves where they are few. A failure later comes with addresses."""
    now = {kind: {k[1]: v for k, v in buffers(mesh, kind).items()} for kind in ("DRAM", "L1", "L1_SMALL", "TRACE")}
    if log:
        parts = []
        for kind, bufs in now.items():
            text = f"{kind}: {len(bufs)} buffers, {sum(bufs.values())} B"
            if prev is not None:
                new = {a: n for a, n in bufs.items() if prev[kind].get(a) != n}
                gone = {a: n for a, n in prev[kind].items() if bufs.get(a) != n}
                for name, d in (("new", new), ("freed", gone)):
                    if d:
                        text += (
                            f"; {name} {len(d)} ({sum(d.values())} B, {min(d)} .. {max(a + n for a, n in d.items())})"
                        )
                        if len(d) <= 12:
                            text += f" {dict(sorted(d.items()))}"
            parts.append(text)
        logger.info(f"HY4_MK11 buffers [{label}]: " + " | ".join(parts))
    return now


def overlaps(a: dict, b: dict) -> list:
    """[(address a, bytes, address b, bytes)] of the address ranges of two {address: bytes} sets that intersect."""
    out = []
    items = sorted(b.items())
    for x, n in sorted(a.items()):
        for y, m in items:
            if y >= x + n:
                break
            if y + m > x:
                out.append((x, n, y, m))
    return out


def late_allocations(base: dict, now: dict, dram_range=None, l1_declared=(), trace_ok=True) -> dict:
    """The rule of PLAN.md section 24 as a check that needs no program: of everything alive in ``now`` (a report())
    that was not there in ``base`` (the report taken right after the FIRST trace capture), what lies outside what was
    declared for it. {class: {address: bytes}} of the violations (empty dicts when the rule holds):
      DRAM      outside ``dram_range`` (lo, hi): the reserve that was held during the capture
      L1        not one of ``l1_declared`` (addresses: the registered tensors and semaphores of the fused step, which
                are restored after every replay of the other trace)
      L1_SMALL  anything (it has neither a reserve nor a restore)
      TRACE     anything unless trace_ok (the trace region is its own allocator: a later trace's buffers are fine)"""
    late = {kind: {a: n for a, n in now[kind].items() if base[kind].get(a) != n} for kind in now}
    lo, hi = dram_range if dram_range is not None else (0, 0)
    declared = set(int(a) for a in l1_declared)
    return {
        "DRAM": {a: n for a, n in late["DRAM"].items() if not (lo <= a and a + n <= hi)},
        "L1": {a: n for a, n in late["L1"].items() if a not in declared},
        "L1_SMALL": dict(late["L1_SMALL"]),
        "TRACE": {} if trace_ok else dict(late["TRACE"]),
    }
