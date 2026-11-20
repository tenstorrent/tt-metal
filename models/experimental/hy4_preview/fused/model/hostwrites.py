# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Every L1 tensor the HOST writes in this process, with what it last wrote: the restore list after something else
(a prefill pass by trace replay, PLAN.md section 26) has written over the fused step's L1.

A reading of the pieces for "what must be restored" misses items silently (run mk11_coexist_cut: the dense FFN's slot
words, host-written once, were classified as written by the step; step B came out wrong with every status empty).
So the list is taken where the writes happen: install() wraps the three ways the pieces put host data into a device
tensor (ttnn.from_torch with a device, ttnn.to_device, ttnn.copy_host_to_device_tensor) and
ttnn.create_global_semaphore; a call from this model's modules whose target is an L1 tensor is registered as
(tensor, the host tensor written, where from). No piece is edited.

    restore()   rewrites every registered tensor that is still allocated with its last host-written contents (but
                those marked per_step: the host writes them before every step anyway) and every global semaphore
                with its initial value
    verify()    reads every registered tensor back and compares it with those contents
    unregistered(tensors)   L1 tensors (model/reserve.py device_tensors) the host never wrote: a step must write
                them before it reads them (a trace's own temporaries are such)

What is NOT decided here by hand: nothing is left out because it "is written by the step anyway". One tensor carries a
declaration of its owner's design instead: scratch(tensor, zero_ranges, defined) (the attention's arena: data ranges a
step writes before it reads them, and status words that must be zero at rest). The host's restore rewrites it whole
all the same; plan() is what a program on the device writes.
"""

from __future__ import annotations

import sys

import torch
from loguru import logger

import ttnn

SCOPE = "models.experimental.hy4_preview.fused"  # the fused model's packages: only their writes are registered
ITEM = {"UINT32": 4, "INT32": 4, "FLOAT32": 4, "BFLOAT16": 2, "UINT16": 2, "UINT8": 1}

_installed = None


class Entry:
    def __init__(self, tensor, host, where, address):
        self.tensor, self.host, self.where, self.address = tensor, host, where, address
        self.writes = 1
        self.per_step = False
        self.scratch = None  # (zero ranges [(core, offset, bytes)], where that is defined): see HostWrites.scratch

    def live(self) -> bool:
        try:
            return self.tensor.is_allocated() and _address(self.tensor) == self.address
        except Exception:
            return False

    def bytes(self) -> int:
        """Per chip."""
        n = 1
        for d in tuple(ttnn.get_device_tensors(self.tensor)[0].shape):
            n *= int(d)
        return n * ITEM.get(str(self.tensor.dtype).split(".")[-1], 4)


def _address(tensor) -> int:
    return int(ttnn.get_device_tensors(tensor)[0].buffer_address())


def _is_l1(tensor) -> bool:
    try:
        return (
            isinstance(tensor, ttnn.Tensor)
            and tensor.storage_type() == ttnn.StorageType.DEVICE
            and tensor.memory_config().buffer_type == ttnn.BufferType.L1
        )
    except Exception:
        return False


def _caller(depth=2):
    frame = sys._getframe(depth)
    module = frame.f_globals.get("__name__", "")
    return (
        module,
        f"{module.split('.')[-2]}.{module.split('.')[-1]}:{frame.f_code.co_name}:{frame.f_lineno}"
        if "." in module
        else module,
    )


class HostWrites:
    def __init__(self):
        self.entries = {}  # L1 address -> Entry
        self.semaphores = []  # (semaphore, initial value, where, its cores)
        self.orig = {}
        self.enabled = True
        self.refused = {}  # calling line outside the fused model's packages -> count of L1 writes not registered

    # ------------------------------------------------------------------------------------------ the wrappers
    def _register(self, tensor, host, where):
        address = _address(tensor)
        old = self.entries.get(address)
        entry = Entry(tensor, host, where, address)
        if old is not None and old.live() and old.tensor is tensor:
            entry.writes, entry.per_step, entry.scratch = old.writes + 1, old.per_step, old.scratch
        self.entries[address] = entry

    def _mine(self, depth=3):
        """(the caller is one of the fused model's modules, where). A host write into an L1 tensor from elsewhere is
        refused (not registered) and logged once per calling line: it is not this model's to restore."""
        module, where = _caller(depth)
        if module.startswith(SCOPE):
            return True, where
        if where not in self.refused:
            logger.warning(f"HY4 host write into an L1 tensor from outside the fused model ({where}): not registered")
        self.refused[where] = self.refused.get(where, 0) + 1
        return False, where

    def install(self):
        orig_from_torch, orig_to_device = ttnn.from_torch, ttnn.to_device
        orig_copy, orig_semaphore = ttnn.copy_host_to_device_tensor, ttnn.create_global_semaphore
        self.orig = dict(
            from_torch=orig_from_torch,
            to_device=orig_to_device,
            copy_host_to_device_tensor=orig_copy,
            create_global_semaphore=orig_semaphore,
        )
        reg = self

        def from_torch(*args, **kw):
            out = orig_from_torch(*args, **kw)
            device = kw.get("device")
            if reg.enabled and device is not None and _is_l1(out):
                mine, where = reg._mine()
                if mine:
                    host_kw = {k: v for k, v in kw.items() if k in ("dtype", "layout", "pad_value", "tile")}
                    if len(args) > 1:
                        host_kw["dtype"] = args[1]
                    host = orig_from_torch(
                        args[0], mesh_mapper=kw.get("mesh_mapper") or ttnn.ReplicateTensorToMesh(device), **host_kw
                    )
                    reg._register(out, host, where)
            return out

        def to_device(tensor, *args, **kw):
            out = orig_to_device(tensor, *args, **kw)
            if reg.enabled and _is_l1(out):
                mine, where = reg._mine()
                if mine:
                    reg._register(out, tensor, where)
            return out

        def copy_host_to_device_tensor(host, device_tensor, *args, **kw):
            out = orig_copy(host, device_tensor, *args, **kw)
            if reg.enabled and _is_l1(device_tensor):
                mine, where = reg._mine()
                if mine:
                    reg._register(device_tensor, host, where)
            return out

        def create_global_semaphore(device, cores, initial_value, *args, **kw):
            sem = orig_semaphore(device, cores, initial_value, *args, **kw)
            if reg.enabled and reg._mine()[0]:
                reg.semaphores.append((sem, int(initial_value), _caller()[1], cores))
            return sem

        ttnn.from_torch, ttnn.to_device = from_torch, to_device
        ttnn.copy_host_to_device_tensor, ttnn.create_global_semaphore = (
            copy_host_to_device_tensor,
            create_global_semaphore,
        )
        return self

    def uninstall(self):
        for name, fn in self.orig.items():
            setattr(ttnn, name, fn)
        self.orig = {}

    # ------------------------------------------------------------------------------------------ the list
    def live(self) -> dict:
        self.entries = {a: e for a, e in self.entries.items() if e.live()}
        return self.entries

    def mark_per_step(self, *tensors):
        """The host writes these before every step: restore() leaves them alone."""
        for t in tensors:
            if t is not None and _is_l1(t) and _address(t) in self.live():
                self.entries[_address(t)].per_step = True

    def scratch(self, tensor, zero_ranges, defined):
        """``tensor`` is scratch: every range of it is written by a step before the step reads it, EXCEPT
        ``zero_ranges`` [(core (x, y), offset, bytes)], which must be zero at rest. ``defined``: where the piece says
        so. A property of the piece that owns the tensor, declared where the Stack takes the tensor over; the host's
        restore still rewrites the whole tensor (its last host-written contents: zeros), a program on the device
        zeroes the ranges only (plan())."""
        entry = self.live().get(_address(tensor))
        assert entry is not None, "scratch: the host never wrote this tensor"
        entry.scratch = ([(tuple(c), int(o), int(n)) for c, o, n in zero_ranges], defined)

    def plan(self) -> dict:
        """What a restore on the device writes, per chip: {"zero": [(address, where, cores or None for every core of
        the tensor, offset, bytes)], "store": [(address, where, bytes per chip)], "semaphores": n, "bytes": {...}}.
        A tensor whose last host-written contents are all zero is zeroed whole; a scratch tensor only in its ranges."""
        zero, store = [], []
        for e in self.live().values():
            if e.per_step:
                continue
            if e.scratch is not None:
                for core, offset, n in e.scratch[0]:
                    zero.append((e.address, e.where, [core], offset, n))
                continue
            shards = ttnn.get_device_tensors(e.host)
            if all(not bool(ttnn.to_torch(sh).contiguous().view(torch.uint8).any()) for sh in shards):
                zero.append((e.address, e.where, None, 0, e.bytes()))
            else:
                store.append((e.address, e.where, e.bytes()))
        return {
            "zero": zero,
            "store": store,
            "semaphores": len(self.semaphores),
            "bytes": {"zero": sum(z[4] for z in zero), "store": sum(x[2] for x in store)},
        }

    def bytes(self, per_step=False) -> int:
        return sum(e.bytes() for e in self.live().values() if per_step or not e.per_step)

    def summary(self) -> str:
        live = self.live()
        rows = sorted(live.values(), key=lambda e: -e.bytes())
        return (
            f"{len(live)} L1 tensors written by the host, {self.bytes(per_step=True)} B per chip ({self.bytes()} B without the "
            f"{sum(e.per_step for e in rows)} written before every step), {len(self.semaphores)} global semaphores; largest: "
            + ", ".join(f"{e.where} {e.bytes()} B x{e.writes}{' (per step)' if e.per_step else ''}" for e in rows[:6])
        )

    def unregistered(self, tensors: dict) -> dict:
        """{path: tensor} of the L1 tensors in ``tensors`` that the host never wrote."""
        live = self.live()
        return {p: t for p, t in tensors.items() if _is_l1(t) and _address(t) not in live}

    # ------------------------------------------------------------------------------------------ restore, verify
    def restore(self, mesh) -> tuple:
        """Returns (tensors written, bytes per chip)."""
        copy = self.orig.get("copy_host_to_device_tensor", ttnn.copy_host_to_device_tensor)
        count = nbytes = 0
        for e in self.live().values():
            if e.per_step:
                continue
            copy(e.host, e.tensor)
            count, nbytes = count + 1, nbytes + e.bytes()
        for sem, value, _, _ in self.semaphores:
            ttnn.reset_global_semaphore_value(sem, value)
        ttnn.synchronize_device(mesh)
        return count, nbytes

    def verify(self, mesh, ranges_only=False, chips=None) -> list:
        """[(where, chip, bytes that differ, L1 address)] of the registered tensors (but the per_step ones) whose contents on the
        device are not what the host last wrote. ranges_only: a scratch tensor is compared in its zero ranges only
        (after a restore on the device, which leaves the rest alone). chips: only these (linear indices)."""
        from models.experimental.hy4_preview.fused.model import reserve as RS

        ttnn.synchronize_device(mesh)
        bad = []
        for e in self.live().values():
            if e.per_step:
                continue
            if ranges_only and e.scratch is not None:
                cores = RS.cores_of(e.tensor)
                per = e.bytes() // len(cores)
                for chip, part in enumerate(ttnn.get_device_tensors(e.tensor)):
                    if chips is not None and chip not in chips:
                        continue
                    got = ttnn.to_torch(part).contiguous().view(torch.uint8).flatten()
                    for core, offset, n in e.scratch[0]:
                        start = cores.index(tuple(core)) * per + offset
                        if bool(got[start : start + n].any()):
                            bad.append(
                                (
                                    e.where,
                                    chip,
                                    f"range {core} {offset}+{n}: {int((got[start : start + n] != 0).sum())}",
                                    e.address,
                                )
                            )
                continue
            shards = ttnn.get_device_tensors(e.host)
            parts = ttnn.get_device_tensors(e.tensor)
            for chip, part in enumerate(parts):
                if chips is not None and chip not in chips:
                    continue
                want = ttnn.to_torch(shards[chip if len(shards) > 1 else 0]).contiguous()
                got = ttnn.to_torch(part).contiguous()
                a, b = want.view(torch.uint8).flatten(), got.view(torch.uint8).flatten()
                if a.numel() != b.numel():
                    bad.append((e.where, chip, f"sizes {a.numel()} / {b.numel()}", e.address))
                elif not torch.equal(a, b):
                    bad.append((e.where, chip, int((a != b).sum()), e.address))
        return bad


def install(mesh_device) -> HostWrites:
    """The process's registry (made and installed at the first call). A registry belongs to one opened mesh: with
    another one (the next test of a session) it starts empty: the tensors and semaphores of a closed mesh are gone."""
    global _installed
    if _installed is None:
        _installed = HostWrites().install()
        logger.debug("HY4 host writes into L1 are registered from here on")
    if getattr(_installed, "mesh", None) is not mesh_device:
        _installed.entries, _installed.semaphores, _installed.refused = {}, [], {}
        _installed.mesh = mesh_device
    return _installed
