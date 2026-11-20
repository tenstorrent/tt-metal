# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""The restore of the fused step's L1 contents as a program on the device (kernels/restore.cpp): it writes what the
host-write registry lists (hostwrites.HostWrites.plan): zeros in the tensors and ranges that are zero at rest, the
global semaphores' initial values, and every buffer with contents from a copy of it in DRAM (the restore image, one
shard per chip, made here from the registry's host tensors). One launch replaces Stack.restore() from the host.

    r = Restore(mesh, stack.hw)        # after the last host write that belongs to the state at rest
    Stack.launch([("restore", *r.program())])

A registered tensor whose contents kernels keep across steps (a frame table in L1 with a twin in DRAM) is not the
host's to restore: its owner passes a hook, ``hooks=[fn]``, called after the launch (fn() launches the owner's own
program that copies the twin back). None exists in the forms built so far.

The image is taken when the Restore is made: after a host write into a stored tensor (a frame table rewritten for a
new binding) stale() is true and the program refuses to launch until refresh() has written the new contents into the
same image tensor.
"""

from __future__ import annotations

from pathlib import Path

import torch

import ttnn
from models.experimental.hy4_preview.fused.layer_common import transport as tp
from models.experimental.hy4_preview.fused.model import reserve as RS

KERNEL_DIR = Path(__file__).resolve().parent / "kernels"
PAGE = 256
MAX_ARGS = 300


class Restore:
    def __init__(self, mesh_device, hw, chips=None, hooks=()):
        """chips: mesh coordinates the program runs on (default: all)."""
        self.mesh, self.hw, self.hooks = mesh_device, hw, list(hooks)
        rows, cols = tuple(mesh_device.shape)
        self.coords = [(r, c) for r in range(rows) for c in range(cols)] if chips is None else [tuple(c) for c in chips]
        self.zero, self.words, self.store, shards, self.n_pages, self.stamp, self.counts = self._plan()
        self.cores = sorted(set(self.zero) | set(self.words) | set(self.store))
        self.image = ttnn.to_device(
            ttnn.from_host_shards(shards, mesh_device.shape), mesh_device, memory_config=ttnn.DRAM_MEMORY_CONFIG
        )
        ttnn.synchronize_device(mesh_device)
        self.max_args = max(len(self._args(core)) for core in self.cores)
        assert self.max_args <= MAX_ARGS, f"{self.max_args} runtime arguments on a core"

    def _plan(self):
        """The registry as it is now: (core -> [(address, bytes)] to zero, core -> [(address, value)] to store,
        core -> [(address, bytes, first page)] to read from the image, the image's host shards (one per chip),
        its pages, {address: writes} of the tensors covered, the counts)."""
        hw = self.hw
        n_chips = self.mesh.get_num_devices()
        zero, words, store = {}, {}, {}
        pages = [[] for _ in range(n_chips)]  # per chip: uint8 pieces of the image
        n_pages = 0
        stamp = {}
        counts = {"zero": 0, "store": 0, "zero_bytes": 0, "store_bytes": 0}
        for e in hw.live().values():
            if e.per_step:
                continue
            stamp[e.address] = e.writes
            cores = RS.cores_of(e.tensor)
            assert cores, f"{e.where}: not an L1 tensor with a shard per core"
            per = e.bytes() // len(cores)
            assert per * len(cores) == e.bytes() and per % 16 == 0, (e.where, e.bytes(), len(cores))
            if e.scratch is not None:
                for core, offset, n in e.scratch[0]:
                    assert tuple(core) in cores and offset + n <= per and n % 4 == 0 and offset % 4 == 0, (
                        e.where,
                        core,
                        offset,
                        n,
                    )
                    zero.setdefault(tuple(core), []).append((e.address + offset, n))
                    counts["zero"] += 1
                    counts["zero_bytes"] += n
                continue
            shards = [
                ttnn.to_torch(t).contiguous().view(torch.uint8).flatten() for t in ttnn.get_device_tensors(e.host)
            ]
            assert all(s.numel() == e.bytes() for s in shards), (e.where, [s.numel() for s in shards][:2], e.bytes())
            if all(not bool(s.any()) for s in shards):
                for core in cores:
                    zero.setdefault(core, []).append((e.address, per))
                counts["zero"] += 1
                counts["zero_bytes"] += e.bytes()
                continue
            per_pages = -(-per // PAGE)
            for i, core in enumerate(cores):
                store.setdefault(core, []).append((e.address, per, n_pages + i * per_pages))
            for chip in range(n_chips):
                data = shards[chip if len(shards) > 1 else 0].reshape(len(cores), per)
                padded = torch.zeros(len(cores), per_pages * PAGE, dtype=torch.uint8)
                padded[:, :per] = data
                pages[chip].append(padded.reshape(-1))
            n_pages += len(cores) * per_pages
            counts["store"] += 1
            counts["store_bytes"] += e.bytes()
        for sem, value, _, crs in hw.semaphores:
            address = ttnn.get_global_semaphore_address(sem)
            for c in ttnn.corerange_to_cores(crs):
                words.setdefault((c.x, c.y), []).append((int(address), int(value)))
        n_pages = max(n_pages, 1)
        shards = []
        for chip in range(n_chips):
            image = torch.cat(pages[chip]) if pages[chip] else torch.zeros(PAGE, dtype=torch.uint8)
            shards.append(
                ttnn.from_torch(
                    image.view(torch.int32).reshape(1, 1, n_pages, PAGE // 4),
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                )
            )
        return zero, words, store, shards, n_pages, stamp, counts

    def refresh(self):
        """After a host write into a stored tensor (a frame table rewritten for a new binding): the image's contents
        again from the registry, written INTO THE SAME DRAM tensor (a host write in place: nothing is allocated, so
        it may happen after every capture). The tensors covered, what is zeroed and what is stored where must be as
        when this Restore was made (else a new one is needed, inside reserved memory)."""
        zero, words, store, shards, n_pages, stamp, counts = self._plan()
        assert (zero, words, store, n_pages) == (
            self.zero,
            self.words,
            self.store,
            self.n_pages,
        ), "the restore's plan changed"
        ttnn.copy_host_to_device_tensor(ttnn.from_host_shards(shards, self.mesh.shape), self.image)
        ttnn.synchronize_device(self.mesh)
        self.stamp, self.counts = stamp, counts

    def _args(self, core):
        z, w, s = self.zero.get(core, []), self.words.get(core, []), self.store.get(core, [])
        args = [self.image.buffer_address(), len(z)]
        for item in z:
            args += list(item)
        args.append(len(w))
        for item in w:
            args += list(item)
        args.append(len(s))
        for item in s:
            args += list(item)
        return [int(a) for a in args]

    def stale(self) -> bool:
        """A stored or zeroed tensor was written by the host, or registered, after this Restore was made."""
        now = {e.address: e.writes for e in self.hw.live().values() if not e.per_step}
        return now != self.stamp

    def summary(self) -> str:
        c = self.counts
        return (
            f"restore program: {len(self.cores)} cores a chip, at most {self.max_args} runtime arguments on a core; zeroes "
            f"{c['zero']} tensors or ranges ({c['zero_bytes']} B per chip), stores {c['store']} tensors from a DRAM image of "
            f"{self.n_pages} pages of {PAGE} B ({c['store_bytes']} B per chip), sets {len(self.hw.semaphores)} semaphores"
        )

    def program(self):
        """(io tensors, MeshProgramDescriptor)."""
        assert not self.stale(), "the registry changed since this Restore was made"
        rt = ttnn.RuntimeArgs()
        for core in self.cores:
            rt[core[0]][core[1]] = self._args(core)
        mesh_desc = ttnn.MeshProgramDescriptor()
        for coord in self.coords:
            kernel = ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "restore.cpp"),
                core_ranges=tp.core_set(self.cores),
                compile_time_args=list(ttnn.TensorAccessorArgs(self.image).get_compile_time_args()),
                runtime_args=rt,
                defines=[("RESTORE_PAGE", f"{PAGE}u")],
                config=ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                ),
            )
            desc = ttnn.ProgramDescriptor(kernels=[kernel], semaphores=[], cbs=[])
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = desc
        return [self.image, self.image], mesh_desc

    def launch(self):
        io, desc = self.program()
        (ttnn.experimental.prepare_generic_op if tp.is_mock() else ttnn.generic_op)(io, desc)
        for hook in self.hooks:
            hook()

    def free(self):
        ttnn.deallocate(self.image)
