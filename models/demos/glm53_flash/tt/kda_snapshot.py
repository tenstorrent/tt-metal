# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""KDA state snapshots of the slot-level prefix cache (device side; design: ``doc/serving/prefix_caching.md``).

The 34 KDA layers keep ONE recurrent state per user slot (``FunctionalDecoder.kda_batched``: ``recurrent``
``[B, H/cols, K, V]`` fp32 tiles + the conv taps ``[taps, B, W]`` bf16 tiles in block mode, per chip, B = users per
row). The traced decode step advances EVERY slot's state in place (idle slots included), so a finished request's
state at a 32-aligned position survives only as a copy: the served adapter snapshots a slot's carries at every
32-aligned decode crossing (the multi-turn hit position) and at eager prefill chunk ends, and restores one before a
prefix-cache hit continues the slot from that position.

Two stores behind one interface (``save(user, entry)`` / ``load(user, entry)``):

* ``DeviceKdaSnapshotRing``: a DRAM ring per KDA layer, ``users_per_row x entries`` entries per chip (every chip
  of a mesh row holds every user's heads: ``memory_plan.snapshot_bytes_per_chip``).
* ``HostKdaSnapshotStore``: the same rows read to the host (one tensor per chip, 35.2 MiB per chip per entry on the
  8x4: 1.1 GiB of host memory per snapshot) and written back per chip; for profiles whose DRAM holds no ring (the
  1 M x 4 profile) at the cost of a PCIe round trip per crossing / hit.

Allocation discipline (doc s.6). Every save / load runs between replays of the decode trace, and a device buffer
allocated while traces exist may sit where a replay's transient buffers were (the next replay overwrites it: the
tt-metal allocator's "potentially unsafe" warning, the 2026-10-27 served hang), so nothing here allocates after the
store is built. Both stores share one set of persistent tensors (``SnapshotDims.plan``, allocated by
``GLM53MeshModel.allocate_kda_snapshots`` before the first capture) and write every intermediate into them with
``output_tensor=`` / ``optional_output_tensor=`` / in-place ops; the per-call selectors (which user, which entry)
are refreshed in place with ``copy_host_to_device_tensor`` (in order on the command queue ahead of the ops that read
them: the traced prefill's ``PrefillSlot.set_user`` pattern). Per KDA layer:

* read a slot: ``ttnn.slice`` of the user's recurrent rows into ``rec_row`` ``[1, H, K, V]`` (tile-aligned begins:
  the native tiled slice writes the preallocated output) and a one-hot ``ttnn.matmul`` ``sel_user [taps, 1, B] @
  conv [taps, B, W]`` into ``conv_row`` ``[taps, 1, W]`` (HiFi4 with fp32 accumulation: every product is x * 1.0 or
  x * 0.0 and the sums add zeros, so the row is copied exactly; a tiled slice on an unaligned row would re-tile
  through ROW_MAJOR and allocate, which the old ``user_conv_state`` / ``conv_to_tiles`` round trips did);
* write a slot: ``paged_fill_cache`` of ``rec_row`` into the batched state's ``[B*H*K/32, 1, 32, V]`` view (identity
  page table, the slot index a device scalar: the traced prefill's own slot write) and the one-hot mask blend of
  the conv row (``conv = conv * keep + conv_row * mask``, exact in bf16) through ``conv_live_rows`` ``[taps, B, W]``;
* the device ring runs the same pair on its ring tensors (``[E, H, K, V]`` / ``[taps, E, W]`` per layer, E = users x
  entries) with the entry selectors; the host store reads ``rec_row`` / ``conv_row`` after the slot read and
  refreshes them from the host before the slot write.

The stores take an ``ops`` namespace (``ttnn`` by default; imported on first device use only) so
``tests/test_serving_prefix_cache_host.py`` drives both on fake ops and checks that no call allocates.
TODO(device): the first (2, 4) block run of this form (``tests/test_serving_prefix_cache_mesh.py``) confirms the
preallocated-output forms of ``slice`` / ``matmul`` / the binary ops on the mesh.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TILE = 32

_DTYPES = ("float32", "bfloat16", "int32")
_LAYOUTS = ("TILE", "ROW_MAJOR")


def _ttnn():
    import ttnn

    return ttnn


def _kda_layers(model) -> list:
    return [dec for dec in model.layers if getattr(dec, "kda_batched", None) is not None]


def one_hot_row(taps: int, n: int, idx: int) -> torch.Tensor:
    """``[taps, 1, n]`` with 1.0 at column ``idx``: the matmul lhs that selects row ``idx`` of a ``[taps, n, W]``."""
    sel = torch.zeros(taps, 1, n)
    sel[:, 0, idx] = 1.0
    return sel


def mask_keep(taps: int, n: int, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``[taps, n, 1]`` one-hot row mask and its complement (the conv row blend)."""
    mask = torch.zeros(taps, n, 1)
    mask[:, idx, 0] = 1.0
    return mask, 1.0 - mask


@dataclass(frozen=True)
class SnapshotDims:
    """The per-chip geometry of the KDA state and of the snapshot ring (``users`` slots x ``entries`` each)."""

    users: int  # B: user slots per mesh row (the batched state's dim 0)
    entries: int  # ring entries per user slot (the host store: the entry range it accepts)
    H: int  # heads per chip
    K: int
    V: int
    taps: int  # conv_kernel_size - 1
    W: int  # conv width per chip

    @classmethod
    def from_model(cls, model, entries: int) -> "SnapshotDims":
        if int(entries) < 1:
            raise ValueError(f"snapshot entries {entries} < 1")
        layers = _kda_layers(model)
        if not layers:
            raise RuntimeError("no KDA layer: nothing to snapshot")
        attn = layers[0]._require_weights().attn
        if not attn.conv_tiled:
            raise NotImplementedError(
                "KDA snapshots need the block KDA decode (tiled conv history, GLM53_KDA_FUSED=1), like the traced prefill"
            )
        live = layers[0].kda_batched
        B, H, K, V = (int(v) for v in live.recurrent.shape)
        taps, Bc, W = (int(v) for v in live.convolution.shape)
        if B != int(model.max_batch) or Bc != B:
            raise ValueError(f"KDA state batch {B} / conv batch {Bc} vs max_batch {model.max_batch}")
        if (H * K) % TILE:
            raise ValueError(f"H*K = {H * K} is not a tile multiple")
        return cls(users=B, entries=int(entries), H=H, K=K, V=V, taps=taps, W=W)

    @property
    def E(self) -> int:
        return self.users * self.entries

    @property
    def nblk(self) -> int:
        """Recurrent-state pages (32-row blocks) per user slot."""
        return self.H * self.K // TILE

    def entry(self, user: int, entry: int) -> int:
        if not (0 <= user < self.users) or not (0 <= entry < self.entries):
            raise ValueError(f"snapshot (user {user}, entry {entry}) outside {self.users} x {self.entries}")
        return user * self.entries + entry

    def plan(self, ring: bool) -> dict[str, tuple[tuple[int, ...], str, str]]:
        """The persistent tensors one store needs besides the ring entries, ``name -> (shape, dtype, layout)``: the
        scratch rows every call writes into, the per-call selectors refreshed in place, the identity page tables.
        Allocated once by the store's constructor (before any trace capture); ~2.5 MiB per chip plus
        ``taps x E x W`` bf16 for the ring's blend rows."""
        B, E, taps, W = self.users, self.E, self.taps, self.W
        plan = {
            "rec_row": ((1, self.H, self.K, self.V), "float32", "TILE"),  # one slot's / entry's recurrent state
            "conv_row": ((taps, 1, W), "bfloat16", "TILE"),  # one slot's / entry's conv taps
            "conv_live_rows": ((taps, B, W), "bfloat16", "TILE"),  # conv_row broadcast onto the live batch's row
            "sel_user": ((taps, 1, B), "bfloat16", "TILE"),  # one-hot row select of the live slot (matmul lhs)
            "user_mask": ((taps, B, 1), "bfloat16", "TILE"),  # 1.0 on the live slot's row
            "user_keep": ((taps, B, 1), "bfloat16", "TILE"),  # 1.0 - user_mask
            "user_idx": ((1,), "int32", "ROW_MAJOR"),  # the live slot (paged_fill_cache batch index)
            "live_pt": ((B, self.nblk), "int32", "ROW_MAJOR"),  # identity page table of the live state's view
        }
        if ring:
            plan.update(
                {
                    "conv_ring_rows": ((taps, E, W), "bfloat16", "TILE"),  # conv_row broadcast onto the ring's row
                    "sel_entry": ((taps, 1, E), "bfloat16", "TILE"),  # one-hot row select of the entry
                    "ring_mask": ((taps, E, 1), "bfloat16", "TILE"),
                    "ring_keep": ((taps, E, 1), "bfloat16", "TILE"),
                    "entry_idx": ((1,), "int32", "ROW_MAJOR"),
                    "ring_pt": ((E, self.nblk), "int32", "ROW_MAJOR"),  # identity page table of the ring's view
                }
            )
        return plan

    def ring_entry_shapes(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Per KDA layer: the ring's recurrent ``[E, H, K, V]`` fp32 and conv ``[taps, E, W]`` bf16 tensors."""
        return (self.E, self.H, self.K, self.V), (self.taps, self.E, self.W)


class SnapshotScratch:
    """The persistent tensors of ``SnapshotDims.plan`` on the device plus the slot read / write primitives both
    stores are built on. ``ops``: ``ttnn`` (default) or the host test's fake namespace."""

    def __init__(self, model, dims: SnapshotDims, *, ring: bool, ops=None):
        self.ops = ops if ops is not None else _ttnn()
        self.model = model
        self.mesh = model.mesh
        self.dims = dims
        self.layers = _kda_layers(model)
        self.plan = dims.plan(ring)
        self.t: dict[str, object] = {}
        for name, (shape, dtype, layout) in self.plan.items():
            self.t[name] = self._alloc(shape, dtype, layout)
        self._refresh("live_pt", torch.arange(dims.users * dims.nblk, dtype=torch.int32))
        if ring:
            self._refresh("ring_pt", torch.arange(dims.E * dims.nblk, dtype=torch.int32))
        self._selected: dict[str, int | None] = {"user": None, "entry": None}
        o = self.ops
        # the one-hot selection matmul: full-mantissa products, fp32 accumulation (exact for x * 1.0 + zeros)
        self.hifi4 = o.init_device_compute_kernel_config(
            self.mesh.device.arch(),
            math_fidelity=o.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )

    # ------------------------------------------------------------------ allocation (construction only)
    def _dtype(self, name: str):
        assert name in _DTYPES, name
        return getattr(self.ops, name)

    def _layout(self, name: str):
        assert name in _LAYOUTS, name
        return getattr(self.ops, f"{name}_LAYOUT")

    def _mapper(self):
        return None if self.mesh.is_single else self.mesh.replicate()

    def _alloc(self, shape: tuple[int, ...], dtype: str, layout: str):
        """A zeroed persistent tensor (replicated on every chip)."""
        if dtype == "int32":  # the tables / indices: an upload of zeros (a 1-D ``zeros`` is not needed on device)
            return self.ops.from_torch(
                torch.zeros(shape, dtype=torch.int32),
                dtype=self._dtype(dtype),
                layout=self._layout(layout),
                device=self.mesh.device,
                memory_config=self.ops.DRAM_MEMORY_CONFIG,
                mesh_mapper=self._mapper(),
            )
        return self.ops.zeros(
            tuple(shape),
            dtype=self._dtype(dtype),
            layout=self._layout(layout),
            device=self.mesh.device,
            memory_config=self.ops.DRAM_MEMORY_CONFIG,
        )

    # ------------------------------------------------------------------ in-place refreshes (no allocation)
    def _refresh(self, name: str, host: torch.Tensor) -> None:
        """Host values -> the persistent tensor ``name`` (replicated), in order on the command queue."""
        shape, dtype, layout = self.plan[name]
        src = host.reshape(shape)
        if dtype == "int32":
            src = src.to(torch.int32)
        h = self.ops.from_torch(
            src.contiguous(), dtype=self._dtype(dtype), layout=self._layout(layout), mesh_mapper=self._mapper()
        )
        self.ops.copy_host_to_device_tensor(h, self.t[name])

    def select_user(self, user: int) -> None:
        """Point the live-slot selectors at ``user`` (refreshed only when the value changes)."""
        if not 0 <= user < self.dims.users:
            raise ValueError(f"user {user} outside {self.dims.users} slots")
        if self._selected["user"] == user:
            return
        taps, B = self.dims.taps, self.dims.users
        mask, keep = mask_keep(taps, B, user)
        self._refresh("sel_user", one_hot_row(taps, B, user))
        self._refresh("user_mask", mask)
        self._refresh("user_keep", keep)
        self._refresh("user_idx", torch.tensor([user], dtype=torch.int32))
        self._selected["user"] = user

    def select_entry(self, idx: int) -> None:
        """Point the ring-entry selectors at flat entry ``idx`` (refreshed only when the value changes)."""
        if "sel_entry" not in self.t:
            raise RuntimeError("no ring selectors (host store)")
        if self._selected["entry"] == idx:
            return
        taps, E = self.dims.taps, self.dims.E
        mask, keep = mask_keep(taps, E, idx)
        self._refresh("sel_entry", one_hot_row(taps, E, idx))
        self._refresh("ring_mask", mask)
        self._refresh("ring_keep", keep)
        self._refresh("entry_idx", torch.tensor([idx], dtype=torch.int32))
        self._selected["entry"] = idx

    # ------------------------------------------------------------------ the slot primitives (no allocation)
    def _view(self, x, shape: tuple[int, ...]):
        return self.ops.experimental.view(x, self.ops.Shape(list(shape)))

    def _read_rows(self, rec, conv, idx: int, sel) -> None:
        """Row ``idx`` of a batched state (``rec [N, H, K, V]``, ``conv [taps, N, W]``) -> ``rec_row`` / ``conv_row``."""
        d, t = self.dims, self.t
        self.ops.slice(rec, [idx, 0, 0, 0], [idx + 1, d.H, d.K, d.V], output_tensor=t["rec_row"])
        self.ops.matmul(sel, conv, compute_kernel_config=self.hifi4, optional_output_tensor=t["conv_row"])

    def _write_rows(self, rec, conv, n: int, page_table, batch_idx, mask, keep, rows) -> None:
        """``rec_row`` / ``conv_row`` -> row ``batch_idx`` of a batched state of ``n`` rows (in place)."""
        d, t = self.dims, self.t
        self.ops.experimental.paged_fill_cache(
            self._view(rec, (n * d.nblk, 1, TILE, d.V)),
            self._view(t["rec_row"], (1, 1, d.H * d.K, d.V)),
            page_table,
            batch_idx_tensor=batch_idx,
        )
        self.ops.multiply(conv, keep, output_tensor=conv)  # zero the row
        self.ops.multiply(t["conv_row"], mask, output_tensor=rows)  # [taps,1,W] x [taps,n,1] -> [taps,n,W]
        self.ops.add(conv, rows, output_tensor=conv)

    def read_live(self, k: int, user: int) -> None:
        """Layer ``k``'s live slot ``user`` -> ``rec_row`` / ``conv_row`` (``select_user`` first)."""
        live = self.layers[k].kda_batched
        self._read_rows(live.recurrent, live.convolution, user, self.t["sel_user"])

    def write_live(self, k: int, user: int) -> None:
        """``rec_row`` / ``conv_row`` -> layer ``k``'s live slot ``user`` (``select_user`` first)."""
        live, t = self.layers[k].kda_batched, self.t
        self._write_rows(
            live.recurrent,
            live.convolution,
            self.dims.users,
            t["live_pt"],
            t["user_idx"],
            t["user_mask"],
            t["user_keep"],
            t["conv_live_rows"],
        )

    def release(self) -> None:
        for x in self.t.values():
            if x.is_allocated():
                self.ops.deallocate(x)
        self.t = {}


class DeviceKdaSnapshotRing(SnapshotScratch):
    """DRAM ring of KDA snapshots: ``entries`` per user slot (flat entry ``user * entries + e`` per chip)."""

    store = "device"

    def __init__(self, model, entries: int, ops=None):
        dims = SnapshotDims.from_model(model, entries)
        super().__init__(model, dims, ring=True, ops=ops)
        self.entries = dims.entries
        self.users = dims.users
        self.total_entries = dims.E
        rec_shape, conv_shape = dims.ring_entry_shapes()
        self.rec = [self._alloc(rec_shape, "float32", "TILE") for _ in self.layers]
        self.conv = [self._alloc(conv_shape, "bfloat16", "TILE") for _ in self.layers]

    def save(self, user: int, entry: int) -> None:
        """The live carries of slot ``user`` -> ring entry ``entry`` (every chip, its own heads)."""
        idx = self.dims.entry(user, entry)
        self.select_user(user)
        self.select_entry(idx)
        t = self.t
        for k in range(len(self.layers)):
            self.read_live(k, user)
            self._write_rows(
                self.rec[k],
                self.conv[k],
                self.dims.E,
                t["ring_pt"],
                t["entry_idx"],
                t["ring_mask"],
                t["ring_keep"],
                t["conv_ring_rows"],
            )

    def load(self, user: int, entry: int) -> None:
        """Ring entry ``entry`` of slot ``user`` -> the live carries of slot ``user``."""
        idx = self.dims.entry(user, entry)
        self.select_user(user)
        self.select_entry(idx)
        for k in range(len(self.layers)):
            self._read_rows(self.rec[k], self.conv[k], idx, self.t["sel_entry"])
            self.write_live(k, user)

    def release(self) -> None:
        for x in self.rec + self.conv:
            if x.is_allocated():
                self.ops.deallocate(x)
        self.rec, self.conv = [], []
        super().release()


class HostKdaSnapshotStore(SnapshotScratch):
    """Host copies of KDA snapshots (``entries`` per user slot; no ring in DRAM, the shared scratch only): per layer
    the per-chip recurrent row ``[chips, 1, H, K, V]`` fp32 and conv row ``[chips, taps, 1, W]`` bf16."""

    store = "host"

    def __init__(self, model, entries: int, ops=None):
        dims = SnapshotDims.from_model(model, entries)
        super().__init__(model, dims, ring=False, ops=ops)
        self.entries = dims.entries
        self.users = dims.users
        self.data: dict[tuple[int, int], list[tuple]] = {}  # per layer: (rec, conv) torch tensors or pending reads
        self.events: dict[tuple[int, int], object] = {}  # the save's recorded event while its reads are in flight

    def _read(self, name: str) -> torch.Tensor:
        return torch.stack([self.ops.to_torch(d) for d in self.mesh.device_tensors(self.t[name])])

    def _read_async(self, name: str):
        """The per-chip reads of scratch ``name`` enqueued without waiting (``Tensor.cpu(blocking=False)``: the next
        layer's slot read into the same scratch is ordered behind them on the command queue); the host tensors are
        materialized by ``_materialize`` after the save's event. Ops without the asynchronous read (the host tests'
        fakes) read synchronously."""
        out = []
        for d in self.mesh.device_tensors(self.t[name]):
            cpu = getattr(d, "cpu", None)
            out.append(cpu(blocking=False) if callable(cpu) else self.ops.to_torch(d))
        if all(isinstance(r, torch.Tensor) for r in out):  # the synchronous fallback: already host data
            return torch.stack(out)
        return out

    @staticmethod
    def _is_torch(reads) -> bool:
        return isinstance(reads, torch.Tensor)

    def _materialize(self, reads) -> torch.Tensor:
        if self._is_torch(reads):
            return reads
        return torch.stack([r if isinstance(r, torch.Tensor) else self.ops.to_torch(r) for r in reads])

    def _write_back(self, name: str, host: torch.Tensor) -> None:
        """Per-chip rows (``[chips, *shape]``) -> the scratch tensor ``name`` (chip i gets chunk i; no allocation)."""
        shape, dtype, layout = self.plan[name]
        if self.mesh.is_single:
            mapper, src = None, host[0]
        else:
            mapper, src = self.mesh.shard_2d(0, 0), host.reshape(-1, *shape[1:])  # chunk i of dim 0 -> chip i
        h = self.ops.from_torch(
            src.contiguous(), dtype=self._dtype(dtype), layout=self._layout(layout), mesh_mapper=mapper
        )
        self.ops.copy_host_to_device_tensor(h, self.t[name])

    def save(self, user: int, entry: int) -> None:
        """The slot's carries -> host, one non-blocking read per chip per layer and one recorded event (the host
        does not wait for the transfer: the synchronous reads cost 725-1194 ms per save at 32 chips x 34 layers,
        served 2026-10-29); ``load`` waits for the event before it uses the data."""
        self.dims.entry(user, entry)
        self.select_user(user)
        out = []
        for k in range(len(self.layers)):
            self.read_live(k, user)
            out.append((self._read_async("rec_row"), self._read_async("conv_row")))
        record = getattr(self.ops, "record_event", None)
        pending = any(not self._is_torch(r) for pair in out for r in pair)
        self.events[(user, entry)] = record(self.mesh.device, 0) if pending and callable(record) else None
        self.data[(user, entry)] = out

    def load(self, user: int, entry: int) -> None:
        self.dims.entry(user, entry)
        if (user, entry) not in self.data:
            raise KeyError(f"no host snapshot for (user {user}, entry {entry})")
        event = self.events.pop((user, entry), None)
        if event is not None:
            self.ops.event_synchronize(event)
        layers = [(self._materialize(rec), self._materialize(conv)) for rec, conv in self.data[(user, entry)]]
        self.data[(user, entry)] = layers
        self.select_user(user)
        for k, (rec_host, conv_host) in enumerate(layers):
            self._write_back("rec_row", rec_host)
            self._write_back("conv_row", conv_host)
            self.write_live(k, user)

    def release(self) -> None:
        self.data = {}
        self.events = {}
        super().release()
