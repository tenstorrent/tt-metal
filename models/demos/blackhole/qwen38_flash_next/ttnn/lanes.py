# SPDX-FileCopyrightText: Copyright (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Lane lifecycle of the batched decode lanes: eviction of one lane's state to a pinned host pool and its
re-admission into any lane, the host page table (session -> lane or host slot) and the residue-aligned admission
scheduler.

The lane state of the batched layout is 237 fixed-address tensors per device (12 QSA packed KV caches
``[1,1,B*C,512]``, 12 compressed index caches ``[B,1,C/4+32,128]``, 12 KV staging tiles ``[1,B,32,512]``, 12
raw-key rings ``[1,B,32,128]``, 36 GDN recurrent states ``[B,12,128,128]``, 144 GDN ring slots ``[1,1,B,2560]``,
9 PLE slots ``[1,B,4,640]``); lane u is a batch index, a dim-1 index, a row of a tile or a ``C``-row region of
every one of them.  Moving lane u tensor by tensor costs one host transfer per tensor (61 / 53 ms per leg at 32k on
measured 2026-09-04, most of it per-transfer overhead), so the pager moves a *packed lane image* instead:

* six small pack buffers (compressed ``[12,1,R,128]``, recurrent ``[36,12,128,128]``, staging ``[12,1,32,512]``,
  ring ``[12,1,32,128]``, PLE ``[9,1,4,640]``, ring slots ``[1,144,1,2560]`` ROW_MAJOR) filled by one *pack trace*
  per lane (slices at the lane index and concats into the persistent buffers) and read with six transfers; written
  back with six transfers and one *unpack trace* per lane (``fill_cache`` at the lane's batch index for the
  batch-indexed families, the selection write ``slot * keep + hit @ row`` for the ring-slot rows);
* the 12 KV regions, the bytes that scale with the context, as 12 whole-lane slabs: ``slice`` of the ``C`` rows into
  an alternating pair of staging buffers and one transfer each; back by one transfer and one whole-slab
  ``update_padded_kv_cache`` at row ``u*C`` (bitwise at 4k / 8k / 32k, 0.15-0.38 ms, measured 2026-09-04).

Every op is data movement or an exact one-term select, so the round trip is bitwise; the traces bake the lane
index, so there are ``B`` pack and ``B`` unpack traces, captured after every lane tensor and pack buffer exists.
The pager never touches the position row: re-admission goes through :meth:`Qwen38TTNNDevicePositionRow.admit`
first (the residue rule), a masked lane reset is the components' ``reset_lane_inplace``.  Page moves run from
:class:`Qwen38LaneMover`, one helper thread, after the step's non-blocking launch (they queue behind the step on
command queue 0) and are joined before the next step's host writes.  The 1-row production path is untouched.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from ttnn.tools.trace_allocation_tracker import corruptible_allocation_scope

import ttnn
from models.demos.blackhole.qwen38_flash_next.config import LAYER_PATTERN
from models.demos.blackhole.qwen38_flash_next.ttnn import gdn as gdn_module
from models.demos.blackhole.qwen38_flash_next.ttnn import ple as ple_module
from models.demos.blackhole.qwen38_flash_next.ttnn import qsa as qsa_module
from models.demos.blackhole.qwen38_flash_next.ttnn.contracts import (
    GDN_RESIDUE_CLASSES,
    MESH_SHAPE,
    TP_SIZE,
    Qwen38MeshContract,
    Qwen38TTNNDevicePositionRow,
    TensorPlacement,
    admission_wait_steps,
    replicate_tensor_2d_mesh_mapper,
    require_lane_count,
    tensor_metadata,
)

QSA_LAYERS = LAYER_PATTERN.count("full_attention")
GDN_LAYERS = LAYER_PATTERN.count("linear_attention")
RING_SLOTS = GDN_LAYERS * gdn_module.CONV_KERNEL_SIZE
KV_WIDTH = 2 * qsa_module.HEAD_DIM
RECURRENT_ROWS = gdn_module.VALUE_HEADS_PER_DEVICE * gdn_module.HEAD_DIM
# ``ttnn.concat`` inputs per launch (the 144 ring slots concatenate as a two-level tree).
CONCAT_FAN_IN = 32
# The KV slabs alternate between two staging buffers so slab l+1 is sliced while slab l is read.
KV_STAGINGS = 2
# Families whose lane is a batch or dim-1 index: written back by ``fill_cache`` at the lane's batch index (through a
# same-buffer view with the lane axis at dim 0 where needed).  The ring slots (lane = a tile row) always take the
# selection write.
FILL_FAMILIES = ("compressed", "recurrent", "staging", "ring", "ple")
# Lane axis of every family's device tensors.
LANE_AXIS = {"kv": 2, "compressed": 0, "recurrent": 0, "staging": 1, "ring": 1, "ple": 1, "conv": 2}


def _tensor_key(tensor) -> tuple[str, int]:
    tensor_id = getattr(tensor, "tensor_id", None)
    if callable(tensor_id):
        tensor_id = tensor_id()
    return ("ttnn", int(tensor_id)) if tensor_id is not None else ("python", id(tensor))


def _deallocate(*tensors) -> None:
    seen: set[tuple[str, int]] = set()
    for tensor in tensors:
        if tensor is not None and _tensor_key(tensor) not in seen:
            seen.add(_tensor_key(tensor))
            ttnn.deallocate(tensor)


def _shape(tensor) -> tuple[int, ...]:
    return tuple(int(value) for value in tensor.shape)


def slice_owned(tensor, start: Sequence[int], end: Sequence[int]):
    """``tensor[start:end]`` in DRAM as a tensor the caller owns and releases.  ``ttnn.slice`` over the whole tensor
    returns its input (the no-op path of ``slice.cpp``): a lane slice of a 1-lane layout or of a 1-layer pack would
    alias the state or the pack, and the pack body's release of its parts would free that (the generic pager over a
    chain's 1-lane state freed every GDN recurrent, 2026-09-20); that case is a clone."""

    start, end = tuple(int(v) for v in start), tuple(int(v) for v in end)
    if all(v == 0 for v in start) and end == _shape(tensor):
        return ttnn.clone(tensor, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.slice(tensor, start, end, memory_config=ttnn.DRAM_MEMORY_CONFIG)


def _landed(result, target, label: str) -> None:
    """An in-place op must hand back the target's own buffer."""

    if result is not None and _tensor_key(result) != _tensor_key(target):
        raise RuntimeError(f"{label} did not land in the persistent target")


def _exact_index(value, limit: int, *, label: str) -> int:
    if isinstance(value, bool) or type(value) is not int or not 0 <= value < limit:
        raise ValueError(f"{label} must be an int in [0,{limit}), got {value!r}")
    return value


# ---------------------------------------------------------------- the lane layout


@dataclass(frozen=True)
class Qwen38LaneFamily:
    """One family of the packed image: the per-device pack shape, its mesh placement, dtype and layout.  The host
    pool holds ``count`` tensors of this shape per slot (12 for the per-layer KV slabs, 1 otherwise)."""

    name: str
    local_shape: tuple[int, ...]
    shard_dim: int | None
    placement: TensorPlacement
    dtype: Any
    layout: Any
    count: int = 1

    @property
    def global_shape(self) -> tuple[int, ...]:
        if self.shard_dim is None:
            return self.local_shape
        return tuple(size * TP_SIZE if dim == self.shard_dim else size for dim, size in enumerate(self.local_shape))

    def bytes_per_device(self) -> int:
        numel = 1
        for size in self.local_shape:
            numel *= size
        return numel * (4 if self.dtype == ttnn.float32 else 2) * self.count

    def mapper(self, mesh_device):
        if self.shard_dim is None:
            return replicate_tensor_2d_mesh_mapper(mesh_device)
        return ttnn.ShardTensor2dMesh(mesh_device, mesh_shape=MESH_SHAPE, dims=(None, self.shard_dim))

    def allocate(self, mesh_device, *, device: bool):
        """A zero tensor of the family's shape: on the mesh (``device``) or on the host (a pool slot)."""

        host = torch.zeros(self.global_shape, dtype=torch.float32 if self.dtype == ttnn.float32 else torch.bfloat16)
        if not device:
            return ttnn.from_torch(host, dtype=self.dtype, layout=self.layout, mesh_mapper=self.mapper(mesh_device))
        return ttnn.from_torch(
            host,
            dtype=self.dtype,
            layout=self.layout,
            device=mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=self.mapper(mesh_device),
        )


def _lane_shapes(lanes: int, context: int) -> dict[str, tuple[tuple[int, ...], TensorPlacement, int | None, Any, Any]]:
    """Per family: the device tensor's local shape at ``lanes`` lanes and ``context`` rows, its placement, shard
    dim, dtype and layout."""

    q, g, p = qsa_module, gdn_module, ple_module
    rows = context // q.COMPRESS_RATIO + ttnn.TILE_SIZE
    kv_pair, head, hidden, replicated = (
        TensorPlacement.KV_PAIR_GROUPED,
        TensorPlacement.HEAD_SHARDED,
        TensorPlacement.HIDDEN_SHARDED,
        TensorPlacement.REPLICATED,
    )
    bf16, fp32, tile, rm = ttnn.bfloat16, ttnn.float32, ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
    return {
        "kv": ((1, 1, lanes * context, KV_WIDTH), kv_pair, 1, bf16, rm),
        "compressed": ((lanes, 1, rows, q.INDEX_HEAD_DIM), replicated, None, bf16, tile),
        "staging": ((1, lanes, q.CACHE_WRITE_ROWS, KV_WIDTH), kv_pair, 1, bf16, tile),
        "ring": ((1, lanes, q.CACHE_WRITE_ROWS, q.INDEX_HEAD_DIM), replicated, None, bf16, tile),
        "recurrent": ((lanes, g.VALUE_HEADS_PER_DEVICE, g.HEAD_DIM, g.HEAD_DIM), head, 1, fp32, tile),
        "conv": ((1, 1, lanes, g.QKV_WIDTH_PER_DEVICE), head, 3, bf16, tile),
        "ple": ((1, lanes, p.RESIDUAL_BRANCHES, p.LOCAL_HIDDEN_SIZE), hidden, 3, bf16, tile),
    }


@dataclass(frozen=True)
class Qwen38LaneLayout:
    """The per-lane device tensors of ``lanes`` batched lanes at ``allocated_context`` rows per lane, by family
    (each tuple in layer order; ``conv`` layer-major, ring slot minor).  The full model has 12 / 12 / 12 / 12 / 36
    / 144 / 9; a component micro-test may hold a subset of layers (the counts must agree within a family)."""

    lanes: int
    allocated_context: int
    kv: tuple[Any, ...]
    compressed: tuple[Any, ...]
    staging: tuple[Any, ...]
    ring: tuple[Any, ...]
    recurrent: tuple[Any, ...]
    conv: tuple[Any, ...]
    ple: tuple[Any, ...]

    def tensors(self) -> tuple[Any, ...]:
        return (*self.kv, *self.compressed, *self.staging, *self.ring, *self.recurrent, *self.conv, *self.ple)

    def families(self) -> tuple[Qwen38LaneFamily, ...]:
        """The pack buffers of this layout (a family with no tensors is absent): lane u's slice of every tensor of
        the family stacked on dim 0 (``conv``: the 144 rows side by side on dim 1, ROW_MAJOR; ``kv``: one
        ``C``-row slab per layer)."""

        shapes = _lane_shapes(1, self.allocated_context)
        families = []
        for name, (shape, placement, shard_dim, dtype, layout) in shapes.items():
            tensors = getattr(self, name)
            if not tensors:
                continue
            if name == "kv":
                families.append(Qwen38LaneFamily(name, shape, shard_dim, placement, dtype, layout, len(tensors)))
            elif name == "conv":
                families.append(
                    Qwen38LaneFamily(
                        name, (1, len(tensors), 1, shape[3]), shard_dim, placement, dtype, ttnn.ROW_MAJOR_LAYOUT
                    )
                )
            else:
                stacked = (len(tensors), *shape[1:]) if LANE_AXIS[name] == 0 else (len(tensors), shape[0], *shape[2:])
                families.append(Qwen38LaneFamily(name, stacked, shard_dim, placement, dtype, layout))
        return tuple(families)

    def image_bytes_per_device(self) -> int:
        """Payload bytes of one lane's image per device (the host pool's bytes per slot per device)."""

        return sum(family.bytes_per_device() for family in self.families())

    def validate(self, mesh_contract: Qwen38MeshContract) -> None:
        """Every tensor of every family in the batched-lanes shapes at this lane count and context (actual vs
        expected in the error); the family counts consistent (one tensor per QSA layer per QSA family, four ring
        slots per recurrent state, 0 or 9 PLE slots); distinct buffers."""

        lanes = require_lane_count(self.lanes, label="lane layout lanes", error_type=RuntimeError)
        context = qsa_module.validate_qsa_cache_capacity(self.allocated_context)
        if len({len(self.kv), len(self.compressed), len(self.staging), len(self.ring)}) != 1:
            raise RuntimeError(
                f"QSA lane families must hold one tensor per QSA layer each, got kv {len(self.kv)} compressed "
                f"{len(self.compressed)} staging {len(self.staging)} ring {len(self.ring)}"
            )
        if len(self.conv) != gdn_module.CONV_KERNEL_SIZE * len(self.recurrent):
            raise RuntimeError(
                f"GDN ring slots must be {gdn_module.CONV_KERNEL_SIZE} per recurrent state, got {len(self.conv)} "
                f"for {len(self.recurrent)}"
            )
        if len(self.ple) not in (0, ple_module.CONV_STATE_LENGTH):
            raise RuntimeError(f"PLE slots must be 0 or {ple_module.CONV_STATE_LENGTH}, got {len(self.ple)}")
        if not self.tensors():
            raise RuntimeError("lane layout holds no tensors")
        keys: set[tuple[str, int]] = set()
        for name, (shape, placement, shard_dim, dtype, layout) in _lane_shapes(lanes, context).items():
            for index, tensor in enumerate(getattr(self, name)):
                if _shape(tensor) != shape or tensor.dtype != dtype or tensor.layout != layout:
                    raise RuntimeError(
                        f"lane {name}[{index}] must be {dtype} {layout} {list(shape)}, got {tensor_metadata(tensor)}"
                    )
                mesh_contract.validate_tensor(tensor, placement=placement, shard_dim=shard_dim)
                keys.add(_tensor_key(tensor))
        if len(keys) != len(self.tensors()):
            raise RuntimeError("lane layout tensors must be distinct buffers")


# ---------------------------------------------------------------- the pinned host pool


@dataclass
class Qwen38LaneHostSlot:
    """One parked lane image on the host: the pack tensors by family (``kv`` holds one per QSA layer) plus the
    host-side lane record the device does not carry (position, committed tokens, PLE n-gram context)."""

    index: int
    tensors: dict[str, tuple[Any, ...]]
    position: int = 0
    committed: tuple[int, ...] = ()
    ple_context: tuple[int, int] | None = None
    session_id: str | None = None


class Qwen38LaneHostPool:
    """``slots`` lane images allocated once on the host (``copy_device_to_host_tensor`` / ``copy_host_to_device_
    tensor`` need the exact device shapes, so every slot mirrors the pager's pack buffers and KV staging)."""

    def __init__(self, mesh_device, layout: Qwen38LaneLayout, slots: int) -> None:
        if isinstance(slots, bool) or type(slots) is not int or slots < 1:
            raise ValueError(f"host pool slots must be a positive int, got {slots!r}")
        self.layout = layout
        self.families = layout.families()
        self.slots = tuple(
            Qwen38LaneHostSlot(
                index,
                {
                    family.name: tuple(family.allocate(mesh_device, device=False) for _ in range(family.count))
                    for family in self.families
                },
            )
            for index in range(slots)
        )

    def bytes_per_slot_per_device(self) -> int:
        return self.layout.image_bytes_per_device()


# ---------------------------------------------------------------- the pager


@dataclass(frozen=True)
class Qwen38LaneMoveTiming:
    """Wall milliseconds of one page move by leg: the pack / unpack trace replay, the small transfers of the pack
    buffers, the KV slabs (their slices or slab writes included), and the whole move."""

    trace_ms: float
    small_transfers_ms: float
    kv_ms: float
    total_ms: float


class Qwen38TTNNLanePager:
    """Evicts one lane's image into a host slot and re-admits a slot into any lane, bitwise (module docstring).

    Construction allocates the pack buffers and the KV staging pair, runs every lane's pack then unpack once
    eagerly (the compile warm-up; a pack followed by the unpack of the same lane is the identity on the lane) and
    captures the ``B`` pack and ``B`` unpack traces.  Build it after every lane tensor exists; run the tracker's
    ``verify_before_replay`` on :attr:`pack_traces` / :attr:`unpack_traces` after any later allocation.  ``evict``
    and ``readmit`` are synchronous and return their leg timing; the caller serializes them with the steps (the
    :class:`Qwen38LaneMover` pattern).  ``fill_families`` names the families written back by ``fill_cache``; the
    others take the selection write (``target * keep + repeat(part) * hit``, exact for finite values).  ``buffers``
    lends another pager's pack buffers and KV staging pair (the per-lane family shapes are the same at any lane count
    and context, so one set serves a lane state and the generic state beside it); the borrower never releases them.
    """

    def __init__(
        self,
        mesh_device,
        mesh_contract: Qwen38MeshContract,
        layout: Qwen38LaneLayout,
        *,
        fill_families: Sequence[str] = FILL_FAMILIES,
        cq_id: int = 0,
        traced: bool = False,
        buffers: "Qwen38TTNNLanePager | None" = None,
    ) -> None:
        mesh_contract.validate_mesh(mesh_device)
        layout.validate(mesh_contract)
        unknown = set(fill_families) - set(FILL_FAMILIES)
        if unknown:
            raise ValueError(f"fill_cache families must be among {FILL_FAMILIES}, got {sorted(unknown)}")
        if buffers is not None and [(f.name, f.local_shape, f.dtype, f.layout) for f in layout.families()] != [
            (f.name, f.local_shape, f.dtype, f.layout) for f in buffers.layout.families()
        ]:
            raise ValueError(
                f"pack buffers of {buffers.layout.lanes} lanes at {buffers.layout.allocated_context} do not fit a "
                f"layout of {layout.lanes} lanes at {layout.allocated_context}"
            )
        self.mesh_device = mesh_device
        self.mesh_contract = mesh_contract
        self.layout = layout
        self.fill_families = frozenset(fill_families)
        self.cq_id = cq_id
        self.traced = bool(traced)
        self.buffers = buffers
        self.compute_config = ttnn.init_device_compute_kernel_config(
            mesh_device.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=False,
        )
        self.families = {family.name: family for family in layout.families()}
        self.packs: dict[str, Any] = {}
        self.kv_stagings: tuple[Any, ...] = ()
        # (lane axis, dtype) -> per lane (hit, keep) one-hot columns of the selection writes.
        self.onehots: dict[tuple[int, Any], tuple[tuple[Any, ...], tuple[Any, ...]]] = {}
        self.views: dict[str, tuple[Any, ...]] = {}
        self.pack_traces: dict[int, Any] = {}
        self.unpack_traces: dict[int, Any] = {}
        self.warmup_ms: dict[str, list[float]] = {"pack": [], "unpack": []}
        self.capture_ms: dict[str, list[float]] = {"pack": [], "unpack": []}
        try:
            self._allocate()
            bodies = (("pack", self._pack_body, self.pack_traces), ("unpack", self._unpack_body, self.unpack_traces))
            for lane in range(layout.lanes):
                for name, body, _ in bodies:
                    started = time.perf_counter_ns()
                    body(lane)
                    ttnn.synchronize_device(mesh_device)
                    self.warmup_ms[name].append((time.perf_counter_ns() - started) / 1e6)
                if self.traced:
                    for name, body, traces in bodies:
                        with corruptible_allocation_scope(mesh_device):
                            trace_id = ttnn.begin_trace_capture(mesh_device, cq_id=cq_id)
                            started = time.perf_counter_ns()
                            body(lane)
                            ttnn.end_trace_capture(mesh_device, trace_id, cq_id=cq_id)
                            self.capture_ms[name].append((time.perf_counter_ns() - started) / 1e6)
                        traces[lane] = trace_id
                ttnn.synchronize_device(mesh_device)
        except BaseException:
            self.release()
            raise

    # -- allocation

    def _allocate(self) -> None:
        mesh, layout, lanes = self.mesh_device, self.layout, self.layout.lanes
        if self.buffers is not None:
            self.packs, self.kv_stagings = dict(self.buffers.packs), tuple(self.buffers.kv_stagings)
        for family in self.families.values():
            if self.buffers is not None:
                continue
            if family.name == "kv":
                self.kv_stagings = tuple(family.allocate(mesh, device=True) for _ in range(KV_STAGINGS))
                for staging in self.kv_stagings:
                    self.mesh_contract.validate_tensor(staging, placement=family.placement, shard_dim=family.shard_dim)
            else:
                self.packs[family.name] = family.allocate(mesh, device=True)
        # Same-buffer views with the lane axis at dim 0 for ``fill_cache`` (the compressed cache already is; the
        # recurrent's heads fold into the rows).
        view = ttnn.experimental.view
        self.views = {
            "compressed": tuple(layout.compressed),
            "recurrent": tuple(view(t, (lanes, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM)) for t in layout.recurrent),
            "staging": tuple(view(t, (lanes, 1, qsa_module.CACHE_WRITE_ROWS, KV_WIDTH)) for t in layout.staging),
            "ring": tuple(
                view(t, (lanes, 1, qsa_module.CACHE_WRITE_ROWS, qsa_module.INDEX_HEAD_DIM)) for t in layout.ring
            ),
            "ple": tuple(
                view(t, (lanes, 1, ple_module.RESIDUAL_BRANCHES, ple_module.LOCAL_HIDDEN_SIZE)) for t in layout.ple
            ),
        }
        needed = {(2, ttnn.bfloat16)} if layout.conv else set()
        for name in FILL_FAMILIES:
            if getattr(layout, name) and name not in self.fill_families:
                needed.add((LANE_AXIS[name], ttnn.float32 if name == "recurrent" else ttnn.bfloat16))
        for axis, dtype in sorted(needed, key=lambda item: (item[0], str(item[1]))):
            hits, keeps = [], []
            for lane in range(lanes):
                hit = torch.zeros(lanes, dtype=torch.float32)
                hit[lane] = 1.0
                shape = [1, 1, 1, 1]
                shape[axis] = lanes
                hits.append(self._replicated(hit.reshape(shape), dtype))
                keeps.append(self._replicated((1.0 - hit).reshape(shape), dtype))
            self.onehots[(axis, dtype)] = (tuple(hits), tuple(keeps))

    def _replicated(self, host: torch.Tensor, dtype):
        return ttnn.from_torch(
            host.to(torch.float32 if dtype == ttnn.float32 else torch.bfloat16),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=self.mesh_device,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=replicate_tensor_2d_mesh_mapper(self.mesh_device),
        )

    def release(self) -> None:
        for traces in (self.pack_traces, self.unpack_traces):
            for trace_id in traces.values():
                ttnn.release_trace(self.mesh_device, trace_id)
            traces.clear()
        owned = () if self.buffers is not None else (*self.packs.values(), *self.kv_stagings)
        _deallocate(*owned, *(tensor for hits, keeps in self.onehots.values() for tensor in (*hits, *keeps)))
        self.packs.clear()
        self.onehots.clear()
        self.kv_stagings = ()

    # -- the traced bodies

    def _slice_lane(self, tensor, lane_axis: int, lane: int, **kwargs):
        start = [0] * 4
        end = list(_shape(tensor))
        start[lane_axis], end[lane_axis] = lane, lane + 1
        if not kwargs:
            return slice_owned(tensor, start, end)
        # Into a given output tensor the whole-tensor case is a copy (``finalize_into_preallocated``).
        return ttnn.slice(tensor, start, end, memory_config=ttnn.DRAM_MEMORY_CONFIG, **kwargs)

    def _concat_tree(self, parts: Sequence[Any], dim: int, *, release_parts: bool, output=None):
        """``parts`` reduced to one tensor by CONCAT_FAN_IN-way concats along ``dim`` (``ttnn.concat`` takes no
        output tensor, so the result is copied into ``output`` when one is given).  Intermediates (and the parts,
        when ``release_parts``) are released.  Returns ``output`` if given, else the reduced tensor."""

        level, owned = list(parts), release_parts
        while len(level) > 1:
            merged = []
            for start in range(0, len(level), CONCAT_FAN_IN):
                group = level[start : start + CONCAT_FAN_IN]
                if len(group) == 1:
                    merged.append(group[0])
                    continue
                merged.append(ttnn.concat(group, dim, memory_config=ttnn.DRAM_MEMORY_CONFIG))
                if owned:
                    _deallocate(*group)
            level, owned = merged, True
        result = level[0]
        if output is None:
            return result
        _landed(ttnn.copy(result, output), output, "lane pack copy")
        if owned:
            _deallocate(result)
        return output

    def _pack_body(self, lane: int) -> None:
        """Lane ``lane``'s image into the pack buffers (device ops only, every intermediate released)."""

        layout = self.layout
        for name in FILL_FAMILIES:
            tensors = getattr(layout, name)
            if not tensors:
                continue
            if len(tensors) == 1:
                # One layer: the lane's slice lands straight in the pack (a lone slice fed to ``ttnn.copy`` as a
                # source is rejected as unallocated, so the copy path is only for the concat of several layers).
                _landed(
                    self._slice_lane(tensors[0], LANE_AXIS[name], lane, output_tensor=self.packs[name]),
                    self.packs[name],
                    f"lane pack {name}",
                )
            else:
                parts = [self._slice_lane(t, LANE_AXIS[name], lane) for t in tensors]
                self._concat_tree(parts, 0, release_parts=True, output=self.packs[name])
        if layout.conv:
            # Every ring slot side by side [1,144,B,2560], to rows, lane u's row of each slot into the pack.
            stacked = self._concat_tree(layout.conv, 1, release_parts=False)
            rows = ttnn.to_layout(stacked, ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            _deallocate(stacked)
            _landed(
                self._slice_lane(rows, 2, lane, output_tensor=self.packs["conv"]), self.packs["conv"], "lane pack conv"
            )
            _deallocate(rows)

    def _unpack_body(self, lane: int) -> None:
        """The pack buffers into lane ``lane`` of every tensor (device ops only)."""

        layout = self.layout
        for name in FILL_FAMILIES:
            for index, target in enumerate(getattr(layout, name)):
                part = self._slice_lane(self.packs[name], 0, index)
                if name in self.fill_families:
                    filled = part
                    if name == "recurrent":
                        filled = ttnn.experimental.view(part, (1, 1, RECURRENT_ROWS, gdn_module.HEAD_DIM))
                    cache_view = self.views[name][index]
                    _landed(
                        ttnn.fill_cache(cache_view, filled, batch_idx=lane),
                        cache_view,
                        f"lane unpack {name}[{index}] fill",
                    )
                else:
                    self._select_write(target, part, lane, LANE_AXIS[name], f"{name}[{index}]")
                _deallocate(part)
        if layout.conv:
            tiles = ttnn.to_layout(self.packs["conv"], ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
            hits, keeps = self.onehots[(2, ttnn.bfloat16)]
            for index, slot in enumerate(layout.conv):
                row = self._slice_lane(tiles, 1, index)
                # slot <- slot * keep + hit @ row: one nonzero term per element (the lane body's selection write).
                placed = ttnn.matmul(
                    hits[lane], row, memory_config=ttnn.DRAM_MEMORY_CONFIG, compute_kernel_config=self.compute_config
                )
                kept = ttnn.multiply(slot, keeps[lane], memory_config=ttnn.DRAM_MEMORY_CONFIG)
                _landed(ttnn.add(kept, placed, output_tensor=slot), slot, f"lane unpack conv[{index}] write")
                _deallocate(row, placed, kept)
            _deallocate(tiles)

    def _select_write(self, target, part, lane: int, lane_axis: int, label: str) -> None:
        """``target[lane] <- part`` as ``target * keep + repeat(part) * hit`` along the lane axis."""

        hits, keeps = self.onehots[(lane_axis, ttnn.float32 if target.dtype == ttnn.float32 else ttnn.bfloat16)]
        repeats = [1, 1, 1, 1]
        repeats[lane_axis] = self.layout.lanes
        expanded = ttnn.repeat(part, tuple(repeats), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        placed = ttnn.multiply(expanded, hits[lane], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        kept = ttnn.multiply(target, keeps[lane], memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _landed(ttnn.add(kept, placed, output_tensor=target), target, f"lane unpack {label} select write")
        _deallocate(expanded, placed, kept)

    # -- the moves

    def _slot(self, slot: Qwen38LaneHostSlot) -> Qwen38LaneHostSlot:
        if not isinstance(slot, Qwen38LaneHostSlot) or set(slot.tensors) != set(self.families):
            raise ValueError("host slot does not match the pager's families")
        return slot

    def _fence(self) -> None:
        ttnn.event_synchronize(ttnn.record_event(self.mesh_device, cq_id=self.cq_id))

    def _run_pack(self, lane: int) -> None:
        """The pack body: one replayed trace when captured, else the eager op sequence.  Eager is the default: the
        pack/unpack bodies allocate transient slices and concat intermediates, and ``verify_before_replay`` refuses
        a replay whose baked scratch overlaps another lane's trace, so the traced form is opt-in (single-lane use)."""

        if self.traced:
            ttnn._ttnn_execute_trace(self.mesh_device, self.pack_traces[lane], cq_id=self.cq_id, blocking=False)
        else:
            self._pack_body(lane)

    def _run_unpack(self, lane: int) -> None:
        if self.traced:
            ttnn._ttnn_execute_trace(self.mesh_device, self.unpack_traces[lane], cq_id=self.cq_id, blocking=False)
        else:
            self._unpack_body(lane)

    def evict(self, lane, slot: Qwen38LaneHostSlot) -> Qwen38LaneMoveTiming:
        """Lane ``lane``'s image into ``slot``: the pack (trace replay or eager body), the six small transfers, the
        KV slabs.  The lane's device state is left as it is (a re-admission or a masked reset overwrites it)."""

        lane, slot = _exact_index(lane, self.layout.lanes, label="evicted lane"), self._slot(slot)
        started = time.perf_counter_ns()
        # A move runs between two replays of the lane traces and consumes every device buffer it allocates before
        # the next one: whatever the ops allocate (the tracker saw the KV region slice and a pack copy outlive the
        # move on a 4x p150 lane, run lfm-disc-a-4883992f) may sit in a trace's scratch and be overwritten later.
        with corruptible_allocation_scope(self.mesh_device):
            self._run_pack(lane)
            self._fence()
            trace_done = time.perf_counter_ns()
            for name, pack in self.packs.items():
                ttnn.copy_device_to_host_tensor(pack, slot.tensors[name][0], blocking=False, cq_id=self.cq_id)
            self._fence()
            small_done = time.perf_counter_ns()
            context = self.layout.allocated_context
            for index, cache in enumerate(self.layout.kv):
                if self.layout.lanes == 1:
                    # One lane: the region is the whole cache, the slot tensor has its shape (no slice, no staging).
                    ttnn.copy_device_to_host_tensor(cache, slot.tensors["kv"][index], blocking=False, cq_id=self.cq_id)
                    continue
                staging = self.kv_stagings[index % KV_STAGINGS]
                _landed(
                    ttnn.slice(
                        cache,
                        (0, 0, lane * context, 0),
                        (1, 1, (lane + 1) * context, KV_WIDTH),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        output_tensor=staging,
                    ),
                    staging,
                    f"lane evict kv[{index}] slice",
                )
                ttnn.copy_device_to_host_tensor(staging, slot.tensors["kv"][index], blocking=False, cq_id=self.cq_id)
            self._fence()
        done = time.perf_counter_ns()
        return Qwen38LaneMoveTiming(
            (trace_done - started) / 1e6,
            (small_done - trace_done) / 1e6,
            (done - small_done) / 1e6,
            (done - started) / 1e6,
        )

    def warm_moves(self, slot: Qwen38LaneHostSlot) -> list[tuple[Qwen38LaneMoveTiming, Qwen38LaneMoveTiming]]:
        """One eviction into ``slot`` and one re-admission of every lane in turn (the identity on each lane): the KV
        slab legs compile here, not on the first move under the lane traces (construction warms the pack and unpack
        bodies only; the slice program is keyed by the lane's row offset: the first eviction on a 4x p150 lane
        compiled 3 programs and took 559 ms, lfm-disc-a-91284b89, and lane 1's slice one more after a lane-0 warm-up,
        lfm-disc-a-73ab5fd7)."""

        return [(self.evict(lane, slot), self.readmit(slot, lane)) for lane in range(self.layout.lanes)]

    def readmit(
        self, slot: Qwen38LaneHostSlot, lane, position_row: Qwen38TTNNDevicePositionRow | None = None
    ) -> Qwen38LaneMoveTiming:
        """``slot``'s image into lane ``lane``: the six small transfers, the unpack (trace replay or eager body),
        the KV slabs.  With a ``position_row`` the residue rule is applied first (``admit`` refuses a misaligned
        position before any device write) and the row takes the slot's position."""

        lane, slot = _exact_index(lane, self.layout.lanes, label="re-admitted lane"), self._slot(slot)
        if position_row is not None:
            position_row.admit(lane, slot.position)
        started = time.perf_counter_ns()
        with corruptible_allocation_scope(self.mesh_device):  # as in evict
            for name, pack in self.packs.items():
                ttnn.copy_host_to_device_tensor(slot.tensors[name][0], pack, cq_id=self.cq_id)
            self._fence()
            small_done = time.perf_counter_ns()
            self._run_unpack(lane)
            self._fence()
            trace_done = time.perf_counter_ns()
            context = self.layout.allocated_context
            for index, cache in enumerate(self.layout.kv):
                if self.layout.lanes == 1:
                    ttnn.copy_host_to_device_tensor(slot.tensors["kv"][index], cache, cq_id=self.cq_id)
                    continue
                staging = self.kv_stagings[index % KV_STAGINGS]
                ttnn.copy_host_to_device_tensor(slot.tensors["kv"][index], staging, cq_id=self.cq_id)
                _landed(
                    ttnn.experimental.deepseek_prefill.update_padded_kv_cache(
                        cache, staging, 0, 0, 1, lane * context, qsa_module.STAGING_AXIS
                    ),
                    cache,
                    f"lane readmit kv[{index}] slab write",
                )
            self._fence()
        done = time.perf_counter_ns()
        return Qwen38LaneMoveTiming(
            (trace_done - small_done) / 1e6,
            (small_done - started) / 1e6,
            (done - trace_done) / 1e6,
            (done - started) / 1e6,
        )


# ---------------------------------------------------------------- the helper thread


class Qwen38LaneMover:
    """One helper thread for page moves: ``submit`` right after the step's non-blocking launch (the move's device
    work queues behind the step on the shared command queue), ``wait`` before the next step's host writes (the
    move's result, or its exception re-raised)."""

    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._outcome: list[tuple[str, Any]] = []

    def submit(self, move: Callable[[], Any]) -> None:
        if self._thread is not None:
            raise RuntimeError("a page move is already in flight; wait for it first")
        self._outcome = []

        def run() -> None:
            try:
                self._outcome.append(("ok", move()))
            except BaseException as error:  # noqa: BLE001 - re-raised by wait()
                self._outcome.append(("error", error))

        self._thread = threading.Thread(target=run, name="qwen38-lane-mover", daemon=True)
        self._thread.start()

    @property
    def busy(self) -> bool:
        return self._thread is not None

    def wait(self) -> Any:
        if self._thread is None:
            raise RuntimeError("no page move in flight")
        self._thread.join()
        self._thread = None
        kind, value = self._outcome[0]
        if kind == "error":
            raise value
        return value


# ---------------------------------------------------------------- the host page table and the admission scheduler


@dataclass
class Qwen38LaneSession:
    """The host record of one session: its position (the residue class is ``position mod 4``), committed tokens,
    PLE n-gram context, and where it lives (a resident lane, a host slot, or neither)."""

    session_id: str
    position: int = 0
    committed: tuple[int, ...] = ()
    ple_context: tuple[int, int] | None = None
    lane: int | None = None
    slot: int | None = None
    last_used_ns: int = 0

    @property
    def residue(self) -> int:
        return self.position % GDN_RESIDUE_CLASSES

    @property
    def resident(self) -> bool:
        return self.lane is not None


class Qwen38LanePageTable:
    """session id -> lane or host slot.  ``lanes`` resident lanes and ``slots`` host slots; every lane and slot has
    at most one owner, every session at most one place.  Pure host bookkeeping: the device moves are the pager's."""

    def __init__(self, lanes: int, slots: int) -> None:
        self.lanes = require_lane_count(lanes, label="page table lanes")
        if isinstance(slots, bool) or type(slots) is not int or slots < 0:
            raise ValueError(f"page table slots must be a non-negative int, got {slots!r}")
        self.slots = slots
        self.sessions: dict[str, Qwen38LaneSession] = {}
        self.lane_owner: list[str | None] = [None] * lanes
        self.slot_owner: list[str | None] = [None] * slots

    def register(
        self, session_id: str, *, position: int = 0, committed: Sequence[int] = (), ple_context=None
    ) -> Qwen38LaneSession:
        if session_id in self.sessions:
            raise ValueError(f"session {session_id!r} is already registered")
        session = Qwen38LaneSession(session_id, int(position), tuple(int(t) for t in committed), ple_context)
        self.sessions[session_id] = session
        return session

    def session(self, session_id: str) -> Qwen38LaneSession:
        try:
            return self.sessions[session_id]
        except KeyError:
            raise KeyError(f"unknown session {session_id!r}") from None

    def free_lanes(self) -> list[int]:
        return [lane for lane, owner in enumerate(self.lane_owner) if owner is None]

    def free_slots(self) -> list[int]:
        return [slot for slot, owner in enumerate(self.slot_owner) if owner is None]

    def place(self, session_id: str, lane: int) -> Qwen38LaneSession:
        """The session takes ``lane`` (fresh, or re-admitted from its slot, which is freed)."""

        session = self.session(session_id)
        lane = _exact_index(lane, self.lanes, label="lane")
        if self.lane_owner[lane] is not None:
            raise ValueError(f"lane {lane} is owned by session {self.lane_owner[lane]!r}")
        if session.lane is not None:
            raise ValueError(f"session {session_id!r} is already resident in lane {session.lane}")
        if session.slot is not None:
            self.slot_owner[session.slot] = None
            session.slot = None
        self.lane_owner[lane] = session_id
        session.lane = lane
        return session

    def park(self, session_id: str, slot: int) -> Qwen38LaneSession:
        """The resident session leaves its lane for host ``slot``."""

        session = self.session(session_id)
        slot = _exact_index(slot, self.slots, label="slot")
        if session.lane is None:
            raise ValueError(f"session {session_id!r} is not resident")
        if self.slot_owner[slot] is not None:
            raise ValueError(f"slot {slot} is owned by session {self.slot_owner[slot]!r}")
        self.lane_owner[session.lane] = None
        session.lane = None
        self.slot_owner[slot] = session_id
        session.slot = slot
        return session

    def stash(self, session_id: str, slot: int) -> Qwen38LaneSession:
        """A session placed nowhere takes host ``slot`` (an image the prefill state produced for it)."""

        session = self.session(session_id)
        slot = _exact_index(slot, self.slots, label="slot")
        if session.lane is not None or session.slot is not None:
            raise ValueError(f"session {session_id!r} is already placed (lane {session.lane}, slot {session.slot})")
        if self.slot_owner[slot] is not None:
            raise ValueError(f"slot {slot} is owned by session {self.slot_owner[slot]!r}")
        self.slot_owner[slot] = session_id
        session.slot = slot
        return session

    def drop(self, session_id: str) -> None:
        session = self.sessions.pop(session_id)
        if session.lane is not None:
            self.lane_owner[session.lane] = None
        if session.slot is not None:
            self.slot_owner[session.slot] = None

    def touch(self, session_id: str, *, position: int, committed: Sequence[int], ple_context, now_ns: int) -> None:
        session = self.session(session_id)
        session.position = int(position)
        session.committed = tuple(int(t) for t in committed)
        session.ple_context = ple_context
        session.last_used_ns = int(now_ns)

    def idle_residents(self, *, now_ns: int, idle_ns: int) -> list[Qwen38LaneSession]:
        """Resident sessions idle for at least ``idle_ns``, least recently used first (the eviction candidates)."""

        return sorted(
            (s for s in self.sessions.values() if s.lane is not None and now_ns - s.last_used_ns >= idle_ns),
            key=lambda s: s.last_used_ns,
        )


@dataclass(frozen=True)
class Qwen38LaneAdmission:
    session_id: str
    lane: int
    position: int
    passed_boundaries: int


class Qwen38LaneAdmissionScheduler:
    """Residue-aligned admission at step boundaries.  Waiting sessions (fresh at 0, forked at ``L0``, re-admitted
    at their committed length) join at the first boundary whose resident residue equals ``position mod 4``
    (``admission_wait_steps``: 0-3 steps), FIFO within a residue class; with no resident lane the first waiter's
    residue becomes the batch's.  ``passed_boundaries`` counts the boundaries a session was passed over at."""

    def __init__(self) -> None:
        self.waiting: list[tuple[str, int, int]] = []  # (session_id, position, boundaries passed)

    def enqueue(self, session_id: str, position: int) -> None:
        if isinstance(position, bool) or type(position) is not int or position < 0:
            raise ValueError(f"admission position must be a non-negative int, got {position!r}")
        if any(item[0] == session_id for item in self.waiting):
            raise ValueError(f"session {session_id!r} is already waiting")
        self.waiting.append((session_id, position, 0))

    def take(self, session_id: str) -> tuple[str, int, int] | None:
        """Remove and return one waiter (a session re-entering the lane it still holds), None when not waiting."""

        for index, item in enumerate(self.waiting):
            if item[0] == session_id:
                return self.waiting.pop(index)
        return None

    def waits(self, resident_residue: int | None) -> dict[str, int]:
        """Steps each waiter still waits from this boundary (0 when it may join now)."""

        return {
            session_id: 0 if resident_residue is None else admission_wait_steps(resident_residue, position)
            for session_id, position, _ in self.waiting
        }

    def admissions(self, resident_residue: int | None, free_lanes: Sequence[int]) -> list[Qwen38LaneAdmission]:
        """The admissions of this boundary: aligned waiters in FIFO order onto ``free_lanes`` in order; the other
        waiters are passed over.  ``resident_residue`` None means an idle batch."""

        if resident_residue is not None and not 0 <= resident_residue < GDN_RESIDUE_CLASSES:
            raise ValueError(f"resident residue must be in [0,{GDN_RESIDUE_CLASSES}), got {resident_residue!r}")
        lanes, residue = list(free_lanes), resident_residue
        admitted: list[Qwen38LaneAdmission] = []
        remaining: list[tuple[str, int, int]] = []
        for session_id, position, passed in self.waiting:
            if residue is None and lanes:
                residue = position % GDN_RESIDUE_CLASSES
            if lanes and admission_wait_steps(residue, position) == 0:
                admitted.append(Qwen38LaneAdmission(session_id, lanes.pop(0), position, passed))
            else:
                remaining.append((session_id, position, passed + 1))
        self.waiting = remaining
        return admitted


__all__ = [
    "FILL_FAMILIES",
    "GDN_LAYERS",
    "LANE_AXIS",
    "QSA_LAYERS",
    "RING_SLOTS",
    "Qwen38LaneAdmission",
    "Qwen38LaneAdmissionScheduler",
    "Qwen38LaneFamily",
    "Qwen38LaneHostPool",
    "Qwen38LaneHostSlot",
    "Qwen38LaneLayout",
    "Qwen38LaneMoveTiming",
    "Qwen38LaneMover",
    "Qwen38LanePageTable",
    "Qwen38LaneSession",
    "Qwen38TTNNLanePager",
]
