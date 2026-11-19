# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Generator primitives for explicit reloads, slot ownership and readback.

Source implementation only: not a serving adapter and not hardware-validated.
Pinned contract: vllm-tt-plugin 042a94863ad27d88207bcf71abdf67122c6f4c27,
DECODE_RELOAD_CONTRACT.md and MODEL_LIFECYCLE.md. Construction/prepare happen
before any live trace. A runtime failure after mutation poisons the helper;
continuing with a guessed request layout is forbidden.
"""

from copy import copy, deepcopy
from dataclasses import dataclass
from math import prod

import torch

import ttnn

from .slot_state import _program, copy_if


@dataclass(frozen=True)
class ReloadCommands:
    reload_inputs: bool = True
    reload_page_table: bool = False
    reload_sampling_params: bool = False
    reset_sampling_state: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in vars(self).values()):
            raise TypeError("Decode reload commands must be booleans")
        if self.reload_inputs and self.reload_page_table:
            raise ValueError("reload_inputs includes page tables; page-only reload is mutually exclusive")
        if self.reset_sampling_state and not self.reload_inputs:
            raise ValueError("Sampling-state reset requires authoritative forward inputs")

    @classmethod
    def from_kwargs(cls, kwargs):
        if "reset_batch" in kwargs:
            raise TypeError("Contract v1 rejects legacy reset_batch")
        if "decode_layout_changed" in kwargs:
            raise TypeError("decode_layout_changed is runner-private, not a v1 command")
        return cls(**{key: kwargs[key] for key in cls.__dataclass_fields__ if key in kwargs})


class _ValidatedRemap(tuple):
    """Immutable model-internal permutation, validated once at its boundary."""

    def __new__(cls, values):
        values = tuple(values)
        if not 1 <= len(values) <= 32 or any(type(value) is not int for value in values):
            raise ValueError("A validated remap requires 1..32 integer slots")
        if sorted(values) != list(range(len(values))):
            raise ValueError("A validated remap must be a complete permutation")
        return tuple.__new__(cls, values)


def normalize_remap(remap, capacity):
    if not 1 <= capacity <= 32:
        raise ValueError("The configured mesh supports 1..32 fixed state slots")
    if type(remap) is _ValidatedRemap:
        if len(remap) != capacity:
            raise ValueError("Validated remap capacity differs from the owner")
        return remap
    if remap is None:
        return _ValidatedRemap(range(capacity))
    raw = torch.as_tensor(remap).reshape(-1).tolist()
    if any(isinstance(value, bool) or int(value) != value for value in raw):
        raise ValueError("slot_remap entries must be integer slot indices")
    values = tuple(int(value) for value in raw)
    if len(values) != capacity or sorted(values) != list(range(capacity)):
        raise ValueError("slot_remap must be a full-capacity permutation, including off-batch slots")
    return _ValidatedRemap(values)


_PERMUTE32 = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t row,uint32_t col) {
 return (row/16)*512+(col/16)*256+(row%16)*16+col%16;
}
void kernel_main() {
 constexpr uint32_t io=get_compile_time_arg_val(0), bytes=get_compile_time_arg_val(1);
 constexpr bool tiled=get_compile_time_arg_val(2)!=0;
 const auto data=TensorAccessor(TensorAccessorArgs<3>(),get_arg_val<uint32_t>(0));
 const auto index=TensorAccessor(TensorAccessorArgs<io>(),get_arg_val<uint32_t>(1));
 const uint32_t first=get_arg_val<uint32_t>(2), stride=get_arg_val<uint32_t>(3), pages=get_arg_val<uint32_t>(4);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* base=reinterpret_cast<volatile tt_l1_ptr uint8_t*>(get_write_ptr(0)+bias);
 auto* indices=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+8192);
 noc.async_read(index,scratch,128,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+8192});
 noc.async_read_barrier();
 const uint32_t page_bytes=tiled?1024*bytes:32*bytes;
 for(uint32_t page=first;page<pages;page+=stride) {
  noc.async_read(data,scratch,page_bytes,{.page_id=page,.offset_bytes=0},{.offset_bytes=bias});
  noc.async_read_barrier();
  for(uint32_t row=0;row<32;++row) {
   const uint32_t source=indices[row];
   for(uint32_t col=0;col<(tiled?32u:1u);++col) {
    const uint32_t from=(tiled?face(source,col):source)*bytes;
    const uint32_t to=(tiled?face(row,col):row)*bytes;
    for(uint32_t b=0;b<bytes;++b) base[4096+to+b]=base[from+b];
   }
  }
  noc.async_write(scratch,data,page_bytes,{.offset_bytes=bias+4096},{.page_id=page,.offset_bytes=0});
  noc.async_write_barrier();
 }
}
"""


class SlotState:
    """Simultaneous permutation of every registered request-owned field.

    Distinct per-slot allocations retain their addresses: snapshots are copied
    before any destination is overwritten. Packed32 row buffers permute whole
    tiles (or a scalar vector) in place after first reading all source rows.
    KV physical pools are shared and never copied; their page mappings move.
    """

    def __init__(self, mesh, capacity):
        self.mesh, self.capacity = mesh, capacity
        normalize_remap(None, capacity)
        self.indices = self._upload(torch.arange(32, dtype=torch.int32), ttnn.uint32)
        self.one = self._upload(torch.ones(1, dtype=torch.int32), ttnn.int32)
        self.groups, self.packed, self.host_fields, self.host_remappers = [], [], [], []
        self.prepared = False
        self.poisoned = False
        self.remaps = 0

    def _upload(self, value, dtype):
        return ttnn.from_torch(
            value,
            dtype=dtype,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=self.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _registration(self):
        if self.prepared:
            raise RuntimeError("Register slot storage before global preparation/capture")

    def register_group(self, name, tensors):
        self._registration()
        tensors = tuple(tensors)
        if len(tensors) != self.capacity:
            raise ValueError(f"{name}: expected one device allocation per configured slot")
        first = tensors[0]
        signature = lambda t: (tuple(t.shape), tuple(t.padded_shape), t.dtype, t.layout, t.memory_config())
        if any(signature(t) != signature(first) for t in tensors):
            raise ValueError(f"{name}: slot tensor layouts differ")
        if len({t.buffer_address() for t in tensors}) != self.capacity:
            raise ValueError(f"{name}: mutable slot allocations may not alias")
        snapshots = tuple(ttnn.empty_like(t) for t in tensors)
        self.groups.append((name, tensors, snapshots))

    def register_packed(self, name, tensor):
        self._registration()
        if tensor.dtype not in (ttnn.bfloat16, ttnn.float32, ttnn.int32, ttnn.uint32):
            raise ValueError(f"{name}: packed remap requires an uncompressed numeric dtype")
        if tensor.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
            raise ValueError(f"{name}: packed remap requires interleaved DRAM")
        tiled = tensor.layout == ttnn.TILE_LAYOUT
        if tiled:
            if len(tensor.shape) < 2 or tensor.shape[-2] != 32 or prod(tuple(tensor.shape)[:-2]) != 1:
                raise ValueError(f"{name}: tiled slots must be exactly the 32 penultimate rows")
        elif tensor.layout != ttnn.ROW_MAJOR_LAYOUT or prod(tensor.shape) != 32 or tensor.shape[-1] != 32:
            raise ValueError(f"{name}: row-major packed slots must be one 32-element vector")
        self.packed.append((name, tensor))

    def register_host(self, name, getter, setter, *, copier=deepcopy):
        """Register a slot-indexed sequence/tensor; getter follows replacements."""
        self._registration()
        self.host_fields.append((name, getter, setter, copier))

    def _packed_remap(self, tensor):
        tiled = tensor.layout == ttnn.TILE_LAYOUT
        element_bytes = 2 if tensor.dtype == ttnn.bfloat16 else 4
        pages = prod(tensor.padded_shape) // 1024 if tiled else 1
        grid = self.mesh.compute_with_storage_grid_size()
        workers = min(8, pages, grid.x * grid.y)
        runtime = [
            (i % grid.x, i // grid.x, [tensor.buffer_address(), self.indices.buffer_address(), i, workers, pages])
            for i in range(workers)
        ]
        a, b = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (tensor, self.indices)]
        args = [3 + len(a), element_bytes, int(tiled)] + a + b
        program = _program(self.mesh, _PERMUTE32, args, runtime, 8384)
        ttnn.generic_op([tensor, self.indices], program)

    def prepare(self):
        if self.prepared:
            return
        # Identity warmup is bit-preserving and needs no request-state reset.
        for _, tensors, snapshots in self.groups:
            for source, snapshot in zip(tensors, snapshots):
                copy_if(source, snapshot, self.one, self.mesh)
            for snapshot, destination in zip(snapshots, tensors):
                copy_if(snapshot, destination, self.one, self.mesh)
        for _, tensor in self.packed:
            self._packed_remap(tensor)
        ttnn.synchronize_device(self.mesh)
        self.prepared = True

    def apply(self, remap):
        if not self.prepared or self.poisoned:
            raise RuntimeError("Slot state is unprepared or was poisoned by a failed mutation")
        order = normalize_remap(remap, self.capacity)
        if order == tuple(range(self.capacity)):
            return
        padded = order + tuple(range(self.capacity, 32))
        # Complete all CPU validation/snapshot construction before any device write.
        updates = []
        for name, getter, setter, copier in self.host_fields:
            current = getter()
            if current is None:
                updates.append((setter, None))
                continue
            if len(current) < self.capacity:
                raise ValueError(f"{name}: host state is shorter than state-slot capacity")
            indices = padded if len(current) == 32 else order
            if len(current) not in (self.capacity, 32):
                raise ValueError(f"{name}: host state length is neither capacity nor 32")
            if isinstance(current, torch.Tensor):
                value = current[list(indices)].clone()
            else:
                value = [copier(current[source]) for source in indices]
            updates.append((setter, value))
        host = ttnn.from_torch(
            torch.tensor(padded, dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        try:
            ttnn.copy_host_to_device_tensor(host, self.indices)
            for _, tensors, snapshots in self.groups:
                for source, snapshot in zip(tensors, snapshots):
                    copy_if(source, snapshot, self.one, self.mesh)
                for destination, source_slot in zip(tensors, order):
                    copy_if(snapshots[source_slot], destination, self.one, self.mesh)
            for _, tensor in self.packed:
                self._packed_remap(tensor)
            for setter, value in updates:
                setter(value)
            for callback in self.host_remappers:
                callback(padded)
            self.remaps += 1
        except BaseException:
            self.poisoned = True
            raise

    @classmethod
    def from_generator(cls, generator):
        """Register this Qwen generator's current persistent state explicitly."""
        owner = cls(generator.mesh, generator.max_batch_size)
        for layer_index, states in enumerate(generator.cache):
            for key in ("recurrent", "conv", "page_table"):
                if key in states[0]:
                    owner.register_group(f"layer{layer_index}.{key}", [s[key] for s in states[: owner.capacity]])
            if "page_table" in states[0]:

                def get_pages(states=states):
                    return [s.get("_host_pages") for s in states[: owner.capacity]]

                def set_pages(values, states=states):
                    for state, value in zip(states, values):
                        state["_host_pages"] = value

                owner.register_host(f"layer{layer_index}.host_pages", get_pages, set_pages)
        owner.register_group("positions", generator.positions)
        owner.register_group("activity", generator.activity)
        for name in (
            "tokens",
            "previous_tokens",
            "incoming_tokens",
            "active_mask",
            "sampling_update_mask",
            "sampling_seed_updates",
        ):
            owner.register_packed(name, getattr(generator, name))
        sampling = generator.sampler.tt_sampling
        for name in ("seeds_tt_tensor", "k_tensor", "p_tensor", "temp_tensor", "_greedy_col", "_scale_logprobs"):
            owner.register_packed("sampling." + name, getattr(sampling, name))
        penalties = generator.sampler.tt_penalties
        for name in (
            "prompt_mask",
            "output_mask",
            "output_counts",
            "output_counts_gathered",
            "presence_penalties",
            "frequency_penalties",
            "repetition_penalties",
            "inverse_repetition_penalties",
        ):
            owner.register_packed("penalties." + name, getattr(penalties, name))
        if sampling.grammar_bitmask_tensor is not None:
            raise ValueError("Register a prepared grammar-row remapper before enabling grammar state")
        owner.register_host(
            "seed_values", lambda: generator.seed_values, lambda value: setattr(generator, "seed_values", value)
        )
        for key in generator.sampling_values:
            owner.register_host(
                "sampling_values." + key,
                lambda key=key: generator.sampling_values[key],
                lambda value, key=key: generator.sampling_values.__setitem__(key, value),
            )

        def clone_rng(rng):
            cloned = torch.Generator(device=rng.device)
            cloned.set_state(rng.get_state())
            return cloned

        owner.register_host(
            "host_rng",
            lambda: generator.host_rng,
            lambda value: setattr(generator, "host_rng", value),
            copier=clone_rng,
        )
        owner.register_host(
            "initial_host_rng_states",
            lambda: generator.initial_host_rng_states,
            lambda value: setattr(generator, "initial_host_rng_states", value),
        )
        owner.register_host(
            "prompt_history",
            lambda: penalties._prompt_tokens_host,
            lambda value: setattr(penalties, "_prompt_tokens_host", value),
        )
        owner.register_host(
            "serving_output_history",
            lambda: getattr(generator, "serving_output_history", None),
            lambda value: setattr(generator, "serving_output_history", value),
        )
        manager = generator.sampler.seed_manager
        # Move even dormant seed-manager objects; the common helper's early
        # return for unseeded state would not preserve every registered stream.
        for name in ("seeds", "seed_counters", "seed_salts", "rngs"):
            owner.register_host(
                "seed_manager." + name,
                lambda name=name: getattr(manager, name),
                lambda value, name=name: setattr(manager, name, value),
                copier=copy,
            )
        logprobs = sampling.log_probs_calculator
        for name in ("logprobs_enabled", "num_logprobs"):
            owner.register_host(
                "logprobs." + name,
                lambda name=name: getattr(logprobs, name),
                lambda value, name=name: setattr(logprobs, name, value),
            )

        def active_layout(order):
            previous = set(generator.active_slots)
            generator.active_slots = tuple(i for i in range(owner.capacity) if order[i] in previous)

        owner.host_remappers.append(active_layout)
        return owner


class ReloadExecutor:
    """Execute commands exactly once, before a single accepted forward.

    Callbacks must be validated before execute() and must update stable buffers.
    reload_parameters must preserve unchanged live RNG streams; reset_sampling
    unconditionally rebuilds commanded RNG/history, including seed=None. The
    callback separation prevents a parameter-only update from becoming a reset.
    """

    def __init__(self, slot_state, *, require_drained=lambda: None):
        self.slots = slot_state
        self.require_drained = require_drained
        self.poisoned = False

    def execute(
        self,
        commands,
        *,
        slot_remap=None,
        reload_forward=None,
        reload_pages=None,
        reload_parameters=None,
        reset_sampling=None,
        forward=None,
    ):
        if self.poisoned:
            raise RuntimeError("Reload executor poisoned after a failed accepted-state mutation")
        if not isinstance(commands, ReloadCommands):
            raise TypeError("Validate the explicit reload command set before submission")
        order = normalize_remap(slot_remap, self.slots.capacity)
        callbacks = (
            (commands.reload_inputs, reload_forward, "reload_inputs"),
            (commands.reload_page_table, reload_pages, "reload_page_table"),
            (commands.reload_sampling_params, reload_parameters, "reload_sampling_params"),
            (commands.reset_sampling_state, reset_sampling, "reset_sampling_state"),
        )
        for enabled, callback, name in callbacks:
            if enabled and not callable(callback):
                raise ValueError(f"{name} requires an explicit implementation")
        if not callable(forward):
            raise ValueError("A validated decode submission callback is required")
        changing_layout = order != tuple(range(self.slots.capacity))
        if (
            changing_layout
            or commands.reload_inputs
            or commands.reload_sampling_params
            or commands.reset_sampling_state
        ):
            self.require_drained()
        # tokens/start_pos are deliberately absent here. A resident request
        # cannot accidentally derive state from their stale values in this helper.
        try:
            self.slots.apply(order)
            for enabled, callback, _ in callbacks:
                if enabled:
                    callback()
            return forward()
        except BaseException:
            self.poisoned = True
            self.slots.poisoned = True
            raise


@dataclass
class DecodeOutput:
    owner: object
    lease: int
    generation: int
    rows: tuple
    is_tokens: bool
    phase: str = "reserved"
    host: object = None
    events: tuple = ()


class DecodeOutputPool:
    """Fixed device snapshots with explicit submission/read/consume ownership.

    A same-CQ copy snapshots the completed model/sampler output before a later
    replay can overwrite it. Reserve before submitting work. No slot is reused
    until host processing completes. Readback is observational and never samples
    or updates model positions. This provides full readback/compact host output,
    not selective device transfer or speculative output support.
    """

    def __init__(self, mesh, *, token_template, logits_template, vocab_size, capacity, depth=2):
        if depth < 1 or not 1 <= capacity <= 32:
            raise ValueError("Invalid output pool geometry")
        self.mesh, self.vocab_size, self.capacity = mesh, vocab_size, capacity
        self.buffers = {
            True: [ttnn.empty_like(token_template) for _ in range(depth)],
            False: [ttnn.empty_like(logits_template) for _ in range(depth)],
        }
        self.leases = [None] * depth
        self.generation = 0
        self.prepared = False

    def prepare(self, *, token_template, logits_template):
        if self.prepared:
            return
        for mode, source in ((True, token_template), (False, logits_template)):
            for destination in self.buffers[mode]:
                ttnn.copy(source, destination)
                host = destination.cpu(blocking=False)
                event = ttnn.record_event(self.mesh, 0)
                ttnn.event_synchronize(event)
                del host, event
        self.prepared = True

    def require_drained(self):
        if any(handle is not None for handle in self.leases):
            raise RuntimeError("Consume pending decode outputs before changing request or sampling state")

    def reserve(self, rows, *, is_tokens):
        if not self.prepared:
            raise RuntimeError("Prepare output snapshots before capturing any trace")
        rows = tuple(int(row) for row in rows)
        if not rows or len(set(rows)) != len(rows) or any(row < 0 or row >= self.capacity for row in rows):
            raise ValueError("Output rows must be unique valid physical slots")
        try:
            lease = self.leases.index(None)
        except ValueError:
            raise RuntimeError("Decode output ring is full; consume a pending output before submitting") from None
        self.generation += 1
        handle = DecodeOutput(self, lease, self.generation, rows, bool(is_tokens))
        self.leases[lease] = handle
        return handle

    def _check(self, handle, phases):
        if not isinstance(handle, DecodeOutput) or handle.owner is not self:
            raise ValueError("Output handle belongs to another generator")
        if self.leases[handle.lease] is not handle or handle.phase not in phases:
            raise RuntimeError("Output handle was consumed, cancelled or used in the wrong phase")

    def cancel_unsubmitted(self, handle):
        self._check(handle, ("reserved",))
        self.leases[handle.lease] = None
        handle.phase = "cancelled"

    def publish(self, handle, output):
        self._check(handle, ("reserved",))
        target = self.buffers[handle.is_tokens][handle.lease]
        if (tuple(output.shape), output.dtype, output.layout) != (tuple(target.shape), target.dtype, target.layout):
            raise ValueError("Decode output differs from its prepared snapshot geometry")
        # A failed copy may already have enqueued work. Such a lease must be
        # drained by discard(), never cancelled as an unsubmitted reservation.
        handle.phase = "publishing"
        ttnn.copy(output, target)
        handle.phase = "published"
        return handle

    def read(self, handle, *, async_read=False, sample_rows=None):
        self._check(handle, ("published",))
        if sample_rows is not None and (handle.is_tokens or tuple(sample_rows) != handle.rows):
            raise ValueError("Read rows must match the immutable submitted host-logit row order")
        handle.host = self.buffers[handle.is_tokens][handle.lease].cpu(blocking=not async_read)
        if async_read:
            handle.events = (ttnn.record_event(self.mesh, 0),)
        handle.phase = "reading"
        return (handle, list(handle.events)) if async_read else handle

    def process(self, handle, *, is_tokens, sample_rows=None, events_synchronized=False):
        # Direct generator users own no external synchronization by default.
        # The vLLM adapter opts in only at its documented post-event boundary.
        if type(events_synchronized) is not bool:
            raise TypeError("events_synchronized must be a boolean")
        self._check(handle, ("published", "reading"))
        if bool(is_tokens) != handle.is_tokens:
            raise ValueError("Output mode differs from the accepted submission")
        if sample_rows is not None and (is_tokens or tuple(sample_rows) != handle.rows):
            raise ValueError("Processing rows differ from submitted output rows")
        if handle.phase == "published":
            self.read(handle, async_read=False, sample_rows=sample_rows)
        if not events_synchronized:
            for event in handle.events:
                ttnn.event_synchronize(event)
        if is_tokens:
            host = ttnn.to_torch(ttnn.get_device_tensors(handle.host)[0]).reshape(-1)
            result = host[list(handle.rows)].to(torch.int64).reshape(-1, 1).clone()
        else:
            host = ttnn.to_torch(handle.host, mesh_composer=ttnn.ConcatMeshToTensor(self.mesh, dim=-1))
            result = host.reshape(32, -1)[list(handle.rows), : self.vocab_size].float().clone()
            result = result[:, None, :]
        # Returned CPU storage is independent and writable. A pool slot is freed
        # only after the transfer/event and conversion have completed.
        handle.host, handle.events = None, ()
        handle.phase = "consumed"
        self.leases[handle.lease] = None
        return result

    def discard(self, handle):
        """Explicit teardown/read-failure drain; never an ordinary output path."""
        self._check(handle, ("reserved", "publishing", "published", "reading"))
        if handle.phase != "reserved":
            # An exceptional .cpu() failure may have enqueued a transfer without
            # returning its event. Drain the queue before releasing its lease.
            ttnn.synchronize_device(self.mesh)
        handle.host, handle.events = None, ()
        handle.phase = "discarded"
        self.leases[handle.lease] = None


_LOGIT_ROW = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
constexpr uint32_t face(uint32_t r,uint32_t c) {
 return (r/16)*512+(c/16)*256+(r%16)*16+c%16;
}
void kernel_main() {
 constexpr uint32_t d=get_compile_time_arg_val(0), m=get_compile_time_arg_val(1);
 const auto source=TensorAccessor(TensorAccessorArgs<2>(),get_arg_val<uint32_t>(0));
 const auto destination=TensorAccessor(TensorAccessorArgs<d>(),get_arg_val<uint32_t>(1));
 const auto metadata=TensorAccessor(TensorAccessorArgs<m>(),get_arg_val<uint32_t>(2));
 const uint32_t first=get_arg_val<uint32_t>(3), stride=get_arg_val<uint32_t>(4), pages=get_arg_val<uint32_t>(5);
 Noc noc; DataflowBuffer scratch(0); scratch.reserve_back(1);
 const uint32_t bias=(64-(get_write_ptr(0)&63))&63;
 auto* words=reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(0)+bias);
 auto* row_ptr=reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(0)+bias+4096);
 noc.async_read(metadata,scratch,4,{.page_id=0,.offset_bytes=0},{.offset_bytes=bias+4096});
 noc.async_read_barrier();
 const uint32_t row=row_ptr[0];
 for(uint32_t page=first;page<pages;page+=stride) {
  noc.async_read(source,scratch,2048,{.page_id=page,.offset_bytes=0},{.offset_bytes=bias});
  noc.async_read(destination,scratch,2048,{.page_id=page,.offset_bytes=0},{.offset_bytes=bias+2048});
  noc.async_read_barrier();
  for(uint32_t col=0;col<32;++col) words[1024+face(row,col)]=words[face(0,col)];
  noc.async_write(scratch,destination,2048,{.offset_bytes=bias+2048},{.page_id=page,.offset_bytes=0});
  noc.async_write_barrier();
 }
}
"""


class PrefillLogitRows:
    """Exact singleton BF16 logit-row staging into the stable 32-row sampler.

    Construct and prepare before trace capture. A runtime scalar selects the
    destination row without a program variant or persistent allocation. Source
    and destination remain vocabulary-sharded with the selected decoder policy.
    """

    def __init__(self, mesh):
        self.mesh = mesh
        self.row = ttnn.from_torch(
            torch.zeros(1, dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        self.prepared = False

    def _geometry(self, source, destination):
        for value in (source, destination):
            if value.dtype != ttnn.bfloat16 or value.layout != ttnn.TILE_LAYOUT:
                raise ValueError("Prefill sampling staging requires tiled BF16 logits")
            if value.memory_config() != ttnn.DRAM_MEMORY_CONFIG:
                raise ValueError("Prefill sampling staging requires interleaved DRAM logits")
        if len(source.shape) < 2 or source.shape[-2] != 1 or prod(tuple(source.shape)[:-2]) != 1:
            raise ValueError("Prefill staging consumes exactly one logical final logit row")
        if (
            len(destination.shape) != 4
            or tuple(destination.shape)[:3] != (1, 1, 32)
            or source.shape[-1] != destination.shape[-1]
        ):
            raise ValueError("Destination must be the matching fixed 32-row sampler allocation")
        if source.buffer_address() == destination.buffer_address():
            raise ValueError("Prefill staging source cannot alias its destination")
        return ttnn.reshape(source, (1, 1, 1, source.shape[-1]))

    def _copy(self, source, destination):
        source = self._geometry(source, destination)
        pages = prod(source.padded_shape) // 1024
        grid = self.mesh.compute_with_storage_grid_size()
        workers = min(8, pages, grid.x * grid.y)
        runtime = [
            (
                i % grid.x,
                i // grid.x,
                [source.buffer_address(), destination.buffer_address(), self.row.buffer_address(), i, workers, pages],
            )
            for i in range(workers)
        ]
        a, b, c = [list(ttnn.TensorAccessorArgs(t).get_compile_time_args()) for t in (source, destination, self.row)]
        program = _program(self.mesh, _LOGIT_ROW, [2 + len(a), 2 + len(a) + len(b)] + a + b + c, runtime, 4192)
        ttnn.generic_op([source, destination, self.row], program)

    def prepare(self, *, source, destination):
        if not self.prepared:
            self._copy(source, destination)
            ttnn.synchronize_device(self.mesh)
            self.prepared = True

    def copy(self, source, destination, slot):
        if not self.prepared:
            raise RuntimeError("Warm prefill logit-row staging before capturing any trace")
        if isinstance(slot, bool) or not isinstance(slot, int) or not 0 <= slot < 32:
            raise ValueError("Destination slot must be an integer in [0,32)")
        self._geometry(source, destination)
        host = ttnn.from_torch(
            torch.tensor([slot], dtype=torch.int32),
            dtype=ttnn.int32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            mesh_mapper=ttnn.ReplicateTensorToMesh(self.mesh),
        )
        ttnn.copy_host_to_device_tensor(host, self.row)
        self._copy(source, destination)
