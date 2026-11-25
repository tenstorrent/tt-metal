# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Stable slot-state plumbing around the COMMON sampler; no sampler math here.

TP4 / sampling_dp=1 / physical capacity32 only. Construct and call prepare()
before the generator's first capture. Persistent scratch, reset updates, gather
indices and warmup snapshots are allocated there, never during request handling.
No trace-allocation exemptions are used. Remaps touch every slot, including a
dormant sampler and off-batch requests. Caller drains pending work first and
applies each accepted remap exactly once; do not ALSO call sampler.apply_slot_remap.

For masked sampling capture before_sample(); common sampling + existing traced
seed advance; after_sample(). refresh_active_slots() runs only at scheduler
transitions. Common penalty decode_src=0 for inactive rows makes scatter_add add
zero; prompt masks are read-only and output counts/masks keep inactive history.
Only32 seed words need a before/after snapshot, avoiding vocab-sized per-token
snapshots. Device PRNG registers are not moved; the Ornith traced stream supplies
an explicit per-slot seed each step, so the next sample reinitializes from the
preserved seed. Do not use this helper with the common SKIP/unseeded PRNG mode.

All buffers referenced by sampling stay at the same address. The scratch is
serially owned by this helper; no overlapping submissions/state transitions.
Snapshot/restore is for warmup/setup, not request-time trace retirement.
"""

import copy
import math
import os
from dataclasses import fields, replace
from numbers import Integral

# ROUND 6: skip all-zero / unused penalty-history uploads (exact); ORNITH_HISTORY_SKIP=0 restores the full upload.
HISTORY_SKIP = os.environ.get("ORNITH_HISTORY_SKIP", "1") == "1"

CAPACITY = 32


def validate_slots(slots, *, permutation=False):
    if callable(getattr(slots, "tolist", None)):
        device = getattr(slots, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cpu":
            raise ValueError("Slot metadata must be host-resident")
        slots = slots.tolist()
    result = tuple(slots)
    if any(isinstance(s, bool) or not isinstance(s, Integral) or not 0 <= s < CAPACITY for s in result):
        raise ValueError("Slot IDs must be integers in [0,32)")
    if len(set(result)) != len(result) or (permutation and set(result) != set(range(CAPACITY))):
        raise ValueError("Expected unique slots, or a full-capacity bijection for remap")
    return tuple(int(s) for s in result)


def seed_selection_indices(slots, *, choose_second):
    """Indices into concatenated first/second32-seed vectors."""
    chosen = set(validate_slots(slots))
    return tuple(i + CAPACITY if (i in chosen) == choose_second else i for i in range(CAPACITY))


def normalize_histories(tokens, *, slots, vocab_size):
    """Input rows are in slots order, never implicitly indexed by slot number."""
    if tokens is None:
        return tuple(() for _ in slots)
    if callable(getattr(tokens, "tolist", None)):
        device = getattr(tokens, "device", None)
        if device is not None and getattr(device, "type", str(device)) != "cpu":
            raise ValueError("History reconstruction requires authoritative host tokens")
        tokens = tokens.tolist()
    if len(tokens) != len(slots):
        raise ValueError("History row count must equal len(slots)")
    result = []
    for row in tokens:
        if any(isinstance(t, bool) or not isinstance(t, Integral) or not -1 <= t < vocab_size for t in row):
            raise ValueError("History tokens must be in vocabulary, or -1 padding")
        result.append(tuple(int(t) for t in row if t >= 0))
    return tuple(result)


def _raw_parameters(params, capacity, *, max_top_k=32):
    """Use common field expansion, but retain RAW temperatures in owner metadata.

    format_sampling_params inverts temperature. Storing that result as a future
    raw request and formatting it again would invert a second time.
    """
    from agentic_models.models.ornith_1_5_9b.vllm.tt.sampling.generator import format_sampling_params

    temperatures = list(params.temperature) if isinstance(params.temperature, list) else [params.temperature]
    if len(temperatures) > capacity:
        raise ValueError("Sampling parameter rows exceed the target slot count")
    temperatures += [0.0] * (capacity - len(temperatures))
    # The common formatter requires its physical 32-row sampler capacity even
    # when this request updates only one or a few logical slots.
    formatted = format_sampling_params(params, CAPACITY, max_top_k=max_top_k)
    result = replace(
        formatted,
        **{
            field.name: getattr(formatted, field.name)[:capacity]
            for field in fields(formatted)
            if getattr(formatted, field.name) is not None
        },
    )
    updates = {"temperature": temperatures}
    if hasattr(result, "num_logprobs") and result.num_logprobs is None:
        updates["num_logprobs"] = [0] * capacity
    return replace(result, **updates)


class _SlotBuffer:
    """Gather an entire slot permutation to scratch before writing destination."""

    def __init__(self, owner, name, tensor):
        import ttnn

        self.owner, self.name, self.tensor = owner, name, tensor
        self.shape = tuple(tensor.shape)
        self.width = math.prod(self.shape) // CAPACITY
        if self.width * CAPACITY != math.prod(self.shape):
            raise ValueError(f"{name}: unexpected sampler tensor shape {self.shape}")
        # Every registered tensor has leading singleton dimensions followed by
        # slot dimension32. Remaining dimensions are per-slot columns.
        nonleading = list(self.shape)
        while len(nonleading) > 1 and nonleading[0] == 1:
            nonleading.pop(0)
        if nonleading[0] != CAPACITY:
            raise ValueError(f"{name}: slot axis is not the first non-singleton axis")
        self.scratch = ttnn.empty(
            (self.width, CAPACITY),
            dtype=tensor.dtype,
            layout=ttnn.TILE_LAYOUT,
            device=owner.mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        self.backup = ttnn.clone(tensor)  # Setup only, before any traces exist.

    def rows(self, tensor=None):
        import ttnn

        tensor = self.tensor if tensor is None else tensor
        # uint32 seeds/k are RM; gather supports their integer dtype in TILE.
        return ttnn.reshape(ttnn.to_layout(tensor, ttnn.TILE_LAYOUT), (CAPACITY, self.width))

    def write_gathered(self, source, indices):
        import ttnn

        transposed = ttnn.transpose(source, 0, 1)
        gathered = ttnn.gather(transposed, 1, indices, out=self.scratch)
        restored = ttnn.transpose(gathered, 0, 1)
        restored = ttnn.to_layout(restored, self.tensor.layout)
        ttnn.copy(ttnn.reshape(restored, self.shape), self.tensor)

    def remap(self, indices):
        self.write_gathered(self.rows(), indices)


class SamplingSlotState:
    def __init__(self, sampler, *, parameter_owner=None):
        import torch

        import ttnn

        self.sampler, self.parameter_owner = sampler, parameter_owner
        self.sampling, self.penalties = sampler.tt_sampling, sampler.tt_penalties
        self.mesh = sampler.mesh_device
        if tuple(self.mesh.shape) != (1, 4) or self.sampling._sampling_dp != 1:
            raise ValueError("SamplingSlotState implements the Ornith1x4 slot layout only")
        if self.sampling.max_batch_size != CAPACITY or self.penalties._total_batch != CAPACITY:
            raise ValueError("The helper requires fixed32 common-sampler slots")
        if self.sampling.grammar_bitmask_tensor is not None:
            raise ValueError("Device grammar needs its own state inventory before using this helper")
        self.buffers = {}
        for scope, obj, names in (
            (
                "sampling",
                self.sampling,
                # The pinned TTSampling has no per-slot log-prob scale (_scale_logprobs); device log-probs are
                # rejected by this model, so there is no such state to move.
                ("k_tensor", "p_tensor", "temp_tensor", "seeds_tt_tensor", "_greedy_col"),
            ),
            (
                "penalties",
                self.penalties,
                (
                    "prompt_mask",
                    "output_mask",
                    "output_counts",
                    "output_counts_gathered",
                    "presence_penalties",
                    "frequency_penalties",
                    "repetition_penalties",
                    "inverse_repetition_penalties",
                    "decode_src",
                ),
            ),
        ):
            for name in names:
                key = f"{scope}.{name}"
                self.buffers[key] = _SlotBuffer(self, key, getattr(obj, name))
        self.permute_indices = {
            width: self._upload(torch.arange(CAPACITY).reshape(1, CAPACITY).repeat(width, 1), dtype=ttnn.uint32)
            for width in {buffer.width for buffer in self.buffers.values()}
        }
        self.keep_rows = self._upload(torch.ones(CAPACITY, 1, dtype=torch.int32), dtype=ttnn.int32)
        # Updates are shape/distribution-identical to the common count tables.
        self.history_updates = {
            name: ttnn.clone(getattr(self.penalties, name))
            for name in ("prompt_mask", "output_mask", "output_counts", "output_counts_gathered")
        }
        self.seed_before = ttnn.clone(self.sampling.seeds_tt_tensor)
        self.seed_updates = ttnn.clone(self.sampling.seeds_tt_tensor)
        self.seed_keep_indices = self._upload(torch.arange(CAPACITY).reshape(1, CAPACITY), dtype=ttnn.uint32)
        self.seed_reset_indices = self._upload(torch.arange(CAPACITY).reshape(1, CAPACITY), dtype=ttnn.uint32)
        self.active_slots = tuple(range(CAPACITY))
        self._prepared = False
        self._snapshot_host = None

    def _upload(self, value, *, dtype, layout=None, device=True, mapper=None):
        import ttnn

        return ttnn.from_torch(
            value.contiguous(),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT if layout is None else layout,
            device=self.mesh if device else None,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=mapper or ttnn.ReplicateTensorToMesh(self.mesh),
        )

    def _copy_host(self, value, target, *, mapper=None):
        import ttnn

        host = self._upload(value, dtype=target.dtype, layout=target.layout, device=False, mapper=mapper)
        ttnn.copy_host_to_device_tensor(host, target)

    def _check_bindings(self):
        for key, buffer in self.buffers.items():
            scope, name = key.split(".", 1)
            current = getattr(getattr(self, scope), name)
            if (
                current.buffer_address() != buffer.tensor.buffer_address()
                or tuple(current.shape) != buffer.shape
                or current.dtype != buffer.tensor.dtype
                or current.layout != buffer.tensor.layout
            ):
                raise RuntimeError(f"Common sampler replaced persistent {key}; prepare a new model before capture")

    def snapshot(self):
        """One snapshot bank, allocated in setup; no nesting or device clone here."""
        import ttnn

        self._check_bindings()
        if self._snapshot_host is not None:
            raise RuntimeError("A sampling warmup snapshot is already live")
        for buffer in self.buffers.values():
            ttnn.copy(buffer.tensor, buffer.backup)
        manager = self.sampler.seed_manager
        calculator = self.sampling.log_probs_calculator
        self._snapshot_host = {
            "seed_lists": {
                name: copy.deepcopy(getattr(manager, name)) for name in ("seeds", "seed_counters", "seed_salts")
            },
            "rngs": [rng.getstate() for rng in manager.rngs],
            "seed_flags": {
                name: getattr(manager, name)
                for name in (
                    "_seed_active",
                    "_reseted",
                    "_needs_skip",
                    "_active_request_seed",
                    "_runtime_seed_buffer_managed",
                )
            },
            "prompt_shadow": copy.deepcopy(self.penalties._prompt_tokens_host),
            "sampling_flags": {
                name: getattr(self.sampler, name)
                for name in ("_penalties_active", "_log_probs_active", "_slot_state_requires_authoritative_reload")
            },
            "force_argmax": self.sampling._force_argmax_sampling,
            "logprobs": {
                name: copy.deepcopy(getattr(calculator, name))
                for name in ("logprobs_enabled", "num_logprobs", "enable_log_probs", "topk_logprobs_needed")
            },
            "parameters": (
                copy.deepcopy(self.parameter_owner.sampling_params) if self.parameter_owner is not None else None
            ),
            "active_slots": self.active_slots,
        }

    def restore(self):
        import ttnn

        self._check_bindings()
        saved = self._snapshot_host
        if saved is None:
            raise RuntimeError("No sampling warmup snapshot to restore")
        for buffer in self.buffers.values():
            ttnn.copy(buffer.backup, buffer.tensor)
        manager = self.sampler.seed_manager
        for name, value in saved["seed_lists"].items():
            setattr(manager, name, value)
        for rng, state in zip(manager.rngs, saved["rngs"]):
            rng.setstate(state)
        for name, value in saved["seed_flags"].items():
            setattr(manager, name, value)
        self.penalties._prompt_tokens_host = saved["prompt_shadow"]
        for name, value in saved["sampling_flags"].items():
            setattr(self.sampler, name, value)
        self.sampling._force_argmax_sampling = saved["force_argmax"]
        for name, value in saved["logprobs"].items():
            setattr(self.sampling.log_probs_calculator, name, value)
        if self.parameter_owner is not None:
            self.parameter_owner.sampling_params = saved["parameters"]
        self.refresh_active_slots(saved["active_slots"])
        self._snapshot_host = None

    def prepare(self):
        """Idempotent warmup of every helper op, restoring all request-owned state."""
        if self._prepared:
            return self
        self.snapshot()
        try:
            self.apply_slot_remap(tuple(reversed(range(CAPACITY))))
            self.reset_history(
                slots=[0, 31], prompt_tokens=[[1, 1, 2], [3]], output_tokens=[[2, 2], [4]], force_upload=True
            )
            self.reset_seed_values(slots=[0, 31], values=[123, 456])
            self.refresh_active_slots([0, 2, 31])
            self.before_sample()
            self.after_sample()
        finally:
            self.restore()
        self._prepared = True
        return self

    def apply_slot_remap(self, remap):
        """Destination i reads source remap[i], including every dormant/offbatch slot."""
        import torch

        self._check_bindings()
        remap = validate_slots(remap, permutation=True)
        for width, index in self.permute_indices.items():
            self._copy_host(torch.tensor(remap).reshape(1, CAPACITY).repeat(width, 1), index)
        for buffer in self.buffers.values():
            buffer.remap(self.permute_indices[buffer.width])
        manager = self.sampler.seed_manager
        for name in ("seeds", "seed_counters", "seed_salts", "rngs"):
            source = getattr(manager, name)
            setattr(manager, name, [copy.copy(source[i]) for i in remap])
        shadow = self.penalties._prompt_tokens_host
        if shadow is not None:
            self.penalties._prompt_tokens_host = shadow[list(remap)].clone()
        calculator = self.sampling.log_probs_calculator
        for name in ("logprobs_enabled", "num_logprobs"):
            source = getattr(calculator, name)
            setattr(calculator, name, [source[i] for i in remap])
        if self.parameter_owner is not None:
            params = _raw_parameters(
                self.parameter_owner.sampling_params, CAPACITY, max_top_k=self.sampler.tt_sampling.max_top_k
            )
            self.parameter_owner.sampling_params = replace(
                params, **{f.name: [getattr(params, f.name)[i] for i in remap] for f in fields(params)}
            )
        active = set(self.active_slots)
        self.refresh_active_slots([destination for destination, source in enumerate(remap) if source in active])

    def _penalties_neutral(self):
        """True when every sampler row has presence 0, frequency 0 and repetition 1 (owner's raw parameters)."""
        params = getattr(self.parameter_owner, "sampling_params", None) if self.parameter_owner is not None else None
        if params is None:
            return False
        for name, neutral in (("presence_penalty", 0.0), ("frequency_penalty", 0.0), ("repetition_penalty", 1.0)):
            value = getattr(params, name, neutral)
            values = value if isinstance(value, (list, tuple)) else [value]
            if any(v is not None and float(v) != neutral for v in values):
                return False
        return True

    def reset_history(self, *, slots, prompt_tokens=None, output_tokens=None, force_upload=False):
        """Replace selected histories; input rows are packed in the supplied slot order."""
        import torch

        import ttnn

        self._check_bindings()
        slots = validate_slots(slots)
        prompts = normalize_histories(prompt_tokens, slots=slots, vocab_size=self.penalties.vocab_size)
        outputs = normalize_histories(output_tokens, slots=slots, vocab_size=self.penalties.vocab_size)
        keep = torch.ones(CAPACITY, 1, dtype=torch.int32)
        keep[list(slots)] = 0
        self._copy_host(keep, self.keep_rows)

        def counts(rows):
            result = torch.zeros(CAPACITY, self.penalties.vocab_size, dtype=torch.int32)
            for slot, row in zip(slots, rows):
                if row:
                    result[slot] = torch.bincount(torch.tensor(row), minlength=self.penalties.vocab_size).to(
                        torch.int32
                    )
            return result

        # ROUND 6: building and uploading the four 32 x vocab int32 tables costs ~45 ms per prefill step. Skip a
        # table's upload when its update is all zero (empty histories) or when no request in the sampler uses a
        # penalty (presence/frequency 0, repetition 1: apply_penalties is then an exact no-op whatever the masks
        # hold). Selected rows are still cleared, so a later penalized request in the slot starts from its own upload.
        neutral = HISTORY_SKIP and not force_upload and self._penalties_neutral()
        need_prompt = not neutral and any(prompts)
        need_output = not neutral and any(outputs)
        prompt_counts = counts(prompts) if need_prompt else None
        output_counts = counts(outputs) if need_output else None
        values = {
            "prompt_mask": (prompt_counts > 0).to(torch.int32) if need_prompt else None,
            "output_mask": (output_counts > 0).to(torch.int32) if need_output else None,
            "output_counts": output_counts,
            "output_counts_gathered": output_counts,
        }
        if not HISTORY_SKIP:
            zero = torch.zeros(CAPACITY, self.penalties.vocab_size, dtype=torch.int32)
            values = {name: zero if value is None else value for name, value in values.items()}
        for name, value in values.items():
            target = getattr(self.penalties, name)
            ttnn.multiply(target, self.keep_rows, output_tensor=target)
            if value is None:
                continue
            dims = (
                self.penalties._shard_dims_gathered
                if name == "output_counts_gathered"
                else self.penalties._shard_dims_mask
            )
            mapper = ttnn.ShardTensor2dMesh(self.mesh, dims=dims, mesh_shape=self.mesh.shape)
            update = self.history_updates[name]
            self._copy_host(value, update, mapper=mapper)
            ttnn.add(target, update, output_tensor=target)
        # Keep common helper's host prompt shadow coherent for future users.
        old = self.penalties._prompt_tokens_host
        width = max(1, max((len(row) for row in prompts), default=0), 0 if old is None else old.shape[-1])
        shadow = torch.full((CAPACITY, width), -1, dtype=torch.int64)
        if old is not None:
            shadow[:, : old.shape[-1]] = old
        for slot, row in zip(slots, prompts):
            shadow[slot] = -1
            if row:
                shadow[slot, : len(row)] = torch.tensor(row)
        self.penalties._prompt_tokens_host = shadow

    def reset_parameters(self, params, *, slots):
        """Merge row-packed parameters, then use the COMMON parameter uploader."""
        from agentic_models.models.ornith_1_5_9b.vllm.tt.sampling.generator import format_sampling_params

        if self.parameter_owner is None:
            raise ValueError(
                "Scoped parameter reset requires parameter_owner.sampling_params as authoritative metadata"
            )
        slots = validate_slots(slots)
        if not slots:
            return
        previous = _raw_parameters(
            self.parameter_owner.sampling_params, CAPACITY, max_top_k=self.sampler.tt_sampling.max_top_k
        )
        update = _raw_parameters(params, len(slots), max_top_k=self.sampler.tt_sampling.max_top_k)
        merged = {}
        for field in fields(previous):
            values = list(getattr(previous, field.name))
            for row, slot in enumerate(slots):
                values[slot] = getattr(update, field.name)[row]
            merged[field.name] = values
        raw = replace(previous, **merged)
        if any(raw.enable_log_probs):
            raise ValueError("OrnithTP4 routes logprobs to host sampling")
        self.sampler.reset_sampling_params(
            format_sampling_params(raw, CAPACITY, max_top_k=self.sampler.tt_sampling.max_top_k)
        )
        self.parameter_owner.sampling_params = raw
        self._check_bindings()

    def reset_seed_values(self, *, slots, values):
        """Use caller's existing request seedstream; preserve all unselected seeds."""
        import torch

        import ttnn

        slots = validate_slots(slots)
        if len(values) != len(slots) or any(not isinstance(v, Integral) or not 1 <= v <= 1_000_000 for v in values):
            raise ValueError("Expected one bounded positive device seed per reset slot")
        update = torch.zeros(CAPACITY, dtype=torch.int64)
        for slot, value in zip(slots, values):
            update[slot] = value
        self._copy_host(update, self.seed_updates)
        self._copy_host(
            torch.tensor(seed_selection_indices(slots, choose_second=True)).reshape(1, CAPACITY),
            self.seed_reset_indices,
        )
        buffer = self.buffers["sampling.seeds_tt_tensor"]
        source = ttnn.concat([buffer.rows(), buffer.rows(self.seed_updates)], dim=0)
        buffer.write_gathered(source, self.seed_reset_indices)

    def refresh_active_slots(self, slots):
        """Scheduler boundary only; no host refresh belongs between sampling replays."""
        import torch

        self.active_slots = validate_slots(slots)
        source = torch.zeros(CAPACITY, 1, dtype=torch.int32)
        source[list(self.active_slots)] = 1
        self._copy_host(source, self.penalties.decode_src)
        self._copy_host(
            torch.tensor(seed_selection_indices(self.active_slots, choose_second=False)).reshape(1, CAPACITY),
            self.seed_keep_indices,
        )

    def before_sample(self):
        """Device-only captured prologue; stable32-word seed backup."""
        import ttnn

        ttnn.copy(self.sampling.seeds_tt_tensor, self.seed_before)

    def after_sample(self):
        """Device-only captured epilogue; keep advanced active seeds, restore peers."""
        import ttnn

        buffer = self.buffers["sampling.seeds_tt_tensor"]
        source = ttnn.concat([buffer.rows(), buffer.rows(self.seed_before)], dim=0)
        buffer.write_gathered(source, self.seed_keep_indices)
