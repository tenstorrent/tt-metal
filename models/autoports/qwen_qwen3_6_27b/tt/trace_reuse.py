# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Conservative reuse of warmed C1 request envelopes with complete allocation checks."""

import os
import threading
from collections import Counter

import torch

import ttnn
from models.common.sampling import format_sampling_params


def tensor_binding(tensor):
    if tensor is None:
        return None
    if not tensor.is_allocated():
        return (id(tensor), "deallocated")
    return (
        id(tensor),
        tensor.buffer_unique_id(),
        tensor.buffer_address(),
        tuple(tensor.shape),
        tuple(tensor.padded_shape),
        str(tensor.dtype),
        str(tensor.layout),
        str(tensor.memory_config()),
    )


class DecodeTraceReuse:
    """No reuse without tracking; every replay checks all surviving allocations.

    Omitting Python GC makes the empty-map test stricter: uncollected cycles
    still appear live. The C++ allocation tracker, including program buffers,
    remains enabled. There are no corruptible exemptions or global hooks.
    """

    def __init__(self, generator):
        self.gen = generator
        self.mesh = generator.mesh_device
        self.tracking = (
            bool(ttnn.TRACE_ALLOC_TRACKING) and os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "0"
        )
        self.enabled = os.environ.get("QWEN36_TRACE_REUSE", "1") == "1"
        self.owner_thread = None
        self.warmed = set()
        self.request_key = None
        self.request_sampling = None
        self.request_slots = ()
        self.preserve_reset = False
        self.pending = False
        self.captured = None
        self.execution = None
        self.before_entries = None
        self.counters = Counter()
        self.last_reason = "no captured request"

    def _assert_owner(self):
        # The serving worker must be the sole submitter to this mesh. This
        # catches cross-thread use of this generator; unrelated external mesh
        # submissions are unsupported and cannot be fenced by a Python guard.
        current = threading.get_ident()
        if self.owner_thread is None:
            self.owner_thread = current
        elif self.owner_thread != current:
            raise RuntimeError("resident trace reuse requires one serialized device-submit thread")

    def _cache_bindings(self):
        return tuple(
            (layer.layer_idx, name, tensor_binding(layer.caches[name]))
            for layer in self.gen.model.layers
            for name in ("conv", "recurrent", "key", "value", "batch_indices")
            if name in layer.caches
        )

    @staticmethod
    def _sampling_signature(params):
        if params is None:
            return None
        formatted = format_sampling_params(params, 32)
        if (
            any(seed is not None for seed in formatted.seed)
            or any(bool(value) for value in formatted.enable_log_probs)
            or any(float(value) != 0 for value in formatted.presence_penalty)
            or any(float(value) != 0 for value in formatted.frequency_penalty)
            or any(float(value) != 1 for value in formatted.repetition_penalty)
            or any(int(value) != 1 for value in formatted.top_k)
        ):
            return None
        return tuple(tuple(getattr(formatted, field)) for field in ("top_k", "top_p", "temperature"))

    def _sampler_slot(self):
        sampler = self.gen.sampling
        if sampler._active_trace_bucket is not None:
            return None
        for key, slot in sampler._trace_states.items():
            if (
                key.penalties_on == sampler._penalties_active
                and key.log_probs_on == getattr(sampler, "_log_probs_active", False)
                and key.force_argmax == sampler.tt_sampling.force_argmax_sampling
                and key.bucket is None
                and slot["id"] is not None
            ):
                return slot
        return None

    def _trace_bindings(self):
        slot = self._sampler_slot()
        if self.gen._decode_trace_id is None or slot is None:
            return None
        # Bucketed sampling traces may exempt their buffers from allocation
        # tracking. Reject every live bucket, including an inactive old one.
        if any(
            key.bucket is not None and state["id"] is not None for key, state in self.gen.sampling._trace_states.items()
        ):
            return None
        self.gen.sampling._validate_trace_inputs(slot, self.gen._trace_logits, self.gen._trace_token)
        sampler_owners = tuple(
            (int(state["id"]), id(state["input"]), id(state["output"]))
            for state in self.gen.sampling._trace_states.values()
            if state["id"] is not None
        )
        return (
            int(self.gen._decode_trace_id),
            sampler_owners,
            tuple(
                tensor_binding(getattr(self.gen, name))
                for name in (
                    "_trace_token",
                    "_trace_position",
                    "_trace_active_mask",
                    "_trace_active_state_mask",
                    "_trace_page_table",
                    "_trace_logits",
                )
            ),
        )

    def _unsafe(self):
        slot = self._sampler_slot()
        if self.gen._decode_trace_id is None or slot is None:
            raise RuntimeError("resident model and sampler traces must both exist before replay")
        unsafe = {}
        trace_ids = [self.gen._decode_trace_id] + [
            state["id"] for state in self.gen.sampling._trace_states.values() if state["id"] is not None
        ]
        for trace_id in trace_ids:
            unsafe.update(ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(self.mesh, trace_id))
        return unsafe

    def invalidate(self, reason):
        self.captured = None
        self.execution = None
        self.pending = False
        self.preserve_reset = False
        self.last_reason = reason

    def begin_prefill(self, *, physical_seq_len, prompt_lens, slots, page_table, kv_cache, sampling_params):
        self._assert_owner()
        self.pending = False
        self.preserve_reset = False
        self.request_key = None
        self.request_sampling = None
        self.request_slots = ()
        try:
            self.request_slots = tuple(int(slot) for slot in slots)
            signature = self._sampling_signature(sampling_params)
            lengths = tuple(int(length) for length in prompt_lens)
            if (
                not self.tracking
                or self.gen.model.batch != 32
                or len(self.request_slots) != 1
                or not 0 <= self.request_slots[0] < 32
                or len(lengths) != 1
                or lengths[0] <= 0
                or signature is None
                or kv_cache is not self.gen.kv_cache
                or not isinstance(page_table, torch.Tensor)
                or tuple(page_table.shape) != tuple(self.gen.page_table_host.shape)
            ):
                self.gen._release_traces()
                return False
            self.request_sampling = signature
            self.request_key = (
                int(physical_seq_len),
                lengths,
                self.request_slots,
                tuple(page_table.shape),
                str(page_table.dtype),
                tensor_binding(self.gen._page_table),
                signature,
            )
            self.before_entries = self.mesh.num_program_cache_entries()
            self.preserve_reset = bool(
                self.enabled
                and self.request_key in self.warmed
                and self.captured is not None
                and self.captured[0] == self._cache_bindings()
                and self.captured[1] == self._trace_bindings()
                and self.captured[2] == signature
                and self.captured[3] == self.before_entries
                and not self._unsafe()
            )
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError):
            self.request_key = None
            self.gen._release_traces()
            raise
        if not self.preserve_reset:
            self.gen._release_traces()
        return self.preserve_reset

    def end_prefill(self):
        self._assert_owner()
        ttnn.synchronize_device(self.mesh)
        preserved = self.preserve_reset
        self.preserve_reset = False
        if self.request_key is None:
            return False
        if len(self.warmed) >= 128:
            self.warmed.clear()
        self.warmed.add(self.request_key)
        if preserved:
            self.pending = bool(
                self.captured is not None
                and self.captured[0] == self._cache_bindings()
                and self.captured[1] == self._trace_bindings()
                and self.mesh.num_program_cache_entries() == self.before_entries
                and not self._unsafe()
            )
            if not self.pending:
                self.counters["guard_recaptures"] += 1
                self.gen._release_traces()
        return self.pending

    def can_reuse_setup(self, *, sampling_params, active_values, page_table, kv_cache):
        self._assert_owner()
        if not self.pending:
            return False
        slots = tuple(torch.as_tensor(active_values).reshape(-1).nonzero().reshape(-1).tolist())
        compatible = bool(
            self.captured is not None
            and slots == self.request_slots
            and kv_cache is self.gen.kv_cache
            and isinstance(page_table, torch.Tensor)
            and tuple(page_table.shape) == tuple(self.gen.page_table_host.shape)
            and self.gen._trace_page_table is self.gen._page_table
            and self._sampling_signature(sampling_params) == self.captured[2]
            and self._cache_bindings() == self.captured[0]
            and self._trace_bindings() == self.captured[1]
            and self.mesh.num_program_cache_entries() == self.captured[3]
            and not self._unsafe()
        )
        self.pending = False
        return compatible

    def captured_setup(self, sampling_params):
        self.record_execution()
        signature = self._sampling_signature(sampling_params)
        if not self.tracking or self.request_key is None or signature is None:
            return
        self._assert_owner()
        ttnn.synchronize_device(self.mesh)
        bindings = self._trace_bindings()
        if bindings is None or self._unsafe():
            raise RuntimeError("new decode capture has unsafe allocations or missing sampler ownership")
        self.captured = (self._cache_bindings(), bindings, signature, self.mesh.num_program_cache_entries())

    def record_execution(self):
        """Keep checked execution for all sampler modes, even without reuse."""
        if not self.tracking:
            return
        self._assert_owner()
        ttnn.synchronize_device(self.mesh)
        bindings = self._trace_bindings()
        if bindings is None or self._unsafe():
            raise RuntimeError("new decode capture has unsafe allocations or missing sampler ownership")
        self.execution = (bindings, self.mesh.num_program_cache_entries())

    def preflight(self):
        """Validate both traces before model state advances; failures abort."""
        if self.execution is None:
            return False
        self._assert_owner()
        if self._trace_bindings() != self.execution[0]:
            raise RuntimeError("resident sampler graph or trace input ownership changed before decode")
        if self.mesh.num_program_cache_entries() != self.execution[1]:
            raise RuntimeError("new programs appeared behind resident decode traces")
        unsafe = self._unsafe()
        if unsafe:
            raise RuntimeError(f"unsafe allocations survive before resident decode replay: {unsafe}")
        return True

    def execute(self, device, trace_id, *, cq_id=None, blocking=True):
        # Recheck the union even for sampler replay. A late failure must abort;
        # recapturing/retrying after model state advanced would double a step.
        if device is not self.mesh or not self.preflight():
            raise RuntimeError("checked trace execution requires the captured generator and mesh")
        slot = self._sampler_slot()
        if trace_id not in (self.gen._decode_trace_id, slot["id"]):
            raise RuntimeError("attempted to replay a trace outside the resident pair")
        return ttnn._ttnn.operations.trace.execute_trace(device, trace_id, cq_id=cq_id, blocking=blocking)
