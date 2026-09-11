# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Instance-only S128/C1/slot-0 greedy decode-trace reuse experiment.

No global hooks, allocator acknowledgments, RNG changes or production defaults.
Install on an otherwise idle adapter and close before tearing its generator down.
Allocation tracking must be enabled at process startup. All uncertain transitions
fall back to the original release/setup path before any replay.
"""

import inspect
import time
import types
from collections import Counter

import torch

import ttnn
from models.autoports.qwen_qwen3_6_27b.doc.prefill_device_analysis.serving_trace_reuse_plan import (
    CapturedContract,
    RequestContract,
    after_prefill,
    before_prefill,
)
from models.common.sampling import format_sampling_params


class ServingTraceReuseExperiment:
    def __init__(self, adapter, *, enabled):
        if not ttnn.TRACE_ALLOC_TRACKING:
            raise RuntimeError("Set TT_METAL_TRACE_ALLOC_TRACKING=1 before starting this guarded experiment")
        self.adapter = adapter
        self.gen = adapter.generator
        self.mesh = self.gen.mesh_device
        self.enabled = bool(enabled)
        self.counters = Counter()
        self.events = []
        self.captured = None
        self.current_request = None
        self.preserve_reset = False
        self.pending_reuse = False
        self.before_entries = None
        self.after_entries = None
        self._restore = []
        self.original_prefill = adapter.prefill_forward
        self.original_setup = self.gen.setup_token_out_decode
        self.original_reset_slots = self.gen.reset_slots
        self.original_release = self.gen._release_traces
        self.original_capture = self.gen._capture_token_out_trace
        self.prefill_signature = inspect.signature(self.original_prefill)
        self.setup_signature = inspect.signature(self.original_setup)
        self._install(adapter, "prefill_forward", self.prefill)
        self._install(self.gen, "setup_token_out_decode", self.setup)
        self._install(self.gen, "reset_slots", self.reset_slots)
        self._install(self.gen, "_release_traces", self.release)
        self._install(self.gen, "_capture_token_out_trace", self.capture)

    def _install(self, owner, name, function):
        self._restore.append((owner, name, name in vars(owner), vars(owner).get(name)))

        def invoke(_instance, *args, **kwargs):
            return function(*args, **kwargs)

        setattr(owner, name, types.MethodType(invoke, owner))

    def close(self):
        self.original_release()
        for owner, name, had_attribute, previous in reversed(self._restore):
            if had_attribute:
                setattr(owner, name, previous)
            else:
                delattr(owner, name)
        self._restore.clear()

    @staticmethod
    def _tensor_binding(tensor):
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

    def _cache_bindings(self):
        return tuple(
            (layer.layer_idx, name, self._tensor_binding(layer.caches[name]))
            for layer in self.gen.model.layers
            for name in ("conv", "recurrent", "key", "value", "batch_indices")
            if name in layer.caches
        )

    def _trace_bindings(self):
        sampler = tuple(
            (id(slot["id"]), id(slot["input"]), tuple(self._output_bindings(slot["output"])))
            for slot in self.gen.sampling._trace_states.values()
            if slot["id"] is not None
        )
        return (
            id(self.gen._decode_trace_id) if self.gen._decode_trace_id is not None else None,
            sampler,
            tuple(
                self._tensor_binding(getattr(self.gen, name))
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

    def _output_bindings(self, output):
        if isinstance(output, ttnn.Tensor):
            yield self._tensor_binding(output)
        elif isinstance(output, (tuple, list)):
            for item in output:
                yield from self._output_bindings(item)

    def _request(self, values):
        try:
            return self._build_request(values)
        except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
            # A guard must never turn an unsupported request into resident
            # replay. Let the original adapter validate it after normal release.
            self.events.append({"event": "contract_rejected", "error": repr(error)})
            return None

    def _build_request(self, values):
        tokens = values["tokens"]
        lengths = tuple(int(x) for x in values["prompt_lens"])
        supplied_slots = values.get("empty_slots")
        slots = tuple(range(len(lengths))) if supplied_slots is None else tuple(int(x) for x in supplied_slots)
        params = values.get("sampling_params")
        if params is None:
            return None
        formatted = format_sampling_params(params, 32)
        if (
            self.gen.model.batch != 32
            or tuple(tokens.shape) != (1, 128)
            or lengths != (128,)
            or slots != (0,)
            or values["kv_cache"] is not self.gen.kv_cache
            or not isinstance(values["page_table"], torch.Tensor)
            or int(formatted.top_k[0]) != 1
            or float(formatted.top_p[0]) != 0.0
        ):
            return None
        start_pos = values.get("start_pos")
        if start_pos is not None and torch.as_tensor(start_pos).ne(0).any():
            return None
        seeded = any(value is not None for value in formatted.seed)
        penalties = any(
            float(x) != default
            for field, default in (
                (formatted.presence_penalty, 0.0),
                (formatted.frequency_penalty, 0.0),
                (formatted.repetition_penalty, 1.0),
            )
            for x in field
        )
        logprobs = any(bool(x) for x in formatted.enable_log_probs)
        return RequestContract(
            prefill_signature=(
                id(self.gen.model),
                id(self.mesh),
                self.gen.model.batch,
                self.gen.model.page_size,
                tuple(tokens.shape),
                str(tokens.dtype),
                lengths,
                tuple(values["page_table"].shape),
                str(values["page_table"].dtype),
            ),
            active_slots=slots,
            sampling_signature=self.adapter._sampling_key(params),
            cache_bindings=self._cache_bindings(),
            page_table_binding=self._tensor_binding(self.gen._page_table),
            seeded=seeded,
            penalties=penalties,
            logprobs=logprobs,
        )

    def _allocations(self):
        # Uncollected cycles conservatively force recapture; collection here is
        # expensive and is unnecessary for the zero-unsafe-allocation guard.
        trace_ids = [self.gen._decode_trace_id]
        trace_ids += [slot["id"] for slot in self.gen.sampling._trace_states.values()]
        unsafe = {}
        for trace_id in trace_ids:
            if trace_id is not None:
                unsafe.update(ttnn._ttnn.operations.trace.get_unsafe_tracked_ids(self.mesh, trace_id))
        ordinary = {key: value for key, value in unsafe.items() if not value.startswith("program_cache:")}
        return {
            "ordinary_count": len(ordinary),
            "program_owned_count": len(unsafe) - len(ordinary),
            "unsafe": {str(key): value for key, value in unsafe.items()},
        }

    def release(self):
        self.counters["release_calls"] += 1
        self.counters["live_trace_releases"] += int(self.gen._decode_trace_id is not None)
        self.captured = None
        self.pending_reuse = False
        self.original_release()

    def capture(self, *args, **kwargs):
        started = time.perf_counter()
        output = self.original_capture(*args, **kwargs)
        self.counters["model_capture_calls"] += 1
        self.events.append({"event": "capture", "elapsed_ms": (time.perf_counter() - started) * 1000})
        return output

    def reset_slots(self, slots):
        if not self.preserve_reset:
            return self.original_reset_slots(slots)
        if tuple(sorted({int(slot) for slot in slots})) != (0,):
            raise RuntimeError("guarded reset was entered for a different active slot")
        ttnn.synchronize_device(self.mesh)
        self.gen.model.reset_slots(slots)
        self.gen._slots_requiring_prefill.update(slots)
        self.counters["resets_preserving_trace"] += 1

    def prefill(self, *args, **kwargs):
        entry = time.perf_counter()
        bound = self.prefill_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        self.current_request = self._request(values)
        self.pending_reuse = False
        decision = before_prefill(self.captured, self.current_request) if self.current_request is not None else None
        self.preserve_reset = bool(
            self.enabled and decision is not None and decision.action == "preserve_until_prefill_finishes"
        )
        self.before_entries = self.mesh.num_program_cache_entries()
        started = time.perf_counter()
        try:
            output = self.original_prefill(*args, **kwargs)
        except BaseException:
            self.gen._release_traces()
            raise
        finally:
            preserved = self.preserve_reset
            self.preserve_reset = False
        prefill_end = time.perf_counter()
        self.after_entries = self.mesh.num_program_cache_entries()
        allocations = self._allocations() if preserved else None
        reason = "control or unmatched request contract"
        if preserved:
            actual = self._request(values)
            if actual is None:
                reason = "request or cache ownership became unreadable after prefill"
            else:
                guard = after_prefill(
                    self.captured,
                    actual,
                    current_trace_bindings=self._trace_bindings(),
                    program_entries_before=self.before_entries,
                    program_entries_after=self.after_entries,
                    allocation_evidence_clear=not allocations["unsafe"],
                    inputs_reload_authorized=True,
                    request_prefill_complete=0 not in self.gen._slots_requiring_prefill,
                )
                reason = guard.reason
                self.pending_reuse = guard.action == "refresh_and_replay"
            if not self.pending_reuse:
                self.counters["guard_recaptures"] += 1
                self.gen._release_traces()
        self.events.append(
            {
                "event": "prefill",
                "elapsed_ms": (time.perf_counter() - entry) * 1000,
                "adapter_work_ms": (prefill_end - started) * 1000,
                "guard_ms": ((started - entry) + (time.perf_counter() - prefill_end)) * 1000,
                "preserved": preserved,
                "reuse_pending": self.pending_reuse,
                "reason": reason,
                "program_entries_before": self.before_entries,
                "program_entries_after": self.after_entries,
                "allocations": allocations,
            }
        )
        return output

    def setup(self, *args, **kwargs):
        entry = time.perf_counter()
        bound = self.setup_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        compatible = False
        if self.pending_reuse:
            try:
                compatible = self._setup_compatible(values)
            except (AttributeError, IndexError, KeyError, RuntimeError, TypeError, ValueError) as error:
                self.events.append({"event": "setup_guard_rejected", "error": repr(error)})
        started = time.perf_counter()
        if compatible:
            if 0 in self.gen._slots_requiring_prefill:
                raise RuntimeError("reset slot must complete prefill before resident reload")
            self.gen.refresh_page_table(values["page_table"])
            self.gen._seed_token_out_trace(values["tokens"], values["positions"])
            self.counters["setup_reuses"] += 1
            output = {
                "token": self.gen._trace_token,
                "position": self.gen._trace_position,
                "page_table": self.gen._trace_page_table,
                "kv_cache": self.gen.kv_cache,
                "active_mask": self.gen._trace_active_mask,
            }
        else:
            self.counters["setup_recaptures"] += 1
            output = self.original_setup(*args, **kwargs)
            if self.current_request is not None:
                self.captured = CapturedContract(self.current_request, self._trace_bindings(), True)
        self.pending_reuse = False
        self.events.append(
            {
                "event": "setup",
                "reused": compatible,
                "elapsed_ms": (time.perf_counter() - entry) * 1000,
                "setup_work_ms": (time.perf_counter() - started) * 1000,
                "guard_ms": (started - entry) * 1000,
            }
        )
        return output

    def _setup_compatible(self, values):
        active = tuple(torch.as_tensor(values["active_mask"]).bool().reshape(-1).tolist())
        return (
            self.gen._decode_trace_id is not None
            and active == (True,) + (False,) * 31
            and values["kv_cache"] is self.gen.kv_cache
            and isinstance(values["page_table"], torch.Tensor)
            and tuple(values["page_table"].shape) == tuple(self.gen.page_table_host.shape)
            and self.gen._trace_page_table is self.gen._page_table
            and self.adapter._sampling_key(values["sampling_params"]) == self.current_request.sampling_signature
            and self._trace_bindings() == self.captured.trace_bindings
            and self._cache_bindings() == self.captured.request.cache_bindings
            and not self._allocations()["unsafe"]
        )
