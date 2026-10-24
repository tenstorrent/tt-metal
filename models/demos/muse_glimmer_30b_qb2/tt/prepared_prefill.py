# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Caller-owned short-prefill traces with explicit prepare/capture phases."""

import time
from contextlib import contextmanager

from ttnn.tools import trace_allocation_tracker

import ttnn


class PreparedPrefill:
    """Trace initial 1..64-token prefills in physical 32/64-row buckets.

    Construct every instance and prepare all eager variants before capture.
    Decode's persistent CCL state must already exist during prepare. Prepare
    this helper before warming remaining eager variants: short AR adds L1 state. Inputs,
    page-table storage and KV bindings remain stable until explicit close.
    Caller input/page-table buffers must predate the first capture and remain
    bound for the session; refresh their contents in place. Trace-produced
    inputs need separately reviewed alias/lifetime ownership. Do not keep fresh
    post-capture device allocations live during replay.
    Consume each returned output before replaying any other live trace.
    Longer or append prefills remain the decoder's ordinary device-only path.
    """

    POLICY = {
        "prefill_norm_min_rows": 32,
        "prefill_norm_block": False,
        "prefill_norm_grid": [8, 8],
        "short_all_reduce": True,
    }
    KIND_POLICY = {
        "sliding_attention": {
            "prefill.input_layernorm.fidelity": "LoFi",
            "prefill.input_layernorm.fp32": False,
        }
    }
    POLICY_KEYS = set(POLICY) | {
        "prefill_norm_block",
        "prefill_norm_grid",
        "prefill.input_layernorm.fidelity",
        "prefill.input_layernorm.fp32",
    }

    def __init__(self, layer, *, mesh, kv_cache, page_table, batch, policy=None):
        self.layer, self.mesh, self.kv_cache = layer, mesh, kv_cache
        self.batch = batch
        if policy and policy.keys() - self.POLICY_KEYS:
            raise ValueError("Unsupported prepared-prefill policy key")
        self.policy = dict(self.POLICY, **(policy or {}))
        # Compute configs are cached by the decoder at construction. Prepare
        # real replacement configs now; changing only its policy is ineffective.
        self.layer_policies = []
        for member in getattr(layer, "layers", [layer]):
            selected = dict(self.KIND_POLICY.get(member.kind, {}), **self.policy)
            old = member.norm_configs["prefill"]["input_layernorm"]
            config = ttnn.init_device_compute_kernel_config(
                mesh.arch(),
                math_fidelity=getattr(
                    ttnn.MathFidelity, selected.get("prefill.input_layernorm.fidelity", old.math_fidelity.name)
                ),
                math_approx_mode=old.math_approx_mode,
                fp32_dest_acc_en=selected.get("prefill.input_layernorm.fp32", old.fp32_dest_acc_en),
                packer_l1_acc=old.packer_l1_acc,
                dst_full_sync_en=old.dst_full_sync_en,
                throttle_level=old.throttle_level,
            )
            self.layer_policies.append((member, selected, config))
        self.captured_policies = []
        self.source_table = page_table
        self.page_table = ttnn.clone(ttnn.slice(page_table, [0, 0], [batch, page_table.shape[1]]))
        self.inputs = {
            size: ttnn.zeros([batch, 1, size, 6656], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
            for size in (32, 64)
        }
        self.traces, self.outputs = {}, {}
        self.prepared = False
        self.capture_count = 0
        self.replay_count = 0
        self.release_count = 0
        self.prepare_ms = self.capture_ms = None

    @contextmanager
    def capture_policy(self):
        # Sharded short norms win with traced dispatch. Keep ordinary eager
        # policy intact; only preparation/capture uses this measured family.
        original = [
            (layer.policy, layer.norm_configs["prefill"]["input_layernorm"]) for layer, _, _ in self.layer_policies
        ]
        try:
            for layer, policy, config in self.layer_policies:
                layer.policy = dict(layer.policy, **policy)
                layer.norm_configs["prefill"]["input_layernorm"] = config
            self.captured_policies = [
                {
                    "layer": layer.layer_idx,
                    "kind": layer.kind,
                    "policy": dict(layer.policy),
                    "input_norm_compute_config": str(config),
                }
                for layer, _, config in self.layer_policies
            ]
            yield
        finally:
            for (layer, _, _), (policy, config) in zip(self.layer_policies, original):
                layer.policy = policy
                layer.norm_configs["prefill"]["input_layernorm"] = config

    @staticmethod
    def bucket(length):
        if not 1 <= length <= 64:
            raise ValueError("Prepared short prefill accepts logical lengths 1..64")
        return 32 if length <= 32 else 64

    def stage_inputs(self, x, page_table, *, slots=None):
        size = self.bucket(x.shape[-2])
        if x.shape[0] != self.batch or list(page_table.shape) != list(self.source_table.shape):
            raise ValueError("Prepared prefill batch/page-table binding changed")
        slots = tuple(range(self.batch)) if slots is None else tuple(slots)
        if (
            len(slots) != self.batch
            or len(set(slots)) != self.batch
            or any(not 0 <= slot < page_table.shape[0] for slot in slots)
        ):
            raise ValueError("Prepared prefill needs distinct slots in the bound table")
        x = ttnn.to_memory_config(x, ttnn.DRAM_MEMORY_CONFIG)
        if x.shape[-2] != size:
            x = ttnn.pad(x, [(0, 0), (0, 0), (0, size - x.shape[-2]), (0, 0)], 0)
        ttnn.copy(x, self.inputs[size])
        if slots != tuple(range(self.batch)) or page_table.shape[0] != self.batch:
            parts = [ttnn.slice(page_table, [slot, 0], [slot + 1, page_table.shape[1]]) for slot in slots]
            page_table = ttnn.concat(parts, dim=0) if len(parts) > 1 else parts[0]
        ttnn.copy(page_table, self.page_table)
        return size

    def prepare(self):
        if self.traces:
            raise RuntimeError("Prepare before capturing any prefill trace")
        started = time.perf_counter()
        for slot in range(self.source_table.shape[0]):
            row = ttnn.slice(self.source_table, [slot, 0], [slot + 1, self.source_table.shape[1]])
            del row
        # Prepare every logical padding/slicing signature, not just aligned inputs.
        for length in range(1, 65):
            size = self.bucket(length)
            x = ttnn.clone(ttnn.slice(self.inputs[size], [0, 0, 0, 0], [self.batch, 1, length, 6656]))
            self.stage_inputs(x, self.source_table, slots=tuple(reversed(range(self.batch))))
            del x
        with self.capture_policy():
            for size, x in self.inputs.items():
                y = self.layer.prefill_forward(x, kv_cache=self.kv_cache, page_table=self.page_table)
                del y
        ttnn.synchronize_device(self.mesh)
        self.prepared = True
        self.prepare_ms = (time.perf_counter() - started) * 1000

    def capture(self):
        if not self.prepared:
            raise RuntimeError("Prepare all model variants before capture")
        started = time.perf_counter()
        with self.capture_policy():
            for size, x in self.inputs.items():
                if size in self.traces:
                    continue
                tid = ttnn.begin_trace_capture(self.mesh, cq_id=0)
                y = self.layer.prefill_forward(x, kv_cache=self.kv_cache, page_table=self.page_table)
                ttnn.end_trace_capture(self.mesh, tid, cq_id=0)
                # Only this exact output backing allocation is exempt. It has no
                # persistent input alias; consumers must finish before another trace.
                trace_allocation_tracker.acknowledge_corruptible(y)
                self.traces[size], self.outputs[size] = tid, y
                self.capture_count += 1
        ttnn.synchronize_device(self.mesh)
        self.capture_ms = (time.perf_counter() - started) * 1000

    def __call__(self, x, *, page_table, slots=None):
        length = x.shape[-2]
        size = self.stage_inputs(x, page_table, slots=slots)
        if size not in self.traces:
            raise RuntimeError("Capture the prepared bucket before request handling")
        ttnn.execute_trace(self.mesh, self.traces[size], cq_id=0, blocking=False)
        self.replay_count += 1
        return ttnn.slice(self.outputs[size], [0, 0, 0, 0], [self.batch, 1, length, 6656])

    def close(self):
        for tid in self.traces.values():
            ttnn.release_trace(self.mesh, tid)
            self.release_count += 1
        self.traces.clear()
        self.outputs.clear()
