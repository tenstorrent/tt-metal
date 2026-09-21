# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded pilot-only full-block timings and original BF16 QKV captures."""

import functools
import hashlib
import statistics
import time

import torch
import ttnn

from models.tt_dit.utils.tensor import to_torch


def map_tensors(value, fn):
    if isinstance(value, ttnn.Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(v, fn) for v in value)
    return value


class Diagnostics:
    def __init__(self, pipeline):
        self.device = pipeline.mesh_device
        self.enabled = False
        self.capture_enabled = False
        self.records = {}
        self.captures = {}
        self.output = None
        for expert, state in enumerate(pipeline.transformer_states):
            for index in (0, 20):
                block = state.model.blocks[index]
                name = f"expert{expert}.block{index}"
                block.forward = self.wrap(name, block.forward)
                block.attn1._attention_capture = functools.partial(self.capture, name)

    def reset(self, output, capture=False):
        self.records = {}
        self.captures = {}
        self.output = output
        self.capture_enabled = capture

    def capture(self, name, attn, q, k, v, logical_n):
        if not self.enabled or not self.capture_enabled or name in self.captures:
            return
        hosts = []
        for i, value in enumerate((q, k, v)):
            end = list(value.shape)
            end[1] = 2
            if i == 0:
                end[2] = 256
            selected = ttnn.slice(value, [0, 0, 0, 0], end)
            if i == 0:
                shards = ttnn.get_device_tensors(selected)
                host = torch.cat([ttnn.to_torch(shards[j]) for j in (0, 4)], dim=1)
            else:
                host = to_torch(selected, mesh_axes=[None, 0, 1, None])
            assert torch.isfinite(host).all()
            hosts.append(host)
        path = self.output / f"{name}-qkv.pt"
        torch.save(dict(zip(("q", "k", "v"), hosts), logical_n=logical_n), path)
        self.captures[name] = dict(
            file=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            heads=[0, 1, 20, 21],
            query_rows="first 256 global rows",
            logical_n=logical_n,
        )

    def wrap(self, name, original):
        @functools.wraps(original)
        def forward(*args, **kwargs):
            if not self.enabled or name in self.records:
                return original(*args, **kwargs)
            saved_args = map_tensors(args, ttnn.clone)
            saved_kwargs = map_tensors(kwargs, ttnn.clone)
            output = original(*args, **kwargs)
            reference = ttnn.to_torch(ttnn.get_device_tensors(output)[0]).clone()
            assert torch.isfinite(reference).all(), name
            original(*saved_args, **saved_kwargs)
            trace = ttnn.begin_trace_capture(self.device, cq_id=0)
            traced = original(*saved_args, **saved_kwargs)
            ttnn.end_trace_capture(self.device, trace, cq_id=0)
            times = []
            try:
                for _ in range(5):
                    ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)
                for _ in range(15):
                    start = time.perf_counter_ns()
                    ttnn.execute_trace(self.device, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter_ns() - start) / 1e6)
                actual = ttnn.to_torch(ttnn.get_device_tensors(traced)[0])
                assert torch.isfinite(actual).all(), name
                exact = torch.equal(actual, reference)
                difference = actual.float() - reference.float()
                l2 = 100 * float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference.float()))
                assert l2 < 0.01, (name, "block replay discrepancy", l2)
            finally:
                ttnn.release_trace(self.device, trace)
            self.records[name] = dict(
                median_ms=statistics.median(times),
                samples_ms=times,
                replay_exact=exact,
                replay_l2_pct=l2,
                warmup_replays=5,
                scope="Full block blocking mesh trace; variant's own two-step pilot inputs",
            )
            print("WAN_BLOCK", name, self.records[name], flush=True)
            return output

        return forward
