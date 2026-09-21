# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Capture real block inputs before model trace capture; time isolated traces."""

import functools
import statistics
import time

import torch
import ttnn

TARGETS = {"dual.0", "dual.3", "dual.7", "single.0", "single.23", "single.47"}


def map_tensors(value, function):
    if isinstance(value, ttnn.Tensor):
        return function(value)
    if isinstance(value, tuple):
        return tuple(map_tensors(x, function) for x in value)
    if isinstance(value, list):
        return [map_tensors(x, function) for x in value]
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    return value


def first_device_outputs(value):
    result = []

    def collect(tensor):
        host = ttnn.to_torch(ttnn.get_device_tensors(tensor)[0]).clone()
        assert bool(torch.isfinite(host).all())
        result.append(host)
        return tensor

    map_tensors(value, collect)
    return result


class BlockBench:
    def __init__(self):
        self.enabled = False
        self.captures = {}

    def install(self, transformer):
        for family, blocks in (
            ("dual", transformer.transformer_blocks),
            ("single", transformer.single_transformer_blocks),
        ):
            for index, block in enumerate(blocks):
                name = f"{family}.{index}"
                if name in TARGETS:
                    block.forward = self.wrap(name, block.forward)

    def wrap(self, name, original):
        @functools.wraps(original)
        def forward(*args, **kwargs):
            if self.enabled and name not in self.captures:
                # These buffers predate trace capture, so later model trace
                # allocations cannot overwrite the captured benchmark inputs.
                self.captures[name] = (original, map_tensors(args, ttnn.clone), map_tensors(kwargs, ttnn.clone))
            return original(*args, **kwargs)

        return forward

    def benchmark(self, device, warmup=250, iterations=50, strict=True):
        if set(self.captures) != TARGETS:
            raise ValueError(f"Incomplete block captures: {sorted(self.captures)}")
        records = []
        for name in sorted(self.captures):
            original, args, kwargs = self.captures[name]
            output = original(*args, **kwargs)
            expected = first_device_outputs(output)
            trace = ttnn.begin_trace_capture(device, cq_id=0)
            traced_output = original(*args, **kwargs)
            ttnn.end_trace_capture(device, trace, cq_id=0)
            times = []
            replay_checks = []

            def check_replay(phase):
                for part, (actual, ref) in enumerate(zip(first_device_outputs(traced_output), expected, strict=True)):
                    exact = torch.equal(actual, ref)
                    check = dict(phase=phase, part=part, exact=exact)
                    if not exact:
                        difference = actual.float() - ref.float()
                        check.update(
                            l2_pct=100
                            * float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref.float())),
                            max_abs=float(difference.abs().max()),
                            unequal=int(torch.count_nonzero(actual != ref)),
                        )
                        print("BLOCK_REPLAY_DIFFERENCE", name, check, flush=True)
                    replay_checks.append(check)
                    if strict:
                        assert exact, f"Block replay differs for {name}: {check}"

            try:
                for _ in range(warmup):
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                check_replay("before_timing")
                for _ in range(iterations):
                    start = time.perf_counter_ns()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    times.append((time.perf_counter_ns() - start) / 1e6)
                check_replay("after_timing")
            finally:
                ttnn.release_trace(device, trace)
            record = dict(
                block=name,
                warmup_replays=warmup,
                median_ms=statistics.median(times),
                min_ms=min(times),
                max_ms=max(times),
                replay_ms=times,
                first_device_output_bitwise_equal=all(c["exact"] for c in replay_checks),
                replay_checks=replay_checks,
                scope="Isolated full block, blocking mesh trace replay; includes attention preprocessing and CCL; excludes input capture and compilation",
            )
            records.append(record)
            print("BLOCK_BENCH", record, flush=True)
        return records
