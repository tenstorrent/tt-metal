# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Exact UINT32 TP4 trace probe for the proposed deferred token-history append.

This tests a data-movement contract, not model accuracy or sampling quality.
Run only after exclusive hardware access and mesh health are established.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn


def probe(mesh, capacity):
    def upload(value):
        return ttnn.from_torch(
            value,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            device=mesh,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )

    # Match the generator's 32 physical token lanes without inventing active users.
    seeds = [torch.arange(32, dtype=torch.int32).reshape(1, 1, 1, 32) + offset for offset in (120000, 130000)]
    seed_buffers = [upload(seed) for seed in seeds]
    tokens = upload(seeds[0])
    empty = upload(torch.zeros(capacity, 1, 1, 32, dtype=torch.int32))
    history = upload(torch.zeros(capacity, 1, 1, 32, dtype=torch.int32))
    cursor = upload(torch.zeros(1, dtype=torch.int32))
    cursor_zero = upload(torch.zeros(1, dtype=torch.int32))
    token_backup = upload(seeds[0])
    cursor_backup = upload(torch.zeros(1, dtype=torch.int32))
    persistent = dict(
        tokens=tokens, history=history, cursor=cursor, token_backup=token_backup, cursor_backup=cursor_backup
    )

    def addresses():
        return {
            name: [shard.buffer_address() for shard in ttnn.get_device_tensors(tensor)]
            for name, tensor in persistent.items()
        }

    original_addresses = addresses()

    def reset(seed, *, clear_history=True):
        ttnn.copy(seed, tokens)
        if clear_history:
            ttnn.copy(empty, history)
        ttnn.copy(cursor_zero, cursor)

    def append():
        # Stand-in for device sampling mutating the persistent input token.
        ttnn.plus_one(tokens)
        updated = ttnn.indexed_fill(cursor, history, tokens, dim=0)
        ttnn.copy(updated, history)
        ttnn.plus_one(cursor)

    def check_state(expected_history, expected_tokens, expected_cursor):
        assert addresses() == original_addresses, "Persistent device buffer addresses changed"
        for shard in ttnn.get_device_tensors(history):
            actual = ttnn.to_torch(shard).to(torch.int64)
            assert torch.equal(actual, expected_history), (capacity, actual, expected_history)
        for shard in ttnn.get_device_tensors(tokens):
            assert torch.equal(ttnn.to_torch(shard).to(torch.int64), expected_tokens)
        for shard in ttnn.get_device_tensors(cursor):
            assert ttnn.to_torch(shard).item() == expected_cursor

    # Warm every variant, including reset copies, before capture.
    reset(seed_buffers[0])
    append()
    reset(seed_buffers[0])
    ttnn.synchronize_device(mesh)
    trace = None
    capturing = False
    try:
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capturing = True
        append()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capturing = False
        runs = []
        for seed, seed_buffer in zip(seeds, seed_buffers):
            reset(seed_buffer)
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(capacity):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - begin

            expected = seed.to(torch.int64) + torch.arange(1, capacity + 1).reshape(capacity, 1, 1, 1)
            check_state(expected, seed.to(torch.int64) + capacity, capacity)
            runs.append(dict(first_seed=int(seed.flatten()[0]), append_us=elapsed * 1e6 / capacity, exact=True))

        # Reuse the same trace for a shorter request, resetting only the history
        # cursor. Seed refresh is request input; the old history tail must survive.
        shorter_steps = capacity // 2
        retained_history = expected.clone()
        reset(seed_buffers[0], clear_history=False)
        for _ in range(shorter_steps):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        retained_history[:shorter_steps] = seeds[0].to(torch.int64) + torch.arange(1, shorter_steps + 1).reshape(
            shorter_steps, 1, 1, 1
        )
        check_state(retained_history, seeds[0].to(torch.int64) + shorter_steps, shorter_steps)
        shorter_reset = dict(steps=shorter_steps, history_cleared=False, unwritten_tail_preserved=True, exact=True)

        capture_restore = dict(skipped=capacity == 1, reason="Cursor 1 is full at capacity 1")
        if capacity > 1:
            ttnn.release_trace(mesh, trace)
            trace = None
            reset(seed_buffers[0])
            append()  # One real output exists before warmup/capture.
            ttnn.copy(tokens, token_backup)
            ttnn.copy(cursor, cursor_backup)
            append()  # Warmup writes the next unused row and advances both inputs.
            ttnn.copy(token_backup, tokens)
            ttnn.copy(cursor_backup, cursor)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            capturing = True
            append()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            capturing = False
            # No reset after capture: the restored cursor/token must govern replay.
            for _ in range(capacity - 1):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            expected = seeds[0].to(torch.int64) + torch.arange(1, capacity + 1).reshape(capacity, 1, 1, 1)
            check_state(expected, seeds[0].to(torch.int64) + capacity, capacity)
            capture_restore = dict(start_cursor=1, replay_steps=capacity - 1, first_row_preserved=True, exact=True)

        return dict(
            capacity=capacity,
            logical_bytes_per_device=capacity * 32 * 4,
            history_shape=[capacity, 1, 1, 32],
            token_shape=[1, 1, 1, 32],
            dtype="uint32",
            replay_blocking=False,
            per_step_host_reads=0,
            per_step_host_writes=0,
            persistent_addresses=original_addresses,
            persistent_addresses_unchanged=True,
            shorter_cursor_reset=shorter_reset,
            capture_cursor_restore=capture_restore,
            runs=runs,
        )
    finally:
        if trace is not None:
            if capturing:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
            ttnn.release_trace(mesh, trace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capacities", nargs="+", type=int, default=[1, 31, 32, 33, 127, 128, 129, 257])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(capacity < 1 for capacity in args.capacities):
        parser.error("History capacities must be positive")
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16000000)
    try:
        result = dict(mesh=[1, 4], scope="Synthetic data-movement contract only", cases=[])
        for capacity in args.capacities:
            result["cases"].append(probe(mesh, capacity))
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("HISTORY_CASE_PASSED", capacity, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
