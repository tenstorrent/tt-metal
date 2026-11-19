# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Minimal serialized TP4 seed-state hardware proof (no model weights).

Run only after obtaining exclusive device ownership under tt-device-usage:

timeout 180 env TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
    PYTHONPATH=/home/ttuser/qwen38-flash/tt-metal python_env/bin/python \
    -m models.autoports.qwen_qwen3_8_flash_next.tests.probe_device_seed \
    --output models/autoports/qwen_qwen3_8_flash_next/doc/full_model/device_seed_device.json

This tests the seed buffer contract, not sampled-token distributions. The
output allocation exactly matches current common TTSampling's sampling_dp=1
seeds_tt_tensor: shape[32], UINT32, ROW_MAJOR, interleaved DRAM, mapper=None.
The script author did not run this device probe.
"""

import argparse
import json
import os
from pathlib import Path

from models.autoports.qwen_qwen3_8_flash_next.tt.device_seed import MASK32, MASK64, reset_words, seed_for_draw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING") != "1":
        parser.error("TT_METAL_TRACE_ALLOC_TRACKING=1 is required before imports")
    if os.environ.get("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "0":
        parser.error("Program-cache tracking must remain enabled")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.device_seed import DeviceSeedState

    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(4)
    report = {
        "passed": False,
        "mesh": [1, 4],
        "trace_tracking": True,
        "program_cache_tracked": True,
        "buffer_source": "shape/dtype/layout/mapper identical to common TTSampling sampling_dp=1",
        "sampler_token_distribution_tested": False,
        "checks": [],
    }
    mesh, trace = None, None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4 * 1024 * 1024)
        output = ttnn.from_torch(
            torch.arange(32).to(torch.uint32),
            device=mesh,
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        seed_state = DeviceSeedState(mesh, output)
        seed_state.prepare()
        seed_state.remap(list(range(32)), warmup=True)
        addresses = [tensor.buffer_address() for tensor in seed_state.tensors]
        rows, active = [], []

        def reset(seeds, counters, slots):
            seed_state.reset_slots(seeds, counters, slots)
            for seed, counter, slot in zip(seeds, counters, slots, strict=True):
                rows[slot] = reset_words(seed, counter)

        def initialize():
            nonlocal rows, active
            rows = [reset_words(0, 0) for _ in range(32)]
            reset([0] * 32, [0] * 32, list(range(32)))
            reset([-1, 2**63 + 123, 2**32 + 9], [MASK32, 7, 99], [0, 1, 2])
            active = [0, 2]
            seed_state.set_active_slots(active)

        def draw(label, *, traced):
            expected = [MASK32] * 32
            for slot in active:
                row = rows[slot]
                assert row[4] == 1, "This exact-formula probe uses explicit seeded rows"
                request_seed = row[0] | (row[1] << 32)
                counter = row[2] | (row[3] << 32)
                expected[slot] = seed_for_draw(request_seed, counter)
                counter = (counter + 1) & MASK64
                row[2], row[3] = counter & MASK32, counter >> 32
            if traced:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            else:
                seed_state.forward()
                ttnn.synchronize_device(mesh)
            device_outputs = ttnn.get_device_tensors(output)
            device_states = ttnn.get_device_tensors(seed_state.state)
            assert len(device_outputs) == len(device_states) == 4
            for rank, (seed_tensor, state_tensor) in enumerate(zip(device_outputs, device_states, strict=True)):
                actual = ttnn.to_torch(seed_tensor).flatten().to(torch.int64).tolist()
                actual_state = ttnn.to_torch(state_tensor).reshape(32, 16).to(torch.int64).tolist()
                assert actual == expected, (label, rank, "seed output", actual, expected)
                assert actual_state == rows, (label, rank, "seed state", actual_state, rows)
            assert addresses == [tensor.buffer_address() for tensor in seed_state.tensors]
            report["checks"].append(
                {"name": label, "traced": traced, "active_slots": list(active), "expected_seeds": expected, "ranks": 4}
            )

        initialize()
        draw("eager_warmup_negative_seed_and_counter_carry", traced=False)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        seed_state.forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        # Capture execution must not determine the logical test starting state.
        initialize()
        draw("trace_initial", traced=True)
        draw("trace_second_draw_inactive_preserved", traced=True)
        first_pair = [item["expected_seeds"] for item in report["checks"][-2:]]

        remap = [2, 0, 1] + list(range(3, 32))
        seed_state.remap(remap)
        rows = [rows[source] for source in remap]
        active = [0, 1]
        seed_state.set_active_slots(active)
        draw("trace_after_three_slot_cycle", traced=True)
        draw("trace_second_after_cycle", traced=True)

        reset([-(2**63) + 9], [MASK32], [2])
        active = [1, 2]
        seed_state.set_active_slots(active)
        draw("trace_partial_reset_counter_carry_other_slots_preserved", traced=True)
        draw("trace_after_partial_reset", traced=True)

        initialize()
        draw("trace_repeat_initial_after_reset", traced=True)
        draw("trace_repeat_second_after_reset", traced=True)
        assert [item["expected_seeds"] for item in report["checks"][-2:]] == first_pair
        report["passed"] = True
        report["trace_replays"] = sum(item["traced"] for item in report["checks"])
        report["persistent_addresses_unchanged"] = True
    except BaseException as error:
        report["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            if trace is not None:
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)
    print(json.dumps({"passed": report["passed"], "output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
