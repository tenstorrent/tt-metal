# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""No-model resident UINT32 selection correctness and warmed trace timing.

Device owner only, accepted lane/mesh descriptor restored, watcher/profilers off:
  TT_METAL_TRACE_ALLOC_TRACKING=1 TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0 \
  timeout --signal=INT --kill-after=30 180 python_env/bin/python -m \
  models.autoports.qwen_qwen3_8_flash_next.tests.probe_token_feedback \
  --tracked-correctness --output /tmp/token_feedback_tracked.json
Run separately with tracking0, no --tracked-correctness for clean timing.
No common sampler mathematics or model weights are involved. One candidate
trace handles every changed active mask; baseline retains one trace per mask.
"""
import argparse
import json
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracked-correctness", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    assert args.iterations > 0 and not args.output.exists()
    from models.autoports.qwen_qwen3_8_flash_next.tests.profile_terminal_row_assembly import preflight

    preflight(args.tracked_correctness)
    import torch
    from ttnn.tools import trace_allocation_tracker as tracker

    import ttnn
    from models.autoports.qwen_qwen3_8_flash_next.tt.device_seed import DeviceSeedState
    from models.autoports.qwen_qwen3_8_flash_next.tt.token_feedback import DeviceTokenFeedback

    if args.tracked_correctness:
        assert tracker.TRACE_ALLOC_TRACKING
    else:
        assert not tracker.TRACE_ALLOC_TRACKING and not tracker.TRACE_ALLOC_DIAGNOSTICS
    report = {
        "passed": False,
        "scope": "no_model_exact_uint32_feedback_candidate",
        "acceptance_eligible": False,
        "tracked_correctness": args.tracked_correctness,
        "checks": [],
        "timings_ms": {},
        "iterations": args.iterations,
    }
    started = time.monotonic()
    mesh, traces = None, {}
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4 * 1024 * 1024)

        def upload(value):
            return ttnn.from_torch(
                value.to(torch.uint32),
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        def refresh(target, value):
            host = ttnn.from_torch(value.to(torch.uint32), dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
            ttnn.copy_host_to_device_tensor(host, target)

        edge = [0, 1, 248319, 2**24 + 1, 2**31 - 1, 2**31, 2**32 - 2, 2**32 - 1]
        current_host = torch.tensor(edge * 4, dtype=torch.int64).reshape(1, 1, 1, 32)
        saved_host = torch.tensor([2**32 - 1 - i * 123457 for i in range(32)], dtype=torch.int64).reshape(1, 1, 1, 32)
        outputs = {name: upload(current_host) for name in ("baseline", "candidate")}
        saved = upload(saved_host)
        seed_output = upload(torch.zeros(32, dtype=torch.int64))
        seed = DeviceSeedState(mesh, seed_output)
        seed.prepare()
        candidate = DeviceTokenFeedback(outputs["candidate"], saved, seed.active)
        candidate.prepare()
        cases = {
            "none": (),
            "all": tuple(range(32)),
            "slot0": (0,),
            "slot31": (31,),
            "sparse": (1, 7, 16, 31),
            "even": tuple(range(0, 32, 2)),
            "odd": tuple(range(1, 32, 2)),
        }

        def baseline(slots):
            rows = [(outputs["baseline"] if i in slots else saved)[:, :, :, i : i + 1] for i in range(32)]
            ttnn.copy(ttnn.concat(rows, dim=3), outputs["baseline"])

        for slots in cases.values():
            baseline(slots)
        candidate.forward()
        ttnn.synchronize_device(mesh)
        count = mesh.num_program_cache_entries()
        addresses = [tensor.buffer_address() for tensor in candidate.tensors]

        def capture(key, body):
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            try:
                body()
            except BaseException:
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                ttnn.release_trace(mesh, trace)
                raise
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            traces[key] = trace

        capture("candidate", candidate.forward)
        for label, slots in cases.items():
            capture(label, lambda slots=slots: baseline(slots))

        def replay(key):
            ttnn.execute_trace(mesh, traces[key], cq_id=0, blocking=False)

        def read_all(tensor):
            # Return host copies only; no device shard views escape this read.
            return [ttnn.to_torch(part).flatten().long().tolist() for part in ttnn.get_device_tensors(tensor)]

        print("CHECK_CHANGED_MASK_REUSED_TRACE", flush=True)
        for cycle in range(2):
            for label, slots in cases.items():
                seed.set_active_slots(slots)
                live = (current_host + cycle * 13).remainder(2**32)
                expected = [int(live.flatten()[i]) if i in slots else int(saved_host.flatten()[i]) for i in range(32)]
                for name in outputs:
                    refresh(outputs[name], live)
                    replay("candidate" if name == "candidate" else label)
                    actual = read_all(outputs[name])
                    assert len(actual) == 4 and all(row == expected for row in actual), (cycle, label, name, actual)
                assert all(row == saved_host.flatten().tolist() for row in read_all(saved))
                assert all(row == [int(i in slots) for i in range(32)] for row in read_all(seed.active))
                report["checks"].append(
                    {"cycle": cycle, "case": label, "all_four_ranks_exact": True, "saved_and_active_unchanged": True}
                )
        assert mesh.num_program_cache_entries() == count
        assert addresses == [tensor.buffer_address() for tensor in candidate.tensors]
        if not args.tracked_correctness:

            def timing(key):
                for _ in range(3):
                    replay(key)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(args.iterations):
                    replay(key)
                ttnn.synchronize_device(mesh)
                return (time.perf_counter() - begin) * 1000 / args.iterations

            for label, slots in cases.items():
                seed.set_active_slots(slots)
                report["timings_ms"][label] = {"baseline": timing(label), "candidate": timing("candidate")}
            report[
                "timing_boundary"
            ] = "Warmed trace submissions and amortized boundary sync; no host uploads/reads inside window"
        report.update(
            program_cache_entries=count,
            candidate_trace_count=1,
            baseline_trace_count=len(cases),
            persistent_addresses=addresses,
            final_replay=True,
        )
        replay("candidate")
        ttnn.synchronize_device(mesh)
        report["passed"] = True
    except BaseException as error:
        report.update(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        report["wall_seconds"] = time.monotonic() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if mesh is not None:
            for trace in traces.values():
                ttnn.release_trace(mesh, trace)
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
