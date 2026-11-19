# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Standalone B2 host-logit layout diagnostic; no model weights or sampler.

Requires exclusive device ownership. Source-only preparation is safe: TTNN and
Torch are imported inside main only after command-line/environment validation.
The capture owns its retained ROW_MAJOR output; no output_tensor argument exists.
"""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Preserve existing evidence; choose a fresh output")
    for name, value in (("TT_METAL_TRACE_ALLOC_TRACKING", "1"), ("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")):
        if os.environ.get(name) != value:
            parser.error(f"Set {name}={value} before launch")
    if os.environ.get("TT_METAL_DEVICE_PROFILER") or os.environ.get("TT_METAL_WATCHER"):
        parser.error("Use an ordinary exclusive run with profiler and watcher unset")
    import torch
    from ttnn.tools import trace_allocation_tracker

    import ttnn

    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "doc/functional_decoder/target_text_config.json").read_text())
    selected = json.loads((root / "doc/datatype_sweep/selected_precision_config.json").read_text())
    # Accept only this diagnostic's selected BF16 sampling storage contract.
    sampling_dtype = selected.get("sampling_dtype", selected.get("policy", {}).get("sampling_dtype"))
    assert sampling_dtype == "bfloat16", selected.keys()
    vocab, ranks, batch = config["vocab_size"], 4, 2
    assert vocab == 248320 and vocab % ranks == 0
    width = vocab // ranks
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    report = dict(
        passed=False,
        scope="synthetic TP4 B2 host-output read/layout diagnostic; no model/sampler or serving/P2 claim",
        sampling_dtype=sampling_dtype,
        input_shape_per_rank=[1, 1, 32, width],
        output_shape_per_rank=[1, 1, batch, width],
        output_tensor_end=[0, 0, batch - 1, width - 1],
        allocation_tracking=True,
        skip_program_cache=False,
        captures=0,
        releases=0,
        runtime_retirements=0,
        signatures=[],
        checks=[],
        measurements=[],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations="Unpadding retains physical slots0/1 only. This cannot handle an active slot>=2 without a different prepared extent or explicit gather. Device model latency, sampler and end-to-end P2 are excluded.",
    )
    mesh = trace = captured = None
    started = time.monotonic()
    try:
        torch.set_num_threads(8)
        assert torch.get_default_dtype() == torch.float32
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, ranks), trace_region_size=0, l1_small_size=32768)
        mapper = ttnn.ShardTensorToMesh(mesh, dim=3)
        columns = torch.arange(vocab).reshape(1, 1, 1, vocab)
        row = torch.arange(32).reshape(1, 1, 32, 1)
        base = ((columns // width) * 128 + row * 2 + (columns % 8) / 8 + 1).bfloat16()
        tensors = [base, -base]
        hosts = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=mapper) for x in tensors]
        source = ttnn.from_torch(
            tensors[0],
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            mesh_mapper=mapper,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
        )
        assert list(source.shape) == [1, 1, 32, width], source.shape

        def addresses(tensor):
            return [part.buffer_address() for part in ttnn.get_device_tensors(tensor)]

        def allocation():
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return view.total_bytes_allocated_per_bank * view.num_banks

        def compose(host):
            return ttnn.to_torch(host, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3)).reshape(-1, vocab)

        def format_rows(values, slots, active_rows):
            result = torch.zeros(len(slots), 1, vocab)
            for row_index in active_rows:
                result[row_index, 0] = values[slots[row_index]]
            return result

        def check(actual, expected, label):
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32)), label
            report["checks"].append(label)

        def untilize():
            return ttnn.untilize_with_unpadding(
                source,
                (0, 0, batch - 1, width - 1),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                use_multicore=True,
            )

        # Prepare every device operation and host conversion before capture.
        warm = untilize()
        ttnn.synchronize_device(mesh)
        assert list(warm.shape) == [1, 1, batch, width]
        assert warm.layout == ttnn.ROW_MAJOR_LAYOUT
        check(compose(warm.cpu()).float(), tensors[0].reshape(32, vocab)[:batch].float(), "eager_untilize_exact")
        check(compose(source.cpu()).float(), tensors[0].reshape(32, vocab).float(), "padded_read_exact")
        ttnn.copy_host_to_device_tensor(hosts[1], source)
        ttnn.copy_host_to_device_tensor(hosts[0], source)
        ttnn.synchronize_device(mesh)
        ttnn.deallocate(warm)
        del warm
        programs_before = mesh.num_program_cache_entries()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        captured = untilize()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        report["captures"] = 1
        report["signatures"] = [
            dict(
                op="untilize_with_unpadding",
                shape=[1, 1, 32, width],
                end=report["output_tensor_end"],
                dtype="bfloat16",
                layout="TILE_to_ROW_MAJOR",
                cq=0,
            )
        ]
        assert mesh.num_program_cache_entries() == programs_before
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        stable_input, stable_output = addresses(source), addresses(captured)
        stable_allocation = allocation()
        report["persistent_addresses"] = dict(input=stable_input, output=stable_output)
        report["program_cache_entries"] = programs_before

        # All B2 row orders/subsets, B1 slot1, and fully inactive outputs.
        cases = [
            ([0, 1], [0, 1]),
            ([1, 0], [0, 1]),
            ([1, 0], [1, 0]),
            ([0, 1], [1]),
            ([1, 0], [0]),
            ([1, 0], []),
            ([1], [0]),
        ]
        for epoch in (0, 1, 0):
            ttnn.copy_host_to_device_tensor(hosts[epoch], source)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            padded = compose(source.cpu(blocking=True)).float()
            compact = compose(captured.cpu(blocking=True)).float()
            check(padded, tensors[epoch].reshape(32, vocab).float(), f"input_epoch{epoch}")
            for slots, active in cases:
                check(
                    format_rows(compact, slots, active),
                    format_rows(padded, slots, active),
                    f"format_epoch{epoch}_slots{slots}_active{active}",
                )

        # Queue two independent host reads before waiting; source/output are
        # overwritten between submissions on the same CQ. Never alias a host
        # handle to the persistent output or finalize a read a second time.
        pending = []
        for epoch, slots, active in ((0, (1, 0), (0, 1)), (1, (0, 1), (1,))):
            ttnn.copy_host_to_device_tensor(hosts[epoch], source)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            host = captured.cpu(blocking=False)
            event = ttnn.record_event(mesh, 0)
            pending.append((host, event, epoch, slots, active))
        assert pending[0][0] is not pending[1][0]
        for host, event, epoch, slots, active in reversed(pending):
            ttnn.event_synchronize(event)
            check(
                format_rows(compose(host).float(), slots, active),
                format_rows(tensors[epoch].reshape(32, vocab).float(), slots, active),
                f"queued_snapshot_epoch{epoch}",
            )
        ttnn.copy_host_to_device_tensor(hosts[0], source)
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        for host, _, epoch, slots, active in pending:
            check(
                format_rows(compose(host).float(), slots, active),
                format_rows(tensors[epoch].reshape(32, vocab).float(), slots, active),
                f"host_snapshot_still_immutable_epoch{epoch}",
            )
        report["queued_before_wait"] = 2

        for repeat in range(args.repeats):
            order = ("padded", "traced_compact") if repeat % 2 == 0 else ("traced_compact", "padded")
            for mode in order:
                ttnn.copy_host_to_device_tensor(hosts[repeat % 2], source)
                ttnn.synchronize_device(mesh)  # Exclude input upload from measurement.
                begin = time.perf_counter_ns()
                if mode == "traced_compact":
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                submitted = time.perf_counter_ns()
                host = (captured if mode == "traced_compact" else source).cpu(blocking=True)
                read = time.perf_counter_ns()
                cpu_values = compose(host)
                composed = time.perf_counter_ns()
                values = cpu_values.float()
                cast = time.perf_counter_ns()
                result = format_rows(values, [1, 0], [0, 1])
                done = time.perf_counter_ns()
                report["measurements"].append(
                    dict(
                        mode=mode,
                        repeat=repeat,
                        trace_submit_ms=(submitted - begin) / 1e6,
                        cpu_blocking_read_ms=(read - submitted) / 1e6,
                        host_compose_ms=(composed - read) / 1e6,
                        float_cast_ms=(cast - composed) / 1e6,
                        format_ms=(done - cast) / 1e6,
                        total_ms=(done - begin) / 1e6,
                    )
                )
                assert result.shape == (2, 1, vocab)
                assert addresses(source) == stable_input and addresses(captured) == stable_output
                assert allocation() == stable_allocation
                assert mesh.num_program_cache_entries() == programs_before
        report["summary_median_ms"] = {
            mode: {
                key: statistics.median(r[key] for r in report["measurements"] if r["mode"] == mode)
                for key in (
                    "trace_submit_ms",
                    "cpu_blocking_read_ms",
                    "host_compose_ms",
                    "float_cast_ms",
                    "format_ms",
                    "total_ms",
                )
            }
            for mode in ("padded", "traced_compact")
        }
        report["stable_device_bytes_per_rank"] = stable_allocation
        report["stable_storage_and_program_cache"] = True
        report["passed"] = True
    except Exception:
        report["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if mesh is not None:
                if trace is not None:
                    ttnn.synchronize_device(mesh)
                    ttnn.release_trace(mesh, trace)
                    report["releases"] += 1
                ttnn.close_mesh_device(mesh)
                report["mesh_close_completed"] = True
        except Exception:
            report["passed"] = False
            report["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            report["seconds"] = time.monotonic() - started
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x") as handle:
                json.dump(report, handle, indent=2)
                handle.write("\n")
    print(json.dumps({key: report[key] for key in ("passed", "captures", "releases", "summary_median_ms")}))


if __name__ == "__main__":
    main()
