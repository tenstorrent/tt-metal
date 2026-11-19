# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Synthetic SwiGLU multicast -> native zeroing kernel-boundary accounting probe.

CPU preparation: --prepare-only /path/to/manifest.json. Hardware execution needs
an external timeout and exclusive ownership of the 1x4 mesh. Run baseline and
--final-ack-barrier in separate clean device lifecycles. The default bounded
diagnostic uses zero_pages' native operations and reports ACK overshoot/timeout
as FAILURE; --zero-mode native invokes the unmodified zero_pages helper and can
hang until the external timeout. Neither path is a serving performance test.
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]
KERNELS = Path(__file__).resolve().parent / "kernels"
NATIVE_HELPER = (
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"
)
SHAPE = (1, 1, 32, 8, 2048)
PARTITIONS = ((0, 52), (52, 103), (103, 154), (154, 205), (205, 256))
EMPTY = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
FIELDS = {
    0: "a_magic",
    1: "a_noc",
    2: "a_entry_sw_sent",
    3: "a_entry_sw_ack",
    4: "a_entry_hw_sent",
    5: "a_entry_hw_ack",
    6: "a_exit_sw_sent",
    7: "a_exit_sw_ack",
    8: "a_exit_hw_sent",
    9: "a_pre_drain_hw_ack",
    10: "a_post_drain_hw_ack",
    11: "a_status",
    12: "mcast_x0",
    13: "mcast_y0",
    14: "mcast_x1",
    15: "mcast_y1",
    16: "receiver_poll_count",
    32: "b_magic",
    33: "b_noc",
    34: "b_entry_sw_sent",
    35: "b_entry_sw_ack",
    36: "b_entry_hw_sent",
    37: "b_entry_hw_ack",
    38: "b_exit_sw_sent",
    39: "b_exit_sw_ack",
    40: "b_exit_hw_sent",
    41: "b_exit_hw_ack",
    42: "b_poll_count",
    43: "b_status",
    44: "page_start",
    45: "page_end",
    46: "output_address",
    47: "scratch_address",
}


def source_manifest():
    files = [
        KERNELS / "vllm_noc_boundary_multicast.cpp",
        KERNELS / "vllm_noc_boundary_zero.cpp",
        REPO / NATIVE_HELPER,
        REPO / "tt_metal/hw/inc/internal/noc_zero_dram.inl",
        REPO / "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h",
    ]
    return {str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def make_programs(ttnn, output, diagnostic, *, final_ack_barrier, poll_limit, native_helper):
    def grid(x0, y0, x1, y1):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))])

    all_cores, sender, untilizers = grid(0, 0, 7, 7), grid(0, 0, 0, 0), grid(1, 0, 4, 0)

    def kernel(source, cores, config, *, compile_args=(), runtime=None, file=False):
        return ttnn.KernelDescriptor(
            kernel_source=str(source),
            source_type=(
                ttnn.KernelDescriptor.SourceType.FILE_PATH if file else ttnn.KernelDescriptor.SourceType.SOURCE_CODE
            ),
            core_ranges=cores,
            config=config,
            compile_time_args=list(compile_args),
            runtime_args=runtime if runtime is not None else ttnn.RuntimeArgs(),
        )

    rt_a = ttnn.RuntimeArgs()
    for y in range(8):
        for x in range(8):
            rt_a[x][y] = [int(x == 0 and y == 0), diagnostic.buffer_address(), 0]
    a = ttnn.ProgramDescriptor(
        kernels=[
            kernel(
                KERNELS / "vllm_noc_boundary_multicast.cpp",
                all_cores,
                ttnn.ReaderConfigDescriptor(),
                compile_args=[int(final_ack_barrier), poll_limit],
                runtime=rt_a,
                file=True,
            ),
            kernel(EMPTY, all_cores, ttnn.WriterConfigDescriptor()),
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=4096,
                core_ranges=all_cores,
                format_descriptors=[ttnn.CBFormatDescriptor(0, ttnn.uint32, 4096)],
            )
        ],
        semaphores=[ttnn.SemaphoreDescriptor(id=0, core_ranges=all_cores, initial_value=0)],
    )
    accessor = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
    b_kernels = []
    for cores, scratch, indices, config in (
        (sender, 7, (0,), ttnn.ReaderConfigDescriptor()),
        (untilizers, 6, range(1, 5), ttnn.WriterConfigDescriptor()),
    ):
        runtime = ttnn.RuntimeArgs()
        for index in indices:
            runtime[index][0] = [output.buffer_address(), *PARTITIONS[index], diagnostic.buffer_address()]
        b_kernels.append(
            kernel(
                KERNELS / "vllm_noc_boundary_zero.cpp",
                cores,
                config,
                compile_args=[scratch, poll_limit, int(native_helper), *accessor],
                runtime=runtime,
                file=True,
            )
        )
    b_kernels += [
        kernel(EMPTY, sender, ttnn.WriterConfigDescriptor()),
        kernel(EMPTY, untilizers, ttnn.ReaderConfigDescriptor()),
    ]
    b = ttnn.ProgramDescriptor(
        kernels=b_kernels,
        cbs=[
            ttnn.CBDescriptor(
                total_size=16384,
                core_ranges=cores,
                format_descriptors=[ttnn.CBFormatDescriptor(scratch, ttnn.uint32, 16384)],
            )
            for cores, scratch in ((sender, 7), (untilizers, 6))
        ],
    )
    return a, b


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", type=Path)
    parser.add_argument("--final-ack-barrier", action="store_true")
    parser.add_argument("--zero-mode", choices=("bounded", "native"), default="bounded")
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--poll-limit", type=int, default=10_000_000)
    parser.add_argument("--trace-region-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/noc_boundary_probe/result.json")
    args = parser.parse_args()
    if not 1 <= args.iterations <= 1_000_000 or not 1 <= args.poll_limit < 2**31:
        parser.error("iterations must be 1..1000000; poll-limit must be 1..2^31-1")
    result = dict(
        status="PREPARED" if args.prepare_only else "RUNNING",
        command=sys.argv,
        experiment="synthetic 8x8 root nonposted multicast -> native zero APIs across kernel boundary",
        mesh=[1, 4],
        multicast_grid=[8, 8],
        num_destinations=63,
        multicast_bytes=[1024, 1024, 1024],
        final_ack_barrier=args.final_ack_barrier,
        zero_mode=args.zero_mode,
        poll_limit=args.poll_limit,
        iterations_requested=args.iterations,
        iterations_completed=0,
        captures=0,
        releases=0,
        sources=source_manifest(),
        status_codes={1: "complete", 2: "ACK overshoot: FAIL", 3: "poll timeout: FAIL"},
        diagnostics_layout="uint32 HEIGHT_SHARDED 64x128, one row per core, row-major core order",
        diagnostic_fields=FIELDS,
        observations=[],
        limitations=[
            "Synthetic count broadcast; no full SwiGLU arithmetic, data-dependent routing, model, sampler or fabric traffic.",
            "63 destinations use the device firmware logical-to-virtual map; no physical core coordinates are hardcoded.",
            "Native mode calls unmodified zero_pages; bounded mode composes its identical native zero APIs but replaces only the final wait with a failure-reporting bounded poll.",
            "A failed bounded wait returns an explicitly invalid diagnostic result; Python stops immediately and never accepts zero output as a pass.",
            "Read/sent/native barriers and device teardown can still stall; the external process timeout is mandatory.",
            "Counter snapshots use on-device normal register reads and local stores to an owned sharded tensor; no host MMIO reads.",
            "Snapshots and receiver waits perturb timing. Failure is diagnostic; a pass does not disprove the full-model boundary hypothesis.",
            "Host observes each complete A->B pair, introducing synchronization between pairs but never between A and B in a retained replay.",
            "No performance or full-model correctness claim; compare baseline and epilogue-drain variants in separate clean lifecycles.",
        ],
    )
    destination = args.prepare_only or args.output
    destination.parent.mkdir(parents=True, exist_ok=True)

    def save():
        destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    if args.prepare_only:
        print(f"CPU_PREPARED {destination}")
        return

    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    if (
        os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1"
        or os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "0"
    ):
        raise RuntimeError("Requires TT_METAL_TRACE_ALLOC_TRACKING=1 and TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE=0")
    import torch

    import ttnn

    mesh = trace = None
    tensors = []
    capture_open = False
    try:
        torch.set_num_threads(4)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        size = mesh.compute_with_storage_grid_size()
        assert size.x >= 8 and size.y >= 8, size
        cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
        diag_memory = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(cores, (1, 128), ttnn.ShardOrientation.ROW_MAJOR),
        )
        diagnostic = ttnn.from_torch(
            torch.zeros((64, 128), dtype=torch.int32),
            dtype=ttnn.uint32,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=diag_memory,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        tensors.append(diagnostic)
        poison = ttnn.from_torch(
            torch.ones(SHAPE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        tensors.append(poison)
        output = ttnn.empty_like(poison)
        tensors.append(output)
        a, b = make_programs(
            ttnn,
            output,
            diagnostic,
            final_ack_barrier=args.final_ack_barrier,
            poll_limit=args.poll_limit,
            native_helper=args.zero_mode == "native",
        )
        result.update(diagnostic_address=diagnostic.buffer_address(), output_address=output.buffer_address())
        save()

        def execute():
            ttnn.generic_op([poison, diagnostic, output], a)
            ttnn.generic_op([poison, diagnostic, output], b)

        def check(label):
            observation = dict(label=label, chips=[])
            failed = False
            for chip_index, chip in enumerate(ttnn.get_device_tensors(diagnostic)):
                data = ttnn.to_torch(chip).reshape(64, 128).to(torch.int64)
                a_status = [int(v) for v in data[:, 11]]
                rows = [
                    dict(core=[x, 0], **{name: int(data[x, index]) for index, name in FIELDS.items()}) for x in range(5)
                ]
                failed |= any(v != 1 for v in a_status) or any(row["b_status"] != 1 for row in rows)
                root = rows[0]
                root["a_outstanding_acks_at_sent_flush_mod32"] = (
                    root["a_exit_sw_ack"] - root["a_pre_drain_hw_ack"]
                ) % 2**32
                root["b_entry_ack_delta_mod32"] = (root["b_entry_hw_ack"] - root["b_entry_sw_ack"]) % 2**32
                observation["chips"].append(dict(chip=chip_index, a_status_all_64=a_status, zeroing_cores=rows))
            # Persist raw accounting before raising or attempting any further device operation.
            result["last_observation"] = observation
            if failed or label in ("warmup", "prepared") or result["iterations_completed"] % 100 == 0:
                result["observations"].append(observation)
            save()
            if failed:
                raise AssertionError(
                    f"{label}: kernel accounting failure; inspect last_observation; zero output is invalid"
                )
            for chip_index, chip in enumerate(ttnn.get_device_tensors(output)):
                data = ttnn.to_torch(chip)
                if not bool(torch.all(data == 0)):
                    raise AssertionError(f"{label}: chip {chip_index} zero output differs")

        ttnn.copy(poison, output)
        execute()
        check("warmup")
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        capture_open = True
        execute()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        capture_open = False
        result.update(captures=1, trace_id=str(trace))
        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
        check("prepared")
        entries = int(mesh.num_program_cache_entries())
        result["prepared_program_entries"] = entries
        started = time.monotonic()
        for iteration in range(1, args.iterations + 1):
            ttnn.copy(poison, output)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            check(f"iteration_{iteration}")
            result["iterations_completed"] = iteration
            if iteration % 100 == 0 or iteration == args.iterations:
                assert int(mesh.num_program_cache_entries()) == entries
                result["seconds"] = time.monotonic() - started
                save()
                print(
                    f"NOC_BOUNDARY_PASS iterations={iteration} final_ack_barrier={args.final_ack_barrier}", flush=True
                )
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        save()
        raise
    finally:
        try:
            if mesh is not None:
                if capture_open:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                    result["releases"] += 1
                for tensor in reversed(tensors):
                    ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException:
            result["status"] = "FAIL"
            result["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
