# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated native combine zero_pages stress probe; NOT a full-combine reproduction.

Run in the model lane environment with an external timeout, after releasing all
other device owners. The default performs 5000 retained-trace replays on a 1x4
mesh. No model, cache, sampler, server, or fabric collective is exercised.
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
NATIVE_HELPER = (
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"
)
SHAPE = (1, 1, 32, 8, 2048)
PAGE_BYTES = 4096
PAGES = 256
SCRATCH_BYTES = 16384
PARTITIONS = ((0, 52), (52, 103), (103, 154), (154, 205), (205, 256))
EMPTY = '#include "api/dataflow/dataflow_api.h"\nvoid kernel_main() {}\n'
SOURCE = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine/device/kernels/dataflow/zero_init_common.hpp"
void kernel_main() {
    constexpr uint32_t scratch_cb = get_compile_time_arg_val(0);
    const auto output = TensorAccessor(TensorAccessorArgs<1>(), get_arg_val<uint32_t>(0));
    const uint32_t page_start = get_arg_val<uint32_t>(1);
    const uint32_t page_end = get_arg_val<uint32_t>(2);
    zero_pages(scratch_cb, page_start, page_end, 4096, output);
}
"""


def make_program(ttnn, output):
    """Match zeroing roles: sender NCRISC/NOC0, four untilizers BRISC/NOC1."""

    def cores(first, last):
        return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(first, 0), ttnn.CoreCoord(last, 0))])

    sender, untilizers, all_cores = cores(0, 0), cores(1, 4), cores(0, 4)
    accessor = list(ttnn.TensorAccessorArgs(output).get_compile_time_args())

    def kernel(source, grid, config, scratch=None, indices=()):
        runtime = ttnn.RuntimeArgs()
        for index in indices:
            runtime[index][0] = [output.buffer_address(), *PARTITIONS[index]]
        return ttnn.KernelDescriptor(
            kernel_source=source,
            source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
            core_ranges=grid,
            config=config,
            compile_time_args=[] if scratch is None else [scratch, *accessor],
            runtime_args=runtime,
        )

    return ttnn.ProgramDescriptor(
        kernels=[
            kernel(SOURCE, sender, ttnn.ReaderConfigDescriptor(), 7, (0,)),
            kernel(EMPTY, sender, ttnn.WriterConfigDescriptor()),
            kernel(EMPTY, untilizers, ttnn.ReaderConfigDescriptor()),
            kernel(SOURCE, untilizers, ttnn.WriterConfigDescriptor(), 6, range(1, 5)),
            kernel(
                '#include "api/compute/common.h"\nvoid kernel_main() {}\n',
                all_cores,
                ttnn.ComputeConfigDescriptor(),
            ),
        ],
        cbs=[
            ttnn.CBDescriptor(
                total_size=SCRATCH_BYTES,
                core_ranges=grid,
                format_descriptors=[ttnn.CBFormatDescriptor(index, ttnn.uint32, SCRATCH_BYTES)],
            )
            for grid, index in ((sender, 7), (untilizers, 6))
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--trace-region-size", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--mode", choices=("trace", "eager"), default="trace")
    parser.add_argument("--output", type=Path, default=ROOT / "readiness_vllm/zeroing_probe/result.json")
    args = parser.parse_args()
    if args.iterations < 5000 or args.iterations > 1_000_000:
        parser.error("--iterations must be between 5000 and 1000000")
    if not 1 <= args.progress_every <= args.iterations:
        parser.error("--progress-every must be between 1 and --iterations")
    if args.trace_region_size <= 0:
        parser.error("--trace-region-size must be positive")

    os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
    os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")
    if os.environ["TT_METAL_TRACE_ALLOC_TRACKING"] != "1":
        raise RuntimeError("TT_METAL_TRACE_ALLOC_TRACKING must be 1")
    if os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] != "0":
        raise RuntimeError("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE must be 0")

    # Lazy imports let --help and CPU syntax validation run without devices.
    import torch

    import ttnn

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="RUNNING",
        experiment="isolated native combine zero_pages; not full combine",
        command=sys.argv,
        mode=args.mode,
        iterations_requested=args.iterations,
        iterations_completed=0,
        progress_every=args.progress_every,
        mesh=[1, 4],
        shape=list(SHAPE),
        output_dtype="BFLOAT16",
        output_layout="ROW_MAJOR",
        output_memory="DRAM_INTERLEAVED",
        bytes_per_device=PAGES * PAGE_BYTES,
        page_bytes=PAGE_BYTES,
        partitions=[list(pair) for pair in PARTITIONS],
        sender=dict(logical_core=[0, 0], processor="NCRISC", noc=0, scratch_cb=7),
        untilizers=dict(logical_cores=[[i, 0] for i in range(1, 5)], processor="BRISC", noc=1, scratch_cb=6),
        scratch_bytes=SCRATCH_BYTES,
        trace_region_size=args.trace_region_size,
        tracking=True,
        skip_program_cache=False,
        captures=0,
        releases=0,
        checkpoints=[],
        sources={
            path: hashlib.sha256((REPO / path).read_bytes()).hexdigest()
            for path in (
                NATIVE_HELPER,
                "tt_metal/hw/inc/internal/noc_zero_dram.inl",
                "tt_metal/hw/inc/internal/tt-1xx/noc_zero_l1.inl",
                "tt_metal/hw/inc/internal/tt-1xx/blackhole/noc_nonblocking_api.h",
            )
        },
        limitations=[
            "Calls the actual native zero_pages helper; does not instantiate native combine routing/compute/semaphores.",
            "Matches zeroing processor roles, page partition and scratch extent, but not full combine L1 addresses/layout.",
            "No model/B2 graph history, request state, cache, sampler, concurrent decode or fabric traffic.",
            "All buffers/program descriptors exist before capture; one retained trace; no per-replay allocations.",
            "Output is re-poisoned with ones before each progress batch and checked exactly on all four chips afterward.",
            "Intermediate calls within a batch write an already-zero output; no claim of per-call output observability.",
            "Checkpoint copies/readbacks introduce synchronization and extra NoC work; this is not a performance benchmark.",
            "External timeout is required to bound a device hang; Python cleanup cannot execute through a blocked device read.",
        ],
    )

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    save()
    mesh = trace = output = poison = None
    capture_open = False
    try:
        torch.set_num_threads(4)
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_region_size)
        assert mesh.compute_with_storage_grid_size().x >= 5
        poison = ttnn.from_torch(
            torch.ones(SHAPE, dtype=torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=mesh,
            memory_config=ttnn.DRAM_MEMORY_CONFIG,
            mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        )
        output = ttnn.empty_like(poison)
        program = make_program(ttnn, output)

        def check(expected, label):
            chips = ttnn.get_device_tensors(output)
            assert len(chips) == 4, len(chips)
            for chip_index, chip in enumerate(chips):
                values = ttnn.to_torch(chip)
                assert values.numel() == PAGES * PAGE_BYTES // 2, values.shape
                if not torch.equal(values, torch.full_like(values, expected)):
                    bad = int((values != expected).sum())
                    raise AssertionError(f"{label}: chip={chip_index}, {bad} elements differ from {expected}")

        def execute():
            # generic_op requires an input and an output even for a fill.
            # The preallocated poison is an unused input, not copied by this kernel.
            ttnn.generic_op([poison, output], program)

        ttnn.copy(poison, output)
        check(1, "initial poison")
        execute()
        check(0, "eager warmup zero")
        # Warm copy, native helper and observation before the only capture.
        ttnn.copy(poison, output)
        if args.mode == "trace":
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            capture_open = True
            execute()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            capture_open = False
            result.update(captures=1, trace_id=str(trace))
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        else:
            execute()
        check(0, "prepared zero")
        entries = int(mesh.num_program_cache_entries())
        result.update(prepared_program_entries=entries, output_address=output.buffer_address())
        save()
        started = time.monotonic()
        for first in range(0, args.iterations, args.progress_every):
            last = min(first + args.progress_every, args.iterations)
            print(f"ZERO_PROBE_BATCH start={first + 1} end={last} mode={args.mode}", flush=True)
            ttnn.copy(poison, output)
            # Verify poisoning each batch so a no-op trace cannot pass silently.
            check(1, f"poison before {first + 1}")
            for _ in range(first, last):
                if trace is None:
                    execute()
                else:
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            check(0, f"zero after {last}")
            current_entries = int(mesh.num_program_cache_entries())
            assert current_entries == entries, (entries, current_entries)
            result["iterations_completed"] = last
            result["checkpoints"].append(
                dict(
                    iteration=last,
                    exact_zero_chips=4,
                    program_entries=current_entries,
                    seconds=time.monotonic() - started,
                )
            )
            save()
            print(f"ZERO_PROBE_PASS iterations={last} exact_zero_chips=4", flush=True)
        result["status"] = "PASS"
    except BaseException:
        result["status"] = "FAIL"
        result["error"] = traceback.format_exc()
        raise
    finally:
        try:
            if mesh is not None:
                if capture_open:
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace is not None:
                    ttnn.release_trace(mesh, trace)
                    result["releases"] += 1
                for tensor in (output, poison):
                    if tensor is not None:
                        ttnn.deallocate(tensor)
                ttnn.close_mesh_device(mesh)
                result["mesh_closed"] = True
        except BaseException:
            result["status"] = "FAIL"
            result["cleanup_error"] = traceback.format_exc()
            raise
        finally:
            save()
    print(json.dumps(dict(status=result["status"], iterations=result["iterations_completed"], output=str(args.output))))


if __name__ == "__main__":
    main()
