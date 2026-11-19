# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Bounded TP4 Linear conventional AGMM repair probe; parent serializes hardware.

--cpu-only imports neither torch nor TTNN. Hardware mode is a focused kernel
proof, not a decoder quality/performance substitute. External run_job.sh owns
hang capture and timeout; this script never resets a device.
"""

import argparse
import hashlib
import json
import math
import os
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]

SKEW_READER = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto input_args = TensorAccessorArgs<0>();
    constexpr auto output_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, get_arg_val<uint32_t>(0), 32);
    const auto output = TensorAccessor(output_args, get_arg_val<uint32_t>(1), 32);
    Noc noc; CircularBuffer scratch(0);
    scratch.reserve_back(1);
    noc.async_read(input, scratch, 32, {.page_id=0}, {.offset_bytes=0});
    noc.async_read_barrier();
    volatile tt_l1_ptr uint32_t* words =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scratch.get_write_ptr());
    const uint32_t requested = words[0];
    const uint32_t bounded = requested > 1350000u ? 1350000u : requested;
    volatile tt_reg_ptr uint32_t* clock =
        reinterpret_cast<volatile tt_reg_ptr uint32_t*>(RISCV_DEBUG_REG_WALL_CLOCK_L);
    const uint32_t start = *clock;
    uint32_t elapsed = 0, iterations = 0;
    // Unsigned subtraction handles a low-word clock wrap. Both the requested
    // clock interval and iteration count are bounded independently.
    while (elapsed < bounded && iterations < 2000000u) {
        elapsed = *clock - start;
        ++iterations;
    }
    for (uint32_t i=0; i<8; ++i) words[i]=0;
    words[0]=requested; words[1]=elapsed; words[2]=iterations;
    words[3]=(elapsed >= bounded && requested <= 1350000u) ? 1u : 0u;
    scratch.push_back(1); scratch.wait_front(1);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(scratch), output, 32,
                    {.offset_bytes=0}, {.page_id=0});
    noc.async_write_barrier(); scratch.pop_front(1);
}
"""


def schedule(rank, size=4):
    result = [(rank, 1, 1)]
    for distance in range(1, size):
        if rank >= distance:
            result.append((rank - distance, 0, distance))
        if rank + distance < size:
            result.append((rank + distance, 1, distance + 1))
    return result


def cpu_contract():
    """Independent specification checks; does not execute the native parser."""
    checks = []
    for rank in range(4):
        entries = schedule(rank)
        assert len(entries) == 4 and sorted(x[0] for x in entries) == list(range(4))
        for _, direction, threshold in entries:
            assert 1 <= threshold <= (rank if direction == 0 else 4 - rank)
        for block in (4, 8):
            for ntiles, per_core_n in ((8, 1), (272, 34), (272, 9)):
                for core in range(math.ceil(ntiles / per_core_n)):
                    column = core * per_core_n
                    a_pages, b_k_rows = [], []
                    for source, _, _ in entries:
                        for local_block in range(40 // block):
                            a = 40 * source + block * local_block
                            b = 40 * ntiles * source + block * ntiles * local_block + column
                            assert (b - column) // ntiles == a
                            a_pages.extend(range(a, a + block))
                            b_k_rows.extend((b - column) // ntiles + k for k in range(block))
                    assert a_pages == b_k_rows and sorted(a_pages) == list(range(160))
                checks.append({"rank": rank, "block": block, "n_tiles": ntiles, "per_core_n": per_core_n})
    return {
        "status": "pass",
        "kind": "independent_cpu_schedule_specification",
        "cases": checks,
        "limitations": ["Does not execute native host validation, descriptor serialization, or device kernels."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--factory", choices=("1d", "2d"), default="2d")
    parser.add_argument("--batch", type=int, choices=(1, 32), default=32)
    parser.add_argument("--n", type=int, choices=(256, 8704), default=256)
    parser.add_argument("--k-block", type=int, choices=(4, 8), default=4)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=("ranked", "source0", "source1", "source2", "source3", "cancellation"),
        default=["ranked"],
    )
    parser.add_argument(
        "--rebind", action="store_true", help="Fresh A/B/scratch/global semaphores with old semaphores poisoned"
    )
    parser.add_argument(
        "--host-negatives",
        action="store_true",
        help="Require precise native rejection of unsupported Linear contracts before first successful fused call",
    )
    parser.add_argument(
        "--queued", type=int, default=0, help="Changed-input/weight calls sharing scratch, no per-call sync"
    )
    parser.add_argument("--trace-replays", type=int, default=0)
    parser.add_argument(
        "--skew", action="store_true", help="Bounded local NCRISC delay before each queued/captured fused call"
    )
    parser.add_argument(
        "--skew-cycles", type=int, default=450000, help="Delay cycles on one rotating rank (135000..1350000)"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.queued == 0 or args.queued >= 50
    assert args.trace_replays >= 0
    assert 135000 <= args.skew_cycles <= 1350000
    assert not args.skew or args.queued or args.trace_replays
    assert not args.output.exists(), "Preserve previous evidence under its original path"
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.cpu_only:
        args.output.write_text(json.dumps(cpu_contract(), indent=2) + "\n")
        return
    watcher = os.getenv("TT_METAL_WATCHER", "0") != "0"
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    assert not (watcher and profiler)
    import torch
    import ttnn

    torch.set_num_threads(4)
    sources = [Path(__file__), ROOT / "tt/fused_ag_candidate.py"]
    native_root = REPO / "ttnn/cpp/ttnn/operations"
    for relative in (
        "ccl/ccl_op_fusion.cpp",
        "ccl/ccl_op_fusion.hpp",
        "ccl/kernel_common/worker_sync_utils.hpp",
        "experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_program_factory.cpp",
        "experimental/ccl/all_gather_matmul_async/device/all_gather_matmul_async_device_operation.cpp",
        "experimental/ccl/all_gather_async/device/all_gather_async_default_program_factory.cpp",
        "experimental/ccl/all_gather_async/device/kernels/minimal_default_writer.cpp",
        "experimental/ccl/all_gather_async/device/kernels/minimal_default_reader.cpp",
        "experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_1d.cpp",
        "experimental/matmul/ccl_fusion/device/ccl_fusion_mcast_2d.cpp",
        "matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        "matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
    ):
        path = native_root / relative
        if path.exists():
            sources.append(path)
    report = {
        "status": "running",
        "mesh": [1, 4],
        "topology": "Linear",
        "fabric": "FABRIC_1D",
        "watcher": watcher,
        "profiler": profiler,
        "arguments": vars(args) | {"output": str(args.output)},
        "allocation_tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0") == "1",
        "skip_program_cache": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
        "source_sha256": {str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "cases": [],
        "cpu_contract": cpu_contract(),
        "limitations": ([] if args.skew else ["No deliberate asymmetric rank skew is injected."])
        + [
            "Focused operation evidence does not establish whole decoder speed or quality.",
            "Program-local and global semaphore values are not directly read back.",
        ],
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        grid = mesh.compute_with_storage_grid_size()
        cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=True,
        )
        common = dict(
            in0_block_w=args.k_block,
            out_subblock_h=1,
            per_core_M=1,
            per_core_N=(9 if args.factory == "1d" else 34) if args.n == 8704 else 1,
            fused_activation=None,
            fuse_batch=True,
        )
        if args.factory == "1d":
            config = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                compute_with_storage_grid_size=(8, 4),
                out_subblock_w=1,
                mcast_in0=True,
                num_global_cb_receivers=0,
                **common,
            )
        else:
            config = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                compute_with_storage_grid_size=(8, 1),
                out_subblock_w=2 if args.n == 8704 else 1,
                transpose_mcast=False,
                **common,
            )
        report["program_config"] = repr(config)
        report["compute_kernel_config"] = repr(ckc)

        def upload(values, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
            )

        def replicate(value):
            return ttnn.from_torch(
                value,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )

        def read(value):
            return [ttnn.to_torch(x).contiguous() for x in ttnn.get_device_tensors(value)]

        def addresses(bank, scratch, output):
            return {
                name: [int(x.buffer_address()) for x in ttnn.get_device_tensors(tensor)]
                for name, tensor in (("a", bank["a"]), ("b", bank["b"]), ("scratch", scratch), ("output", output))
            }

        def new_sems():
            return [ttnn.create_global_semaphore(mesh, cores, 0) for _ in range(2)], ttnn.create_global_semaphore(
                mesh, cores, 0
            )

        skew_bank = []
        if args.skew:
            # A separate one-core program on the same ordered CQ. No CCL or
            # cross-rank event/fence is present; each rank reads only its local
            # sharded count. All allocations and eager compilation precede loops.
            skew_core = ttnn.CoreCoord(0, 0)
            skew_ranges = ttnn.CoreRangeSet({ttnn.CoreRange(skew_core, skew_core)})
            for delayed_rank in range(4):
                counts = [args.skew_cycles if rank == delayed_rank else 0 for rank in range(4)]
                host = torch.tensor([count for count in counts for _ in range(8)], dtype=torch.int32)
                source = ttnn.from_torch(
                    host.reshape(1, 1, 1, 32),
                    device=mesh,
                    dtype=ttnn.uint32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                destination = ttnn.empty_like(source)
                runtime = ttnn.RuntimeArgs()
                runtime[0][0] = [source.buffer_address(), destination.buffer_address()]
                compile_args = ttnn.TensorAccessorArgs(source).get_compile_time_args()
                compile_args += ttnn.TensorAccessorArgs(destination).get_compile_time_args()
                kernel = ttnn.KernelDescriptor(
                    kernel_source=SKEW_READER,
                    source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                    core_ranges=skew_ranges,
                    compile_time_args=compile_args,
                    runtime_args=runtime,
                    config=ttnn.ReaderConfigDescriptor(),
                )
                cb = ttnn.CBDescriptor(
                    total_size=32,
                    core_ranges=skew_ranges,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.uint32, page_size=32)],
                )
                descriptor = ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[])
                skew_bank.append((source, destination, descriptor, counts))

        def inject_skew(step):
            if args.skew:
                source, destination, descriptor, _ = skew_bank[step % 4]
                ttnn.generic_op([source, destination], descriptor)

        def check_skew():
            records = []
            for _, destination, _, expected_counts in skew_bank:
                words = [value.flatten().tolist() for value in read(destination)]
                assert all(
                    row[0] == count and row[1] >= count and row[2] <= 2000000 and row[3] == 1
                    for row, count in zip(words, expected_counts)
                ), (words, expected_counts)
                records.append(words)
            return records

        if args.skew:
            for step in range(4):
                inject_skew(step)
            ttnn.synchronize_device(mesh)
            report["skew"] = {
                "kind": "one_local_ncrisc_core_per_rank_before_fused_call",
                "rank_selection": "rotating one of ranks 0,1,2,3",
                "cycles": args.skew_cycles,
                "maximum_cycles": 1350000,
                "maximum_loop_iterations": 2000000,
                "nominal_ms_at_1_35GHz": args.skew_cycles / 1350000,
                "no_interrank_fence": True,
                "eager_records": check_skew(),
            }

        def score(actual, expected):
            assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
            x, y = actual.float().flatten(), expected.float().flatten()
            if torch.equal(x, y):
                return 1.0
            x, y = x - x.mean(), y - y.mean()
            value = float(torch.dot(x, y) / (torch.linalg.vector_norm(x) * torch.linalg.vector_norm(y)))
            assert math.isfinite(value)
            return value

        def host_bank(pattern, generation):
            gen = torch.Generator().manual_seed(3917 + generation * 101)
            values = []
            for rank in range(4):
                value = torch.randn(1, 1, args.batch, 1280, generator=gen) * (0.125 + rank / 16)
                value += (rank - 1.5) / 8 + generation / 32
                if pattern.startswith("source") and rank != int(pattern[-1]):
                    value.zero_()
                values.append(value.bfloat16())
            weights = []
            for rank in range(4):
                w = torch.randn(1, 1, 5120, args.n, generator=gen) / 32
                # Independent source and destination fingerprints expose A/B K permutation.
                for source in range(4):
                    w[:, :, source * 1280 : (source + 1) * 1280] *= (source + 1) * (rank + 2) / 8
                if pattern == "cancellation":
                    # Two opposing source terms plus independent surviving terms. Nonzero
                    # random remainder avoids an undefined PCC on an all-zero reference.
                    values[1] = values[0].clone()
                    w[:, :, 1280:2560] = -w[:, :, :1280]
                weights.append(w.bfloat16())
            a, b = upload(values), upload(weights, ttnn.bfloat8_b)
            gathered = torch.cat(values, -1)
            quantized = read(b)
            golden = [(gathered.float() @ w.float()).bfloat16() for w in quantized]
            return {"a": a, "b": b, "gathered": gathered, "golden": golden}

        def fused(bank, scratch, sems, barrier, **overrides):
            kwargs = dict(
                persistent_output_buffer=scratch,
                dim=3,
                multi_device_global_semaphore=sems,
                barrier_semaphore=barrier,
                all_gather_core_grid_offset=(0, 5) if args.factory == "1d" else (0, 2),
                num_links=1,
                num_workers_per_link=2,
                topology=ttnn.Topology.Linear,
                memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                program_config=config,
                compute_kernel_config=ckc,
                dtype=ttnn.bfloat16,
            )
            kwargs.update(overrides)
            return ttnn.experimental.all_gather_matmul_async(bank["a"], bank["b"], **kwargs)

        def host_negatives(bank, *, warm=False):
            """Only exact native RuntimeError messages satisfy these checks.

            Constructors, tensor conversion, and API TypeErrors occur outside the
            expected rejection handler. No unsupported call is synchronized or
            retried after it unexpectedly returns successfully.
            """
            if not warm:
                report["host_negatives"] = []
            scratch = replicate(torch.zeros(1, 1, args.batch, 5120, dtype=torch.bfloat16))
            sems, barrier = new_sems()

            def reject(name, expected, *, changed_bank=None, changed_scratch=None, **overrides):
                record = {
                    "name": name,
                    "phase": "warm" if warm else "cold",
                    "status": "running",
                    "expected_native_message": expected,
                }
                report["host_negatives"].append(record)
                save()
                try:
                    fused(
                        bank if changed_bank is None else changed_bank,
                        scratch if changed_scratch is None else changed_scratch,
                        sems,
                        barrier,
                        **overrides,
                    )
                except RuntimeError as error:
                    message = str(error)
                    record["exception_type"] = type(error).__name__
                    record["native_message"] = message
                    record["status"] = "pass" if expected in message else "wrong-rejection"
                    save()
                    assert expected in message, (name, expected, message)
                else:
                    record["status"] = "unexpected-acceptance"
                    save()
                    raise AssertionError(f"Unsupported Linear AGMM contract accepted: {name}")

            # Both directions share the same physical M32 padding but must not
            # change the public logical row count. Test again after a valid call.
            exact_shape_message = (
                "Linear AGMM persistent scratch must match the exact gathered logical and padded shapes"
            )
            for input_rows, scratch_rows in ((32, 1), (1, 32)):
                changed_a = upload([torch.zeros(1, 1, input_rows, 1280, dtype=torch.bfloat16) for _ in range(4)])
                changed_scratch = replicate(torch.zeros(1, 1, scratch_rows, 5120, dtype=torch.bfloat16))
                reject(
                    f"logical_M{input_rows}_scratch_M{scratch_rows}",
                    exact_shape_message,
                    changed_bank=bank | {"a": changed_a},
                    changed_scratch=changed_scratch,
                )
            # This is a shared-storage view on the same mesh, not a new device.
            subset = ttnn.get_device_tensors(scratch)[0]
            assert len(subset.device_coords()) == 1 and subset.shape == scratch.shape and subset.dtype == scratch.dtype
            reject(
                "scratch_coordinate_subset",
                "Linear AGMM persistent scratch must occupy the complete operand mesh",
                changed_scratch=subset,
            )
            if warm:
                assert len(report["host_negatives"]) == 15 and all(
                    x["status"] == "pass" for x in report["host_negatives"]
                )
                save()
                return

            shape_message = (
                "Linear AGMM initially supports one physical M tile, local K1280 and full K5120 without outer batches"
            )
            m64 = upload([torch.zeros(1, 1, 64, 1280, dtype=torch.bfloat16) for _ in range(4)])
            m64_scratch = replicate(torch.zeros(1, 1, 64, 5120, dtype=torch.bfloat16))
            reject("physical_M64", shape_message, changed_bank=bank | {"a": m64}, changed_scratch=m64_scratch)
            k1248 = upload([torch.zeros(1, 1, args.batch, 1248, dtype=torch.bfloat16) for _ in range(4)])
            k4992_scratch = replicate(torch.zeros(1, 1, args.batch, 4992, dtype=torch.bfloat16))
            k4992_weight = upload(
                [torch.zeros(1, 1, 4992, args.n, dtype=torch.bfloat16) for _ in range(4)], ttnn.bfloat8_b
            )
            reject(
                "local_K1248_full_K4992",
                shape_message,
                changed_bank=bank | {"a": k1248, "b": k4992_weight},
                changed_scratch=k4992_scratch,
            )

            one_core = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))})
            a_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(one_core, [32, 1280], ttnn.ShardOrientation.ROW_MAJOR),
            )
            sharded_a = ttnn.to_memory_config(bank["a"], a_memory)
            reject(
                "sharded_A",
                "Linear AGMM requires interleaved DRAM A, B, gathered scratch and matmul output",
                changed_bank=bank | {"a": sharded_a},
            )
            four_cores = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
            scratch_memory = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(four_cores, [32, 1280], ttnn.ShardOrientation.ROW_MAJOR),
            )
            sharded_scratch = ttnn.to_memory_config(scratch, scratch_memory)
            reject(
                "sharded_scratch",
                "Linear AGMM persistent scratch must be interleaved DRAM with ordinary tiles",
                changed_scratch=sharded_scratch,
            )
            reject("absent_barrier", "Linear AGMM requires a global admission barrier", barrier_semaphore=None)

            def unsupported_1d(**changes):
                fields = dict(
                    compute_with_storage_grid_size=(8, 4),
                    in0_block_w=args.k_block,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    per_core_M=1,
                    per_core_N=9,
                    fuse_batch=True,
                    mcast_in0=True,
                    gather_in0=False,
                    num_global_cb_receivers=0,
                )
                fields.update(changes)
                return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(**fields)

            one_d_message = "Linear AGMM 1D requires ordinary mcast_in0 with enough cores for a single N sweep"
            reject("1d_mcast_in1", one_d_message, program_config=unsupported_1d(mcast_in0=False))
            reject("1d_gather_in0", one_d_message, program_config=unsupported_1d(gather_in0=True))
            reject(
                "repeated_output_block",
                "Linear AGMM requires Kblock4/8 and exactly one output h/w block per core on a rectangular grid",
                program_config=unsupported_1d(out_block_w=3),
            )
            reject(
                "AG_overlaps_physical_mcast_rectangle",
                "Linear AGMM all-gather workers must avoid every physical multicast destination",
                all_gather_core_grid_offset=(0, 0),
            )
            assert len(report["host_negatives"]) == 12 and all(x["status"] == "pass" for x in report["host_negatives"])
            save()

        def validate(output, bank, *, gathered=None, separate=None):
            scores = [score(a, e) for a, e in zip(read(output), bank["golden"])]
            rows = [
                [score(a[:, :, row], e[:, :, row]) for row in range(args.batch)]
                for a, e in zip(read(output), bank["golden"])
            ]
            result = {"host_pcc_by_rank": scores, "host_pcc_per_row_min": [min(x) for x in rows]}
            assert min(scores) >= 0.995 and min(min(x) for x in rows) >= 0.995, result
            if gathered is not None:
                result["gather_exact_by_rank"] = [torch.equal(x, bank["gathered"]) for x in read(gathered)]
                assert all(result["gather_exact_by_rank"]), result
            if separate is not None:
                result["separate_pcc_by_rank"] = [score(a, b) for a, b in zip(read(output), read(separate))]
                assert min(result["separate_pcc_by_rank"]) >= 0.995, result
            return result

        for pattern in args.patterns:
            entry = {"pattern": pattern, "checks": []}
            report["cases"].append(entry)
            save()
            bank = host_bank(pattern, 0)
            if args.host_negatives and len(report["cases"]) == 1:
                host_negatives(bank)
            oracle_sems, oracle_barrier = new_sems()
            ag = ttnn.experimental.all_gather_async(
                bank["a"],
                dim=3,
                multi_device_global_semaphore=oracle_sems,
                barrier_semaphore=oracle_barrier,
                num_links=1,
                num_workers_per_link=2,
                topology=ttnn.Topology.Linear,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )
            separate = ttnn.matmul(
                ag,
                bank["b"],
                program_config=config,
                compute_kernel_config=ckc,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            ttnn.synchronize_device(mesh)
            assert all(torch.equal(x, bank["gathered"]) for x in read(ag))
            scratch = replicate(torch.zeros(1, 1, args.batch, 5120, dtype=torch.bfloat16))
            sems, barrier = new_sems()
            gathered, output = fused(bank, scratch, sems, barrier)
            ttnn.synchronize_device(mesh)
            entry["checks"].append({"kind": "eager", **validate(output, bank, gathered=gathered, separate=separate)})
            if args.host_negatives and len(report["cases"]) == 1:
                host_negatives(bank, warm=True)
            previous_addresses = addresses(bank, scratch, output)
            entry["checks"][-1]["addresses"] = previous_addresses
            cache_after_first = mesh.num_program_cache_entries()
            if args.rebind:
                # Keep allocations live so addresses must differ, and poison retired
                # semaphores to detect a cache callback that accidentally reuses them.
                retained = [(bank, scratch, sems, barrier, output)]
                for generation in (1, 2):
                    for old in [*sems, barrier]:
                        ttnn.reset_global_semaphore_value(old, 0x40000000)
                    ttnn.synchronize_device(mesh)
                    bank = host_bank(pattern, generation)
                    scratch = replicate(torch.zeros(1, 1, args.batch, 5120, dtype=torch.bfloat16))
                    sems, barrier = new_sems()
                    gathered, output = fused(bank, scratch, sems, barrier)
                    ttnn.synchronize_device(mesh)
                    count = mesh.num_program_cache_entries()
                    current_addresses = addresses(bank, scratch, output)
                    assert all(
                        all(a != b for a, b in zip(current_addresses[name], previous_addresses[name]))
                        for name in current_addresses
                    ), (previous_addresses, current_addresses)
                    entry["checks"].append(
                        {
                            "kind": "cache_rebind",
                            "generation": generation,
                            "program_cache_entries": count,
                            "addresses": current_addresses,
                            **validate(output, bank, gathered=gathered),
                        }
                    )
                    assert count == cache_after_first, (count, cache_after_first)
                    retained.append((bank, scratch, sems, barrier, output))
                    previous_addresses = current_addresses
            if args.queued:
                banks = [bank, host_bank(pattern, 5), host_bank(pattern, 6)]
                queued = []
                # No host upload, readback, or synchronization in this launch loop.
                # Every output stays live, while every invocation shares gather scratch.
                for step in range(args.queued):
                    source = banks[step % len(banks)]
                    inject_skew(step)
                    _, value = fused(source, scratch, sems, barrier)
                    queued.append((source, value))
                ttnn.synchronize_device(mesh)
                results = [validate(value, source) for source, value in queued]
                last = queued[-1][0]
                assert all(torch.equal(x, last["gathered"]) for x in read(scratch))
                entry["checks"].append(
                    {
                        "kind": "queued_changed_input_and_weight",
                        "launches": args.queued,
                        "checked_outputs": len(results),
                        "minimum_host_pcc": min(min(x["host_pcc_by_rank"]) for x in results),
                        "final_scratch_exact": True,
                    }
                )
                if args.skew:
                    entry["checks"][-1]["skew_records"] = check_skew()
            if args.trace_replays:
                # A three-call changed-input trace validates intermediate output generations.
                banks = [bank, host_bank(pattern, 7), host_bank(pattern, 8)]
                for item in banks:
                    fused(item, scratch, sems, barrier)
                ttnn.synchronize_device(mesh)
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                trace_outputs = []
                for step, item in enumerate(banks):
                    inject_skew(step)
                    trace_outputs.append(fused(item, scratch, sems, barrier)[1])
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                try:
                    for _ in range(args.trace_replays):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    results = [validate(value, source) for source, value in zip(banks, trace_outputs)]
                    entry["checks"].append(
                        {
                            "kind": "queued_changed_input_trace",
                            "replays": args.trace_replays,
                            "calls_per_replay": 3,
                            "checks": results,
                        }
                    )
                    if args.skew:
                        entry["checks"][-1]["skew_records"] = check_skew()
                finally:
                    ttnn.release_trace(mesh, trace)
            save()
        report["status"] = "pass"
    except BaseException:
        report["status"] = "fail"
        report["exception"] = traceback.format_exc()
        raise
    finally:
        save()
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
