# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Exact TP4 SDPA-head concat/sigmoid-gate family, ending at o_proj's input.

CPU checks use no Torch/TTNN. Parent owns hardware scheduling. All inputs,
outputs and native intermediates are allocated before any trace is captured.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

from models.autoports.qwen_qwen3_6_27b.tt import attention_gate_candidate as candidate

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[2]


def cpu_check():
    checks = ownership = 0
    for batch in range(1, 33):
        for cores in (1, 8, 17, 48):
            tiles = [t for start, count in candidate.tile_plan(batch, cores) for t in range(start, start + count)]
            assert tiles == list(range(48))
            ownership += len(tiles)
        for user in range(batch):
            for head in range(6):
                for col in range(8):
                    for face in range(2):
                        page, aligned, skew = candidate.face_address(user, head, col, face)
                        assert page == user * 8 + col and aligned % 64 == 0 and aligned + 64 <= 2048
                        assert skew in (0, 32)
                        for j in range(16):
                            src = aligned + skew + j * 2
                            assert src == (head // 16) * 1024 + face * 512 + (head % 16) * 32 + j * 2
                            dst = (user // 16) * 1024 + face * 512 + (user % 16) * 32 + j * 2
                            assert 0 <= dst < 2048
                            tile = head * 8 + col
                            storage_core, storage_tile = divmod(tile, 6)
                            assert 0 <= storage_core < 8 and 0 <= storage_tile < 6
                            checks += 1
        for residue in range(64):
            shift = (-residue) % 64
            assert shift + (batch * 2) * 64 <= 4160
    negative = 0
    for callback in (
        lambda: candidate.tile_plan(0),
        lambda: candidate.tile_plan(33),
        lambda: candidate.tile_plan(1, 0),
        lambda: candidate.tile_plan(1, True),
        lambda: candidate.face_address(0, 6, 0, 0),
    ):
        try:
            callback()
        except ValueError:
            negative += 1
        else:
            raise AssertionError("Invalid geometry accepted")
    sdpa = (
        REPO / "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/sdpa_decode_device_operation.cpp"
    ).read_text()
    assert "uint32_t num_q_heads = q_shape_unpadded[2];" in sdpa
    assert "output_shape[2] = num_q_heads;" in sdpa
    sigmoid = (REPO / "tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_sigmoid.h").read_text()
    assert (
        "if constexpr (!is_fp32_dest_acc_en)" in sigmoid
        and "sfpi::convert<sfpi::vFloat16b>(result, sfpi::RoundMode::Nearest)" in sigmoid
    )
    assert "sigmoid_tile<VectorMode::RC,false>(0)" in candidate.COMPUTE
    assert "mul_binary_tile<false>(0,1,0)" in candidate.COMPUTE
    return dict(
        pass_cpu=True,
        address_checks=checks,
        tile_ownership_checks=ownership,
        expected_rejections=negative,
        native_logical_heads=6,
        physical_heads=32,
        scratch_bytes_per_worker=4160,
        cb_bytes_per_worker=3 * 4096 + 4160,
        output_bytes_per_device=32 * 1536 * 2,
        precision="Native nonapprox BF16 sigmoid and BF16 multiply; hardware exactness still unverified",
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cpu-check", action="store_true")
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 3, 17, 32])
    ap.add_argument("--all-batches", action="store_true")
    ap.add_argument("--cores", nargs="+", type=int, default=[8, 48])
    ap.add_argument("--row-bands", nargs="+", type=int, choices=(1, 2), default=[1])
    ap.add_argument("--logical-heads", nargs="+", type=int, choices=(6, 32), default=[6])
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--queued", type=int, default=50)
    ap.add_argument("--samples", type=int, default=7)
    ap.add_argument("--replays", type=int, default=100)
    ap.add_argument("--no-timing", action="store_true")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.cpu_check:
        print(json.dumps(cpu_check(), indent=2))
        return
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert args.output and not args.output.exists()
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert all(1 <= b <= 32 for b in args.batches) and all(1 <= c <= 96 for c in args.cores)
    assert args.steps >= 2 and min(args.queued, args.samples, args.replays) > 0
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracked = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    if tracked:
        assert os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE") == "0"
    timed = not (watcher or profiler or tracked or args.no_timing)
    import torch
    import ttnn
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    files = [
        Path(__file__),
        ROOT / "tt/attention_gate_candidate.py",
        ROOT / "tt/packed_swiglu_candidate.py",
        ROOT / "tt/multichip_decoder.py",
    ]
    report = dict(
        status="running",
        mesh=[1, 4],
        cases=[],
        cpu=cpu_check(),
        arguments=vars(args) | {"output": str(args.output)},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        watcher=watcher,
        profiler=profiler,
        allocation_tracking=tracked,
        timed=timed,
        skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "unset"),
        boundary="Native SDPA DRAM heads + compact DRAM gate through canonical8-core width-L1 o_proj input",
        timing_scope="Complete boundary, native baseline intermediates preallocated; excludes input copies and o_proj matmul",
        poison="Preserved IEEE NaNs in unused head rows6..31 and padded gate user rowsB..31",
        reference="Current native layout/concat/sigmoid-multiply chain, not Torch sigmoid/HF",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        grid = mesh.compute_with_storage_grid_size()
        full_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        canonical = candidate.canonical_output_memory(mesh)
        owner = OptimizedDecoder()

        def empty(shape, memory=ttnn.DRAM_MEMORY_CONFIG):
            return ttnn.empty(shape, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, memory_config=memory)

        def upload(values):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                preserve_nan_values=True,
            )

        def read(tensor):
            return [ttnn.to_torch(shard).contiguous() for shard in ttnn.get_device_tensors(tensor)]

        def physical(tensor):
            return [shard.cpu().to_torch_with_padded_shape().contiguous() for shard in ttnn.get_device_tensors(tensor)]

        def bits(a, b):
            assert a.shape == b.shape and a.dtype == b.dtype == torch.bfloat16
            return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())

        def raw_copy(source, destination):
            assert tuple(source.padded_shape) == tuple(destination.padded_shape)
            assert source.memory_config() == destination.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            assert source.dtype == destination.dtype == ttnn.bfloat16
            pages = 1
            for dim in source.padded_shape:
                pages *= dim
            pages //= 1024
            assert pages % 8 == 0 and source.buffer_address() != destination.buffer_address()
            core = ttnn.CoreCoord(0, 0)
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            rt = ttnn.RuntimeArgs()
            rt[0][0] = [source.buffer_address(), destination.buffer_address(), pages]
            ct = list(ttnn.TensorAccessorArgs(source).get_compile_time_args()) + list(
                ttnn.TensorAccessorArgs(destination).get_compile_time_args()
            )
            source_code = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto sa=TensorAccessorArgs<0>();
    constexpr auto da=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(sa,get_arg_val<uint32_t>(0),2048);
    const auto dst=TensorAccessor(da,get_arg_val<uint32_t>(1),2048);
    const uint32_t pages=get_arg_val<uint32_t>(2);
    Noc noc;CircularBuffer cb(0);
    for(uint32_t base=0;base<pages;base+=8) {
        cb.reserve_back(8);
        for(uint32_t i=0;i<8;i++) noc.async_read(src,cb,2048,{.page_id=base+i},{.offset_bytes=i*2048});
        noc.async_read_barrier();cb.push_back(8);cb.wait_front(8);
        for(uint32_t i=0;i<8;i++) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(cb),dst,2048,
            {.offset_bytes=i*2048},{.page_id=base+i});
        noc.async_write_barrier();cb.pop_front(8);
    }
}
"""
            kernel = ttnn.KernelDescriptor(
                kernel_source=source_code,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=rt,
                config=ttnn.ReaderConfigDescriptor(),
            )
            cb = ttnn.CBDescriptor(
                total_size=16384,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
            ttnn.generic_op([source, destination], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))

        positive_done = False
        for batch in args.batches:
            for logical_heads in args.logical_heads:
                entry = dict(batch=batch, logical_heads=logical_heads, checks=[], poison_checks=[], trace_us={})
                report["cases"].append(entry)
                save()
                shape = (1, batch, logical_heads, 256)
                heads_mem = owner._compact_height_mem(owner._compact_cores(batch), 32, 256)
                concat_mem = owner._compact_height_mem(owner._compact_cores(6), 32, 256)
                concat_mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1, concat_mem.shard_spec
                )
                heads_l1 = empty(shape, heads_mem)
                concat_l1 = empty((1, 1, 32, 1536), concat_mem)
                concat_dram = empty((1, 1, 32, 1536))
                # B32's native slice is a no-op. Do not give the baseline an
                # artificial preallocated copy at that boundary.
                sliced = empty((1, 1, batch, 1536)) if batch < 32 else concat_dram
                compact = ttnn.reshape(sliced, (1, batch, 1536))
                assert compact.buffer_address() == sliced.buffer_address(), "Compact reshape must be a metadata view"
                product = empty((1, batch, 1536))
                paths = [("native", None, None, None)] + [
                    (f"{kind}_c{cores}" + (f"_bands{bands}" if bands != 1 else ""), kind, cores, bands)
                    for bands in args.row_bands
                    for cores in args.cores
                    if cores <= 48 * bands
                    for kind in ("dram_shard", "direct_l1")
                ]
                outputs = [empty((1, batch, 1536), canonical) for _ in paths]
                intermediates = {name: empty((1, batch, 1536)) for name, kind, _, _ in paths if kind == "dram_shard"}
                live = (empty(shape), empty((1, batch, 1536)))
                rebound = (empty(shape), empty((1, batch, 1536)))
                banks, hosts = [], []
                for step in range(args.steps):
                    hh, gg = [], []
                    for rank in range(4):
                        rng = torch.Generator().manual_seed(17039 + batch * 31 + rank * 997 + step * 5113)
                        h = torch.full((1, batch, 32, 256), float("nan"), dtype=torch.bfloat16)
                        g = torch.full((1, 32, 1536), float("nan"), dtype=torch.bfloat16)
                        h[:, :, :6] = (torch.randn((1, batch, 6, 256), generator=rng) * (0.25 + rank)).bfloat16()
                        g[:, :batch] = (torch.randn((1, batch, 1536), generator=rng) * (1 + step * 4)).bfloat16()
                        g[:, :batch, :10] = torch.tensor(
                            [-80, -16, -8, -1, -0.0, 0.0, 1, 8, 16, 80], dtype=torch.bfloat16
                        )
                        h[:, :, :6, :4] = torch.tensor([0.0, -0.0, 1.0, -1.0], dtype=torch.bfloat16)
                        hh.append(h)
                        gg.append(g)
                    full_h, full_g = upload(hh), upload(gg)
                    pair = (empty(shape), empty((1, batch, 1536)))
                    raw_copy(full_h, pair[0])
                    raw_copy(full_g, pair[1])
                    banks.append(pair)
                    hosts.append((hh, gg))
                    for label, tensors, expected in (("heads", pair[0], hh), ("gate", pair[1], gg)):
                        actual = physical(tensors)
                        counts = [bits(a, b) for a, b in zip(actual, expected)]
                        entry["poison_checks"].append(
                            dict(
                                stage="upload_raw_copy",
                                step=step,
                                tensor=label,
                                bit_errors=counts,
                                nan_count=[int(torch.isnan(a).sum()) for a in actual],
                            )
                        )
                        assert counts == [0] * 4
                    ttnn.deallocate(full_h)
                    ttnn.deallocate(full_g)
                save()

                def set_inputs(step, destination=live):
                    for src, dst in zip(banks[step], destination):
                        raw_copy(src, dst)

                def invoke(index, inputs=live):
                    name, kind, cores, bands = paths[index]
                    h, g = inputs
                    if kind is None:
                        ttnn.to_memory_config(h, heads_mem, output_tensor=heads_l1)
                        ttnn.experimental.nlp_concat_heads_decode(
                            heads_l1, num_heads=6, output_tensor=concat_l1, sub_core_grids=full_grid
                        )
                        ttnn.to_memory_config(concat_l1, ttnn.DRAM_MEMORY_CONFIG, output_tensor=concat_dram)
                        if batch < 32:
                            ttnn.slice(concat_dram, [0, 0, 0, 0], [1, 1, batch, 1536], output_tensor=sliced)
                        ttnn.multiply(
                            g, compact, input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID], output_tensor=product
                        )
                        ttnn.to_memory_config(product, canonical, output_tensor=outputs[index])
                    elif kind == "dram_shard":
                        candidate.attention_gate(h, g, output=intermediates[name], cores=cores, row_bands=bands)
                        ttnn.to_memory_config(intermediates[name], canonical, output_tensor=outputs[index])
                    else:
                        candidate.attention_gate(h, g, output=outputs[index], cores=cores, row_bands=bands)
                    return outputs[index]

                def verify(index, expected, label):
                    actual = read(outputs[index])
                    checks = []
                    for rank, (a, b) in enumerate(zip(actual, expected)):
                        for user in range(batch):
                            mismatches = bits(a[:, user], b[:, user])
                            score = 1.0 if mismatches == 0 else float(pcc(a[:, user].float(), b[:, user].float()))
                            checks.append(
                                dict(
                                    rank=rank,
                                    user=user,
                                    bit_errors=mismatches,
                                    pcc=score,
                                    finite=bool(torch.isfinite(a[:, user]).all()),
                                )
                            )
                    entry["checks"].append(dict(path=paths[index][0], label=label, values=checks))
                    save()
                    assert all(c["bit_errors"] == 0 and c["finite"] and c["pcc"] >= 0.995 for c in checks)

                eager = []
                for step in range(args.steps):
                    set_inputs(step)
                    invoke(0)
                    reference = read(outputs[0])
                    eager.append(reference)
                    for index in range(len(paths)):
                        invoke(index)
                        verify(index, reference, f"eager{step}")
                    # Source padding and all input bits remain immutable.
                    assert all(
                        bits(a, b) == 0
                        for tensor, host in zip(live, hosts[step])
                        for a, b in zip(physical(tensor), host)
                    )
                set_inputs(1, rebound)
                for index in range(len(paths)):
                    invoke(index, rebound)
                    verify(index, eager[1], "rebound")
                # Validate direct output's physical dead rows before any capture.
                for index, (_, kind, _, _) in enumerate(paths):
                    if kind is not None:
                        assert all(torch.count_nonzero(a[:, batch:]) == 0 for a in physical(outputs[index]))
                addresses = [t.buffer_address() for t in outputs]
                traces = []
                try:
                    set_inputs(0)
                    for index in range(len(paths)):
                        invoke(index)  # all program-cache entries and buffers warm before capture
                    for index in range(len(paths)):
                        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                        invoke(index)
                        ttnn.end_trace_capture(mesh, trace, cq_id=0)
                        traces.append(trace)
                    if tracked and not positive_done:
                        unsafe = empty((1, 32, 32))
                        detected = False
                        try:
                            ttnn.execute_trace(mesh, traces[-1], cq_id=0, blocking=True)
                        except RuntimeError as exc:
                            if not any(word in str(exc).lower() for word in ("unsafe", "allocation")):
                                raise
                            detected = True
                        finally:
                            ttnn.deallocate(unsafe)
                        entry["allocation_positive_control"] = detected
                        save()
                        assert detected
                        positive_done = True
                    for step in range(args.steps):
                        set_inputs(step)
                        for index in reversed(range(len(paths))):
                            ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=True)
                            verify(index, eager[step], f"trace{step}")
                    for step in range(args.queued):
                        set_inputs(step % args.steps)
                        for trace in traces:
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    for index in range(len(paths)):
                        verify(index, eager[(args.queued - 1) % args.steps], "queued_final")
                        if paths[index][1] is not None:
                            assert all(torch.count_nonzero(a[:, batch:]) == 0 for a in physical(outputs[index]))
                    assert all(
                        bits(a, b) == 0
                        for tensor, host in zip(live, hosts[(args.queued - 1) % args.steps])
                        for a, b in zip(physical(tensor), host)
                    )
                    if timed:
                        # Rotate order each sample to avoid systematically giving one family the hot slot.
                        samples = [[] for _ in paths]
                        for sample in range(args.samples):
                            for offset in range(len(paths)):
                                index = (sample + offset) % len(paths)
                                ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=True)
                                start = time.perf_counter_ns()
                                for _ in range(args.replays):
                                    ttnn.execute_trace(mesh, traces[index], cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                samples[index].append((time.perf_counter_ns() - start) / 1000 / args.replays)
                        for index, (name, _, _, _) in enumerate(paths):
                            entry["trace_us"][name] = dict(
                                samples=samples[index], median=statistics.median(samples[index])
                            )
                            verify(index, eager[(args.queued - 1) % args.steps], "timed_final")
                    assert addresses == [t.buffer_address() for t in outputs]
                finally:
                    for trace in traces:
                        ttnn.release_trace(mesh, trace)
                entry.update(pass_all=True, queued=args.queued, preallocated_output_addresses=addresses)
                save()
        report.update(status="pass", pass_all=True)
    except BaseException:
        report.update(status="fail", exception=traceback.format_exc())
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()


if __name__ == "__main__":
    main()
