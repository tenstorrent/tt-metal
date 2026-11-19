# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized TP4 packed split/QK norm/RoPE/V/gate whole-family probe."""

import argparse
import hashlib
import json
import os
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 3, 17, 32])
    parser.add_argument("--all-batches", action="store_true")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--replays", type=int, default=50)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--weight-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.all_batches:
        args.batches = list(range(1, 33))
    assert all(1 <= b <= 32 for b in args.batches)
    assert args.steps >= 2 and args.samples > 0 and args.replays > 0 and args.epsilon > 0
    assert args.output.resolve().is_relative_to((ROOT / "doc/optimized_multichip_decoder").resolve())
    assert not args.output.exists()
    watcher = "TT_METAL_WATCHER" in os.environ
    profiler = os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0"
    tracked = os.getenv("TT_METAL_TRACE_ALLOC_TRACKING") == "1"
    assert not (watcher and profiler)
    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    from models.autoports.qwen_qwen3_6_27b.tt.parallel_qk_norm_candidate import compact_qk_norm
    from models.autoports.qwen_qwen3_6_27b.tt.parallel_qk_rope_candidate import compact_qk_rope
    from models.autoports.qwen_qwen3_6_27b.tt.packed_qk_rope_candidate import packed_qk_rope
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
    from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import _local_qkv_split
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    hf_config = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_config)
    sources = [
        Path(__file__),
        *[
            ROOT / "tt" / f
            for f in (
                "packed_qk_rope_candidate.py",
                "parallel_qk_rope_candidate.py",
                "parallel_qk_norm_candidate.py",
                "multichip_decoder.py",
                "optimized_decoder.py",
            )
        ],
    ]
    report = dict(
        status="running",
        mesh=[1, 4],
        watcher=watcher,
        profiler=profiler,
        allocation_tracking=tracked,
        skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "unset"),
        arguments=vars(args) | {"output": str(args.output)},
        cases=[],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        paths=["split_parallel_norm_native_rope", "split_fused_norm_rope", "direct_packed_norm_rope"],
        output_order=["Q", "K", "V", "gate"],
        timing_scope="Whole family from packed DRAM input through four outputs; no projection/SDPA/cache update",
        trace_allocation_policy="All paths allocate their outputs at capture; direct persistent-output replay additionally checked",
        poison="Exact IEEE BF16 NaN source rows B..31; host tilize and raw full-tile copies preserve bits",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    mesh = None
    save()
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            packer_l1_acc=True,
        )
        wdtype, host_wdtype = getattr(ttnn, args.weight_dtype), getattr(torch, args.weight_dtype)

        def upload(values, dtype=ttnn.bfloat16):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                preserve_nan_values=True,
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def read_outputs(outputs):
            return [read(t) for t in outputs]

        def bits(a, b):
            assert tuple(a.shape) == tuple(b.shape), (a.shape, b.shape)
            return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())

        def raw_packed_copy(source, destination):
            """Setup-only 112-tile DRAM copy independent of public logical B."""
            assert source.buffer_address() != destination.buffer_address()
            for tensor in (source, destination):
                assert tuple(tensor.padded_shape) == (1, 32, 3584)
                assert tensor.dtype == ttnn.bfloat16 and tensor.layout == ttnn.TILE_LAYOUT
                assert tensor.memory_config() == ttnn.DRAM_MEMORY_CONFIG
            core = ttnn.CoreCoord(0, 0)
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
            runtime = ttnn.RuntimeArgs()
            runtime[0][0] = [source.buffer_address(), destination.buffer_address()]
            ct = []
            for tensor in (source, destination):
                ct.extend(ttnn.TensorAccessorArgs(tensor).get_compile_time_args())
            kernel = ttnn.KernelDescriptor(
                kernel_source=r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc.h"
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr auto sa=TensorAccessorArgs<0>();
    constexpr auto da=TensorAccessorArgs<sa.next_compile_time_args_offset()>();
    const auto src=TensorAccessor(sa,get_arg_val<uint32_t>(0),2048);
    const auto dst=TensorAccessor(da,get_arg_val<uint32_t>(1),2048);
    Noc noc;CircularBuffer tiles(0);
    for(uint32_t base=0;base<112;base+=8) {
        tiles.reserve_back(8);
        for(uint32_t i=0;i<8;i++) noc.async_read(src,tiles,2048,{.page_id=base+i},{.offset_bytes=i*2048});
        noc.async_read_barrier();tiles.push_back(8);tiles.wait_front(8);
        for(uint32_t i=0;i<8;i++) noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(tiles),dst,2048,
            {.offset_bytes=i*2048},{.page_id=base+i});
        noc.async_write_barrier();tiles.pop_front(8);
    }
}
""",
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=ct,
                runtime_args=runtime,
                config=ttnn.ReaderConfigDescriptor(),
            )
            cb = ttnn.CBDescriptor(
                total_size=16384,
                core_ranges=ranges,
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=0, data_format=ttnn.bfloat16, page_size=2048)],
            )
            ttnn.generic_op([source, destination], ttnn.ProgramDescriptor(kernels=[kernel], cbs=[cb], semaphores=[]))

        for batch in args.batches:
            entry = dict(
                batch=batch,
                checks=[],
                trace_checks=[],
                timings_ms={},
                eager_allocation_timings_ms={},
                poison_checks=[],
                queued_checks=[],
                rebind=[],
            )
            report["cases"].append(entry)
            save()
            owner = OptimizedDecoder()
            heads_mem = owner._compact_height_mem(owner._compact_cores(batch), 32, 256)
            k_mem = owner._compact_height_mem(owner._compact_cores(batch, offset=32), 32, 256)
            rot_mem = owner._compact_rope_mem(batch, mesh.compute_with_storage_grid_size())
            generators = [torch.Generator().manual_seed(13531 + batch * 37 + rank) for rank in range(4)]
            weights_host = [
                [(torch.randn(1, 1, 256, generator=g) * 0.15 + 1).to(host_wdtype) for g in generators] for _ in range(2)
            ]
            weights = [upload(v, wdtype) for v in weights_host]
            banks, oracle, physical_host = [], [], []

            def check_poison(tensor, expected, stage, step):
                diagnostics = []
                for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                    actual = shard.cpu().to_torch_with_padded_shape()
                    bit_count = bits(actual, expected[rank])
                    nan_count = int(torch.isnan(actual[:, batch:]).sum())
                    diagnostic = dict(
                        stage=stage,
                        step=step,
                        rank=rank,
                        bits=bit_count,
                        padding_nan=nan_count,
                        expected_padding_nan=(32 - batch) * 3584,
                    )
                    diagnostics.append(diagnostic)
                    assert tuple(actual.shape) == (1, 32, 3584)
                    assert bit_count == 0 and nan_count == (32 - batch) * 3584, diagnostic
                entry["poison_checks"].extend(diagnostics)
                save()

            for step in range(args.steps):
                host = []
                for rank, g in enumerate(generators):
                    tensor = torch.full((1, 32, 3584), float("nan"), dtype=torch.bfloat16)
                    tensor[:, :batch] = (torch.randn(1, batch, 3584, generator=g) * 0.25 + rank * 0.0625).bfloat16()
                    if step == 1:
                        tensor[0, 0, :256] = 0
                    host.append(tensor)
                full = upload(host)
                check_poison(full, host, "physical_upload", step)
                packed = ttnn.empty(
                    (1, batch, 3584),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                )
                raw_packed_copy(full, packed)
                check_poison(packed, host, "logical_raw_copy", step)
                physical_host.append(host)
                trig_host, positions = [[], []], []
                for rank in range(4):
                    pos = torch.arange(batch) * 977 + rank * 37 + step * 31 + 17
                    cos_host, sin_host = hf_rotary(torch.zeros(batch, 1, 256, dtype=torch.bfloat16), pos[:, None])
                    trig_host[0].append(cos_host.unsqueeze(1).bfloat16())
                    trig_host[1].append(sin_host.unsqueeze(1).bfloat16())
                    positions.append(pos.tolist())
                trig = [upload(v) for v in trig_host]
                banks.append([packed, *trig])
                entry.setdefault("positions", []).append(positions)
                refs = [[], [], [], []]
                for rank in range(4):
                    logical = host[rank][:, :batch]
                    for branch, (start, heads) in enumerate(((0, 6), (1536, 1))):
                        raw = logical[..., start : start + heads * 256].reshape(1, batch, heads, 256).float()
                        norm = (
                            raw
                            * torch.rsqrt(raw.square().mean(-1, keepdim=True) + args.epsilon)
                            * weights_host[branch][rank].float()
                        ).bfloat16()
                        c, s = [v[rank].permute(2, 0, 1, 3) for v in trig_host]
                        prefix = norm[..., :64]
                        rotated = torch.cat((-prefix[..., 32:], prefix[..., :32]), -1)
                        result = (
                            (prefix.float() * c.float()).bfloat16().float()
                            + (rotated.float() * s.float()).bfloat16().float()
                        ).bfloat16()
                        refs[branch].append(torch.cat((result, norm[..., 64:]), -1))
                    refs[2].append(logical[..., 1792:2048].reshape(1, batch, 1, 256).clone())
                    refs[3].append(logical[..., 2048:].clone())
                oracle.append(refs)
            live = [ttnn.empty_like(t) for t in banks[0]]
            checked_steps = set()

            def update(step):
                raw_packed_copy(banks[step][0], live[0])
                for src, dst in zip(banks[step][1:], live[1:]):
                    ttnn.copy(src, dst)
                if step not in checked_steps:
                    check_poison(live[0], physical_host[step], "live_raw_copy", step)
                    checked_steps.add(step)

            def invoke(index, tensors=None, selected_weights=None, output_buffers=None):
                packed, cos, sin = live if tensors is None else tensors
                ws = weights if selected_weights is None else selected_weights
                if index == 2:
                    return packed_qk_rope(packed, *ws, cos, sin, epsilon=args.epsilon, outputs=output_buffers)
                gate = packed[:, :, 2048:]
                projected = packed[:, :, :2048]
                q, k, v = _local_qkv_split(ttnn.reshape(projected, (1, 1, batch, 2048)), memory_config=heads_mem)
                if index == 1:
                    q, k = compact_qk_rope(q, k, *ws, cos, sin, epsilon=args.epsilon)
                else:
                    q, k = compact_qk_norm(q, k, *ws, epsilon=args.epsilon)
                    trig = tuple(ttnn.to_memory_config(ttnn.permute(t, (2, 0, 1, 3)), rot_mem) for t in (cos, sin))

                    def rotate(t):
                        prefix = ttnn.to_memory_config(t[..., :64], rot_mem)
                        prefix = ttnn.experimental.rotary_embedding_hf(
                            prefix, *trig, is_decode_mode=True, memory_config=rot_mem, compute_kernel_config=ckc
                        )
                        return ttnn.concat(
                            [ttnn.to_memory_config(prefix, ttnn.DRAM_MEMORY_CONFIG), t[..., 64:]], dim=-1
                        )

                    q, k = rotate(q), ttnn.to_memory_config(rotate(k), k_mem)
                return q, k, v, gate

            update(0)
            persistent = invoke(2)
            eager = []
            for step in range(args.steps):
                update(step)
                outputs = [invoke(i) for i in range(3)]
                values = [read_outputs(t) for t in outputs]
                eager.append(values)
                result = dict(
                    step=step, native_pcc=[], selected_pcc=[], torch_pcc=[], selected_bits=[], tail_bits=[], peruser=[]
                )
                for branch in range(4):
                    for rank in range(4):
                        native, selected, actual = [v[branch][rank] for v in values]
                        assert all(bool(torch.isfinite(t).all()) for t in (native, selected, actual))
                        result["native_pcc"].append(pcc(actual, native))
                        result["selected_pcc"].append(pcc(actual, selected))
                        result["torch_pcc"].append(pcc(actual, oracle[step][branch][rank]))
                        result["selected_bits"].append(bits(actual, selected))
                        if branch < 2:
                            result["tail_bits"].append(bits(actual[..., 64:], selected[..., 64:]))
                        else:
                            assert bits(actual, oracle[step][branch][rank]) == 0
                        for user in range(batch):
                            # Zero Q head is embedded among six heads; V/gate remain nonconstant.
                            value = pcc(actual[:, user], selected[:, user])
                            result["peruser"].append(value)
                            assert value >= 0.995
                entry["checks"].append(result)
                save()
                assert min(result["native_pcc"] + result["selected_pcc"] + result["torch_pcc"]) >= 0.995
                assert max(result["selected_bits"] + result["tail_bits"]) == 0
                for branch, heads in ((0, 6), (1, 1), (2, 1)):
                    for shard in ttnn.get_device_tensors(outputs[2][branch]):
                        padding = shard.cpu().to_torch_with_padded_shape()[:, :, heads:]
                        assert bool(torch.isfinite(padding).all()) and bool((padding == 0).all())
                for rank, shard in enumerate(ttnn.get_device_tensors(outputs[2][3])):
                    assert bits(shard.cpu().to_torch_with_padded_shape(), physical_host[step][rank][..., 2048:]) == 0
                check_poison(live[0], physical_host[step], "input_after_all_paths", step)
                for tensor, source in zip(live[1:], banks[step][1:]):
                    assert all(bits(a, b) == 0 for a, b in zip(read(tensor), read(source)))
                for tensor, source in zip(weights, weights_host):
                    assert all(bits(a, b) == 0 for a, b in zip(read(tensor), source))
                del outputs

            # All addresses rebound, values changed, old buffers retained.
            update(0)
            invoke(2, output_buffers=persistent)
            ttnn.synchronize_device(mesh)
            replacement_live = [ttnn.empty_like(t) if i == 0 else ttnn.clone(t) for i, t in enumerate(banks[1])]
            raw_packed_copy(banks[1][0], replacement_live[0])
            replacement_weights = [
                upload([(v.float() * 0.9375 + 0.03125).to(host_wdtype) for v in group], wdtype)
                for group in weights_host
            ]
            replacement_outputs = [ttnn.empty_like(t) for t in persistent]
            old = [*live, *weights, *persistent]
            new = [*replacement_live, *replacement_weights, *replacement_outputs]
            assert all(a.buffer_address() != b.buffer_address() for a, b in zip(old, new))
            reference = read_outputs(invoke(1, replacement_live, replacement_weights))
            ttnn.synchronize_device(mesh)
            before = mesh.num_program_cache_entries()
            actual = read_outputs(invoke(2, replacement_live, replacement_weights, replacement_outputs))
            after = mesh.num_program_cache_entries()
            errors = [bits(actual[b][r], reference[b][r]) for b in range(4) for r in range(4)]
            entry["rebind"].append(
                dict(
                    errors=errors,
                    before=before,
                    after=after,
                    old_addresses=[t.buffer_address() for t in old],
                    new_addresses=[t.buffer_address() for t in new],
                )
            )
            save()
            assert max(errors) == 0 and before == after
            # Stress queued generic-op launches with shared persistent outputs.
            for repeat in range(50):
                step = repeat % args.steps
                update(step)
                invoke(2, output_buffers=persistent)
            ttnn.synchronize_device(mesh)
            actual = read_outputs(persistent)
            errors = [bits(actual[b][r], eager[49 % args.steps][2][b][r]) for b in range(4) for r in range(4)]
            entry["queued_checks"].append(dict(kind="50_generic_op", errors=errors))
            assert max(errors) == 0
            if not watcher and not profiler and not tracked:
                for index in (0, 1, 2):
                    samples = []
                    for _ in range(args.samples):
                        ttnn.synchronize_device(mesh)
                        start = time.perf_counter()
                        output = invoke(index)  # All four outputs allocated, including direct.
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter() - start) * 1000)
                        del output
                    entry["eager_allocation_timings_ms"][report["paths"][index]] = samples
            for round_idx, order in enumerate(((0, 1, 2), (2, 1, 0))):
                for index in order:
                    update(0)
                    output = invoke(index)
                    ttnn.synchronize_device(mesh)
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    output = invoke(index)
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    try:
                        if tracked and round_idx == 0 and index == 0:
                            unsafe = ttnn.empty((1, 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh)
                            positive = False
                            try:
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            except RuntimeError as exc:
                                if not any(word in str(exc).lower() for word in ("unsafe", "allocation")):
                                    raise
                                positive = True
                            finally:
                                ttnn.deallocate(unsafe)
                            assert positive
                            entry["allocation_positive_control"] = positive
                        for step in range(args.steps):
                            update(step)
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            actual = read_outputs(output)
                            errors = [bits(actual[b][r], eager[step][index][b][r]) for b in range(4) for r in range(4)]
                            entry["trace_checks"].append(dict(round=round_idx, path=index, step=step, errors=errors))
                            save()
                            assert max(errors) == 0
                        for repeat in range(50):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        actual = read_outputs(output)
                        errors = [
                            bits(actual[b][r], eager[args.steps - 1][index][b][r]) for b in range(4) for r in range(4)
                        ]
                        entry["queued_checks"].append(dict(kind="50_trace", path=index, round=round_idx, errors=errors))
                        assert max(errors) == 0
                        if not watcher and not profiler and not tracked:
                            samples = []
                            for _ in range(args.samples):
                                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                                start = time.perf_counter()
                                for _ in range(args.replays):
                                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                                ttnn.synchronize_device(mesh)
                                samples.append((time.perf_counter() - start) * 1000 / args.replays)
                            entry["timings_ms"].setdefault(report["paths"][index], []).extend(samples)
                    finally:
                        ttnn.release_trace(mesh, trace)
                        del output
            # Separate explicit persistent-output trace verifies descriptor operands.
            update(0)
            invoke(2, output_buffers=persistent)
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            invoke(2, output_buffers=persistent)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                for step in range(args.steps):
                    update(step)
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = read_outputs(persistent)
                    errors = [bits(actual[b][r], eager[step][2][b][r]) for b in range(4) for r in range(4)]
                    entry["trace_checks"].append(dict(kind="persistent_outputs", step=step, errors=errors))
                    assert max(errors) == 0
            finally:
                ttnn.release_trace(mesh, trace)
            entry["timing_medians_ms"] = {
                name: statistics.median(samples) for name, samples in entry["timings_ms"].items()
            }
            entry["eager_allocation_medians_ms"] = {
                name: statistics.median(samples) for name, samples in entry["eager_allocation_timings_ms"].items()
            }
            if profiler:
                ttnn.ReadDeviceProfiler(mesh)
            save()
        report["status"] = "pass"
        report["pass"] = True
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
