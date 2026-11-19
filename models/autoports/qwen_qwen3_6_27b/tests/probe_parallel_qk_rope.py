# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Parent-serialized TP4 QK norm/partial-RoPE correctness and whole-chain timing."""

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
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--replays", type=int, default=20)
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
    from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    torch.set_num_threads(4)
    hf_config = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
    hf_rotary = Qwen3_5TextRotaryEmbedding(hf_config)
    sources = [
        Path(__file__),
        ROOT / "tt/parallel_qk_rope_candidate.py",
        ROOT / "tt/parallel_qk_norm_candidate.py",
        ROOT / "tt/multichip_decoder.py",
    ]
    report = dict(
        status="running",
        mesh=[1, 4],
        watcher=watcher,
        profiler=profiler,
        allocation_tracking=tracked,
        skip_program_cache=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "unset"),
        cases=[],
        arguments=vars(args) | {"output": str(args.output)},
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        paths=["native_norm_rope", "parallel_norm_native_rope", "fused_norm_rope"],
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

        def upload(values, dtype=ttnn.bfloat16, *, preserve_nan_values=True):
            return ttnn.from_torch(
                torch.cat(values, -1).contiguous(),
                device=mesh,
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                # The default device tilize/typecast path may canonicalize IEEE
                # NaN to the hardware sentinel. Host construction preserves bits.
                preserve_nan_values=preserve_nan_values,
            )

        def read(t):
            return [ttnn.to_torch(v).contiguous() for v in ttnn.get_device_tensors(t)]

        def read_pair(pair):
            return [read(t) for t in pair]

        def bit_errors(a, b):
            return int((a.contiguous().view(torch.int16) != b.contiguous().view(torch.int16)).sum())

        def copy_physical_shards(source, destination, batch):
            """Setup-only raw tile copy; preserve dead rows despite logical head count.

            A public reshape may route through codegen and initialize padding.
            Copy exactly each owner's eight existing BF16 tiles, without a TM
            operator interpreting the destination's logical head count.
            """
            assert source.buffer_address() != destination.buffer_address()
            assert source.memory_config() == destination.memory_config()
            cores = [ttnn.CoreCoord(i % 8, i // 8) for i in range(batch)]
            ranges = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])
            runtime = ttnn.RuntimeArgs()
            for core in cores:
                physical_core = mesh.worker_core_from_logical_core(core)
                runtime[core.x][core.y] = [
                    source.buffer_address(),
                    destination.buffer_address(),
                    physical_core.x,
                    physical_core.y,
                ]
            kernel_source = r"""
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/endpoints.h"
#include "api/dataflow/noc.h"
void kernel_main() {
    const uint32_t src=get_arg_val<uint32_t>(0),dst=get_arg_val<uint32_t>(1);
    const uint32_t nx=get_arg_val<uint32_t>(2),ny=get_arg_val<uint32_t>(3);
    Noc noc;UnicastEndpoint local;CircularBuffer tiles(0);
    tiles.reserve_back(8);
    noc.async_read(local,tiles,16384,{.noc_x=nx,.noc_y=ny,.addr=src},{.offset_bytes=0});
    noc.async_read_barrier();tiles.push_back(8);tiles.wait_front(8);
    noc.async_write(use<CircularBuffer::AddrSelector::READ_PTR>(tiles),local,16384,
        {.offset_bytes=0},{.noc_x=nx,.noc_y=ny,.addr=dst});
    noc.async_write_barrier();tiles.pop_front(8);
}
"""
            kernel = ttnn.KernelDescriptor(
                kernel_source=kernel_source,
                source_type=ttnn.KernelDescriptor.SourceType.SOURCE_CODE,
                core_ranges=ranges,
                compile_time_args=[],
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
            entry = dict(batch=batch, checks=[], trace_checks=[], timings_ms={}, rebind=[])
            report["cases"].append(entry)
            save()
            generators = [torch.Generator().manual_seed(5231 + batch * 37 + rank) for rank in range(4)]
            weights_host = [
                [(torch.randn(1, 1, 256, generator=g) * 0.15 + 1).to(host_wdtype) for g in generators] for _ in range(2)
            ]
            weights = [upload(v, wdtype) for v in weights_host]
            owner = OptimizedDecoder()
            input_mem = OptimizedDecoder._compact_height_mem(
                owner, OptimizedDecoder._compact_cores(owner, batch), 32, 256
            )
            k_mem = OptimizedDecoder._compact_height_mem(
                owner, OptimizedDecoder._compact_cores(owner, batch, offset=32), 32, 256
            )
            rot_mem = OptimizedDecoder._compact_rope_mem(owner, batch, mesh.compute_with_storage_grid_size())
            banks, oracle = [], []
            for step in range(args.steps):
                host_qk, device_qk = [], []
                for heads in (6, 1):
                    padded = []
                    logical = []
                    for rank, g in enumerate(generators):
                        full = torch.full((1, batch, 32, 256), float("nan"), dtype=torch.bfloat16)
                        full[:, :, :heads] = (
                            torch.randn(1, batch, heads, 256, generator=g) * 0.25 + rank * 0.0625
                        ).bfloat16()
                        if step == 1:
                            full[0, 0, 0] = 0
                        padded.append(full)
                        logical.append(full[:, :, :heads].clone())

                    def padding_sample(value):
                        flat = value.contiguous().reshape(-1)
                        unique_bits = torch.unique(flat.view(torch.int16)).tolist()
                        sample_values = []
                        for v in flat[:8].float().tolist():
                            sample_values.append(v if v == v and abs(v) != float("inf") else str(v))
                        return dict(
                            unique_bf16_bits=[f"0x{(v & 0xffff):04x}" for v in unique_bits[:16]],
                            unique_bit_patterns=len(unique_bits),
                            first_values=sample_values,
                            nan_count=int(torch.isnan(flat).sum()),
                            finite_count=int(torch.isfinite(flat).sum()),
                            zero_count=int((flat == 0).sum()),
                        )

                    if not report.get("default_nan_transport_diagnostic"):
                        default_upload = upload(padded, preserve_nan_values=False)
                        diagnostic = []
                        for rank, shard in enumerate(ttnn.get_device_tensors(default_upload)):
                            actual = shard.cpu().to_torch_with_padded_shape()
                            diagnostic.append(
                                dict(
                                    rank=rank,
                                    expected=padding_sample(padded[rank][:, :, heads:, :]),
                                    actual=padding_sample(actual[:, :, heads:, :]),
                                )
                            )
                        report["default_nan_transport_diagnostic"] = diagnostic
                        report["poison_upload_policy"] = dict(
                            preserve_nan_values=True,
                            reason="Disable device tilize/typecast NaN canonicalization; require exact raw IEEE BF16 padding bits",
                        )
                        save()
                        del default_upload

                    def check_padding(tensor, stage):
                        counts = []
                        for rank, shard in enumerate(ttnn.get_device_tensors(tensor)):
                            actual = shard.cpu().to_torch_with_padded_shape().contiguous()
                            count = int(torch.isnan(actual[:, :, heads:, :]).sum())
                            valid_bits = bit_errors(actual[:, :, :heads], padded[rank][:, :, :heads])
                            padding_bits = bit_errors(actual[:, :, heads:], padded[rank][:, :, heads:])
                            diagnostic = dict(
                                step=step,
                                heads=heads,
                                stage=stage,
                                rank=rank,
                                logical_shape=list(tensor.shape),
                                padded_read_shape=list(actual.shape),
                                address=tensor.buffer_address(),
                                padding_nan=count,
                                padding_zero=int((actual[:, :, heads:, :] == 0).sum()),
                                valid_bit_errors=valid_bits,
                                padding_bit_errors=padding_bits,
                                padding_sample=padding_sample(actual[:, :, heads:, :]),
                            )
                            entry.setdefault("padding_diagnostics", []).append(diagnostic)
                            save()
                            assert tuple(actual.shape) == (1, batch, 32, 256), diagnostic
                            assert count == batch * (32 - heads) * 256 and valid_bits == padding_bits == 0, diagnostic
                            counts.append(count)
                        return counts

                    uploaded = upload(padded)
                    check_padding(uploaded, "full_physical_upload")
                    physical = ttnn.to_memory_config(uploaded, input_mem)
                    check_padding(physical, "full_physical_sharded")
                    tensor = ttnn.empty(
                        (1, batch, heads, 256),
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        device=mesh,
                        memory_config=input_mem,
                    )
                    copy_physical_shards(physical, tensor, batch)
                    padding_counts = check_padding(tensor, "raw_copy_into_logical_shard")
                    entry.setdefault("poisoned_padding", []).append(dict(step=step, heads=heads, counts=padding_counts))
                    host_qk.append(logical)
                    device_qk.append(tensor)
                # Original public [B,1,1,64] trig, distinct per rank/user/generation.
                trig_host = [[], []]
                positions = []
                for rank in range(4):
                    pos = torch.arange(batch) * 977 + rank * 37 + step * 31 + 17
                    cos_host, sin_host = hf_rotary(torch.zeros(batch, 1, 256, dtype=torch.bfloat16), pos[:, None])
                    trig_host[0].append(cos_host.unsqueeze(1).bfloat16())
                    trig_host[1].append(sin_host.unsqueeze(1).bfloat16())
                    positions.append(pos.tolist())
                trig = [upload(v) for v in trig_host]
                banks.append([*device_qk, *trig])
                refs = []
                for branch in range(2):
                    values = []
                    for rank in range(4):
                        raw = host_qk[branch][rank].float()
                        norm = (
                            raw
                            * torch.rsqrt(raw.square().mean(-1, keepdim=True) + args.epsilon)
                            * weights_host[branch][rank].float()
                        ).bfloat16()
                        c, s = [v[rank].permute(2, 0, 1, 3) for v in trig_host]
                        prefix = norm[..., :64]
                        rotated = torch.cat((-prefix[..., 32:], prefix[..., :32]), -1)
                        prefix = (
                            (prefix.float() * c.float()).bfloat16().float()
                            + (rotated.float() * s.float()).bfloat16().float()
                        ).bfloat16()
                        values.append(torch.cat((prefix, norm[..., 64:]), -1))
                    refs.append(values)
                oracle.append(refs)
                entry.setdefault("positions", []).append(positions)
            live = [ttnn.empty_like(t) for t in banks[0]]
            candidate_outputs = None
            checked_live_steps = set()

            def update(step):
                for index, (src, dst) in enumerate(zip(banks[step], live)):
                    if index < 2:
                        copy_physical_shards(src, dst, batch)
                    else:
                        ttnn.copy(src, dst)
                if step not in checked_live_steps:
                    for heads, tensor in zip((6, 1), live[:2]):
                        counts = [
                            int(torch.isnan(shard.cpu().to_torch_with_padded_shape()[:, :, heads:, :]).sum())
                            for shard in ttnn.get_device_tensors(tensor)
                        ]
                        entry.setdefault("live_poison_checks", []).append(dict(step=step, heads=heads, counts=counts))
                        save()
                        assert all(count == batch * (32 - heads) * 256 for count in counts), counts
                    checked_live_steps.add(step)

            def invoke(index, tensors=None, selected_weights=None, output_buffers=None):
                q, k, cos, sin = live if tensors is None else tensors
                ws = weights if selected_weights is None else selected_weights
                if index == 2:
                    return compact_qk_rope(
                        q,
                        k,
                        *ws,
                        cos,
                        sin,
                        epsilon=args.epsilon,
                        outputs=candidate_outputs if output_buffers is None else output_buffers,
                    )
                if index == 1:
                    normalized = compact_qk_norm(q, k, *ws, epsilon=args.epsilon)
                else:
                    normalized = tuple(
                        ttnn.rms_norm(
                            ttnn.to_memory_config(t, ttnn.DRAM_MEMORY_CONFIG),
                            weight=w,
                            epsilon=args.epsilon,
                            compute_kernel_config=ckc,
                        )
                        for t, w in zip((q, k), ws)
                    )
                trig = tuple(ttnn.to_memory_config(ttnn.permute(t, (2, 0, 1, 3)), rot_mem) for t in (cos, sin))

                def rotate(t):
                    prefix = ttnn.to_memory_config(t[..., :64], rot_mem)
                    prefix = ttnn.experimental.rotary_embedding_hf(
                        prefix, *trig, is_decode_mode=True, memory_config=rot_mem, compute_kernel_config=ckc
                    )
                    return ttnn.concat([ttnn.to_memory_config(prefix, ttnn.DRAM_MEMORY_CONFIG), t[..., 64:]], dim=-1)

                q_out, k_out = (rotate(t) for t in normalized)
                return q_out, ttnn.to_memory_config(k_out, k_mem)

            update(0)
            candidate_outputs = invoke(2)
            eager = []
            for step in range(args.steps):
                update(step)
                values = [read_pair(invoke(i)) for i in range(3)]
                eager.append(values)
                result = dict(step=step, native_pcc=[], selected_pcc=[], torch_pcc=[], selected_bits=[], tail_bits=[])
                for branch in range(2):
                    for rank in range(4):
                        native, selected, actual = [v[branch][rank] for v in values]
                        assert all(bool(torch.isfinite(t).all()) for t in (native, selected, actual))
                        result["native_pcc"].append(pcc(actual, native))
                        result["selected_pcc"].append(pcc(actual, selected))
                        result["torch_pcc"].append(pcc(actual, oracle[step][branch][rank]))
                        result["selected_bits"].append(bit_errors(actual, selected))
                        result["tail_bits"].append(bit_errors(actual[..., 64:], selected[..., 64:]))
                        for user in range(batch):
                            assert pcc(actual[:, user], selected[:, user]) >= 0.995
                entry["checks"].append(result)
                save()
                assert min(result["native_pcc"] + result["selected_pcc"] + result["torch_pcc"]) >= 0.995
                assert max(result["selected_bits"] + result["tail_bits"]) == 0, result
                for t, source in zip(live, banks[step]):
                    assert all(bit_errors(a, b) == 0 for a, b in zip(read(t), read(source)))

            # Rebind every address; retain originals so allocator reuse cannot fake the test.
            update(0)
            invoke(2)
            ttnn.synchronize_device(mesh)
            before_cache = mesh.num_program_cache_entries()
            replacement_live = [ttnn.empty_like(t) if i < 2 else ttnn.clone(t) for i, t in enumerate(banks[1])]
            for source, destination in zip(banks[1][:2], replacement_live[:2]):
                copy_physical_shards(source, destination, batch)
            replacement_weights = [
                upload([(v.float() * 0.9375 + 0.03125).to(host_wdtype) for v in group], wdtype)
                for group in weights_host
            ]
            replacement_outputs = [ttnn.empty_like(t) for t in candidate_outputs]
            for old, new in zip(
                [*live, *weights, *candidate_outputs], [*replacement_live, *replacement_weights, *replacement_outputs]
            ):
                assert old.buffer_address() != new.buffer_address()
            changed_reference = read_pair(invoke(1, replacement_live, replacement_weights))
            ttnn.synchronize_device(mesh)
            # Reference/setup may add cache entries; measure generic_op rebinding itself.
            before_rebind = mesh.num_program_cache_entries()
            actual = read_pair(invoke(2, replacement_live, replacement_weights, replacement_outputs))
            after_rebind = mesh.num_program_cache_entries()
            errors = [bit_errors(actual[b][r], changed_reference[b][r]) for b in range(2) for r in range(4)]
            entry["rebind"].append(
                dict(
                    errors=errors,
                    before_setup=before_cache,
                    before_rebind=before_rebind,
                    after_rebind=after_rebind,
                    old_addresses=[t.buffer_address() for t in [*live, *weights, *candidate_outputs]],
                    new_addresses=[
                        t.buffer_address() for t in [*replacement_live, *replacement_weights, *replacement_outputs]
                    ],
                )
            )
            save()
            assert max(errors) == 0 and before_rebind == after_rebind
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
                                if not any(x in str(exc).lower() for x in ("unsafe", "allocation")):
                                    raise
                                positive = True
                            finally:
                                ttnn.deallocate(unsafe)
                            assert positive
                            entry["allocation_positive_control"] = positive
                        for step in range(args.steps):
                            update(step)
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                            actual = read_pair(output)
                            errors = [
                                bit_errors(actual[b][r], eager[step][index][b][r]) for b in range(2) for r in range(4)
                            ]
                            entry["trace_checks"].append(dict(round=round_idx, path=index, step=step, errors=errors))
                            save()
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
            entry["timing_medians_ms"] = {
                name: statistics.median(samples) for name, samples in entry["timings_ms"].items()
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
