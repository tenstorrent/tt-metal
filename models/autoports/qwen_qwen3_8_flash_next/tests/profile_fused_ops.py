# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fused-stage per-operation collection with bounded profiler drains.

Warm/setup work drains every64 device operations. Prefill collection also drains
between segments, so its per-op kernels are diagnostic and its host/gap time must
not rank candidates. The separate whole-trace harness measures full latency.
Decode replay has no intermediate drains. Inline kernel source is materialized
unchanged into files to avoid Tracy's unescaped multiline source-name CSV defect.

Run externally with python -m tracy -r -p -v. Final PCC comes from the paired
correctness probes; this runner never substitutes host wall time for device
performance. All state buffers, snapshots and input shapes predate capture.
"""

import argparse
import contextlib
import hashlib
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import torch
from tracy import signpost
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_linear_contract import runtime_audit
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import load_reference as load_ple
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_ple_decoder import token_rows
from models.autoports.qwen_qwen3_8_flash_next.tt.functional_decoder import FunctionalDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.packed_embedding import PackedEmbedding


@contextlib.contextmanager
def bounded_profiler(mesh, record):
    """Drain at host operation boundaries outside trace capture/replay."""
    previous = int(ttnn._ttnn.get_device_operation_id())
    busy = False

    def wrap(fn):
        def call(*args, **kwargs):
            nonlocal previous, busy
            result = fn(*args, **kwargs)
            current = int(ttnn._ttnn.get_device_operation_id())
            if not busy and current - previous >= 64:
                busy = True
                ttnn.synchronize_device(mesh)
                ttnn.ReadDeviceProfiler(mesh)
                record.append(current)
                previous = current
                busy = False
            return result

        return call

    with contextlib.ExitStack() as stack:
        for name in ("from_torch", "add", "multiply", "sum", "typecast", "concat", "generic_op", "slice", "reshape"):
            stack.enter_context(patch.object(ttnn, name, wrap(getattr(ttnn, name))))
        yield


@contextlib.contextmanager
def materialized_kernels(directory, record):
    """Change only source transport; preserve every byte and descriptor config."""
    original = ttnn.generic_op
    directory.mkdir(parents=True, exist_ok=True)

    def call(tensors, program, *args, **kwargs):
        kernels = program.kernels
        for kernel in kernels:
            if kernel.source_type == ttnn.KernelDescriptor.SourceType.SOURCE_CODE:
                source = kernel.kernel_source
                digest = hashlib.sha256(source.encode()).hexdigest()
                path = directory / ("kernel_" + digest + ".cpp")
                if digest not in record:
                    path.write_text(source)
                    record[digest] = str(path)
                kernel.kernel_source = str(path.resolve())
                kernel.source_type = ttnn.KernelDescriptor.SourceType.FILE_PATH
        program.kernels = kernels
        return original(tensors, program, *args, **kwargs)

    with patch.object(ttnn, "generic_op", call):
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 1, 3), required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/ttuser/qwen38-flash/packed_ple_nf4_absmax"))
    parser.add_argument(
        "--streaming-profile",
        action="store_true",
        help="Record exact raw program-ID ranges for the separate streaming CSV merger",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("TT_METAL_WATCHER"):
        raise ValueError("Watcher and profiling must be separate runs")
    if args.streaming_profile:
        if os.environ.get("TT_METAL_STREAMING_PROFILER") != "1" or os.environ.get("TT_METAL_DEVICE_PROFILER"):
            raise ValueError("Streaming collection needs TT_METAL_STREAMING_PROFILER=1 and no TT_METAL_DEVICE_PROFILER")
        if not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
            raise ValueError("Streaming contract requires TT_METAL_TRACE_ALLOC_TRACKING=1")
        if not os.environ.get("TT_METAL_STREAMING_PROFILER_OPS_CSV"):
            raise ValueError("TT_METAL_STREAMING_PROFILER_OPS_CSV is required")
    torch.set_num_threads(8)
    torch.manual_seed(2071 + args.layer)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    if args.layer == 1:
        reference, weights = load_ple(args.snapshot, config)
        del reference
    else:
        weights = load_weights(args.snapshot, args.layer, "")
    report = {
        "layer": args.layer,
        "mode": args.mode,
        "weights": "real",
        "seq_len": args.seq_len,
        "capacity": args.capacity if args.layer == 3 else None,
        "regime": "serial",
        "visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "snapshot": str(args.snapshot),
        "runtime_audit": True,
        "collection": "segmented_prefill" if args.mode == "prefill" else "uninterrupted_traced_decode",
        "end_to_end_latency_authority": "separate whole-trace harness",
        "bounded_drain_op_ids": [],
        "materialized_kernel_sha256": {},
        "streaming_profile": args.streaming_profile,
        "device_gap_measured": args.mode == "decode" and not args.streaming_profile,
        "expert_dtype": args.expert_dtype,
        "note": "Host milliseconds are diagnostic; use signposted tt-perf-report CSV for device latency.",
    }
    if args.streaming_profile:
        report["streaming_environment"] = {
            name: os.environ.get(name)
            for name in (
                "TT_METAL_STREAMING_PROFILER",
                "TT_METAL_STREAMING_PROFILER_DRAM_MB",
                "TT_METAL_STREAMING_PROFILER_FIFO_MB",
                "TT_METAL_STREAMING_PROFILER_OPS_CSV",
                "TT_METAL_STREAMING_PROFILER_ZONE_CSV",
                "TT_METAL_STREAMING_PROFILER_TRACY",
            )
        }
        report["streaming_selection"] = {
            "counter_semantics": "raw device operation IDs in half-open [begin,end); host CSV IDs encode (raw<<10)|device",
            "mode": args.mode,
            "trace_replay_calls": [],
        }
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    source_context = materialized_kernels(args.output.parent / "kernels", report["materialized_kernel_sha256"])
    source_context.__enter__()
    try:
        with bounded_profiler(mesh, report["bounded_drain_op_ids"]):
            packed = (
                PackedEmbedding.from_checkpoint(args.snapshot, cache_dir=args.cache_dir, mesh_device=mesh)
                if args.layer == 1
                else None
            )
            model = FunctionalDecoder.from_state_dict(
                weights,
                hf_config=config,
                layer_idx=args.layer,
                mesh_device=mesh,
                packed_embedding=packed,
                expert_dtype=getattr(ttnn, args.expert_dtype),
            )
            state = model.create_state(capacity=args.capacity, hash_lengths=(1, 32, args.seq_len % 32 or 32))

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        prompt = device((torch.randn(1, 1, args.seq_len, 10240) * 0.02).bfloat16())
        decode = device((torch.randn(1, 1, 1, 10240) * 0.02).bfloat16())
        prompt_ids = (
            device(token_rows(torch.randint(config.vocab_size, (1, args.seq_len))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if args.layer == 1
            else None
        )
        decode_ids = (
            device(token_rows(torch.randint(config.vocab_size, (1, 1))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if args.layer == 1
            else None
        )
        if args.layer == 3:
            buffers = [
                getattr(state.attention, name)
                for name in ("k_cache", "v_cache", "raw_history", "compressed_keys", "position")
            ]
        else:
            names = ("conv", "recurrent", "ple_conv", "token_history") if args.layer == 1 else ("conv", "recurrent")
            buffers = [getattr(state, name) for name in names]
        initial = [ttnn.clone(tensor) for tensor in buffers]
        prefix = [ttnn.clone(tensor) for tensor in buffers]

        def restore(saved):
            for src, dst in zip(saved, buffers):
                ttnn.copy(src, dst)

        def drain_profiler():
            # The standard API is a no-op in streaming mode; the resident
            # receiver drains continuously and flushes its CSV at device close.
            if not args.streaming_profile:
                ttnn.ReadDeviceProfiler(mesh)

        def operation_counter():
            # Host dispatcher bookkeeping, not a device/tensor readback.
            return int(ttnn._ttnn.get_device_operation_id())

        def prefill():
            with runtime_audit():
                return model.prefill_forward(prompt, state=state, token_ids=prompt_ids)

        def step():
            with runtime_audit():
                return model.decode_forward(decode, state=state, token_ids=decode_ids)

        # Observe actual sparse operands during unmeasured warmup. Retain only
        # the mask references; host inspection occurs after the runtime audit
        # and every reference is released before the measured call/capture.
        sparse_masks = []
        original_sparse_matmul = ttnn.sparse_matmul

        def observe_sparse(*operands, **options):
            sparse_masks.append((options["sparsity"], options.get("indices")))
            return original_sparse_matmul(*operands, **options)

        with patch.object(ttnn, "sparse_matmul", observe_sparse):
            with bounded_profiler(mesh, report["bounded_drain_op_ids"]):
                warm = prefill()
        ttnn.synchronize_device(mesh)
        active_counts = [
            int(indices.shape[-1]) if indices is not None else int(torch.count_nonzero(ttnn.to_torch(mask)))
            for mask, indices in sparse_masks
        ]
        sparse_masks.clear()
        report["prefill_sparse_active_experts_per_op"] = active_counts
        report["sparse_count_scope"] = "Actual warmup mask/index slots; read outside runtime audit and measurement"
        if len(active_counts) == 2 and len(set(active_counts)) == 1:
            report["prefill_sparse_active_experts"] = active_counts[0]
        ttnn.deallocate(warm)
        drain_profiler()
        for src, dst in zip(buffers, prefix):
            ttnn.copy(src, dst)
        if args.mode == "prefill":
            restore(initial)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            if args.streaming_profile:
                report["streaming_selection"]["begin"] = operation_counter()
            start = time.perf_counter()
            with bounded_profiler(mesh, report["bounded_drain_op_ids"]):
                output = prefill()
            if args.streaming_profile:
                report["streaming_selection"]["end"] = operation_counter()
            ttnn.synchronize_device(mesh)
            report["warmed_prefill_host_ms"] = (time.perf_counter() - start) * 1000
            signpost("PERF_PREFILL_END")
            drain_profiler()
        else:
            warm = step()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm)
            drain_profiler()
            restore(prefix)
            if args.streaming_profile:
                signpost("STREAMING_CAPTURE_BEGIN")
                report["streaming_selection"]["begin"] = operation_counter()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            output = step()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            if args.streaming_profile:
                report["streaming_selection"]["end"] = operation_counter()
                signpost("STREAMING_CAPTURE_END")
            trace_allocation_tracker.acknowledge_corruptible(output)
            restore(prefix)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            if args.streaming_profile:
                report["streaming_selection"]["trace_replay_calls"].append("unmeasured_warmup")
            ttnn.to_torch(output)
            drain_profiler()
            restore(prefix)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            start = time.perf_counter()
            with runtime_audit():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
            if args.streaming_profile:
                report["streaming_selection"]["trace_replay_calls"].append("measured_final")
            report["warmed_traced_decode_host_ms"] = (time.perf_counter() - start) * 1000
            signpost("PERF_DECODE_END")
            drain_profiler()
        report["finite"] = bool(torch.isfinite(ttnn.to_torch(output)).all())
        assert report["finite"]
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        source_context.__exit__(None, None, None)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
