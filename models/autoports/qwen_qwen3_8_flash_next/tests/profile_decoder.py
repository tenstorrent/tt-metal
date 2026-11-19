# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Serial warmed decoder windows for Tracy, separate from watcher/correctness.

Run externally with python -m tracy -r -p -v. Final PCC comes from the paired
correctness probes; this runner never substitutes host wall time for device
performance. All state buffers, snapshots and input shapes predate capture.
"""
import argparse
import json
import os
import time
from pathlib import Path

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
        "streaming_profile": args.streaming_profile,
        "device_gap_measured": not args.streaming_profile,
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
    try:
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

        warm = prefill()
        ttnn.synchronize_device(mesh)
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
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
