# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""One whole-decoder trace replay for the small-buffer trace-only profiler.

Prefill and decode are separate processes. This measures the complete replay's
device interval, including internal idle gaps; it provides no per-op timings.
Numerical evidence here is exact eager/replay output AND persistent-state
equality. HF acceptance remains bound to the separate correctness artifacts.
"""

import argparse
import hashlib
import json
import os
from contextlib import nullcontext
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
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device-only", action="store_true", help="Explicit harness protocol; no Tracy host client")
    parser.add_argument("--profiler-dir", type=Path, help="Fresh native device CSV directory for --device-only")
    parser.add_argument(
        "--clean-timing",
        action="store_true",
        help="Disable runtime audits; require allocation tracking and diagnostics off",
    )
    args = parser.parse_args()
    if not 1 <= args.seq_len <= 32:
        parser.error("Whole-trace profiling fixture uses a complete1..32-token public prefill")
    if args.layer == 1 and args.cache_dir is None:
        parser.error("Real full-resident PLE requires --cache-dir")
    if args.device_only:
        if args.profiler_dir is None:
            parser.error("--device-only requires --profiler-dir")
        if (args.profiler_dir / ".logs/profile_log_device.csv").exists():
            raise ValueError("Use a fresh profiler directory; existing evidence is preserved")
        if Path(os.environ.get("TT_METAL_PROFILER_DIR", "")).resolve() != args.profiler_dir.resolve():
            raise ValueError("Set TT_METAL_PROFILER_DIR to --profiler-dir before starting Python")
    required = {"TT_METAL_DEVICE_PROFILER": "1", "TT_METAL_TRACE_PROFILER": "1"}
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise ValueError("Use the trace-only capture wrapper with the classic device profiler enabled")
    for key in (
        "TT_METAL_WATCHER",
        "TT_METAL_STREAMING_PROFILER",
        "TT_METAL_PROFILER_MID_RUN_DUMP",
        "TT_METAL_DEVICE_PROFILER_DISPATCH",
        "TT_METAL_PROFILER_ACCUMULATE",
        "TT_METAL_PROFILER_CPP_POST_PROCESS",
        "TT_METAL_PROFILER_DISABLE_DUMP_TO_FILES",
    ):
        if os.environ.get(key) not in (None, "", "0"):
            raise ValueError(f"Incompatible profiling setting: {key}")
    if args.clean_timing:
        if trace_allocation_tracker.TRACE_ALLOC_TRACKING or trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS:
            raise ValueError("Clean timing requires trace allocation tracking and diagnostics disabled")
        if any(
            os.environ.get(key) != "0" for key in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS")
        ):
            raise ValueError("Explicitly set trace allocation tracking and tracebacks to0 for clean timing")
    elif not trace_allocation_tracker.TRACE_ALLOC_TRACKING:
        raise ValueError("Tracked correctness profiling requires TT_METAL_TRACE_ALLOC_TRACKING=1")
    audit = nullcontext if args.clean_timing else runtime_audit
    torch.set_num_threads(8)
    torch.manual_seed(2071 + args.layer)
    config = AutoConfig.from_pretrained(args.snapshot, local_files_only=True).text_config
    if args.layer == 1:
        reference, weights = load_ple(args.snapshot, config)
        del reference
    else:
        weights = load_weights(args.snapshot, args.layer, "")
    report = {
        "mode": args.mode,
        "layer": args.layer,
        "seq_len": args.seq_len,
        "capacity": args.capacity if args.layer == 3 else None,
        "snapshot": str(args.snapshot),
        "cache_dir": str(args.cache_dir) if args.cache_dir else None,
        "expert_dtype": args.expert_dtype,
        "weights": "real; complete layer and full PLE table when applicable",
        "profiler_mode": "whole_trace_aggregate",
        "scope": __doc__,
        "runtime_audit": False,
        "allocation_tracking": bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING),
        "allocation_diagnostics": bool(trace_allocation_tracker.TRACE_ALLOC_DIAGNOSTICS),
        "measurement_regime": "clean_timing" if args.clean_timing else "tracked_correctness",
        "runtime_audit_scope": "disabled; separate tracked correctness evidence required"
        if args.clean_timing
        else "all forwards and replays",
        "trace_replay_calls": [],
        "per_op_timings_available": False,
        "metadata_authority": "harness_trace_protocol_v1" if args.device_only else "tracy_host_capture",
        "environment": {
            **{
                key: value
                for key, value in os.environ.items()
                if key.startswith("TT_METAL_PROFILER") or key in required
            },
            **{
                key: os.environ.get(key)
                for key in ("TT_METAL_TRACE_ALLOC_TRACKING", "TT_METAL_TRACE_ALLOC_TRACEBACKS", "TT_METAL_WATCHER")
            },
        },
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
        state = model.create_state(capacity=args.capacity, hash_lengths=(1, 32, args.seq_len))
        protocol = None
        if args.device_only:
            from models.autoports.qwen_qwen3_8_flash_next.tests.profile_trace_protocol import (
                prepare_protocol,
                record_event,
                sample_predrain,
                save_partial,
            )

            protocol = prepare_protocol(mesh, model)
            report["measurement_protocol"] = protocol

        def device(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(value, dtype=dtype, layout=layout, device=mesh)

        prompt = device((torch.randn(1, 1, args.seq_len, 10240) * 0.02).bfloat16())
        token = device((torch.randn(1, 1, 1, 10240) * 0.02).bfloat16())
        prompt_ids = (
            device(token_rows(torch.randint(config.vocab_size, (1, args.seq_len))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if args.layer == 1
            else None
        )
        token_ids = (
            device(token_rows(torch.randint(config.vocab_size, (1, 1))), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
            if args.layer == 1
            else None
        )
        owner = state.attention if args.layer == 3 else state
        names = (
            ("k_cache", "v_cache", "raw_history", "compressed_keys", "position", "page_table")
            if args.layer == 3
            else (("conv", "recurrent", "ple_conv", "token_history") if args.layer == 1 else ("conv", "recurrent"))
        )
        buffers = {name: getattr(owner, name) for name in names}

        def snapshot():
            return {name: ttnn.to_torch(value).clone() for name, value in buffers.items()}

        def prepared(saved):
            return {
                name: ttnn.from_torch(value.contiguous(), dtype=buffers[name].dtype, layout=buffers[name].layout)
                for name, value in saved.items()
            }

        def restore(saved):
            for name, value in saved.items():
                ttnn.copy_host_to_device_tensor(value, buffers[name])

        def memory():
            ttnn.synchronize_device(mesh)
            view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
            return {
                "allocated_bytes": view.num_banks * view.total_bytes_allocated_per_bank,
                "free_bytes": view.num_banks * view.total_bytes_free_per_bank,
            }

        def prefill():
            return model.prefill_forward(prompt, state=state, token_ids=prompt_ids)

        def decode():
            return model.decode_forward(token, state=state, token_ids=token_ids)

        initial = snapshot()
        with audit():
            eager = prefill()
        prefix = snapshot()
        if args.mode == "decode":
            ttnn.deallocate(eager)
            with audit():
                eager = decode()
        baseline_output = ttnn.to_torch(eager).clone()
        baseline_state = snapshot()
        ttnn.deallocate(eager)
        reset = prepared(initial if args.mode == "prefill" else prefix)
        restore(reset)
        ttnn.ReadDeviceProfiler(mesh)
        report["before_capture_memory"] = memory()
        forward = prefill if args.mode == "prefill" else decode
        report["capture_begin_raw_op_id"] = int(ttnn._ttnn.get_device_operation_id())
        if protocol is not None:
            record_event(protocol, "capture_begin", raw_op_id=report["capture_begin_raw_op_id"])
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with audit():
            output = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        report["capture_end_raw_op_id"] = int(ttnn._ttnn.get_device_operation_id())
        report["trace_id"] = int(trace)
        if protocol is not None:
            record_event(protocol, "capture_end", raw_op_id=report["capture_end_raw_op_id"], trace_id=int(trace))
        if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
            trace_allocation_tracker.acknowledge_corruptible(output)
        report["after_capture_memory"] = memory()

        def correctness():
            actual = ttnn.to_torch(output)
            observed = snapshot()
            return {
                "finite": bool(torch.isfinite(actual).all()),
                "output_exact_vs_eager": torch.equal(actual, baseline_output),
                "state_exact_vs_eager": {name: torch.equal(observed[name], baseline_state[name]) for name in names},
            }

        restore(reset)
        if protocol is not None:
            record_event(
                protocol,
                "replay_begin",
                ordinal=1,
                trace_id=int(trace),
                raw_op_id=int(ttnn._ttnn.get_device_operation_id()),
            )
        with audit():
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
        if protocol is not None:
            record_event(
                protocol,
                "replay_end",
                ordinal=1,
                trace_id=int(trace),
                raw_op_id=int(ttnn._ttnn.get_device_operation_id()),
            )
            sample_predrain(protocol, 1)
            save_partial(report, args.output)
        report["trace_replay_calls"].append("unmeasured_warmup")
        # Drain before any reset/device computation can extend the trace's final
        # marker. Host reads and state comparisons are outside measured windows.
        ttnn.ReadDeviceProfiler(mesh)
        report["warm_replay"] = correctness()
        restore(reset)
        ttnn.synchronize_device(mesh)
        if protocol is None:
            signpost(f"PERF_{args.mode.upper()}")
        else:
            record_event(protocol, f"PERF_{args.mode.upper()}")
            record_event(
                protocol,
                "replay_begin",
                ordinal=2,
                trace_id=int(trace),
                raw_op_id=int(ttnn._ttnn.get_device_operation_id()),
            )
        with audit():
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
        if protocol is None:
            signpost(f"PERF_{args.mode.upper()}_END")
        else:
            record_event(
                protocol,
                "replay_end",
                ordinal=2,
                trace_id=int(trace),
                raw_op_id=int(ttnn._ttnn.get_device_operation_id()),
            )
            record_event(protocol, f"PERF_{args.mode.upper()}_END")
            sample_predrain(protocol, 2)
            save_partial(report, args.output)
        report["trace_replay_calls"].append("measured_final")
        ttnn.ReadDeviceProfiler(mesh)
        report["measured_replay"] = correctness()
        report["runtime_audit"] = not args.clean_timing
        report["passed"] = all(
            check["finite"] and check["output_exact_vs_eager"] and all(check["state_exact_vs_eager"].values())
            for check in (report["warm_replay"], report["measured_replay"])
        )
        assert report["passed"], "Captured whole pass differs from eager output or state"
    except Exception as error:
        report["error"] = str(error).split("backtrace:")[0]
        raise
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        if args.device_only:
            raw = args.profiler_dir / ".logs/profile_log_device.csv"
            report["native_csv"] = {
                "path": str(raw),
                "exists": raw.exists(),
                "sha256": hashlib.sha256(raw.read_bytes()).hexdigest() if raw.exists() else None,
            }
            report["timing_acceptance"] = "Pending strict offline official-parser and known-core inventory validation"
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
