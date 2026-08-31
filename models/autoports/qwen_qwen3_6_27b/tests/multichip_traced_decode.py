# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed TP4 trace replay against sequential optimized 1x1 TTNN outputs."""

import argparse
import json
import time
from pathlib import Path

import torch
from tracy import signpost
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import LAYER as FULL_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.full_attention_synthetic_pcc import _state as full_state
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import LAYER as LINEAR_LAYER
from models.autoports.qwen_qwen3_6_27b.tests.linear_attention_synthetic_pcc import _state as linear_state
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID, _to_device
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
from models.autoports.qwen_qwen3_6_27b.tt.optimized_decoder import OptimizedDecoder
from models.common.utility_functions import comp_pcc


def upload(value, mesh, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
        dtype=dtype,
        layout=layout,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def copy_host(value, destination, *, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, mesh_mapper=None):
    host = ttnn.from_torch(value.contiguous(), dtype=dtype, layout=layout, mesh_mapper=mesh_mapper)
    ttnn.copy_host_to_device_tensor(host, destination, cq_id=0)


def runtime_args(mesh, batch):
    pages = torch.arange(batch, dtype=torch.int32).reshape(batch, 1).flip(0)
    page = upload(pages, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
    pos = upload(torch.zeros(batch, dtype=torch.uint32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
    return page, pos


def dram_snapshot(mesh):
    ttnn.synchronize_device(mesh)
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    banks = int(view.num_banks)
    return {
        "num_banks": banks,
        "total_bytes": banks * int(view.total_bytes_per_bank),
        "allocated_bytes": banks * int(view.total_bytes_allocated_per_bank),
        "free_bytes": banks * int(view.total_bytes_free_per_bank),
        "largest_contiguous_bytes_free_per_bank": int(view.largest_contiguous_bytes_free_per_bank),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=1)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--baseline-candidate", default="default")
    parser.add_argument("--result-json")
    parser.add_argument("--forbid-program-cache-misses", action="store_true")
    args = parser.parse_args()
    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer = FULL_LAYER if args.kind == "full" else LINEAR_LAYER
    state = full_state(config) if args.kind == "full" else linear_state(config)
    row_offset = torch.arange(args.batch, dtype=torch.float32).reshape(args.batch, 1, 1) * 0.01
    tokens = [
        ((torch.randn(args.batch, 1, 5120) * 0.2) + row_offset + step * 0.03).bfloat16() for step in range(args.steps)
    ]

    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=4_000_000)
    baseline_trace_id = None
    baseline_trace_ended = False
    try:
        baseline = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer,
            mesh_device=one,
            batch=args.batch,
            max_context=64,
            candidate=args.baseline_candidate,
        )
        x = _to_device(tokens[0].reshape(1, 1, args.batch, 5120), mesh_device=one)
        page, pos = runtime_args(one, args.batch)
        cache_names = ("key", "value") if args.kind == "full" else ("conv", "recurrent")
        initial = {name: ttnn.to_torch(ttnn.get_device_tensors(baseline.caches[name])[0]) for name in cache_names}

        def baseline_decode():
            return baseline.decode_forward(hidden_states=x, page_table=page, current_positions=pos)

        for _ in range(2):
            baseline_decode()
            ttnn.synchronize_device(one)
            for name in cache_names:
                copy_host(
                    initial[name],
                    baseline.caches[name],
                    dtype=baseline.caches[name].dtype,
                    layout=baseline.caches[name].layout,
                )
            ttnn.synchronize_device(one)
        baseline_trace_id = ttnn.begin_trace_capture(one, cq_id=0)
        baseline_trace_output = baseline_decode()
        ttnn.end_trace_capture(one, baseline_trace_id, cq_id=0)
        baseline_trace_ended = True

        # Capture records the commands; replay token zero to seed the cache and
        # validate the capture-position result before timed autoregressive steps.
        for name in cache_names:
            copy_host(
                initial[name],
                baseline.caches[name],
                dtype=baseline.caches[name].dtype,
                layout=baseline.caches[name].layout,
            )
        copy_host(tokens[0].reshape(1, 1, args.batch, 5120), x)
        copy_host(torch.zeros(args.batch, dtype=torch.uint32), pos, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(one)
        ttnn.execute_trace(one, baseline_trace_id, cq_id=0, blocking=True)
        expected = [ttnn.to_torch(ttnn.get_device_tensors(baseline_trace_output)[0]).reshape(args.batch, 1, 5120)]
        baseline_times = []
        for step in range(1, args.steps):
            copy_host(tokens[step].reshape(1, 1, args.batch, 5120), x)
            copy_host(
                torch.full((args.batch,), step, dtype=torch.uint32),
                pos,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            started = time.perf_counter()
            ttnn.execute_trace(one, baseline_trace_id, cq_id=0, blocking=True)
            baseline_times.append((time.perf_counter() - started) * 1000)
            expected.append(
                ttnn.to_torch(ttnn.get_device_tensors(baseline_trace_output)[0]).reshape(args.batch, 1, 5120)
            )
    finally:
        if baseline_trace_id is not None and baseline_trace_ended:
            ttnn.release_trace(one, baseline_trace_id)
        ttnn.close_mesh_device(one)

    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=4_000_000)
    trace_id = None
    trace_ended = False
    try:
        decoder = MultichipDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer,
            mesh_device=mesh,
            batch=args.batch,
            max_context=64,
            candidate=args.candidate,
        )
        memory_snapshots = {"after_load": dram_snapshot(mesh)}
        x = upload(tokens[0].reshape(1, 1, args.batch, 5120), mesh)
        page, pos = runtime_args(mesh, args.batch)
        cache_names = ("key", "value") if args.kind == "full" else ("conv", "recurrent")
        initial = {name: ttnn.to_torch(ttnn.get_device_tensors(decoder.caches[name])[0]) for name in cache_names}

        def decode():
            return decoder.decode_forward(hidden_states=x, page_table=page, current_positions=pos)

        decode()
        ttnn.synchronize_device(mesh)
        memory_snapshots["after_first_warm"] = dram_snapshot(mesh)
        for name in cache_names:
            copy_host(
                initial[name],
                decoder.caches[name],
                dtype=decoder.caches[name].dtype,
                layout=decoder.caches[name].layout,
            )
        # Re-run the exact path immediately before capture. Some TTNN program
        # signatures depend on transient allocation state, so an earlier first
        # compile can leave a cold variant at the capture boundary.
        ttnn.synchronize_device(mesh)
        decode()
        ttnn.synchronize_device(mesh)
        memory_snapshots["after_second_warm"] = dram_snapshot(mesh)
        for name in cache_names:
            copy_host(
                initial[name],
                decoder.caches[name],
                dtype=decoder.caches[name].dtype,
                layout=decoder.caches[name].layout,
            )
        # Mesh host writes are asynchronous and must finish before capture.
        ttnn.synchronize_device(mesh)
        if args.forbid_program_cache_misses:
            mesh.set_program_cache_misses_allowed(False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        trace_ended = True
        memory_snapshots["after_capture"] = dram_snapshot(mesh)
        for name in cache_names:
            copy_host(
                initial[name],
                decoder.caches[name],
                dtype=decoder.caches[name].dtype,
                layout=decoder.caches[name].layout,
            )
        copy_host(tokens[0].reshape(1, 1, args.batch, 5120), x)
        copy_host(torch.zeros(args.batch, dtype=torch.uint32), pos, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        ttnn.synchronize_device(mesh)
        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
        capture_replicas = [
            ttnn.to_torch(t).reshape(args.batch, 1, 5120) for t in ttnn.get_device_tensors(trace_output)
        ]
        capture_passed, capture_message = comp_pcc(expected[0].float(), capture_replicas[0].float(), 0.995)
        assert capture_passed, capture_message
        first_run_outputs = [capture_replicas[0].clone()]
        replay_times, pcc = [], []
        signpost("PERF_DECODE")
        for step in range(1, args.steps):
            copy_host(tokens[step].reshape(1, 1, args.batch, 5120), x)
            copy_host(
                torch.full((args.batch,), step, dtype=torch.uint32),
                pos,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            started = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            replay_times.append((time.perf_counter() - started) * 1000)
            replicas = [ttnn.to_torch(t).reshape(args.batch, 1, 5120) for t in ttnn.get_device_tensors(trace_output)]
            passed, message = comp_pcc(expected[step].float(), replicas[0].float(), 0.995)
            assert passed, message
            assert all(comp_pcc(replicas[0].float(), other.float(), 0.99999)[0] for other in replicas[1:])
            pcc.append(str(message))
            first_run_outputs.append(replicas[0].clone())
        signpost("PERF_DECODE_END")
        # Rewind mutable state and replay the same token/position stream.  This
        # stresses trace determinism as well as stateful cache reset semantics.
        for name in cache_names:
            copy_host(
                initial[name],
                decoder.caches[name],
                dtype=decoder.caches[name].dtype,
                layout=decoder.caches[name].layout,
            )
        deterministic_pcc = []
        for step in range(args.steps):
            copy_host(tokens[step].reshape(1, 1, args.batch, 5120), x)
            copy_host(
                torch.full((args.batch,), step, dtype=torch.uint32),
                pos,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            repeated = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).reshape(args.batch, 1, 5120)
            passed, message = comp_pcc(first_run_outputs[step].float(), repeated.float(), 0.99999)
            assert passed, message
            deterministic_pcc.append(str(message))
        memory_snapshots["after_replays"] = dram_snapshot(mesh)
        baseline_ms = float(torch.tensor(baseline_times).median())
        multichip_ms = float(torch.tensor(replay_times).median())
        result = {
            "kind": args.kind,
            "candidate": args.candidate,
            "baseline_candidate": args.baseline_candidate,
            "batch": args.batch,
            "steps": args.steps,
            "pcc": pcc,
            "single_chip_median_ms": baseline_ms,
            "multichip_trace_median_ms": multichip_ms,
            "speedup": baseline_ms / multichip_ms,
            "efficiency": baseline_ms / multichip_ms / 4,
            "capture_position_pcc": str(capture_message),
            "fallback_audit": True,
            "single_chip_trace_replay": True,
            "trace_replay": True,
            "deterministic_replay": True,
            "deterministic_pcc": deterministic_pcc,
            "dram_memory_snapshots": memory_snapshots,
        }
        print("MULTICHIP_TRACED_DECODE", json.dumps(result, sort_keys=True))
        if args.result_json:
            Path(args.result_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.result_json).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        if trace_id is not None and trace_ended:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
