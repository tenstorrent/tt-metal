# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Trace a real TP4 linear->full decoder stack under one residual contract."""

import argparse
import json
import statistics
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
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import MODEL_ID
from models.autoports.qwen_qwen3_6_27b.tt.multichip_decoder import TARGET_FABRIC, MultichipDecoder
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


def upload_fractured(value, mesh):
    return ttnn.from_torch(
        value.contiguous(),
        device=mesh,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


def copy_host(value, destination, *, dtype=None, layout=ttnn.TILE_LAYOUT):
    host = ttnn.from_torch(value.contiguous(), dtype=dtype or destination.dtype, layout=layout)
    ttnn.copy_host_to_device_tensor(host, destination, cq_id=0)


def snapshot(decoder, names):
    return {name: [ttnn.to_torch(t) for t in ttnn.get_device_tensors(decoder.caches[name])] for name in names}


def restore(decoder, values):
    for name, per_rank in values.items():
        for destination, value in zip(ttnn.get_device_tensors(decoder.caches[name]), per_rank):
            copy_host(value, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--residual", choices=("replicated", "fractured"), required=True)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--reference-json", help="replicated result used for fractured output PCC")
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()
    if args.steps < 4:
        raise ValueError("at least four replay steps are required")

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    state = linear_state(config) | full_state(config)
    offsets = torch.arange(args.batch, dtype=torch.float32).reshape(args.batch, 1, 1) * 0.01
    tokens = [((torch.randn(args.batch, 1, 5120) * 0.2) + offsets + i * 0.03).bfloat16() for i in range(args.steps)]
    pages = torch.arange(args.batch, dtype=torch.int32).reshape(args.batch, 1).flip(0)

    reference = None
    if args.reference_json:
        reference = json.loads(Path(args.reference_json).read_text())["outputs"]

    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=8_000_000)
    trace_id = None
    trace_ended = False
    try:
        linear = MultichipDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=LINEAR_LAYER,
            mesh_device=mesh,
            batch=args.batch,
            max_context=64,
        )
        full = MultichipDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=FULL_LAYER,
            mesh_device=mesh,
            batch=args.batch,
            max_context=64,
        )
        initial_token = tokens[0].reshape(1, 1, args.batch, 5120)
        x = upload_fractured(initial_token, mesh) if args.residual == "fractured" else upload(initial_token, mesh)
        page = upload(pages, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        pos = upload(torch.zeros(args.batch, dtype=torch.uint32), mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)
        linear_initial = snapshot(linear, ("conv", "recurrent"))
        full_initial = snapshot(full, ("key", "value"))

        def decode():
            if args.residual == "fractured":
                boundary = linear.decode_forward_fractured(hidden_states=x, page_table=page, current_positions=pos)
                return full.decode_forward_fractured(hidden_states=boundary, page_table=page, current_positions=pos)
            boundary = linear.decode_forward(hidden_states=x, page_table=page, current_positions=pos)
            return full.decode_forward(hidden_states=boundary, page_table=page, current_positions=pos)

        for _ in range(2):
            decode()
            ttnn.synchronize_device(mesh)
            restore(linear, linear_initial)
            restore(full, full_initial)
            ttnn.synchronize_device(mesh)

        mesh.set_program_cache_misses_allowed(False)
        trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
        trace_output = decode()
        ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
        trace_ended = True
        restore(linear, linear_initial)
        restore(full, full_initial)
        ttnn.synchronize_device(mesh)

        samples = []
        output_summaries = []
        pcc = []
        signpost("PERF_DECODE_STACK")
        for step in range(args.steps):
            token = tokens[step].reshape(1, 1, args.batch, 5120)
            if args.residual == "fractured":
                host = ttnn.from_torch(
                    token.contiguous(),
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                ttnn.copy_host_to_device_tensor(host, x, cq_id=0)
            else:
                copy_host(token, x)
            copy_host(
                torch.full((args.batch,), step, dtype=torch.uint32),
                pos,
                dtype=ttnn.uint32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
            )
            started = time.perf_counter()
            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
            samples.append((time.perf_counter() - started) * 1000.0)
            if args.residual == "fractured":
                host = torch.cat(
                    [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(trace_output)], dim=-1
                )
            else:
                host = ttnn.to_torch(ttnn.get_device_tensors(trace_output)[0]).float()
            summary = {"mean": float(host.mean()), "std": float(host.std()), "sum": float(host.sum())}
            output_summaries.append(summary)
            if reference is not None:
                expected = torch.tensor(reference[step]["tensor"], dtype=torch.float32).reshape(host.shape)
                passed, message = comp_pcc(expected, host, 0.995)
                if not passed:
                    raise AssertionError(message)
                pcc.append(str(message))
            summary["tensor"] = host.flatten().tolist()
        signpost("PERF_DECODE_STACK_END")

        result = {
            "residual": args.residual,
            "batch": args.batch,
            "steps": args.steps,
            "mesh": [1, 4],
            "trace_median_ms": statistics.median(samples[1:]),
            "samples_ms": samples,
            "pcc_vs_replicated": pcc,
            "result_gather_outside_trace": args.residual == "fractured",
            "inter_layer_collective": False,
            "inter_layer_width": 1280 if args.residual == "fractured" else 5120,
            "fallback_audit": True,
            "outputs": output_summaries,
        }
        path = Path(args.result_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("MULTICHIP_STACK_TRACE", json.dumps({k: v for k, v in result.items() if k != "outputs"}, sort_keys=True))
    finally:
        if trace_id is not None and trace_ended:
            ttnn.release_trace(mesh, trace_id)
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
