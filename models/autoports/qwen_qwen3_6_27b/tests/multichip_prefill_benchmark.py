# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""Warmed TP4 prefill latency and PCC for both Qwen3.6 layer kinds."""

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("full", "linear"), required=True)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=7)
    parser.add_argument("--candidate", default="default")
    parser.add_argument("--result-json")
    args = parser.parse_args()
    if args.sequence < 1 or args.warmup < 1 or args.iterations < 3:
        raise ValueError("sequence/warmup must be positive and iterations must be at least three")

    ttnn.CONFIG.throw_exception_on_fallback = True
    torch.manual_seed(20260813)
    config = AutoConfig.from_pretrained(MODEL_ID).text_config
    layer = FULL_LAYER if args.kind == "full" else LINEAR_LAYER
    state = full_state(config) if args.kind == "full" else linear_state(config)
    hidden = (torch.randn(1, args.sequence, config.hidden_size) * 0.2).bfloat16()
    input_host = hidden.unsqueeze(0)
    page_host = torch.arange((args.sequence + 63) // 64, dtype=torch.int32).reshape(1, -1)
    position_host = torch.arange(args.sequence, dtype=torch.int64).to(torch.uint32).reshape(1, -1)

    one = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:
        baseline = OptimizedDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer,
            mesh_device=one,
            batch=1,
            max_context=max(64, args.sequence),
            candidate="default" if args.candidate.startswith("multichip_") else args.candidate,
        )
        expected_tt = baseline.prefill_forward(
            hidden_states=_to_device(input_host, mesh_device=one),
            page_table=_to_device(page_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.int32),
            current_positions=_to_device(
                position_host, mesh_device=one, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.uint32
            ),
        )
        ttnn.synchronize_device(one)
        expected = ttnn.to_torch(ttnn.get_device_tensors(expected_tt)[0]).float()
    finally:
        ttnn.close_mesh_device(one)

    ttnn.set_fabric_config(TARGET_FABRIC)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        decoder = MultichipDecoder.from_state_dict(
            state,
            hf_config=config,
            layer_idx=layer,
            mesh_device=mesh,
            batch=1,
            max_context=max(64, args.sequence),
            candidate=args.candidate,
        )
        hidden_states = upload(input_host, mesh)
        page_table = upload(page_host, mesh, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
        positions = upload(position_host, mesh, dtype=ttnn.uint32, layout=ttnn.ROW_MAJOR_LAYOUT)

        def prefill():
            return decoder.prefill_forward(
                hidden_states=hidden_states, page_table=page_table, current_positions=positions
            )

        output = prefill()
        ttnn.synchronize_device(mesh)
        replicas = [ttnn.to_torch(value).float() for value in ttnn.get_device_tensors(output)]
        passed, message = comp_pcc(expected, replicas[0], 0.995)
        if not passed:
            raise AssertionError(message)
        if not all(comp_pcc(replicas[0], other, 0.99999)[0] for other in replicas[1:]):
            raise AssertionError("TP4 prefill replicas differ")

        for _ in range(args.warmup):
            prefill()
        ttnn.synchronize_device(mesh)
        samples = []
        signpost("PERF_PREFILL")
        for _ in range(args.iterations):
            started = time.perf_counter()
            prefill()
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - started) * 1000.0)
        signpost("PERF_PREFILL_END")
        result = {
            "kind": args.kind,
            "candidate": args.candidate,
            "batch": 1,
            "logical_sequence": args.sequence,
            "mesh": [1, 4],
            "warmed_prefill_median_ms": statistics.median(samples),
            "samples_ms": samples,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "pcc": str(message),
            "replicas_equal": True,
            "fallback_audit": True,
        }
        print("MULTICHIP_PREFILL", json.dumps(result, sort_keys=True))
        if args.result_json:
            path = Path(args.result_json)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


if __name__ == "__main__":
    main()
