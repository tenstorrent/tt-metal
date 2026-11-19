# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Experimental decode-only compact expert graph, compared with the fused512 graph."""

import argparse
import json
import statistics
import time
from copy import copy
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import SparseMoe, WeightLoader
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedPreciseProjection, FusedSparseMoe


def compact_moe(owner, x):
    assert owner.compact_experts
    return owner(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert-dtype", choices=["bfloat16", "bfloat8_b"], default="bfloat16")
    p.add_argument("--pack-experts", action="store_true")
    p.add_argument("--packed-gate-cores", type=int, choices=(5, 10, 20, 40), default=5)
    a = p.parse_args()
    assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
    torch.set_num_threads(8)
    torch.manual_seed(618)
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    weights = load_weights(a.snapshot, a.layer, "mlp.")
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    report = dict(
        rows=[],
        passed=False,
        layer=a.layer,
        expert_dtype=a.expert_dtype,
        pack_experts=a.pack_experts,
        packed_gate_cores=a.packed_gate_cores,
    )
    trace = None
    try:
        loader = WeightLoader(weights, device)
        original = SparseMoe(loader, cfg, expert_dtype=getattr(ttnn, a.expert_dtype))
        owner = FusedSparseMoe.__new__(FusedSparseMoe)
        owner.__dict__.update(original.__dict__)
        owner.fuse_expert_activation = owner.fuse_shared_activation = True
        owner.pack_projections(shared=True)
        owner.prepare_fused_reduce(device)
        owner.compact_sparsity = ttnn.from_torch(
            torch.ones(1, 1, 1, 512, dtype=torch.bfloat16), device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        owner.precise_router = FusedPreciseProjection(owner.precise_router, owner.compute, 64)
        baseline = copy(owner)
        owner.compact_experts = True
        if a.pack_experts:
            owner.pack_projections(experts=True)
            # Packing doubles N. Match the original per-core output width with
            # ten cores, then compare narrower column partitions of that fusion.
            owner.program = copy(owner.program)
            owner.program.decode_gate_up_cores = (5, 1) if a.packed_gate_cores == 5 else (10, a.packed_gate_cores // 10)
        for case in range(4):
            x = loader.tensor((torch.randn(1, 1, 1, 2560) * 0.02).bfloat16())
            expected = ttnn.to_torch(baseline(x))
            actual = ttnn.to_torch(compact_moe(owner, x))
            pcc = float(torch.corrcoef(torch.stack([expected.float().flatten(), actual.float().flatten()]))[0, 1])
            row = dict(
                case=case,
                pcc=pcc,
                exact=torch.equal(expected, actual),
                max_abs=float((expected.float() - actual.float()).abs().max()),
                times={},
            )
            for name, fn in [("expanded", lambda: baseline(x)), ("compact", lambda: compact_moe(owner, x))]:
                fn()
                trace = ttnn.begin_trace_capture(device, cq_id=0)
                output = fn()
                ttnn.end_trace_capture(device, trace, cq_id=0)
                trace_allocation_tracker.acknowledge_corruptible(output)
                samples = []
                for _ in range(12):
                    start = time.perf_counter()
                    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - start) * 1000)
                replay = ttnn.to_torch(output)
                assert torch.equal(replay, expected if name == "expanded" else actual)
                row["times"][name] = statistics.median(samples[2:])
                ttnn.release_trace(device, trace)
                trace = None
            report["rows"].append(row)
            print(row, flush=True)
        report["passed"] = all(r["pcc"] >= 0.995 for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(device, trace)
        ttnn.close_mesh_device(device)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
