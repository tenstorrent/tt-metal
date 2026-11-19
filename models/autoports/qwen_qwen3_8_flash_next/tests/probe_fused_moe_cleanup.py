# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Isolated tests of remaining redundant MoE casts/layouts and gate activation."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.probe_components import load_weights
from models.autoports.qwen_qwen3_8_flash_next.tt import fused_decoder as fused
from models.autoports.qwen_qwen3_8_flash_next.tt.decoder_components import SparseMoe, WeightLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--clean-timing", action="store_true")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--expert-dtype", choices=("bfloat16", "bfloat8_b"), default="bfloat16")
    a = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(7139)
    assert bool(trace_allocation_tracker.TRACE_ALLOC_TRACKING) != a.clean_timing
    cfg = AutoConfig.from_pretrained(a.snapshot, local_files_only=True).text_config
    weights = load_weights(a.snapshot, a.layer, "mlp.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    report = dict(
        rows=[],
        passed=False,
        layer=a.layer,
        expert_dtype=a.expert_dtype,
        clean_timing=a.clean_timing,
        watcher=os.environ.get("TT_METAL_WATCHER"),
        candidate_sources={},
    )
    try:
        loader = WeightLoader(weights, mesh)
        original = SparseMoe(loader, cfg, expert_dtype=getattr(ttnn, a.expert_dtype))
        owner = fused.FusedSparseMoe.__new__(fused.FusedSparseMoe)
        owner.__dict__.update(original.__dict__)
        del original
        owner.compact_experts = True
        owner.fuse_expert_activation = (
            owner.fuse_shared_activation
        ) = owner.fuse_packed_activation = owner.early_packed_activation = True
        owner.pack_projections(
            shared=True, experts=True, host_experts=weights["mlp.experts.gate_up_proj"], packed_gate_cores=(10, 4)
        )
        owner.prepare_fused_reduce(mesh)
        owner.precise_router = fused.FusedPreciseProjection(
            owner.precise_router, owner.compute, 64, widen_in_unpack=True, host_weight=weights["mlp.gate.weight"]
        )
        owner.fuse_moe_cleanup = False
        cleaned = fused.FusedSparseMoe.__new__(fused.FusedSparseMoe)
        cleaned.__dict__.update(owner.__dict__)
        cleaned.fuse_moe_cleanup = True
        models = {"current": owner, "all": cleaned}
        path = a.output.with_name(a.output.stem + "_runtime.py.txt")
        source = inspect.getsource(fused.FusedSparseMoe.__call__)
        path.write_text(source)
        report["candidate_sources"] = dict(
            path=str(path),
            sha256=hashlib.sha256(source.encode()).hexdigest(),
            runtime_class="FusedSparseMoe",
            policies={"current": False, "all": True},
        )
        for count in (1, 2, 7, 16, 17, 31, 32):
            x = loader.tensor((torch.randn(1, 1, count, 2560) * 0.02).bfloat16())
            expected = ttnn.to_torch(owner(x))
            for name, model in models.items():
                actual = ttnn.to_torch(model(x))
                row = dict(
                    count=count,
                    variant=name,
                    exact=torch.equal(actual, expected),
                    max_abs=float((actual.float() - expected.float()).abs().max()),
                )
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                try:
                    out = model(x)
                finally:
                    # Even a rejected operation must leave CQ capture/bypass
                    # before release and mesh close; release alone cannot do it.
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                if trace_allocation_tracker.TRACE_ALLOC_TRACKING:
                    trace_allocation_tracker.acknowledge_corruptible(out)
                samples = []
                for _ in range(16):
                    start = time.perf_counter()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    samples.append((time.perf_counter() - start) * 1000)
                row["replay_exact"] = torch.equal(ttnn.to_torch(out), actual)
                row["traced_ms"] = statistics.median(samples[4:])
                ttnn.release_trace(mesh, trace)
                trace = None
                report["rows"].append(row)
                print(row, flush=True)
        report["passed"] = all(r["exact"] and r["replay_exact"] for r in report["rows"])
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)
        a.output.write_text(json.dumps(report, indent=2) + "\n")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
