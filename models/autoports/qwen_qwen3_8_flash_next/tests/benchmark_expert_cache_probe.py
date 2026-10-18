# SPDX-License-Identifier: Apache-2.0
"""Standalone cold/warm real-expert serialization qualification.

Populate setup cache on the owned mesh, compare selected expert tensors on all
four chips, and compare a B32 MoE output before/after replacing every expert
weight with its warm-loaded equivalent. No live serving instrumentation.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt import precision
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.expert_weight_cache import ExpertWeightCache
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, ParallelMoE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    args = parser.parse_args()
    if os.environ.get("QWEN_FLASH_EXPERT_CACHE"):
        raise ValueError("Unset QWEN_FLASH_EXPERT_CACHE so the control is constructed from checkpoint")
    if any(os.environ.get(name, "0") not in ("", "0") for name in ("TT_METAL_WATCHER", "TT_METAL_DEVICE_PROFILER")):
        raise ValueError("Run the isolated cache probe without watcher or profiler")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    report = {"scope": "one MoE component; full-model B1/B32 serving qualification still required", "roundtrip": []}
    # Source sealing is host-only setup work, separately timed in cache stats.
    cache = ExpertWeightCache(args.checkpoint, args.cache, "fill")
    report["identity"] = cache.identity
    cfg = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True).text_config
    state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.")
    selected = precision.for_layer(precision.load_policy(), args.layer)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    try:
        ops = MeshOps(mesh)
        model = ParallelMoE(state, cfg, ops)
        for name in ("shared_gu", "shared_down"):
            precision.cast_attr(model, name, selected["weight_groups"]["shared_expert"])
        model.shared_compute = precision.compute(mesh, selected["compute_fidelities"]["shared_expert"])
        model.router_compute = precision.compute(mesh, selected["compute_fidelities"]["router"])
        ops.precision_ccl = selected["ccl_dtype"]
        torch.manual_seed(20261016)
        source = torch.randn(1, 32, 1, 2560).bfloat16()
        device = ops.upload(source)
        baseline_tensor = model(device)
        baseline = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(baseline_tensor)], -1)
        del baseline_tensor
        filled = cache.load_layer(args.layer, state, ops)
        report["fill"] = dict(cache.stats)
        del filled
        warm = ExpertWeightCache(args.checkpoint, args.cache, "read")

        class NoCheckpointReads:
            def __getitem__(self, name):
                raise AssertionError(f"Warm cache tried to access checkpoint weight {name}")

        loaded = warm.load_layer(args.layer, NoCheckpointReads(), ops)
        report["read"] = dict(warm.stats)
        for role, original, restored in zip(("gate", "up", "down"), (model.gates, model.ups, model.downs), loaded):
            for expert in (0, 63, 127):
                for chip, (a, b) in enumerate(
                    zip(ttnn.get_device_tensors(original[expert]), ttnn.get_device_tensors(restored[expert]))
                ):
                    before, after = ttnn.to_torch(a), ttnn.to_torch(b)
                    metrics = pcc_metrics(before, after)
                    report["roundtrip"].append({"role": role, "expert": expert, "chip": chip, **metrics})
                    assert metrics["exact_equal"]
        model.gates, model.ups, model.downs = loaded
        actual_tensor = model(device)
        actual = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(actual_tensor)], -1)
        report["component_comparison"] = pcc_metrics(baseline, actual)
        torch.save({"input": source, "baseline": baseline, "cached": actual}, args.output / "outputs.pt")
        report["output_sha256"] = hashlib.sha256((args.output / "outputs.pt").read_bytes()).hexdigest()
        assert report["component_comparison"]["exact_equal"]
        report["status"] = "component_pass"
    finally:
        ttnn.close_mesh_device(mesh)
        (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "fill": report["fill"], "read": report["read"]}), flush=True)


if __name__ == "__main__":
    main()
