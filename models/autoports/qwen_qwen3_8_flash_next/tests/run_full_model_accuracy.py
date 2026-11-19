# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Run the unmodified readiness scorers on one resident full-model generator."""

import argparse
import importlib
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tt.generator import build_generator


def memory(mesh):
    view = ttnn.get_memory_view(mesh, ttnn.BufferType.DRAM)
    return {
        key: getattr(view, key)
        for key in (
            "num_banks",
            "total_bytes_per_bank",
            "total_bytes_allocated_per_bank",
            "total_bytes_free_per_bank",
            "largest_contiguous_bytes_free_per_bank",
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity", type=int, default=262144)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    model_dir = Path(__file__).resolve().parents[1]
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0, l1_small_size=32768)
    result = {"passed": False, "scope": "all_48_layers", "regime": "serial", "capacity": args.capacity}
    generator = None
    started = time.monotonic()
    try:
        generator = build_generator(model_dir, mesh, max_seq_len=args.capacity)
        result["memory_loaded"] = memory(mesh)
        generator.prepare()
        result["memory_prepared"] = memory(mesh)
        result["setup_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        for name in ("run_prefill_check", "run_teacher_forcing"):
            module = importlib.import_module("readiness_check." + name)
            # Reuse weights and traces; the runner and all its scoring are unchanged.
            with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                result[name] = getattr(module, name)(
                    model_dir=model_dir, reference_path=model_dir / "readiness_aime24_chat.refpt", mesh_device=mesh
                )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            assert all(row["top5"] >= 0.98 and row["top100"] == 1 for row in result[name]), result[name]
        if args.resume:
            module = importlib.import_module("readiness_check.run_prefill_resume_check")
            with patch.object(module, "_import_build_generator", return_value=lambda **kwargs: generator):
                resume = module.run_prefill_resume_check(
                    model_dir=model_dir,
                    mesh_device=mesh,
                    lengths=[257],
                    block_sizes=[32],
                    sliding_window=None,
                    tile=32,
                    window=128,
                    controls=module.CONTROL_NAMES,
                    thresholds=module.DEFAULT_THRESHOLDS,
                    vocab_size=generator.model.config.vocab_size,
                    seed=0,
                )
            output = model_dir / "readiness/prefill_resume_check.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(resume, indent=2) + "\n")
            result["resume_status"] = resume["status"]
            assert resume["status"] == "pass", resume["reasons"]
        result["memory_final"] = memory(mesh)
        result["counters"] = dict(generator.counters)
        result["passed"] = True
    finally:
        result["wall_seconds"] = time.monotonic() - started
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        if generator is not None:
            generator.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
