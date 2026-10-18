# SPDX-License-Identifier: Apache-2.0
"""Real HC norm weights; original padding versus compact rows, TP4 exclusive."""

import argparse
import hashlib
import json
import os
import subprocess
import traceback
from pathlib import Path

os.environ.setdefault("TT_METAL_TRACE_ALLOC_TRACKING", "1")
os.environ.setdefault("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0")

import torch
from ttnn.tools import trace_allocation_tracker

import ttnn
from models.autoports.qwen_qwen3_8_flash_next.tests import benchmark_prefill_norm_candidate as candidate
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_batch_moe_probe import kernel_identity
from models.autoports.qwen_qwen3_8_flash_next.tests.benchmark_prefill_moe_group_probe import measured
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import MeshOps, _hidden_parts
from models.autoports.qwen_qwen3_8_flash_next.tt.precision import load_policy
from models.autoports.qwen_qwen3_8_flash_next.tt.prefill_view import _ops

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--lengths", default="128,256,512")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--trace-bytes", type=int, default=268435456)
    args = parser.parse_args()
    lengths = list(map(int, args.lengths.split(",")))
    assert lengths and all(n in (1, 128, 256, 512) for n in lengths)
    assert args.repeats > 0 and args.warmup > 0
    assert all(
        os.environ.get(k, "0") in ("", "0")
        for k in ("TT_METAL_DEVICE_PROFILER", "TT_METAL_WATCHER", "TT_METAL_WATCHER_DUMP_ALL")
    )
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    cfg = json.loads((args.checkpoint / "config.json").read_text())["text_config"]
    policy = load_policy()
    assert policy["config_id"] == "combined_head4_gdn4_lofi"
    report = dict(
        status="preparing",
        scope="isolated prefill HC RMSNorm, real stream-specific weights; no full-model claim",
        checkpoint=str(args.checkpoint.resolve()),
        layer=args.layer,
        policy=policy,
        context=cfg["max_position_embeddings"],
        results=[],
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        timing_boundary="warmed whole norm chain including compaction/restoration and weight multiply through standalone completion synchronize_device",
        poison="NaN implicit tile padding in both input and real norm weight; all logical values finite",
    )
    native = Path("ttnn/cpp/ttnn/operations/normalization/layernorm_distributed/device")
    sources = [
        Path(__file__),
        Path(candidate.__file__),
        ROOT / "tt/prefill_view.py",
        ROOT / "tt/multichip_decoder.py",
        ROOT / "tests/benchmark_prefill_moe_group_probe.py",
        native / "layernorm_pre_all_gather_program_factory.cpp",
        native / "layernorm_post_all_gather_program_factory.cpp",
    ]
    report["source_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    for path in sources:
        (args.output / (path.name + ".txt")).write_bytes(path.read_bytes())

    def save():
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    mesh = None
    try:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=args.trace_bytes)
        assert ttnn._ttnn.operations.trace.trace_allocation_tracking_enabled()
        assert trace_allocation_tracker.TRACE_ALLOC_TRACKING
        assert os.environ["TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE"] == "0"
        report["tracking"] = dict(native=True, python=True, skip_program_cache=False)
        ops = _ops(MeshOps(mesh))
        ops.precision_ccl = policy["ccl_dtype"]
        state = CheckpointState(args.checkpoint, f"model.language_model.layers.{args.layer}.")
        host_weight = 1 + state["attn_hyper_connection.hc_norm.weight"].float()
        weight_shards = [p.reshape(1, 1, 4, 640) for p in _hidden_parts(host_weight)]
        weight = ops.shards(weight_shards)
        report["actual_precision"] = dict(
            compute=kernel_identity(ops.compute),
            weight_dtype=str(weight.dtype),
            activation_dtype="bfloat16",
            statistics_dtype="float32",
            ccl=ops.precision_ccl,
            weight_shape=list(weight.shape),
            weight_padded_shape=list(weight.padded_shape),
        )
        torch.save(dict(norm_weight_shards=weight_shards), args.output / "weights.pt")
        report["weight_sha256"] = hashlib.sha256((args.output / "weights.pt").read_bytes()).hexdigest()
        for n in lengths:
            weight = ttnn.fill_implicit_tile_padding(weight, 0.0)
            rng = torch.Generator().manual_seed(2026101700 + n)
            host = [
                (
                    torch.randn(1, n, 4, 640, generator=rng) * 0.2
                    + torch.arange(4).reshape(1, 1, 4, 1) * 0.05
                    + rank * 0.01
                ).bfloat16()
                for rank in range(4)
            ]
            clean = ttnn.to_memory_config(ops.shards(host), ttnn.L1_MEMORY_CONFIG)
            clean_reference = None
            for poison in (False, True):
                value = ttnn.fill_implicit_tile_padding(clean, float("nan")) if poison else clean
                w = ttnn.fill_implicit_tile_padding(weight, float("nan")) if poison else weight
                calls = {
                    "baseline": lambda: ops.norm(value, w, cfg["rms_norm_eps"]),
                    "candidate": lambda: candidate.compact_norm(ops, value, w, cfg["rms_norm_eps"]),
                }
                for call in calls.values():
                    output = call()
                    ttnn.synchronize_device(mesh)
                    del output
                raw = {}
                timing = {}
                for name, call in calls.items():
                    eager, replay, times = measured(mesh, call, args.repeats, args.warmup)
                    assert torch.isfinite(eager).all() and torch.isfinite(replay).all()
                    assert torch.equal(eager, replay), (n, poison, name, "eager/replay")
                    raw[name] = dict(eager=eager, replay=replay)
                    timing[name] = times
                if not poison:
                    clean_reference = {name: v["replay"].clone() for name, v in raw.items()}
                comparison = pcc_metrics(raw["baseline"]["replay"], raw["candidate"]["replay"])
                record = dict(
                    sequence=n,
                    padding="nan" if poison else "zero",
                    input_shape=list(value.shape),
                    input_padded_shape=list(value.padded_shape),
                    input_memory=str(value.memory_config()),
                    compact_shape=[1, 1, n * 4, 640],
                    timing=timing,
                    comparison=comparison,
                    clean_vs_poison={name: pcc_metrics(clean_reference[name], v["replay"]) for name, v in raw.items()},
                    speedup=timing["baseline"]["median_seconds"] / timing["candidate"]["median_seconds"],
                )
                path = args.output / f's{n}_{record["padding"]}.pt'
                torch.save(dict(input_shards=host, weight_shards=weight_shards, outputs=raw), path)
                record["raw"] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report["results"].append(record)
                save()
                print(json.dumps(record), flush=True)
                assert comparison["exact_equal"], (n, poison, comparison)
                assert all(v["exact_equal"] for v in record["clean_vs_poison"].values())
                del value, w
            del clean
        report["status"] = "passed_exact_component"
        save()
    except BaseException:
        report.update(status="failed", error=traceback.format_exc())
        save()
        raise
    finally:
        if mesh is not None:
            ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
