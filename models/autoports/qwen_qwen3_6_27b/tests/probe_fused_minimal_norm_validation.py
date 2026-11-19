# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Native RMS admission negatives followed by healthy TP4 B1/B32 rebinding.

Source-only preparation/--cpu-only never imports Torch or TTNN. Expected native
validation messages and unchanged program-cache entry counts identify rejection
before native program construction/launch; generic API failures never count.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import traceback

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "links_zero": "num_links must be greater than 0",
    "links_two": "supports exactly one link",
    "stats_absent": "requires a pre-allocated stats tensor",
    "stats_bf16": "stats tensor dtype",
    "stats_dram_interleaved": "requires WIDTH_SHARDED L1 stats",
    "stats_row_major": "requires TILE stats",
    "stats_multi_core": "requires stats on one sender core",
    "stats_wrong_sender": "stats core must be the first input shard core",
    "stats_two_tiles": "requires padded stats shape [1,1,32,32*num_devices]",
    "stats_rank3": "requires rank-4 stats",
    "stats_column_major": "requires ROW_MAJOR stats shard orientation",
    "stats_outer_batch2": "requires padded stats shape [1,1,32,32*num_devices]",
    "output_batch2": "persistent output logical shape must match input logical shape",
    "output_float32": "persistent output dtype must match requested output dtype",
    "output_row_major": "persistent output must be TILE",
    "output_dram_interleaved": "persistent output memory/shard config must match requested output memory config",
    "output_l1_interleaved": "persistent output memory/shard config must match requested output memory config",
    "output_shifted_grid": "persistent output memory/shard config must match requested output memory config",
    "output_column_major": "persistent output memory/shard config must match requested output memory config",
    "output_wider_shard": "persistent output memory/shard config must match requested output memory config",
    "output_host": "Device Operations expect device tensors as inputs",
    "output_other_mesh": "persistent output must be on the input mesh device",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cpu-only", action="store_true")
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "prepared",
        "mesh": [1, 4],
        "negative_plan": CASES,
        "positive_batches": [1, 32],
        "threshold": 0.995,
        "negative_execution": "expected source-validation RuntimeError plus unchanged program-cache count; generic API/construction errors fail",
        "watcher": "TT_METAL_WATCHER" in os.environ,
        "profiler": os.getenv("TT_METAL_DEVICE_PROFILER", "0") != "0",
        "negative_results": [],
        "positive_results": [],
        "source_sha256": {
            str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [Path(__file__), ROOT / "tt/fused_minimal_norm_candidate.py"]
        },
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    if args.cpu_only:
        assert len(CASES) == 22 and len(set(CASES)) == 22
        report.update(status="cpu-plan-pass", hardware_validated=False)
        save()
        return

    import torch
    import ttnn
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import load_layer
    from models.autoports.qwen_qwen3_6_27b.tests.multichip_assertions import pcc

    mesh = None
    output_submesh = None
    try:
        assert not (report["watcher"] and report["profiler"])
        torch.set_num_threads(8)
        torch.manual_seed(4169)
        config = Qwen3_5TextConfig(**json.loads((ROOT / "doc/hf_config.json").read_text())["text_config"])
        base_gamma = 1 + load_layer(args.layer)["input_layernorm.weight"].float()
        # Deliberately distinct gamma values make stale gamma bindings observable.
        # Owner0 uses the real model gamma; owner1 is an admission/cache challenge.
        multiplier = torch.where(torch.arange(5120) % 2 == 0, 0.5, 1.5)
        host_gammas = [base_gamma, base_gamma * multiplier]
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4))

        def grid(x0, x1):
            return ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, 0))})

        inputs_grid, sender_grid = grid(0, 7), grid(0, 0)

        def memory(cores, shape, orientation=None):
            return ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cores, shape, ttnn.ShardOrientation.ROW_MAJOR if orientation is None else orientation),
            )

        input_mem = memory(inputs_grid, (32, 160))
        stats_mem = memory(sender_grid, (32, 128))
        ckc = ttnn.init_device_compute_kernel_config(
            mesh.arch(),
            math_fidelity=ttnn.MathFidelity.HiFi4,
            math_approx_mode=False,
            fp32_dest_acc_en=True,
            packer_l1_acc=True,
        )
        program = ttnn.LayerNormShardedMultiCoreProgramConfig(
            compute_with_storage_grid_size=(8, 1), subblock_w=1, block_h=1, block_w=5, inplace=False
        )

        def allocate_stats(shape=(1, 1, 32, 128), dtype=None, layout=None, mem=None):
            return ttnn.empty(
                shape,
                dtype=ttnn.float32 if dtype is None else dtype,
                layout=ttnn.TILE_LAYOUT if layout is None else layout,
                device=mesh,
                memory_config=stats_mem if mem is None else mem,
            )

        gamma = [
            ttnn.from_torch(
                g.reshape(1, 1, 160, 32),
                dtype=ttnn.float32,
                layout=ttnn.ROW_MAJOR_LAYOUT,
                device=mesh,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=2),
            )
            for g in host_gammas
        ]
        owners = {}
        host_inputs = {}
        for batch in (1, 32):
            for bank in range(2):
                host = torch.randn(1, 1, batch, 5120).bfloat16()
                host += (torch.arange(5120) // 1280).reshape(1, 1, 1, 5120).bfloat16() * 0.125
                host_inputs[batch, bank] = host
                inp = ttnn.from_torch(
                    host,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    device=mesh,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=3),
                )
                inp = ttnn.to_memory_config(inp, input_mem)
                owners[batch, bank] = {
                    "input": inp,
                    "stats": [allocate_stats() for _ in range(2)],
                    "sems": [ttnn.create_global_semaphore(mesh, inputs_grid, 0) for _ in range(2)],
                    "outputs": [
                        ttnn.empty(
                            (1, 1, batch, 1280),
                            dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT,
                            device=mesh,
                            memory_config=input_mem,
                        )
                        for _ in range(2)
                    ],
                }
        bad_stats = {
            "stats_absent": None,
            "stats_bf16": allocate_stats(dtype=ttnn.bfloat16),
            "stats_dram_interleaved": allocate_stats(mem=ttnn.DRAM_MEMORY_CONFIG),
            "stats_row_major": allocate_stats(layout=ttnn.ROW_MAJOR_LAYOUT),
            "stats_multi_core": allocate_stats((1, 1, 32, 256), mem=memory(grid(0, 1), (32, 128))),
            "stats_wrong_sender": allocate_stats(mem=memory(grid(1, 1), (32, 128))),
            "stats_two_tiles": allocate_stats((1, 1, 32, 64), mem=memory(sender_grid, (32, 64))),
            "stats_rank3": allocate_stats((1, 32, 128)),
            "stats_column_major": allocate_stats(mem=memory(sender_grid, (32, 128), ttnn.ShardOrientation.COL_MAJOR)),
            "stats_outer_batch2": allocate_stats((2, 1, 32, 128), mem=memory(sender_grid, (64, 128))),
        }

        # Every malformed output is a real constructible tensor. Construction or
        # generic API failures do not count as the expected native rejection.
        def allocate_output(shape=(1, 1, 1, 1280), *, dtype=None, layout=None, mem=None, device=None):
            return ttnn.empty(
                shape,
                dtype=ttnn.bfloat16 if dtype is None else dtype,
                layout=ttnn.TILE_LAYOUT if layout is None else layout,
                device=mesh if device is None else device,
                memory_config=input_mem if mem is None else mem,
            )

        output_submesh = mesh.create_submesh(ttnn.MeshShape(1, 1), ttnn.MeshCoordinate(0, 0))
        tall_mem = memory(inputs_grid, (64, 160))
        tall_base = allocate_output((1, 1, 64, 1280), mem=tall_mem)
        # TensorSpec prevents this inconsistent physical view before the native
        # op is reachable. Classify that exact construction rejection separately;
        # it is not evidence that the native padded-output guard executed.
        try:
            ttnn.reshape(tall_base, ttnn.Shape((1, 1, 1, 1280)), ttnn.Shape((1, 1, 64, 1280)))
        except RuntimeError as error:
            assert "Shard height 64 must match physical height 32 for width sharded" in str(error), str(error)
            report["construction_rejections"] = [
                {
                    "case": "output_padded_height64",
                    "stage": "TensorSpec construction",
                    "expected_message": "Shard height 64 must match physical height 32 for width sharded",
                    "native_guard_executed": False,
                    "pass": True,
                }
            ]
        else:
            raise AssertionError("Inconsistent width-sharded physical height unexpectedly constructed")
        bad_outputs = {
            "output_batch2": allocate_output((1, 1, 2, 1280)),
            "output_float32": allocate_output(dtype=ttnn.float32),
            "output_row_major": allocate_output(layout=ttnn.ROW_MAJOR_LAYOUT, mem=memory(inputs_grid, (1, 160))),
            "output_dram_interleaved": allocate_output(mem=ttnn.DRAM_MEMORY_CONFIG),
            "output_l1_interleaved": allocate_output(mem=ttnn.L1_MEMORY_CONFIG),
            "output_shifted_grid": allocate_output(mem=memory(grid(1, 8), (32, 160))),
            "output_column_major": allocate_output(mem=memory(inputs_grid, (32, 160), ttnn.ShardOrientation.COL_MAJOR)),
            "output_wider_shard": allocate_output(mem=memory(inputs_grid, (32, 320))),
            "output_host": ttnn.from_torch(torch.zeros((1, 1, 1, 1280)), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT),
            "output_other_mesh": allocate_output(device=output_submesh),
        }
        ttnn.synchronize_device(mesh)

        def invoke(batch, bank, phase, *, links=1, stats_override=False, stats=None, output=None, output_memory=None):
            owner = owners[batch, bank]
            return ttnn.fused_rms_minimal(
                owner["input"],
                program,
                1,
                mesh,
                owner["sems"][phase],
                persistent_output_tensor=owner["outputs"][phase] if output is None else output,
                num_links=links,
                topology=ttnn.Topology.Linear,
                dtype=ttnn.bfloat16,
                compute_kernel_config=ckc,
                memory_config=input_mem if output_memory is None else output_memory,
                epsilon=config.rms_norm_eps,
                weight=gamma[bank],
                stats=stats if stats_override else owner["stats"][phase],
            )

        report["status"] = "running"
        save()
        for name, expected in CASES.items():
            before = mesh.num_program_cache_entries()
            rejected = False
            message = None
            try:
                unexpected = invoke(
                    1,
                    0,
                    0,
                    links=0 if name == "links_zero" else 2 if name == "links_two" else 1,
                    stats_override=name in bad_stats,
                    stats=bad_stats.get(name),
                    output=bad_outputs.get(name),
                    output_memory=None,
                )
                del unexpected
            except RuntimeError as error:
                message = str(error)
                if expected not in message:
                    report["negative_results"].append(
                        {"case": name, "pass": False, "expected": expected, "unexpected_error": message}
                    )
                    save()
                    raise
                rejected = True
            after = mesh.num_program_cache_entries()
            row = {
                "case": name,
                "expected": expected,
                "message": message,
                "rejection_stage": (
                    "device_operation input validation" if name == "output_host" else "RMS native validation"
                ),
                "native_guard_executed": name != "output_host",
                "cache_entries_before": before,
                "cache_entries_after": after,
                "pass": rejected and before == after,
            }
            report["negative_results"].append(row)
            save()
            assert row["pass"], row
        ttnn.synchronize_device(mesh)
        for batch in (1, 32):
            cached = None
            for phase in range(2):
                for bank in range(2):
                    result = invoke(batch, bank, phase)
                    ttnn.synchronize_device(mesh)
                    after = mesh.num_program_cache_entries()
                    if cached is None:
                        cached = after
                    assert after == cached, "Fresh bindings unexpectedly created another native program"
                    values = [ttnn.to_torch(t).float().reshape(batch, 1280) for t in ttnn.get_device_tensors(result)]
                    assert len(values) == 4
                    host = host_inputs[batch, bank].double().reshape(batch, 5120)
                    expected = host * torch.rsqrt(host.square().mean(-1, keepdim=True) + config.rms_norm_eps)
                    expected *= host_gammas[bank].double()
                    scores = [
                        [pcc(expected[u, rank * 1280 : (rank + 1) * 1280], values[rank][u]) for u in range(batch)]
                        for rank in range(4)
                    ]
                    row = {
                        "batch": batch,
                        "bank": bank,
                        "phase": phase,
                        "per_rank_per_user": scores,
                        "cache_entries": after,
                        "pass": min(v for rank in scores for v in rank) >= 0.995,
                    }
                    report["positive_results"].append(row)
                    save()
                    assert row["pass"], row
                    del result
        report.update(
            status="pass",
            hardware_validated=True,
            negatives_before_first_native_fused_launch=True,
            valid_after_rejections=True,
            fresh_bindings_reuse_native_program=True,
            gamma_challenge="owner0 actual model FP32 gamma; owner1 alternating0.5/1.5 multiplier for cache testing",
            timing="disabled; admission/health probe only",
        )
    except BaseException:
        report.update(status="fail", traceback=traceback.format_exc())
        save()
        raise
    finally:
        if output_submesh is not None:
            ttnn.close_mesh_device(output_submesh)
        if mesh is not None:
            ttnn.close_mesh_device(mesh)
            report["mesh_closed"] = True
        save()


if __name__ == "__main__":
    main()
