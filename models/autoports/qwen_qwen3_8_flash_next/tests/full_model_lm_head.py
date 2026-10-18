"""Precision-locked common LMHead1D geometry candidates on real recorded input."""

import argparse
import gc
import json
import math
import time
import traceback
from pathlib import Path
import torch
import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig
from models.autoports.qwen_qwen3_8_flash_next.tt.model import QwenFlashNextModel
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.multichip_decoder import _hidden_parts
from models.autoports.qwen_qwen3_8_flash_next.tests.metrics import pcc_metrics


def host(tensor):
    return torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(tensor)], -1)


def measure(mesh, fn):
    for _ in range(2):
        out = fn()
    ttnn.synchronize_device(mesh)
    expected = host(out)
    del out
    tid = ttnn.begin_trace_capture(mesh, cq_id=0)
    out = fn()
    ttnn.end_trace_capture(mesh, tid, cq_id=0)
    times = []
    try:
        for _ in range(3):
            tick = time.perf_counter()
            for _ in range(32):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            times.append((time.perf_counter() - tick) * 1000 / 32)
        assert torch.equal(expected, host(out))
    finally:
        ttnn.release_trace(mesh, tid)
    return expected, times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--recorded",
        default="/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/multichip/recorded_inputs/layer3.pt",
    )
    ap.add_argument("--tag", default="lm_head_candidates")
    ap.add_argument("--candidate", default=None)
    ap.add_argument("--family", choices=["dram", "interleaved", "all"], default="dram")
    a = ap.parse_args()
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=160000000)
    rows = []
    dest = Path(f"models/autoports/qwen_qwen3_8_flash_next/doc/full_model/{a.tag}.json")
    try:
        model = QwenFlashNextModel(a.checkpoint, mesh, layer_indices=[])
        recorded = torch.load(a.recorded, weights_only=True)["inputs"][1]
        residual = model.ops.shards(_hidden_parts(recorded.unsqueeze(0)))
        mixed = model.terminal(residual)
        pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
            compute_with_storage_grid_size=(10, 8),
            in0_block_w=4,
            out_subblock_h=1,
            out_subblock_w=1,
            per_core_M=1,
            per_core_N=25,
            fuse_batch=True,
            mcast_in0=True,
        )

        def baseline():
            return ttnn.linear(
                ttnn.to_memory_config(mixed, ttnn.L1_MEMORY_CONFIG),
                model.lm_head,
                program_config=pc,
                compute_kernel_config=model.ops.compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
            )

        reference, times = measure(mesh, baseline)
        rows.append(dict(name="interleaved80_K4", ms=times, median_ms=sorted(times)[1], pcc=1.0))
        if a.family in ("interleaved", "all"):
            for gx, gy, block, subblock in (
                (11, 10, 4, 1),
                (11, 10, 8, 3),
                (11, 10, 10, 3),
                (11, 10, 16, 3),
                (10, 10, 8, 4),
                (10, 10, 10, 4),
                (10, 10, 16, 4),
                (10, 8, 8, 1),
                (10, 8, 10, 1),
            ):
                name = f"interleaved_{gx}x{gy}_k{block}_sub{subblock}"
                row = dict(name=name, dtype="BF16", fidelity="HiFi4", fp32_dest=True)
                try:
                    pc = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=block,
                        out_subblock_h=1,
                        out_subblock_w=subblock,
                        per_core_M=1,
                        per_core_N=math.ceil(1940 / (gx * gy)),
                        fuse_batch=True,
                        mcast_in0=True,
                    )
                    actual, times = measure(mesh, baseline)
                    metrics = pcc_metrics(reference, actual)
                    row.update(
                        status="pass" if metrics["pcc"] >= 0.999 else "accuracy_fail",
                        metrics=metrics,
                        ms=times,
                        median_ms=sorted(times)[1],
                        top1_equal=bool(torch.equal(reference.argmax(-1), actual.argmax(-1))),
                    )
                except (RuntimeError, ValueError, TypeError, AssertionError) as error:
                    traceback.print_exc()
                    row.update(status="contract_error", error=str(error))
                rows.append(row)
                dest.write_text(json.dumps(dict(recorded=a.recorded, rows=rows), indent=2) + "\n")
                print("LM_HEAD_CANDIDATE", json.dumps(row), flush=True)
            if a.family == "interleaved":
                return
        source = CheckpointState(a.checkpoint)["lm_head.weight"].T
        banks = mesh.dram_grid_size().x
        dg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))])
        candidates = [
            (1, 3, 1, 10),
            (2, 2, 1, 10),
            (2, 3, 1, 10),
            (2, 3, 2, 10),
            (4, 2, 2, 10),
            (4, 3, 2, 10),
            (4, 3, 4, 10),
            (1, 1, 1, 10),
            (1, 2, 2, 10),
            (1, 3, 2, 10),
            (2, 1, 2, 10),
            (2, 2, 4, 10),
            (2, 3, 4, 10),
            (2, 3, 8, 10),
            (4, 1, 4, 10),
            (4, 2, 8, 10),
            (4, 3, 8, 10),
            (4, 3, 16, 5),
        ]
        for splits, readers, block, cores in candidates:
            name = f"common_s{splits}_r{readers}_k{block}_c{cores}"
            if a.candidate and name != a.candidate:
                continue
            row = dict(
                name=name,
                splits=splits,
                readers=readers,
                in0_block_w=block,
                input_cores=cores,
                dtype="BF16",
                fidelity="HiFi4",
                fp32_dest=True,
            )
            head = None
            try:
                sizes = [62080 // splits] * splits
                weights = []
                memories = []
                programs = []
                for i, n in enumerate(sizes):
                    parts = [source[:, r * 62080 + i * n : r * 62080 + (i + 1) * n] for r in range(4)]
                    packed = torch.cat(parts, -1).contiguous()
                    shard_n = math.ceil(n / (32 * banks * readers)) * 32 * readers
                    memory = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(dg, [2560, shard_n], ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    weights.append(
                        LazyWeight(
                            source=packed,
                            dtype=ttnn.bfloat16,
                            device=mesh,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=memory,
                            mesh_mapper_config=ttnn.MeshMapperConfig(
                                placements=[ttnn.PlacementShard(-1)], mesh_shape_override=ttnn.MeshShape([4])
                            ),
                        )
                    )
                    memories.append(memory)
                    programs.append(
                        ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=block,
                            per_core_M=1,
                            per_core_N=shard_n // 32 // readers,
                            num_workers_per_dram_bank=readers,
                        )
                    )
                coords = [ttnn.CoreCoord(i % min(8, cores), i // min(8, cores)) for i in range(cores)]
                ag = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in coords])
                imem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(ag, [32, 2560 // cores], ttnn.ShardOrientation.ROW_MAJOR),
                )
                head = LMHead1D.from_config(
                    LMHead1DConfig(
                        output_weights=weights,
                        mesh_device=mesh,
                        dim=2560,
                        max_batch_size=32,
                        program_configs=programs,
                        compute_kernel_config=model.ops.compute,
                        output_split_sizes=sizes,
                        lm_head_dtype=ttnn.bfloat16,
                        output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
                        input_memcfg=imem,
                        weights_memcfgs=memories,
                    )
                )
                head.load_device_weights()
                actual, times = measure(mesh, lambda: head.forward(ttnn.to_memory_config(mixed, imem)))
                metrics = pcc_metrics(reference, actual)
                row.update(
                    status="pass" if metrics["pcc"] >= 0.999 else "accuracy_fail",
                    metrics=metrics,
                    ms=times,
                    median_ms=sorted(times)[1],
                    top1_equal=bool(torch.equal(reference.argmax(-1), actual.argmax(-1))),
                )
            except (RuntimeError, ValueError, TypeError, AssertionError) as error:
                traceback.print_exc()
                row.update(status="contract_error", error=str(error))
            finally:
                del head
                if "weights" in locals():
                    del weights
                gc.collect()
            rows.append(row)
            dest.write_text(json.dumps(dict(recorded=a.recorded, rows=rows), indent=2) + "\n")
            print("LM_HEAD_CANDIDATE", json.dumps(row), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
