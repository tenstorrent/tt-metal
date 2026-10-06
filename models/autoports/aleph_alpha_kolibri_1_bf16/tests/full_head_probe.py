# SPDX-License-Identifier: Apache-2.0
"""Precision-locked LM-head geometry comparison using the pinned real weights."""

import argparse
import json
import os
import time
from pathlib import Path

import torch

import ttnn
from models.common.modules.lazy_weight import LazyWeight
from models.common.modules.lm_head.lm_head_1d import LMHead1D, LMHead1DConfig, _create_dram_sharded_mem_config

from ..tt.checkpoint import load_weights
from .full_provenance import provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cores", type=int, default=40)
    parser.add_argument("--block", type=int, default=2)
    parser.add_argument("--readers", type=int, default=1)
    parser.add_argument("--splits", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(7)
    root = Path(os.environ.get("FULL_ARTIFACT_DIR", Path(__file__).resolve().parents[1] / "doc/full_model"))
    result = dict(provenance=provenance(), candidate=vars(args), rows=[])
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=10000000)
    try:
        weights = load_weights("lm_head.")["lm_head.weight"].T.contiguous()
        compute = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False, fp32_dest_acc_en=True, packer_l1_acc=True
        )
        common = dict(
            mesh_device=mesh,
            dim=2560,
            lm_head_dtype=ttnn.float32,
            compute_kernel_config=compute,
            output_memcfg=ttnn.DRAM_MEMORY_CONFIG,
        )
        baseline = LMHead1D.from_config(
            LMHead1DConfig(output_weights=[LazyWeight(source=weights, dtype=ttnn.float32)], **common)
        )
        baseline.load_device_weights()
        dram = mesh.dram_grid_size()
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
        input_grid = (
            ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, args.cores // 8 - 1))})
            if args.cores % 8 == 0
            else ttnn.num_cores_to_corerangeset(args.cores, mesh.compute_with_storage_grid_size(), row_wise=True)
        )
        input_mem = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
            ttnn.BufferType.L1,
            ttnn.ShardSpec(input_grid, (32, 2560 // args.cores), ttnn.ShardOrientation.ROW_MAJOR),
        )
        logical_width = 32000 // args.splits
        padding_quantum = 32 * dram.x * args.readers
        physical_width = ((logical_width + padding_quantum - 1) // padding_quantum) * padding_quantum
        split_weights = []
        for split in range(args.splits):
            shards = [
                torch.nn.functional.pad(
                    weights[:, chip * 32000 + split * logical_width : chip * 32000 + (split + 1) * logical_width],
                    (0, physical_width - logical_width),
                )
                for chip in range(4)
            ]
            split_weights.append(LazyWeight(source=torch.cat(shards, dim=-1), dtype=ttnn.float32))
        weight_mem = _create_dram_sharded_mem_config(2560, physical_width, dram_grid, dram_cores=dram.x)
        candidate = LMHead1D.from_config(
            LMHead1DConfig(
                output_weights=split_weights,
                program_configs=[
                    ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=args.block,
                        per_core_M=1,
                        per_core_N=(physical_width // 32 + args.cores - 1) // args.cores,
                        num_workers_per_dram_bank=args.readers,
                    )
                    for _ in range(args.splits)
                ],
                input_memcfg=input_mem,
                weights_memcfgs=[weight_mem] * args.splits,
                output_split_sizes=[logical_width] * args.splits,
                **common,
            )
        )
        candidate.load_device_weights()
        for batch in (1, 32):
            # Geometry probe only; full-model real-prompt gates qualify adoption.
            host = torch.randn(1, 1, batch, 2560).bfloat16()
            x = ttnn.from_torch(
                host,
                device=mesh,
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            reference = host.float() @ weights.float()
            outputs = {}
            for name, head in (("auto", baseline), ("dram", candidate)):

                def forward():
                    inp = ttnn.to_memory_config(x, input_mem) if name == "dram" else x
                    return head(inp)

                y = forward()
                actual = torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(y)], dim=-1)
                outputs[name] = actual
                pcc = torch.corrcoef(torch.stack([actual.flatten(), reference.flatten()]))[0, 1].item()
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                y = forward()
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                for _ in range(5):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                tick = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - tick) * 10
                ttnn.release_trace(mesh, trace)
                result["rows"].append(
                    dict(
                        batch=batch,
                        path=name,
                        ms=ms,
                        pcc=pcc,
                        top1_matches_reference=(actual.argmax(-1) == reference.argmax(-1)).float().mean().item(),
                    )
                )
                print("HEAD_ROW", result["rows"][-1], flush=True)
                (root / f"head_c{args.cores}_k{args.block}_r{args.readers}_s{args.splits}.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
            assert torch.allclose(outputs["auto"], outputs["dram"], rtol=0.02, atol=0.03), (
                (outputs["auto"] - outputs["dram"]).abs().max()
            )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
