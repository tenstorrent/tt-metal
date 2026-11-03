# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precision-locked real-weight LM-head geometry experiment.

All variants include input resharding and output DRAM concatenation. Recorded HF layer3 inputs isolate geometry, not full-model accuracy.
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from ..tt.model import Checkpoint, K2HorizonModel
from .multichip_activations import recorded_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    output = args.output
    rows = []
    try:
        model = K2HorizonModel(mesh, layer_indices=[], max_seq_len=1024, precision_config=args.config)
        x = model.put(recorded_inputs(3, 1, 32).reshape(1, 1, 32, 2560).bfloat16())
        host = Checkpoint().load("lm_head.")["lm_head.weight"]
        host = torch.nn.functional.pad(host, (0, 0, 0, 262144 - model.vocab_size)).reshape(4, 65536, 2560)

        def baseline():
            return ttnn.linear(
                x,
                model.lm_head,
                dtype=ttnn.bfloat16,
                compute_kernel_config=model.head_compute,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )

        reference = ttnn.to_torch(baseline(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()

        def measure(name, fn):
            warm = fn()
            got = ttnn.to_torch(warm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            pcc = float(torch.corrcoef(torch.stack([got.flatten(), reference.flatten()]))[0, 1])
            agreement = float((got.argmax(-1) == reference.argmax(-1)).float().mean())
            del warm
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            result = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            try:
                ttnn.synchronize_device(mesh)
                start = time.perf_counter()
                for _ in range(100):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                row = dict(
                    policy=model.precision,
                    measurement_regime="isolated real-weight head; recorded layer3 activations; 100 traced replays; not full-model ranking",
                    name=name,
                    ms=(time.perf_counter() - start) * 10,
                    pcc=pcc,
                    argmax_agreement=agreement,
                )
                rows.append(row)
                print(row, flush=True)
            finally:
                ttnn.release_trace(mesh, tid)
            del result
            output.write_text(json.dumps(rows, indent=2) + "\n")

        measure("interleaved_8x8", baseline)
        banks = mesh.dram_grid_size().x
        grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for width in (32768, 16384):
            config = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.DRAM,
                ttnn.ShardSpec(grid, (2560, width // banks), ttnn.ShardOrientation.ROW_MAJOR),
            )
            weights = []
            for start in range(0, 65536, width):
                value = host[:, start : start + width].reshape(4 * width, 2560).T.contiguous()[None, None]
                weights.append(
                    ttnn.from_torch(
                        value,
                        device=mesh,
                        dtype=getattr(ttnn, model.precision["lm_head_dtype"]),
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=config,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    )
                )
            for cores, block in ((8, 1), (8, 2), (8, 5), (8, 10), (4, 10), (4, 20), (2, 20), (2, 40), (1, 40), (1, 80)):
                for workers, pipeline in ((4, True),):
                    program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                        in0_block_w=block,
                        per_core_M=1,
                        per_core_N=width // (32 * 8),
                        num_workers_per_dram_bank=workers,
                        use_multicore_pipeline=pipeline,
                    )

                    def candidate():
                        activation = ttnn.interleaved_to_sharded(
                            x,
                            (cores, 1),
                            (32, 2560 // cores),
                            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                            ttnn.ShardOrientation.ROW_MAJOR,
                        )
                        parts = [
                            ttnn.to_memory_config(
                                ttnn.linear(
                                    activation,
                                    weight,
                                    dtype=ttnn.bfloat16,
                                    compute_kernel_config=model.head_compute,
                                    program_config=program,
                                    memory_config=ttnn.MemoryConfig(
                                        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
                                    ),
                                ),
                                ttnn.DRAM_MEMORY_CONFIG,
                            )
                            for weight in weights
                        ]
                        return ttnn.concat(parts, dim=-1, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                    name = f"dram_width{width}_cores{cores}_block{block}_workers{workers}_pipeline{pipeline}"
                    try:
                        measure(name, candidate)
                    except RuntimeError as error:
                        # Validation/compile failures are experiment outcomes;
                        # runtime hangs are handled externally with live evidence.
                        rows.append(dict(name=name, error=str(error)))
                        output.write_text(json.dumps(rows, indent=2) + "\n")
                        print(name, str(error), flush=True)
            del weights
        measure("interleaved_8x8_return", baseline)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
