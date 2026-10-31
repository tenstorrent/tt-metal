"""Real-weight terminal geometry controls; reduced model activations, TP4."""
import json
import math
import time
from pathlib import Path

import torch

import ttnn

from ..tt.generator import Gemma4Generator


def main():
    torch.set_num_threads(8)
    root = Path("models/autoports/google_gemma_4_31b_it/doc/optimized_full_model")
    prompt = json.loads((root.parent / "full_model/aime24_prompt.json").read_text())["tokens"]
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=128 * 1024 * 1024)
    results = json.loads((root / "lm_head_candidates.json").read_text())
    try:
        gen = Gemma4Generator(mesh, layer_indices=[0, 5], head_mode="interleaved")
        model = gen.model
        gen.prefill.stage(torch.tensor(prompt), slot=0, start=0, size=192, valid=len(prompt))
        hidden = gen.prefill.hidden(size=192, start=0, kv_cache=gen.kv_cache)
        hidden = gen.prefill.last_row(hidden)
        x = ttnn.rms_norm(
            ttnn.to_memory_config(hidden, ttnn.DRAM_MEMORY_CONFIG),
            epsilon=model.config.rms_norm_eps,
            weight=model.norm_weight,
            compute_kernel_config=model.layers[0].kernel,
        )

        def baseline():
            return ttnn.linear(
                x,
                model.lm_head,
                dtype=ttnn.bfloat16,
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                compute_kernel_config=model.kernel,
                core_grid=ttnn.CoreGrid(x=8, y=8),
            )

        ref = ttnn.to_torch(baseline(), mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()

        def evaluate(name, fn):
            out = fn()
            actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()
            pcc = float(torch.corrcoef(torch.stack([ref.flatten(), actual.flatten()]))[0, 1])
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = fn()
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            t = time.perf_counter()
            for _ in range(64):
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            us = (time.perf_counter() - t) * 1e6 / 64
            ttnn.release_trace(mesh, tid)
            result = dict(
                candidate=name,
                traced_host_us=us,
                pcc=pcc,
                weight_dtype="bfloat8_b",
                fidelity="HiFi2",
                scope="LM_head_only_reduced_model_real_activation",
            )
            results.append(result)
            print(result, flush=True)
            (root / "lm_head_candidates.json").write_text(json.dumps(results, indent=2) + "\n")

        evaluate(dict(kind="interleaved"), baseline)
        dram = mesh.dram_grid_size()
        dg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))])
        for width in (65536,):
            weights = []
            for offset in range(0, 65536, width):
                w = ttnn.slice(model.lm_head, (0, offset), (5376, offset + width))
                cfg = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dg, (5376, width // 8), ttnn.ShardOrientation.ROW_MAJOR),
                )
                weights.append(ttnn.to_memory_config(w, cfg))
            for cores in (8, 4):
                a = ttnn.to_memory_config(x, model.layers[0]._working_memory(x, cores))
                for readers in (1, 2):
                    for block in (1, 3, 7) if cores == 8 else (1, 3, 7):
                        name = dict(kind="dram_sharded", width=width, cores=cores, readers=readers, block=block)
                        pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=block,
                            per_core_M=1,
                            per_core_N=math.ceil(width / 32 / cores),
                            num_workers_per_dram_bank=readers,
                        )

                        def forward():
                            parts = [
                                ttnn.to_memory_config(
                                    ttnn.linear(
                                        a,
                                        w,
                                        dtype=ttnn.bfloat16,
                                        program_config=pc,
                                        compute_kernel_config=model.kernel,
                                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                    ),
                                    ttnn.DRAM_MEMORY_CONFIG,
                                )
                                for w in weights
                            ]
                            return ttnn.concat(parts, dim=-1)

                        try:
                            evaluate(name, forward)
                        except RuntimeError as e:
                            results.append(dict(candidate=name, error=str(e)[:2000]))
                            print("REJECT", name, str(e)[:300], flush=True)
                            (root / "lm_head_candidates.json").write_text(json.dumps(results, indent=2) + "\n")
                del a
            del weights, w
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
