# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Isolated traced projection search on captured decoder activations and real weights."""
import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import provenance, decoder_class
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mlp-dtype", default="bfloat8_b")
    ap.add_argument("--fidelity", default="HiFi2")
    ap.add_argument("--pipeline", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(17)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    weights = load_layer(args.layer)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []

    def save():
        args.output.write_text(
            json.dumps(
                {
                    "provenance": provenance(),
                    "layer": args.layer,
                    "batch": args.batch,
                    "weights": "real_checkpoint",
                    "activation_source": "captured eager decoder decode, seed17, zero initial state",
                    "mlp_dtype": args.mlp_dtype,
                    "fidelity": args.fidelity,
                    "pipeline": args.pipeline,
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )

    try:
        layer = decoder_class().from_state_dict(
            weights,
            hf_config=config,
            layer_idx=args.layer,
            mesh_device=mesh,
            policy={"mlp_dtype": args.mlp_dtype, "mlp_fidelity": args.fidelity, "attention_fidelity": args.fidelity},
        )
        captured = {}
        original = layer._linear
        read = lambda t: ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        def capture(x, key):
            captured[key] = read(x).reshape(1, args.batch, -1)
            return original(x, key)

        layer._linear = capture
        x = torch.randn(args.batch, 1, 5120, dtype=torch.bfloat16) * 0.1
        state = (
            layer.create_recurrent_state(args.batch)
            if layer.kind == "linear_attention"
            else layer.create_paged_state(torch.arange(args.batch).reshape(args.batch, 1))
        )
        kw = {}
        if layer.kind == "full_attention":
            rot = Qwen3_5TextRotaryEmbedding(config)
            kw = {
                "rope": tuple(
                    layer._upload(v.unsqueeze(1)) for v in rot(x, torch.zeros(args.batch, 1, dtype=torch.long))
                ),
                "current_pos": ttnn.from_torch(
                    torch.zeros(args.batch, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
            }
        layer.decode_forward(layer._upload(x), state=state, **kw)
        layer._linear = original
        dg = mesh.dram_grid_size()
        banks = dg.x * dg.y
        dgrid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dg.x - 1, dg.y - 1))})
        for key, host in captured.items():
            w = layer.weights[key]
            k, n = w.shape
            reference = host.float() @ read(w).float()
            group = "mlp" if key.startswith("mlp.") else "attention"
            tx = layer._upload(host)
            # Alternating order makes the one-reader control bracket wider reader candidates.
            for cores in [8, 4, 16]:
                if k % (cores * 32):
                    continue
                grid = ttnn.num_cores_to_corerangeset(cores, mesh.compute_with_storage_grid_size(), row_wise=True)
                mem = ttnn.create_sharded_memory_config(
                    (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                )
                sx = ttnn.to_memory_config(tx, mem)
                shard_tiles = k // cores // 32
                blocks = [
                    v
                    for v in [2, 4, 5, 8, 10, 16, 17, 20, 32, 34, 40, 64, 68]
                    if k // 32 % v == 0 and (shard_tiles % v == 0 or v % shard_tiles == 0)
                ]
                for readers in [1, 2, 3]:
                    wn = math.ceil(n / 32 / banks / readers) * 32 * readers
                    wm = ttnn.MemoryConfig(
                        ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                        ttnn.BufferType.DRAM,
                        ttnn.ShardSpec(dgrid, (k, wn), ttnn.ShardOrientation.ROW_MAJOR),
                    )
                    sw = ttnn.to_memory_config(w, wm)
                    for block in blocks:
                        row = {
                            "key": key,
                            "shape": [args.batch, k, n],
                            "cores": cores,
                            "readers": readers,
                            "block": block,
                            "weight_dtype": str(w.dtype),
                            "fidelity": args.fidelity,
                            "pipeline": args.pipeline,
                            "weight_shard": [k, wn],
                            "input_shard": [32, k // cores],
                        }
                        tid = None
                        try:
                            pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                                in0_block_w=block,
                                per_core_M=1,
                                per_core_N=math.ceil(n / 32 / cores),
                                num_workers_per_dram_bank=readers,
                                use_multicore_pipeline=args.pipeline,
                            )

                            def run():
                                return ttnn.linear(
                                    sx,
                                    sw,
                                    program_config=pc,
                                    compute_kernel_config=layer.matmul_configs[group],
                                    memory_config=ttnn.MemoryConfig(
                                        ttnn.TensorMemoryLayout.WIDTH_SHARDED, ttnn.BufferType.L1
                                    ),
                                    dtype=ttnn.bfloat16,
                                )

                            y = run()
                            actual = read(y)
                            del y
                            row["pcc"] = pcc(reference, actual)
                            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                            y = run()
                            ttnn.end_trace_capture(mesh, tid, cq_id=0)
                            elapsed = []
                            for _ in range(20):
                                t = time.perf_counter()
                                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                                elapsed.append((time.perf_counter() - t) * 1000)
                            row["trace_ms"] = statistics.median(elapsed)
                            row["correct"] = row["pcc"] >= 0.995
                            row["bit_equal"] = torch.equal(actual, read(y))
                        except RuntimeError as exc:
                            row["error"] = str(exc).split("backtrace:")[0]
                        finally:
                            if tid is not None:
                                ttnn.release_trace(mesh, tid)
                        rows.append(row)
                        save()
                    del sw
                del sx
            print(key, "complete", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
