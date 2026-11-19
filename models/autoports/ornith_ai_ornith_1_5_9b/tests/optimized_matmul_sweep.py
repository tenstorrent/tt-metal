# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-role traced geometry timings on actual decoder boundary activations."""
import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import upload
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--bfp4", action="store_true")
    args = ap.parse_args()
    Path(args.output).with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "implementation_sha256": hashlib.sha256(
                    (Path(__file__).resolve().parents[1] / "tt/optimized_decoder.py").read_bytes()
                ).hexdigest(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "chip": 0,
                "regime": "serial",
                "layer": args.layer,
                "mlp_bfp4": args.bfp4,
                "weights": "real checkpoint",
                "activations": "recorded at actual decoder projection boundaries on seeded BF16 probe",
                "pcc_reference": "first geometry of the same weight dtype; whole-layer HF precision scores are in diagnostic reports",
                "tracking": False,
                "watcher": False,
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(8)
    torch.manual_seed(23)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    records = []
    try:

        class Probe(OptimizedDecoder):
            mlp_fidelity = ttnn.MathFidelity.LoFi

            def linear(self, x, name):
                if x.shape[2] == 1 and name in self.dram_weights:
                    self.inputs[name] = ttnn.clone(x)
                return super().linear(x, name)

        sd = weights(args.layer)
        layer = Probe.from_state_dict(sd, hf_config=config(), layer_idx=args.layer, mesh_device=mesh)
        layer.inputs = {}
        state = layer.allocate_state(context=1056)
        pt = upload(torch.arange(33, dtype=torch.int32)[None], mesh, True)
        tx = upload((torch.randn(1, 1, 1024, 4096) * 0.1).bfloat16(), mesh)
        layer.prefill_forward(tx, state=state, page_table=pt)
        dx = upload((torch.randn(1, 1, 1, 4096) * 0.1).bfloat16(), mesh)
        pos = upload(torch.tensor([1024], dtype=torch.int32), mesh, True)
        layer.decode_forward(dx, state=state, page_table=pt, current_position=pos)
        for name, x in layer.inputs.items():
            weight = layer.dram_weights[name]
            compute = layer.matmul_compute(name)
            if args.bfp4 and name in ("gate_up", "mlp.down_proj", "mlp.gate_proj", "mlp.up_proj"):
                host = (
                    torch.cat([sd["mlp.gate_proj.weight"], sd["mlp.up_proj.weight"]], 0)
                    if name == "gate_up"
                    else sd[name + ".weight"]
                )
                weight = ttnn.from_torch(
                    host.T.contiguous(),
                    device=mesh,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=weight.memory_config(),
                )
            k, n = tuple(weight.shape)[-2:]
            expected = None
            for readers, cores, block in [
                (r, c, b) for r in (1, 2, 3, 4, 6, 8) for c in (32, 64) for b in (1, 2, 4, 8, 16)
            ]:
                row = dict(
                    role=name,
                    readers=readers,
                    cores=cores,
                    block=block,
                    shape=[1, k, n],
                    weight_dtype=str(weight.dtype),
                    fidelity=str(compute.math_fidelity),
                )

                def layout(width):
                    sw = math.ceil(width / (32 * cores))
                    nc = math.ceil(width / (32 * sw))
                    grid = ttnn.num_cores_to_corerangeset(nc, mesh.compute_with_storage_grid_size(), True)
                    return (
                        ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                            ttnn.BufferType.L1,
                            ttnn.ShardSpec(grid, (32, sw * 32), ttnn.ShardOrientation.ROW_MAJOR),
                        ),
                        sw,
                    )

                im, iw = layout(k)
                om, ow = layout(n)
                cfg = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=ow,
                    num_workers_per_dram_bank=readers,
                    use_multicore_pipeline=readers > 1,
                )

                def run():
                    return ttnn.to_memory_config(
                        ttnn.linear(
                            ttnn.to_memory_config(x, im),
                            weight,
                            program_config=cfg,
                            compute_kernel_config=compute,
                            memory_config=om,
                        ),
                        ttnn.DRAM_MEMORY_CONFIG,
                    )

                trace = None
                try:
                    out = run()
                    ttnn.synchronize_device(mesh)
                    actual = ttnn.to_torch(out)
                    if expected is None:
                        expected = actual
                    row["pcc"] = pcc(expected, actual)
                    del out
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = run()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    samples = []
                    for _ in range(5):
                        start = time.perf_counter()
                        for _ in range(50):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        samples.append((time.perf_counter() - start) * 1000 / 50)
                    row.update(
                        ms=statistics.median(samples),
                        samples_ms=samples,
                        input_shard_k_tiles=iw,
                        output_shard_n_tiles=ow,
                    )
                except RuntimeError as exc:
                    row["error"] = str(exc)[:1800]
                finally:
                    if trace is not None:
                        ttnn.release_trace(mesh, trace)
                records.append(row)
                Path(args.output).write_text(json.dumps(records, indent=2) + "\n")
            legal = [r for r in records if r["role"] == name and "ms" in r and r["pcc"] >= 0.995]
            print(name, sorted(legal, key=lambda r: r["ms"])[:3], flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
