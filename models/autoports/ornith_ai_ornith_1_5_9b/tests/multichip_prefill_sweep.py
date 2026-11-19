# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 prefill kernel-family table, real weights and unchanged precision."""

import argparse
import itertools
import json
import math
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_stage4_reference import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, pcc, weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--lengths", default="64,128,256,1024")
    ap.add_argument("--role")
    ap.add_argument("--activation", choices=["bf16", "bfp8"], default="bf16")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(49)
    output = Path(args.output)
    rows = json.loads(output.read_text()) if output.exists() else []
    for row in rows:
        row.setdefault("allocation_policy", "original_L1_input_retained")
    done = {
        tuple(r["key"]) for r in rows
        if r["key"][-1] != "DRAM" or r["allocation_policy"] == "only_requested_placement_live"
    }
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=16 * 1024 * 1024)
    try:
        layer = MultichipDecoder.from_state_dict(
            weights(args.layer), hf_config=config(), layer_idx=args.layer, mesh_device=mesh
        )
        roles = [
            "packed_input",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            "linear_attn.out_proj" if args.layer == 0 else "self_attn.o_proj",
            "packed_mlp",
        ]
        for name, nrows in itertools.product(roles, map(int, args.lengths.split(","))):
            if args.role and name != args.role:
                continue
            w = layer.weights[name + ".weight"]
            k, n = w.shape
            inp = ttnn.from_torch(
                (torch.randn(1, 1, nrows, k) * 0.1).bfloat16(),
                device=mesh,
                layout=ttnn.TILE_LAYOUT,
                mesh_mapper=ttnn.ReplicateTensorToMesh(mesh),
            )
            if args.activation == "bfp8":
                inp = ttnn.typecast(inp, ttnn.bfloat8_b)
            x = ttnn.to_memory_config(inp, ttnn.L1_MEMORY_CONFIG)
            swiglu = name == "packed_mlp"
            ref = ttnn.experimental.minimal_matmul(
                x,
                w,
                fuse_swiglu=swiglu,
                config=layer.prefill_program,
                compute_kernel_config=layer.matmul_compute(name),
                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                dtype=ttnn.bfloat16,
            )
            expected = [ttnn.to_torch(v) for v in ttnn.get_device_tensors(ref)]
            del ref, x
            candidates = []
            for mb, kb, nb, sb, mem in itertools.product(
                (1, 2, 4, 8), (4, 8, 16, 32), (4, 8, 16), (1, 2), ("L1", "DRAM")
            ):
                candidates.append(("minimal", mb, kb, nb, sb, 11, 10, mem))
            for gx, gy, kb, sb, mem in itertools.product((8, 11), (4, 8, 10), (2, 4, 8, 16), (1, 2), ("L1", "DRAM")):
                candidates.append(("2d", 0, kb, 0, sb, gx, gy, mem))
            for kb, sb, mem in itertools.product((1, 2, 4), (1, 2), ("L1", "DRAM")):
                candidates.append(("1d", 0, kb, 0, sb, 8, 8, mem))
            def priority(candidate):
                family, mb, kb, nb, sb, gx, gy, placement = candidate
                if family == "minimal" and (mb, kb, nb, sb) == (4, 16, 8, 2):
                    return (0, placement)
                if family == "2d" and (kb, gx, gy, sb) == (8, 8, 8, 2):
                    return (1, placement)
                if family == "1d" and (kb, sb) == (2, 1):
                    return (2, placement)
                return (3, family, abs(mb - 4), abs(kb - 16), abs(nb - 8), -sb, gx, gy, placement)

            candidates.sort(key=priority)
            for family, mb, kb, nb, sb, gx, gy, placement in candidates:
                if (output.parent / "STOP").exists():
                    raise SystemExit(75)
                key = (name, nrows, family, mb, kb, nb, sb, gx, gy, placement)
                if key in done:
                    continue
                row = dict(
                    key=key,
                    layer=args.layer,
                    role=name,
                    logical_M=nrows,
                    K=k,
                    N=n,
                    family=family,
                    M_block=mb,
                    K_block=kb,
                    N_block=nb,
                    subblock_w=sb,
                    grid=[gx, gy],
                    input_memory=placement,
                    allocation_policy="only_requested_placement_live",
                    dtype="BFP8",
                    activation_dtype=args.activation,
                    output_dtype="BF16",
                    fidelity="LoFi" if swiglu or name.startswith("mlp.") else "HiFi2",
                    devices=[0, 1, 2, 3],
                    regime="serial",
                )
                print("CANDIDATE", key, flush=True)
                tx = ttnn.to_memory_config(inp, ttnn.L1_MEMORY_CONFIG) if placement == "L1" else inp

                def mm(weight):
                    outn = weight.shape[-1]
                    if family == "minimal":
                        return ttnn.experimental.minimal_matmul(
                            tx,
                            weight,
                            fuse_swiglu=swiglu,
                            config=ttnn.MinimalMatmulConfig(
                                M_block_size=mb,
                                K_block_size=kb,
                                N_block_size=nb,
                                subblock_h=1,
                                subblock_w=sb,
                                compute_with_storage_grid_size=(gx, gy),
                            ),
                            compute_kernel_config=layer.matmul_compute(name),
                            memory_config=ttnn.DRAM_MEMORY_CONFIG,
                            dtype=ttnn.bfloat16,
                        )
                    pcN = math.ceil(outn / (32 * (gx if family == "2d" else gx * gy) * sb)) * sb
                    common = dict(
                        compute_with_storage_grid_size=(gx, gy),
                        in0_block_w=kb,
                        out_subblock_h=1,
                        out_subblock_w=sb,
                        per_core_M=math.ceil(nrows / (32 * gy)) if family == "2d" else math.ceil(nrows / 32),
                        per_core_N=pcN,
                        fused_activation=None,
                    )
                    cfg = (
                        ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(**common, transpose_mcast=False)
                        if family == "2d"
                        else ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            **common, fuse_batch=True, mcast_in0=True
                        )
                    )
                    return ttnn.linear(
                        tx,
                        weight,
                        program_config=cfg,
                        compute_kernel_config=layer.matmul_compute(name),
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        dtype=ttnn.bfloat16,
                    )

                def forward():
                    if not swiglu or family == "minimal":
                        return mm(w)
                    gate = mm(layer.weights["mlp.gate_proj.weight"])
                    up = mm(layer.weights["mlp.up_proj.weight"])
                    if gate is None:
                        return None
                    return ttnn.multiply(gate, up, input_tensor_a_activations=[ttnn.UnaryOpType.SILU])

                try:
                    out = forward()
                    if out is None:
                        continue
                    ttnn.synchronize_device(mesh)
                    score = min(pcc(a, ttnn.to_torch(b)) for a, b in zip(expected, ttnn.get_device_tensors(out)))
                    del out
                    trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                    out = forward()
                    ttnn.end_trace_capture(mesh, trace, cq_id=0)
                    times = []
                    for _ in range(3):
                        ttnn.synchronize_device(mesh)
                        begin = time.perf_counter()
                        for _ in range(20):
                            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                        ttnn.synchronize_device(mesh)
                        times.append((time.perf_counter() - begin) * 1000 / 20)
                    ttnn.release_trace(mesh, trace)
                    del out
                    row.update(pcc=score, ms=statistics.median(times), samples_ms=times, passed=score >= 0.995)
                except RuntimeError as error:
                    row.update(error=str(error), passed=False)
                    rows.append(row)
                    output.write_text(json.dumps(rows, indent=2) + "\n")
                    raise
                rows.append(row)
                output.write_text(json.dumps(rows, indent=2) + "\n")
                print("RESULT", row["ms"], row["pcc"], flush=True)
                del tx
            del inp
        print("PREFILL_SWEEP_PASS", len(rows), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
