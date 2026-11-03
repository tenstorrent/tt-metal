# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Alternating-reader exact-role microbenchmark on recorded HF activations.

Constructor/oracle work and output reads are outside trace timing. This is
supplementary evidence; whole-layer candidates decide whether a setting wins.
"""

import argparse
import hashlib
import inspect
import json
import math
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch

import ttnn

from .fused_reference import ROOT, pcc, reference
from .optimized_activations import recorded_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--source", type=Path)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--options-file", type=Path, required=True)
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    options = json.loads(args.options_file.read_text())
    torch.set_num_threads(8)
    c, hf, rope, sd = reference(args.layer)
    x = recorded_inputs(args.layer, 1, 257)
    inputs = {}
    prefix = "mlp.shared_experts" if args.layer >= 3 else "mlp"
    names = ["self_attn.q_proj", "self_attn.o_proj", prefix + ".gate_proj", prefix + ".up_proj", prefix + ".down_proj"]
    hooks = []
    for name in names:

        def save(module, values, name=name):
            inputs[name] = values[0][:, -1:].detach().clone()

        hooks.append(hf.get_submodule(name).register_forward_pre_hook(save))
    with torch.no_grad():
        hf(
            x,
            position_embeddings=rope(x, torch.arange(257)[None]),
            attention_mask=torch.full((257, 257), -float("inf")).triu(1)[None, None],
        )
    for hook in hooks:
        hook.remove()
    del hf
    if args.source:
        cls = SourceFileLoader("micro_candidate", str(args.source)).load_module().OptimizedDecoder
    else:
        from ..tt.optimized_decoder import OptimizedDecoder

        cls = OptimizedDecoder
    source = Path(inspect.getfile(cls))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        model = cls.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, options=options)
        mapping = {
            "attention_packed": inputs["self_attn.q_proj"],
            "self_attn.o_proj": inputs["self_attn.o_proj"],
            prefix + ".down_proj": inputs[prefix + ".down_proj"],
        }
        if args.layer >= 3:
            if options.get("split_shared"):
                mapping["mlp.shared_gate_up"] = inputs[prefix + ".gate_proj"]
                mapping["mlp.gate"] = inputs[prefix + ".gate_proj"]
            else:
                mapping["mlp.shared_router"] = inputs[prefix + ".gate_proj"]
        elif options.get("packed_dense"):
            mapping["mlp.gate_up"] = inputs[prefix + ".gate_proj"]
        else:
            mapping[prefix + ".gate_proj"] = inputs[prefix + ".gate_proj"]
            mapping[prefix + ".up_proj"] = inputs[prefix + ".up_proj"]
        banks = mesh.dram_grid_size().x
        dg = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(banks - 1, 0))})
        for name, input_cpu in mapping.items():
            w = model.weights[name + ".weight"]
            k, n = tuple(w.shape)[-2:]
            role = model._role(name)
            cores = options.get(role + "_cores", options.get("dense_cores", 8))
            cg = ttnn.num_cores_to_corerangeset(cores, mesh.compute_with_storage_grid_size(), True)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(cg, [32, k // cores], ttnn.ShardOrientation.ROW_MAJOR),
            )
            host = torch.zeros(1, 1, 32, k, dtype=torch.bfloat16)
            host[:, :, 0, :] = input_cpu.reshape(1, 1, k)
            activation_dtype = (
                getattr(ttnn, options.get("attention_activation" if role == "attn" else "mlp_activation", "bfloat16"))
                if role in ("attn", "up")
                else ttnn.bfloat16
            )
            a = ttnn.from_torch(host, device=mesh, dtype=activation_dtype, layout=ttnn.TILE_LAYOUT, memory_config=mem)
            compute = model.matmul_compute[model._weight_group(name)]
            block = options.get("dram_" + role + "_block", k // 32 // cores)
            variants = {}
            control = None
            for readers in (1, 2, 3):
                padded_n = math.ceil(n / (32 * banks * readers)) * 32 * banks * readers
                pw = w
                if padded_n != n:
                    pw = ttnn.typecast(w, ttnn.bfloat16) if w.dtype == ttnn.bfloat4_b else w
                    pw = ttnn.pad(pw, [(0, 0)] * (len(w.shape) - 1) + [(0, padded_n - n)], 0)
                    if pw.dtype != w.dtype:
                        pw = ttnn.typecast(pw, w.dtype)
                wm = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.DRAM,
                    ttnn.ShardSpec(dg, [k, padded_n // banks], ttnn.ShardOrientation.ROW_MAJOR),
                )
                pw = ttnn.to_memory_config(pw, wm)
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=math.ceil(padded_n / 32 / cores),
                    num_workers_per_dram_bank=readers,
                    fused_activation=ttnn.UnaryWithParam(ttnn.UnaryOpType.SILU) if name.endswith("gate_proj") else None,
                )

                def forward(pw=pw, pc=pc, output=None):
                    return ttnn.linear(
                        a,
                        pw,
                        optional_output_tensor=output,
                        program_config=pc,
                        compute_kernel_config=compute,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if output is None else output.memory_config(),
                    )

                warm = forward()
                cpu = ttnn.to_torch(warm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[..., :n]
                if control is None:
                    control = cpu
                score = pcc(control, cpu)
                assert score > 0.999
                fixed = lambda forward=forward, warm=warm: forward(output=warm)
                fixed()
                variants[readers] = (fixed, pw)
                tile_bytes = {"bfloat4_b": 576, "bfloat8_b": 1088}.get(str(w.dtype).split(".")[-1].lower(), 2048)
                # TTNN string spellings vary; explicit enum comparison is authoritative.
                tile_bytes = 576 if w.dtype == ttnn.bfloat4_b else (1088 if w.dtype == ttnn.bfloat8_b else 2048)
                rows.append(
                    dict(
                        role=name,
                        input_dtype=str(activation_dtype),
                        readers=readers,
                        shape=[32, k, n],
                        padded_n=padded_n,
                        weight_dtype=str(w.dtype),
                        compute=str(compute),
                        program=str(pc),
                        input_memory=str(mem),
                        weight_memory=str(wm),
                        input_cores=cores,
                        input_shard_tiles=k // 32 // cores,
                        block=block,
                        per_bank_tiles=padded_n // 32 // banks,
                        per_reader_row_bytes=padded_n // 32 // banks // readers * tile_bytes,
                        physical_weight_bytes=k // 32 * padded_n // 32 * tile_bytes,
                        logical_weight_bytes=k // 32 * n // 32 * tile_bytes,
                        pcc_vs_one_reader=score,
                        times_ms=[],
                    )
                )
            for readers, (forward, pw) in list(variants.items()):
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = forward()
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                variants[readers] = (tid, out, pw)
            for step, readers in enumerate((1, 2, 3) if args.profile else (1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1)):
                tid = variants[readers][0]
                ttnn.synchronize_device(mesh)
                from tracy import signpost

                marker = "MICRO_" + name.replace(".", "_") + "_R" + str(readers)
                if args.profile:
                    signpost(marker)
                iterations = 1 if args.profile else 100
                start = time.perf_counter()
                for _ in range(iterations):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                ms = (time.perf_counter() - start) * 1000 / iterations
                if args.profile:
                    signpost(marker + "_END")
                row = next(r for r in rows if r["role"] == name and r["readers"] == readers)
                row["times_ms"].append(ms)
            for tid, out, pw in variants.values():
                actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))[..., :n]
                assert pcc(control, actual) > 0.999, "Replay corrupted microbenchmark output"
                ttnn.release_trace(mesh, tid)
                out.deallocate(True)
                pw.deallocate(True)
            for row in [r for r in rows if r["role"] == name]:
                row["mean_ms"] = sum(row["times_ms"]) / len(row["times_ms"])
                row["physical_GBs"] = row["physical_weight_bytes"] / row["mean_ms"] / 1e6
                row["logical_GBs"] = row["logical_weight_bytes"] / row["mean_ms"] / 1e6
                row["peak_fraction"] = row["physical_GBs"] / 512
            print("ROLE_DONE", name, [(r["readers"], r["mean_ms"]) for r in rows if r["role"] == name], flush=True)
        dest = ROOT / f"doc/optimized_decoder/dram_microbench_{args.tag}_layer{args.layer}.json"
        dest.write_text(
            json.dumps(
                dict(
                    options=options,
                    implementation_source=str(source),
                    implementation_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    effective_options=model.options,
                    rows=rows,
                    metric="warmed traced host time, one synchronization per100 replays",
                    peak_GBs=512,
                    activation_source="recorded checkpoint HF",
                    firmware="19.8.1",
                ),
                indent=2,
            )
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
