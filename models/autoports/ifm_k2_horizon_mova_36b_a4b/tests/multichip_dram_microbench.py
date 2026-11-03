# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""TP4 exact-weight reader microbench; layer timing remains the selection metric."""

import argparse
import hashlib
import json
import math
import os
import time

import torch

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from .fused_reference import ROOT, pcc, reference
from .multichip_activations import recorded_inputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--profile", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    c, hf, rope, sd = reference(args.layer)
    x = recorded_inputs(args.layer, 1, 257)
    prefix = "mlp.shared_experts" if args.layer >= 3 else "mlp"
    inputs, hooks = {}, []
    for name in ["self_attn.q_proj", "self_attn.o_proj", prefix + ".gate_proj", prefix + ".down_proj"]:

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
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=32 * 1024 * 1024)
    rows = []
    try:
        model = MultichipDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        mapping = {
            "attention_packed" if args.layer >= 3 else "attention_decode": (inputs["self_attn.q_proj"], False),
            "self_attn.o_proj": (inputs["self_attn.o_proj"], True),
            "mlp.shared_router" if args.layer >= 3 else "mlp.gate_up": (inputs[prefix + ".gate_proj"], False),
            prefix + ".down_proj": (inputs[prefix + ".down_proj"], True),
        }
        for name, (value, fractured) in mapping.items():
            weight = model.decode_weights[name + ".weight"]
            k, n = tuple(weight.shape)[-2:]
            role = model._role(name)
            grid = ttnn.num_cores_to_corerangeset(8, mesh.compute_with_storage_grid_size(), True)
            mem = ttnn.MemoryConfig(
                ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                ttnn.BufferType.L1,
                ttnn.ShardSpec(grid, [32, k // 8], ttnn.ShardOrientation.ROW_MAJOR),
            )
            shards = value.chunk(4, dim=-1) if fractured else [value] * 4
            host = torch.zeros(4, 1, 32, k, dtype=torch.bfloat16)
            for rank, v in enumerate(shards):
                host[rank, 0, 0, : v.numel()] = v.reshape(-1)
            dtype = ttnn.bfloat8_b if role in ("attn", "attn_decode", "up") else ttnn.bfloat16
            a = ttnn.from_torch(
                host,
                device=mesh,
                mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0),
                dtype=dtype,
                layout=ttnn.TILE_LAYOUT,
                memory_config=mem,
            )
            base = model.matmul_compute[model._weight_group(name)]
            compute = ttnn.WormholeComputeKernelConfig(
                math_fidelity=base.math_fidelity,
                math_approx_mode=base.math_approx_mode,
                fp32_dest_acc_en=True,
                packer_l1_acc=base.packer_l1_acc,
            )
            block = model.options["dram_" + role + "_block"]
            variants, control = {}, None
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)
            for readers in (1, 2, 3):
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    use_multicore_pipeline=True,
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=n // 32 // 8,
                    num_workers_per_dram_bank=readers,
                )

                def forward(pc=pc, output=None):
                    return ttnn.linear(
                        a,
                        weight,
                        program_config=pc,
                        compute_kernel_config=compute,
                        dtype=ttnn.bfloat16,
                        memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG if output is None else output.memory_config(),
                        optional_output_tensor=output,
                    )

                out = forward()
                now = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                if control is None:
                    control = now.clone()
                scores = [pcc(control[r], now[r]) for r in range(4)]
                assert min(scores) >= 0.995, scores
                forward(output=out)
                if args.profile:
                    ttnn.ReadDeviceProfiler(mesh)
                tid = ttnn.begin_trace_capture(mesh, cq_id=0)
                forward(output=out)
                ttnn.end_trace_capture(mesh, tid, cq_id=0)
                variants[readers] = (tid, out)
                tile_bytes = 576 if weight.dtype == ttnn.bfloat4_b else 1088
                bank_tiles = n // 32 // 8
                rows.append(
                    dict(
                        role=name,
                        readers=readers,
                        shape=[32, k, n],
                        weight_dtype=str(weight.dtype),
                        input_dtype=str(dtype),
                        compute=str(compute),
                        program=str(pc),
                        input_memory=str(mem),
                        weight_memory=str(weight.memory_config()),
                        block=block,
                        input_cores=8,
                        per_bank_tiles=bank_tiles,
                        per_reader_row_bytes_min=bank_tiles // readers * tile_bytes,
                        per_reader_row_bytes_max=math.ceil(bank_tiles / readers) * tile_bytes,
                        physical_weight_bytes=k // 32 * n // 32 * tile_bytes,
                        logical_weight_bytes=k
                        * (
                            model.Q_WIDTH + 2 * model.KV_WIDTH + model.Q_WIDTH
                            if name == "attention_decode"
                            else model.weights[name + ".weight"].shape[-1]
                        )
                        / 1024
                        * tile_bytes,
                        pcc_per_rank_vs_one_reader=scores,
                        times_ms=[],
                    )
                )
            for readers in (1, 2, 3) if args.profile else (1, 2, 3, 3, 2, 1, 1, 2, 3, 3, 2, 1):
                tid, out = variants[readers]
                ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
                if args.profile:
                    ttnn.ReadDeviceProfiler(mesh)
                from tracy import signpost

                marker = "MICRO_" + name.replace(".", "_") + "_R" + str(readers)
                if args.profile:
                    signpost(marker)
                start = time.perf_counter()
                for _ in range(1 if args.profile else 200):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                elapsed = (time.perf_counter() - start) * 1000 / (1 if args.profile else 200)
                if args.profile:
                    signpost(marker + "_END")
                    ttnn.ReadDeviceProfiler(mesh)
                next(r for r in rows if r["role"] == name and r["readers"] == readers)["times_ms"].append(elapsed)
            for readers, (tid, out) in variants.items():
                actual = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
                assert min(pcc(control[r], actual[r]) for r in range(4)) >= 0.995
                ttnn.release_trace(mesh, tid)
                out.deallocate(True)
            a.deallocate(True)
            print("ROLE_DONE", name, flush=True)
        for row in rows:
            row["mean_ms"] = sum(row["times_ms"]) / len(row["times_ms"])
            row["physical_GBs_per_chip_host_timing"] = row["physical_weight_bytes"] / row["mean_ms"] / 1e6
            row["peak_fraction_host_timing"] = row["physical_GBs_per_chip_host_timing"] / 512
        dest = (
            ROOT
            / "doc/optimized_multichip_decoder"
            / f'dram_microbench_{"profile" if args.profile else "wall"}_l{args.layer}.json'
        )
        dest.write_text(
            json.dumps(
                dict(
                    mesh=[1, 4],
                    rows=rows,
                    peak_GBs_per_chip=512,
                    metric="isolated warmed trace host time; final whole-layer results govern selection",
                    tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING"),
                    source_sha256=hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                    activation_source="real checkpoint HF hook inputs; column TP inputs replicated, row TP inputs fractured",
                ),
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
