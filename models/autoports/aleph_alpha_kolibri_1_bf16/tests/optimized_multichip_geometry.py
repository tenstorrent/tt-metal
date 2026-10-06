# SPDX-License-Identifier: Apache-2.0
"""Real-activation, alternating-reader DRAM projection geometry measurements."""

import argparse
import gc
import json
import math
import time

import torch
from tracy import signpost

import ttnn

from .multichip_coverage import OUT, Harness
from .reference import load_weights
from .run_decoder import pcc


def run(layer, readers_only=False, repetitions=100):
    torch.set_num_threads(8)
    h = Harness(layer)
    h.provenance = dict(mesh=[1, 4], source="recorded target-model inputs through TP4 decoder", layer=layer)
    rows = []
    try:
        pages = torch.arange(h.blocks, dtype=torch.int32)[None]
        x = h.input(129)
        c, s = h.angles(torch.arange(128))
        h.model.prefill_chunk_forward(
            h.tt(x[:, :128][None]),
            kv_cache=h.cache,
            page_table=h.integer(pages),
            chunk_page_table=h.integer(pages[:, :4]),
            chunk_start=h.integer(torch.tensor([0], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        observed = {}
        controls = {}
        original = h.model._linear

        def record(a, w, **kwargs):
            info = h.model.projection_info.get(id(w))
            if info:
                observed[info["role"]] = torch.cat(
                    [ttnn.to_torch(t) for t in ttnn.get_device_tensors(a)], dim=1
                ).clone()
            out = original(a, w, **kwargs)
            if info:
                controls[info["role"]] = torch.cat(
                    [ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], dim=1
                ).clone()
            return out

        h.model._linear = record
        c, s = h.angles(torch.tensor([128]))
        h.model.decode_forward(
            h.tt(x[:, 128:].reshape(1, 1, 1, 2560)),
            kv_cache=h.cache,
            page_table=h.integer(pages),
            current_pos=h.integer(torch.tensor([128], dtype=torch.int32)),
            cos=h.tt(c),
            sin=h.tt(s),
        )
        h.model._linear = original
        w = {k.removeprefix(f"model.layers.{layer}."): v for k, v in load_weights(layer).items()}

        def columns(parts):
            return torch.stack([torch.cat([v.chunk(4, dim=-1)[r] for v in parts], dim=-1) for r in range(4)])[None]

        values = {
            "qkv": columns([w["self_attn." + r + ".weight"].T for r in ("q_proj", "k_proj", "v_proj")]),
            "o_proj": torch.stack(w["self_attn.o_proj.weight"].T.chunk(4, dim=0))[None],
            "gate_up": columns([w["mlp.shared_experts." + r + ".weight"].T for r in ("gate_proj", "up_proj")]),
            "down_proj": torch.stack(w["mlp.shared_experts.down_proj.weight"].T.chunk(4, dim=0))[None],
        }
        dram = h.mesh.dram_grid_size()
        banks = dram.x * dram.y
        dram_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(dram.x - 1, dram.y - 1))})
        for role, value in values.items():
            k, n = value.shape[-2:]
            actual = observed[role]
            expected = actual.float() @ value.float()
            role_index = ("qkv", "o_proj", "gate_up", "down_proj").index(role)
            base = (h.model.policy.projection_cores[role_index], h.model.policy.projection_k[role_index])
            cores_list = {
                "qkv": (1, 2, 4, 8, 16, 20),
                "o_proj": (2, 4, 8, 16, 24),
                "gate_up": (1, 2, 4, 5, 10, 20),
                "down_proj": (1, 2, 4),
            }[role]
            geometries = [base] if readers_only else []
            if not readers_only:
                for cores in cores_list:
                    width = k // 32 // cores
                    for block in sorted({v for v in range(1, width + 1) if width % v == 0}):
                        if block > 0 and width % block == 0:
                            geometries.append((cores, block))
            for cores, block in dict.fromkeys(geometries):
                group = []
                for readers in (1, 2, 3):
                    row = dict(
                        role=role,
                        shape=[1, k, n],
                        dtype="bfloat4_b",
                        fidelity="LoFi",
                        fp32_dest_acc_en=h.model.policy.projection_fp32[role_index],
                        cores=cores,
                        kblock=block,
                        readers=readers,
                        shard_k_tiles=k // 32 // cores,
                    )
                    try:
                        shard_n = math.ceil(n / 32 / banks / readers) * 32 * readers
                        physical_n = shard_n * banks
                        row.update(
                            per_core_M=1,
                            per_core_N=math.ceil(physical_n / 32 / cores),
                            output_subblock="DRAM-sharded op selected",
                            physical_n=physical_n,
                            bank_n_tiles=shard_n // 32,
                            reader_row_bytes=shard_n // 32 // readers * 576,
                            stored_weight_bytes=k // 32 * (physical_n // 32) * 576,
                            logical_weight_bytes=k // 32 * math.ceil(n / 32) * 576,
                        )
                        weight_mem = ttnn.MemoryConfig(
                            ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                            ttnn.BufferType.DRAM,
                            ttnn.ShardSpec(dram_grid, (k, shard_n), ttnn.ShardOrientation.ROW_MAJOR),
                        )
                        weight = ttnn.from_torch(
                            torch.nn.functional.pad(value, (0, physical_n - n)).contiguous(),
                            device=h.mesh,
                            mesh_mapper=ttnn.ShardTensorToMesh(h.mesh, dim=1),
                            dtype=ttnn.bfloat4_b,
                            layout=ttnn.TILE_LAYOUT,
                            memory_config=weight_mem,
                        )
                        grid = ttnn.num_cores_to_corerangeset(
                            cores, h.mesh.compute_with_storage_grid_size(), row_wise=True
                        )
                        mem = ttnn.create_sharded_memory_config(
                            (32, k // cores), grid, ttnn.ShardStrategy.WIDTH, use_height_and_width_as_shard_shape=True
                        )
                        a = ttnn.to_memory_config(h.tt(actual, dim=1), mem)
                        program = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                            in0_block_w=block,
                            per_core_M=1,
                            per_core_N=math.ceil(physical_n / 32 / cores),
                            num_workers_per_dram_bank=readers,
                        )
                        compute = ttnn.WormholeComputeKernelConfig(
                            math_fidelity=ttnn.MathFidelity.LoFi,
                            math_approx_mode=False,
                            fp32_dest_acc_en=h.model.policy.projection_fp32[role_index],
                            packer_l1_acc=True,
                        )

                        def call():
                            return ttnn.linear(
                                a,
                                weight,
                                dtype=ttnn.bfloat16,
                                memory_config=ttnn.L1_WIDTH_SHARDED_MEMORY_CONFIG,
                                program_config=program,
                                compute_kernel_config=compute,
                            )

                        out = call()
                        row["pcc"] = pcc(
                            expected,
                            torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], dim=1)[..., :n],
                        )
                        row["same_dtype_control_pcc"] = pcc(
                            controls[role],
                            torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], dim=1)[..., :n],
                        )
                        # The stage gate is full-decoder PCC. Geometry must preserve
                        # the already-passing production BFP4 projection, not undo
                        # its intentional quantization relative to FP32 matmul.
                        assert row["same_dtype_control_pcc"] >= 0.9999, row
                        del out
                        ttnn.synchronize_device(h.mesh)
                        tid = ttnn.begin_trace_capture(h.mesh, cq_id=0)
                        try:
                            out = call()
                        finally:
                            ttnn.end_trace_capture(h.mesh, tid, cq_id=0)
                        group.append((row, tid, out, a, weight))
                        row["samples_ms"] = []
                    except Exception as error:
                        row["error"] = str(error)
                        rows.append(row)
                        print(json.dumps(row), flush=True)
                for repeat in range(3):
                    for row, tid, out, a, weight in group if repeat % 2 == 0 else reversed(group):
                        ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=True)
                        tag = f"DENSE_{role}_C{cores}_K{block}_R{row['readers']}_ROUND{repeat}"
                        signpost(tag)
                        start = time.monotonic()
                        for _ in range(repetitions):
                            ttnn.execute_trace(h.mesh, tid, cq_id=0, blocking=False)
                        ttnn.synchronize_device(h.mesh)
                        row["samples_ms"].append((time.monotonic() - start) * 1000 / repetitions)
                        signpost(tag + "_END")
                        assert (
                            pcc(
                                controls[role],
                                torch.cat([ttnn.to_torch(t) for t in ttnn.get_device_tensors(out)], dim=1)[..., :n],
                            )
                            >= 0.9999
                        )
                for row, tid, *_ in group:
                    ttnn.release_trace(h.mesh, tid)
                    row["median_ms"] = sorted(row["samples_ms"])[1]
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                group.clear()
                gc.collect()
                suffix = "_readers" if readers_only else ""
                (OUT / f"dense_geometry_{layer}{suffix}.json").write_text(
                    json.dumps(dict(provenance=h.provenance, repetitions=repetitions, rows=rows), indent=2) + "\n"
                )
    finally:
        h.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--readers-only", action="store_true")
    p.add_argument("--repetitions", type=int, default=100)
    a = p.parse_args()
    run(a.layer, a.readers_only, a.repetitions)
