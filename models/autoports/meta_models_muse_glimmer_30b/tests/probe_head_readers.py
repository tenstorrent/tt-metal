# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Alternating-order same-weight LM-head reader and working-shard probe."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.meta_models_muse_glimmer_30b.tt.generator import build_generator

ROOT = Path(__file__).parents[1]


def main(args):
    torch.set_num_threads(8)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    gen, traces = None, {}
    try:
        gen = build_generator(ROOT, mesh, layer_indices=[0, 3], prepare=False, terminal_policy=json.loads(args.policy))
        gen._write(torch.zeros(1, dtype=torch.int32), gen.positions)
        gen._decode_graph()  # establish the production persistent CCL/L1 state
        gen._write(torch.full((1, 128), 42, dtype=torch.int32), gen.prefill[128]["tokens"])
        gen._prefill_graph(128)
        x = ttnn.slice(gen.prefill[128]["hidden"], (0, 0, 127, 0), (1, 1, 128, 6656))
        model = gen.model
        original_pc = model.lm_head.config.program_configs[0]
        variants = {}
        blocked = []
        for cores in (52, 13):
            shard_k = 208 // cores
            if args.extended:
                geometries = []
                for readers in (1, 2, 3):
                    for block in (b for b in range(1, 209) if 208 % b == 0):
                        key = f"cores{cores}_readers{readers}_block{block}"
                        if (block <= shard_k and shard_k % block) or (block > shard_k and block % shard_k):
                            blocked.append(
                                dict(
                                    key=key,
                                    reason="K block must divide a storage shard or gather an integral number of shards",
                                )
                            )
                            continue
                        n_tiles = model.padded_vocab_size // 128
                        n_per_worker = (n_tiles + 8 * readers - 1) // (8 * readers)
                        weight_tile = {"bfloat4_b": 576, "bfloat8_b": 1088, "bfloat16": 2048}[
                            model.terminal_policy["weight_dtype"]
                        ]
                        # Factory triples B and doubles A except a single K block;
                        # FP32 partials cannot alias the BF16 output DFB. Using
                        # unpadded N gives a conservative lower bound, excluding
                        # resident tensors, alignment, output storage and staging.
                        dfb_lower_bound = (
                            block * 2048 * (2 if block < 208 else 1)
                            + block * n_per_worker * weight_tile * (3 if block < 208 else 1)
                            + n_per_worker * (6144 if model.terminal_policy["fp32"] else 2048)
                        )
                        if dfb_lower_bound > 1461120:
                            blocked.append(
                                dict(
                                    key=key,
                                    reason="DFBs alone exceed physical allocatable L1",
                                    modeled_minimum_dfb_bytes=dfb_lower_bound,
                                    per_core_l1_bytes=1461120,
                                )
                            )
                            continue
                        geometries.append((readers, block))
            else:
                geometries = ((2, 1), (3, 1), (3, 2))
            for readers, block in geometries:
                key = f"cores{cores}_readers{readers}_block{block}"
                mem = ttnn.MemoryConfig(
                    ttnn.TensorMemoryLayout.WIDTH_SHARDED,
                    ttnn.BufferType.L1,
                    ttnn.ShardSpec(
                        ttnn.num_cores_to_corerangeset(cores, ttnn.CoreCoord(8, 8), row_wise=True),
                        [32, 6656 // cores],
                        ttnn.ShardOrientation.ROW_MAJOR,
                    ),
                )
                # Output storage remains the production geometry so this
                # measures the input working shard, not different output padding.
                pc = ttnn.MatmulMultiCoreReuseMultiCastDRAMShardedProgramConfig(
                    in0_block_w=block,
                    per_core_M=1,
                    per_core_N=original_pc.per_core_N,
                    num_workers_per_dram_bank=readers,
                )
                variants[key] = (mem, pc)

        def forward(key):
            model.lm_input_mem, pc = variants[key]
            model.lm_head.config.program_configs = [pc]
            ttnn.copy(model.head(x, decode=True), gen.logits)

        for key in list(variants):
            try:
                forward(key)
                ttnn.synchronize_device(mesh)
            except RuntimeError as error:
                if not args.extended or not any(
                    word in str(error).lower()
                    for word in (
                        "statically allocated dataflow buffers",
                        "statically allocated circular buffers",
                        "not enough space",
                        "failed to allocate",
                    )
                ):
                    raise
                blocked.append(dict(key=key, reason="runtime L1 allocation limit", error=str(error)))
                del variants[key]
                ttnn.synchronize_device(mesh)
        ttnn.synchronize_device(mesh)
        for key in variants:
            tid = ttnn.begin_trace_capture(mesh, cq_id=0)
            forward(key)
            ttnn.end_trace_capture(mesh, tid, cq_id=0)
            traces[key] = tid
        baseline = None
        rows = []
        keys = list(variants)
        for repeat in range(6):
            for key in keys if repeat % 2 == 0 else reversed(keys):
                start = time.perf_counter()
                for _ in range(64):
                    ttnn.execute_trace(mesh, traces[key], cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                latency = (time.perf_counter() - start) * 1000 / 64
                host = gen._read_logits(gen.logits).flatten().float()
                if baseline is None:
                    baseline = host
                pcc = torch.corrcoef(torch.stack((baseline, host)))[0, 1].item()
                assert pcc > 0.999
                rows.append(
                    dict(
                        key=key,
                        repeat=repeat,
                        ms=latency,
                        pcc=pcc,
                        top1=int(host.argmax()),
                        exact_equal=bool(torch.equal(baseline, host)),
                    )
                )
        result = dict(
            policy=model.terminal_policy,
            extended_geometry=args.extended,
            rejected_geometries=blocked,
            blocker_source="ttnn/cpp/ttnn/operations/matmul/device/factory/matmul_multicore_reuse_mcast_dram_sharded_program_factory.cpp:296-326,480-550",
            geometry_contract="K tiles208; all divisors considered, including multi-shard blocks; reject only incompatible storage-shard coverage or physical/runtime L1 limits",
            layers=[0, 3],
            input="real128-token last hidden row",
            rows=rows,
            medians={key: statistics.median(r["ms"] for r in rows if r["key"] == key) for key in keys},
        )
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print("HEAD_READER_PROBE_PASS", result["medians"], flush=True)
    finally:
        for tid in traces.values():
            ttnn.release_trace(mesh, tid)
        if gen:
            gen.close()
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default='{"weight_dtype":"bfloat8_b","fidelity":"HiFi2"}')
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extended", action="store_true")
    main(parser.parse_args())
