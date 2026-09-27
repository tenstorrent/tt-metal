# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Paired optimized TP1 / TP4 checks with identical real weights and inputs."""

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

import ttnn
from models.autoports.google_gemma_4_26b_a4b_it.tests.run_decoder import load_layer
from models.autoports.google_gemma_4_26b_a4b_it.tests.runtime_audit import device_only
from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import CollectiveBufferPool, MultichipDecoder
from models.autoports.google_gemma_4_26b_a4b_it.tt.optimized_decoder import OptimizedDecoder


def _pcc_values_pass(values):
    return bool(values) and all(math.isfinite(value) and value >= 0.995 for value in values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--length", type=int, default=65)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--persistent-ccl", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--optimized-decode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ccl-l1", action="store_true")
    parser.add_argument("--ccl-workers", type=int, choices=[1, 2, 4])
    parser.add_argument("--ccl-buffers", type=int, choices=[1, 2])
    parser.add_argument("--ccl-chunks", type=int, choices=[1, 4])
    parser.add_argument("--attention-bfp4", choices=["qkv", "output", "both"])
    parser.add_argument("--expert-fidelity", choices=["LoFi", "HiFi2", "HiFi4"])
    parser.add_argument("--expert-gate-dtype", choices=["bfloat4_b", "bfloat8_b"])
    parser.add_argument("--attention-precision", choices=["baseline", "qkv", "output", "both"])
    parser.add_argument("--activation-bfp8", choices=["attention", "shared"])
    parser.add_argument("--sharded-moe-bfp8", action="store_true")
    parser.add_argument("--fused-mmrs", action="store_true")
    parser.add_argument("--output-agmm", action="store_true")
    parser.add_argument("--split-qkv", action="store_true")
    parser.add_argument("--residual-l1", action="store_true")
    parser.add_argument("--diagnose-residual", action="store_true")
    parser.add_argument("--dram-readers", type=int, choices=[1, 2, 3], default=1)
    parser.add_argument("--dram-storage-cores", type=int, choices=[4, 8], default=8)
    parser.add_argument("--shared-prefill-k", type=int)
    parser.add_argument("--shared-prefill-l1", action="store_true")
    parser.add_argument("--shared-prefill-subblock", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--tp", type=int, choices=[1, 4])
    parser.add_argument("--expert-parallel", action="store_true")
    parser.add_argument("--fused-tail", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimized-shared", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shared-dram", action="store_true")
    parser.add_argument("--attention-dram", choices=["qkv", "output"])
    parser.add_argument("--shared-geometry", type=int, choices=[0, 1, 2], default=None)
    parser.add_argument("--output-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="LoFi")
    parser.add_argument("--qkv-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="LoFi")
    parser.add_argument(
        "--dense-geometry",
        choices=["baseline", "qkv-n1", "qkv-n2", "qkv-n4", "output-n2", "output-n4", "router-n2", "router-n4"],
        default="baseline",
    )
    parser.add_argument("--sharded-decode-rope", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--moe-ccl-bfp8", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--expert-gate-bfp4", action="store_true")
    parser.add_argument("--shared-gate-bfp4", action="store_true")
    parser.add_argument("--shared-down-bfp4", action="store_true")
    parser.add_argument("--expert-activation-bfp8", action="store_true")
    parser.add_argument("--split-expert-gate", action="store_true")
    parser.add_argument("--split-shared-gate", action="store_true")
    parser.add_argument("--projection-k", type=int)
    parser.add_argument("--projection-role", choices=["qkv", "output", "router"], default="qkv")
    parser.add_argument("--attention-ccl-dtype", choices=["float32", "bfloat16", "bfloat8_b"], default="bfloat16")
    parser.add_argument("--full-attention-ccl-dtype", choices=["float32", "bfloat16", "bfloat8_b"], default="bfloat8_b")
    parser.add_argument("--grouped-moe-reduce", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hybrid-experts", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sparse-gate-geometry", choices=["baseline", "n1-k88", "n2-k44", "n2-k88"], default="baseline"
    )
    parser.add_argument("--sparse-down-geometry", choices=["baseline", "n2-k6"], default="baseline")
    parser.add_argument("--check-cache", action="store_true")
    parser.add_argument("--repeat-input", action="store_true")
    parser.add_argument("--reserve-full-stack", action="store_true")
    parser.add_argument("--prefill-timing-samples", type=int, default=3)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--duplicate-replays", type=int, default=1)
    parser.add_argument("--sharded-residual", action="store_true")
    parser.add_argument("--ring", action="store_true")
    parser.add_argument("--fused-agmm", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.reserve_full_stack and not args.hybrid_experts:
        parser.error("--reserve-full-stack uses the selected hybrid expert memory plan")
    if args.duplicate_replays < 1:
        parser.error("--duplicate-replays must be positive")
    if args.dense_geometry != "baseline" and (args.attention_dram or args.fused_agmm):
        parser.error("Dense geometry trials require the interleaved projection backend")
    if args.expert_parallel and (args.sparse_gate_geometry != "baseline" or args.sparse_down_geometry != "baseline"):
        parser.error("Sparse geometry candidates require TP or hybrid indexed decode experts")
    if args.expert_parallel and args.hybrid_experts:
        parser.error("--expert-parallel requires --no-hybrid-experts")
    if args.shared_geometry and not args.optimized_shared:
        parser.error("--no-optimized-shared requires --shared-geometry 0")
    if args.sharded_residual and args.grouped_moe_reduce:
        parser.error("--sharded-residual requires --no-grouped-moe-reduce")
    if args.attention_dram == "qkv" and args.fused_agmm:
        parser.error("--attention-dram qkv and --fused-agmm are separate backends")
    if args.shared_dram and not args.optimized_shared:
        parser.error("--shared-dram requires --optimized-shared")
    if args.shared_dram and args.shared_geometry:
        parser.error("--shared-dram requires --shared-geometry 0")
    if not 0 <= args.steps <= 128:
        parser.error("--steps must be between0 and128 for the recorded fixture")
    if args.profile and not args.steps:
        parser.error("--profile requires a decode window; steps0 is a capacity check")
    if args.fused_agmm and not (args.ring and args.sharded_residual):
        parser.error("--fused-agmm requires --ring and --sharded-residual")
    runtime_hash = hashlib.sha256(
        Path(__file__).parents[1].joinpath("tt/multichip_decoder.py").read_bytes()
    ).hexdigest()
    runner_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    source_hashes = dict(runtime_sha256=runtime_hash, runner_sha256=runner_hash)
    print("SOURCE_HASHES", json.dumps(source_hashes), flush=True)

    def fail(details):
        report = dict(command=sys.argv, **source_hashes, layer=args.layer, **details)
        args.output.with_suffix(".failure.json").write_text(json.dumps(report, indent=2) + "\n")
        raise AssertionError(json.dumps(report))

    torch.set_num_threads(8)
    torch.manual_seed(42)
    root = Path(__file__).parents[1]
    config = AutoConfig.from_pretrained(Path(__file__).parent).text_config
    hf = load_layer(config, args.layer, True)
    fixture = torch.load(root / f"doc/optimized_decoder/actual_text_layer{args.layer}_4096_128.pt", weights_only=True)
    source_prefill = fixture["prefill"]
    if args.repeat_input:
        source_prefill = source_prefill.repeat(
            1, (args.length + source_prefill.shape[1] - 1) // source_prefill.shape[1], 1
        )
    x = source_prefill[:, : args.length]
    assert x.shape[1] == args.length, "Long inputs require --repeat-input or a matching fixture"
    assert args.length + args.steps <= config.max_position_embeddings
    decode = fixture["decode"][:, : args.steps]
    extent = (args.length + args.steps + 1023) // 1024 * 1024
    cos, sin = Gemma4TextRotaryEmbedding(config)(
        x, torch.arange(extent)[None], layer_type=config.layer_types[args.layer]
    )
    block = 32
    pages = extent // block
    table = torch.randperm(pages, dtype=torch.int32)[None]
    outputs = {}
    caches = {}
    timings = {}
    capacity = {}
    for tp, cls in ((1, OptimizedDecoder), (4, MultichipDecoder)):
        if args.tp is not None and tp != args.tp:
            continue
        fabric = ttnn.FabricConfig.FABRIC_1D_RING if args.ring else ttnn.FabricConfig.FABRIC_1D
        ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED if tp == 1 else fabric)
        mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, tp), trace_region_size=16777216)
        try:
            mapper = ttnn.ReplicateTensorToMesh(mesh)

            def upload(value, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
                return ttnn.from_torch(value, device=mesh, dtype=dtype, layout=layout, mesh_mapper=mapper)

            def input_upload(value):
                return ttnn.from_torch(
                    value,
                    device=mesh,
                    dtype=ttnn.bfloat16,
                    layout=ttnn.TILE_LAYOUT,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1) if tp == 4 and args.sharded_residual else mapper,
                    memory_config=(
                        decoder.decode_residual_memory
                        if tp == 4 and args.residual_l1 and value.shape[-2] == 1
                        else ttnn.DRAM_MEMORY_CONFIG
                    ),
                )

            def read(value, phase, *, step=None, replay="first"):
                if args.diagnose_residual and tp == 4 and phase == "decode" and step == 0:
                    captured = {
                        name: [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(t)]
                        for name, t in decoder.debug_tensors.items()
                    }
                    captured["norms"] = [
                        (
                            ttnn.to_torch(ttnn.get_device_tensors(a)[0]).float(),
                            ttnn.to_torch(ttnn.get_device_tensors(b)[0]).float(),
                        )
                        for a, b in decoder.debug_norms
                    ]
                    captured["layer_scalar"] = decoder.layer.layer_scalar
                    captured["epsilon"] = config.rms_norm_eps
                    torch.save(captured, args.output.with_suffix(".tensors.pt"))
                if tp == 4 and args.sharded_residual:
                    parts = [ttnn.to_torch(value, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=-1)).float()]
                else:
                    parts = [ttnn.to_torch(v).float() for v in ttnn.get_device_tensors(value)]
                finite_counts = [int(torch.isfinite(part).sum()) for part in parts]
                replicas_equal = all(torch.equal(parts[0], part) for part in parts[1:])
                nonfinite = any(count != part.numel() for count, part in zip(finite_counts, parts))
                if nonfinite or not replicas_equal:
                    ranks = []
                    for rank, part in enumerate(parts):
                        finite_pair = torch.isfinite(parts[0]) & torch.isfinite(part)
                        ranks.append(
                            dict(
                                rank=rank,
                                elements=part.numel(),
                                finite=finite_counts[rank],
                                nan=int(torch.isnan(part).sum()),
                                positive_infinity=int(torch.isposinf(part).sum()),
                                negative_infinity=int(torch.isneginf(part).sum()),
                                equal_to_rank0=torch.equal(parts[0], part),
                                bits_equal_to_rank0=torch.equal(parts[0].view(torch.int32), part.view(torch.int32)),
                                changed_vs_rank0=int((parts[0] != part).sum()),
                                finite_max_abs_diff=(
                                    float((parts[0][finite_pair] - part[finite_pair]).abs().max())
                                    if finite_pair.any()
                                    else None
                                ),
                            )
                        )
                    fail(
                        dict(
                            kind="nonfinite" if nonfinite else "replica_difference",
                            tp=tp,
                            phase=phase,
                            step=step,
                            replay=replay,
                            absolute_position=args.length + step if step is not None else None,
                            replicas_equal=replicas_equal,
                            ranks=ranks,
                        )
                    )
                return parts[0]

            decoder = cls.from_state_dict(
                hf.state_dict(),
                hf_config=config,
                layer_idx=args.layer,
                mesh_device=mesh,
                **(
                    {
                        "sharded_residual": args.sharded_residual,
                        "persistent_ccl": args.persistent_ccl,
                        "optimized_decode": args.optimized_decode,
                        "attention_precision": args.attention_precision,
                        "expert_gate_dtype": getattr(ttnn, args.expert_gate_dtype) if args.expert_gate_dtype else None,
                        "collective_buffer_pool": CollectiveBufferPool(mesh) if args.reserve_full_stack else None,
                        "fused_agmm": args.fused_agmm,
                        "topology": ttnn.Topology.Ring if args.ring else ttnn.Topology.Linear,
                        "fused_tail": args.fused_tail,
                        "optimized_shared": args.optimized_shared,
                        "shared_dram": args.shared_dram,
                        "attention_dram": args.attention_dram,
                        "shared_geometry": args.shared_geometry,
                        "grouped_moe_reduce": args.grouped_moe_reduce,
                        "qkv_fidelity": getattr(ttnn.MathFidelity, args.qkv_fidelity),
                        "output_fidelity": getattr(ttnn.MathFidelity, args.output_fidelity),
                        "attention_ccl_dtype": getattr(ttnn, args.attention_ccl_dtype),
                        "full_attention_ccl_dtype": (
                            getattr(ttnn, args.full_attention_ccl_dtype) if args.full_attention_ccl_dtype else None
                        ),
                        "hybrid_experts": args.hybrid_experts,
                        **({"expert_parallel": True} if args.expert_parallel else {}),
                    }
                    if tp == 4
                    else {}
                ),
            )
            if tp == 4 and args.attention_bfp4:
                attention = decoder.layer.self_attn
                if args.attention_bfp4 in ("qkv", "both"):
                    projection = attention.source.weights.wqkv
                    projection.weight = ttnn.typecast(projection.weight, ttnn.bfloat4_b)
                if args.attention_bfp4 in ("output", "both"):
                    attention.source.weights = replace(
                        attention.source.weights,
                        o_proj=ttnn.typecast(attention.source.weights.o_proj, ttnn.bfloat4_b),
                    )
            if tp == 4 and args.ccl_l1:
                decoder.collective_memory = ttnn.L1_MEMORY_CONFIG
            if tp == 4:
                ccl_tuning = {
                    key: value
                    for key, value in (
                        ("num_workers_per_link", args.ccl_workers),
                        ("num_buffers_per_channel", args.ccl_buffers),
                        ("chunks_per_sync", args.ccl_chunks),
                    )
                    if value is not None
                }
                if ccl_tuning:
                    if not decoder.persistent_ccl:
                        raise ValueError("CCL tuning controls use the selected persistent path")
                    decoder.ccl_tuning = ccl_tuning
            if tp == 4 and args.residual_l1:
                decoder.debug_residual = args.diagnose_residual
                decoder.debug_norms = []
                decoder.decode_residual_memory = ttnn.create_sharded_memory_config(
                    (32, 704),
                    ttnn.CoreGrid(x=4, y=1),
                    ttnn.ShardStrategy.WIDTH,
                    ttnn.ShardOrientation.ROW_MAJOR,
                    use_height_and_width_as_shard_shape=True,
                )
                norm_program = ttnn.LayerNormShardedMultiCoreProgramConfig(
                    compute_with_storage_grid_size=(4, 1),
                    subblock_w=2,
                    block_h=1,
                    block_w=22,
                    inplace=False,
                )
                original_normalize = decoder.normalize

                def normalize(value, epsilon, weight=None):
                    if value.shape[-2] != 1:
                        return original_normalize(value, epsilon, weight)
                    value = ttnn.to_memory_config(ttnn.typecast(value, ttnn.float32), decoder.decode_residual_memory)
                    original_value = value
                    value = ttnn.rms_norm(
                        value,
                        epsilon=epsilon,
                        program_config=norm_program,
                        compute_kernel_config=decoder.layer.self_attn.compute,
                        memory_config=decoder.decode_residual_memory,
                    )
                    if args.diagnose_residual:
                        decoder.debug_norms.append(
                            (
                                ttnn.to_memory_config(original_value, ttnn.DRAM_MEMORY_CONFIG),
                                ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG),
                            )
                        )
                    return (
                        value
                        if weight is None
                        else ttnn.mul(value, weight, memory_config=decoder.decode_residual_memory)
                    )

                decoder.normalize = normalize
            if tp == 4 and args.activation_bfp8:
                if args.activation_bfp8 == "attention":
                    decoder.layer.self_attn.source.weights.wqkv.input_bfp8 = True
                    decoder.layer.self_attn.output_input_bfp8 = True
                else:
                    decoder.layer.shared_mlp.input_bfp8 = True
            if tp == 4 and args.sharded_moe_bfp8:
                if not args.sharded_residual:
                    raise ValueError("Sharded MoE payload candidate requires the sharded residual family")
                decoder.sharded_moe_bfp8 = True
                decoder.layer.shared_mlp.reduce = lambda value: decoder.reduce_scatter(
                    ttnn.typecast(value, ttnn.bfloat8_b)
                )
            if tp == 4 and args.fused_mmrs:
                if not (args.ring and args.sharded_residual):
                    raise ValueError("Fused output requires a Ring mesh and carried sharded residual")
                attention = decoder.layer.self_attn
                dtype = decoder.attention_ccl_dtype
                intermediate = upload(torch.zeros(1, 1, 32, 2816), dtype)
                reduced_buffer = upload(torch.zeros(1, 1, 32, 704), dtype)
                semaphores = decoder.ccl.get_rs_ping_pong_semaphore()
                barrier = decoder.ccl.get_barrier_semaphore()
                program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                    compute_with_storage_grid_size=(11, 6),
                    in0_block_w=16,
                    out_subblock_h=1,
                    out_subblock_w=4,
                    per_core_M=1,
                    per_core_N=8,
                    transpose_mcast=False,
                    fused_activation=None,
                    fuse_batch=False,
                )

                def fused_output(value):
                    value = ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG)
                    value = ttnn.pad(value, [(0, 0), (0, 0), (0, 31), (0, 0)], 0)
                    _, reduced = ttnn.experimental.matmul_reduce_scatter_async(
                        value,
                        attention.source.weights.o_proj,
                        persistent_intermediate_buffer=intermediate,
                        persistent_output_buffer=reduced_buffer,
                        multi_device_global_semaphore=semaphores,
                        barrier_semaphore=barrier,
                        reduce_scatter_core_grid_offset=(0, 6),
                        dim=3,
                        num_links=1,
                        topology=ttnn.Topology.Ring,
                        subdevice_id=ttnn.SubDeviceId(0),
                        memory_config_rs=ttnn.DRAM_MEMORY_CONFIG,
                        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=program,
                        compute_kernel_config=attention.output_compute,
                        dtype=dtype,
                    )
                    return reduced[:, :, :1, :]

                attention.decode_output_fused = fused_output
            if tp == 4 and args.output_agmm:
                if not args.ring or args.fused_mmrs:
                    raise ValueError("Output AGMM requires Ring and excludes MMRS")
                attention = decoder.layer.self_attn
                weight = ttnn.from_torch(
                    hf.state_dict()["self_attn.o_proj.weight"].T[None, None],
                    device=mesh,
                    dtype=ttnn.bfloat8_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                )
                if args.attention_bfp4 in ("output", "both"):
                    weight = ttnn.typecast(weight, ttnn.bfloat4_b)
                width = attention.config.num_attention_heads * attention.config.head_dim * 4
                gathered = upload(torch.zeros(1, 1, 1, width), ttnn.bfloat16)
                semaphores = decoder.ccl.get_ag_ping_pong_semaphore()
                barrier = decoder.ccl.get_barrier_semaphore()
                program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 1),
                    in0_block_w=16,
                    out_subblock_h=1,
                    out_subblock_w=2,
                    per_core_M=1,
                    per_core_N=2,
                    fuse_batch=True,
                    mcast_in0=True,
                )

                def gathered_output(value):
                    _, projected = ttnn.experimental.all_gather_matmul_async(
                        ttnn.to_memory_config(value, ttnn.DRAM_MEMORY_CONFIG),
                        weight,
                        persistent_output_buffer=gathered,
                        dim=3,
                        multi_device_global_semaphore=semaphores,
                        all_gather_core_grid_offset=(0, 8),
                        barrier_semaphore=barrier,
                        num_links=1,
                        topology=ttnn.Topology.Ring,
                        memory_config_ag=ttnn.DRAM_MEMORY_CONFIG,
                        memory_config_mm=ttnn.DRAM_MEMORY_CONFIG,
                        program_config=program,
                        compute_kernel_config=attention.output_compute,
                        dtype=ttnn.float32,
                    )
                    projected = projected[:, :, :1, :]
                    return (
                        projected
                        if args.sharded_residual
                        else decoder.gather(ttnn.typecast(projected, decoder.attention_ccl_dtype))
                    )

                attention.decode_output_fused = gathered_output
            if tp == 4 and (args.shared_prefill_k or args.shared_prefill_l1):
                shared = decoder.layer.shared_mlp
                weight = shared.down.__closure__[0].cell_contents

                def shared_down(value):
                    rows = value.padded_shape[-2]
                    per_m = (rows // 32 + 7) // 8
                    sub_h = args.shared_prefill_subblock
                    while per_m % sub_h:
                        sub_h //= 2
                    program = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
                        compute_with_storage_grid_size=(11, 8),
                        in0_block_w=args.shared_prefill_k or 1,
                        out_subblock_h=sub_h,
                        out_subblock_w=2,
                        per_core_M=per_m,
                        per_core_N=8,
                        transpose_mcast=False,
                        fused_activation=None,
                        fuse_batch=False,
                    )
                    if args.shared_prefill_l1:
                        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
                    return ttnn.linear(value, weight, program_config=program, memory_config=ttnn.DRAM_MEMORY_CONFIG)

                shared.down = shared_down
            if tp == 4 and args.attention_dram:
                from models.autoports.google_gemma_4_26b_a4b_it.tt.multichip_decoder import _DramAttentionProjection

                attention = decoder.layer.self_attn
                owner = attention.source.weights.wqkv if args.attention_dram == "qkv" else attention
                attribute = "decode_dram" if args.attention_dram == "qkv" else "decode_output_dram"
                weight = (
                    attention.source.weights.wqkv.weight
                    if args.attention_dram == "qkv"
                    else attention.source.weights.o_proj
                )
                setattr(
                    owner,
                    attribute,
                    _DramAttentionProjection(
                        weight,
                        mesh,
                        weight.shape[-2] // (32 * args.dram_storage_cores),
                        readers=args.dram_readers,
                        storage_cores=args.dram_storage_cores,
                    ),
                )
            if tp == 4 and args.shared_dram and args.dram_readers != 1:
                decoder.layer.shared_mlp.configure_decode(
                    hf.state_dict(),
                    mesh,
                    config.layer_types[args.layer] == "sliding_attention",
                    readers=args.dram_readers,
                )
            if tp == 4 and args.expert_fidelity:
                experts = getattr(decoder.layer.moe.experts, "decode", decoder.layer.moe.experts)
                experts.decode_compute = ttnn.init_device_compute_kernel_config(
                    mesh.arch(),
                    math_fidelity=getattr(ttnn.MathFidelity, args.expert_fidelity),
                    math_approx_mode=False,
                    fp32_dest_acc_en=False,
                    packer_l1_acc=False,
                )
            if tp == 4 and (args.sparse_gate_geometry != "baseline" or args.sparse_down_geometry != "baseline"):
                experts = decoder.layer.moe.experts
                experts = getattr(experts, "decode", experts)
                if experts.indexed_router is None or experts.expert_split or experts.config.top_k != 8:
                    raise ValueError("Sparse geometry candidates require packed indexed top-8 decode")
                if tuple(experts.gate_up.shape) != (1, 128, 2816, 384):
                    raise ValueError("Unexpected local gate/up weight geometry")
                if tuple(experts.down.shape) != (1, 128, 192, 2816):
                    raise ValueError("Unexpected local down weight geometry")

                def sparse_geometry(grid, block):
                    available = mesh.compute_with_storage_grid_size()
                    if grid[0] > available.x or grid[1] > available.y:
                        raise ValueError("Sparse geometry grid exceeds available workers")
                    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=grid,
                        in0_block_w=block,
                        out_subblock_h=1,
                        out_subblock_w=2,
                        out_block_h=1,
                        out_block_w=2,
                        per_core_M=1,
                        per_core_N=2,
                        fuse_batch=False,
                        mcast_in0=True,
                    )

                if args.sparse_gate_geometry != "baseline":
                    gate_block = 44 if args.sparse_gate_geometry == "n2-k44" else 88
                    if args.sparse_gate_geometry == "n1-k88":
                        experts.gate_config.in0_block_w = 88
                    else:
                        experts.gate_config = sparse_geometry((6, 1), gate_block)
                if args.sparse_down_geometry != "baseline":
                    experts.down_config = sparse_geometry((11, 4), 6)
            if tp == 4:
                attention = decoder.layer.self_attn
                if args.sharded_decode_rope is not None:
                    attention.sharded_decode_rope = args.sharded_decode_rope
                if args.moe_ccl_bfp8 is not None:
                    decoder.moe_ccl_bfp8 = args.moe_ccl_bfp8
                if args.dense_geometry != "baseline":
                    role, width = args.dense_geometry.split("-n")
                    width = int(width)
                    if role == "qkv":
                        owner, field = attention.source.weights.wqkv, "program"
                        grid = (8, (64 if config.layer_types[args.layer] == "sliding_attention" else 96) // width // 8)
                        if width == 1 and config.layer_types[args.layer] == "full_attention":
                            grid = (11, 9)
                    elif role == "output":
                        owner, field = attention, "output_program"
                        grid = (11, 88 // width // 11)
                    else:
                        owner, field = decoder.layer.moe.router, "projection_program"
                        grid = (4 // width, 1)
                    previous = getattr(owner, field)
                    setattr(
                        owner,
                        field,
                        ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                            compute_with_storage_grid_size=grid,
                            in0_block_w=previous.in0_block_w,
                            out_subblock_h=1,
                            out_subblock_w=width,
                            out_block_h=1,
                            out_block_w=width,
                            per_core_M=1,
                            per_core_N=width,
                            fuse_batch=True,
                            fused_activation=None,
                            mcast_in0=True,
                        ),
                    )
            if tp == 4 and args.expert_gate_bfp4:
                experts = getattr(decoder.layer.moe.experts, "decode", decoder.layer.moe.experts)
                if args.expert_parallel or tuple(experts.gate_up.shape) != (1, 128, 2816, 384):
                    raise ValueError("Gate precision candidate requires TP indexed expert layout")
                fused = hf.state_dict()["experts.gate_up_proj"]
                gate, up = fused.chunk(2, dim=-2)
                gate, up = (torch.nn.functional.pad(t.transpose(-2, -1), (0, 64)) for t in (gate, up))
                packed = torch.cat(
                    [torch.cat((g, u), dim=-1) for g, u in zip(gate.chunk(4, -1), up.chunk(4, -1))], dim=-1
                )
                experts.gate_up = ttnn.from_torch(
                    packed.unsqueeze(0),
                    device=mesh,
                    dtype=ttnn.bfloat4_b,
                    layout=ttnn.TILE_LAYOUT,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                )
            if tp == 4 and args.expert_activation_bfp8:
                experts = getattr(decoder.layer.moe.experts, "decode", decoder.layer.moe.experts)
                if args.expert_parallel:
                    raise ValueError("Activation candidate requires indexed TP experts")
                experts.decode_activation_dtype = ttnn.bfloat8_b
            if tp == 4 and (args.shared_gate_bfp4 or args.shared_down_bfp4):
                shared = decoder.layer.shared_mlp
                if args.shared_dram or shared.decode_weights is None:
                    raise ValueError("Shared precision trial requires the interleaved backend")
                weights = list(shared.decode_weights)
                state = hf.state_dict()
                if args.shared_gate_bfp4:
                    gate, up = (
                        torch.nn.functional.pad(state[name].transpose(-2, -1), (0, 64))
                        for name in ("mlp.gate_proj.weight", "mlp.up_proj.weight")
                    )
                    packed = torch.cat(
                        [torch.cat((u, g), dim=-1) for u, g in zip(up.chunk(4, -1), gate.chunk(4, -1))], dim=-1
                    )
                    weights[0] = ttnn.from_torch(
                        packed[None, None],
                        device=mesh,
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-1),
                    )
                if args.shared_down_bfp4:
                    down = torch.nn.functional.pad(state["mlp.down_proj.weight"].transpose(-2, -1), (0, 0, 0, 64))
                    weights[1] = ttnn.from_torch(
                        down[None, None],
                        device=mesh,
                        dtype=ttnn.bfloat4_b,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                        mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=-2),
                    )
                shared.decode_weights = tuple(weights)
            if tp == 4 and args.split_expert_gate:
                experts = getattr(decoder.layer.moe.experts, "decode", decoder.layer.moe.experts)
                if args.expert_parallel or experts.indexed_router is None:
                    raise ValueError("Split expert trial requires indexed TP decode")
                experts.gate = experts.gate_up[..., : experts.width]
                experts.up = experts.gate_up[..., experts.width :]
                experts.separate_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(6, 1),
                    in0_block_w=44,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=1,
                    out_block_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=False,
                    mcast_in0=True,
                )
                experts.expert_split = True
            if tp == 4 and args.split_shared_gate:
                shared = decoder.layer.shared_mlp
                if args.shared_dram or shared.decode_weights is None:
                    raise ValueError("Split shared trial requires the interleaved decode backend")

                class SplitShared(type(shared)):
                    def __call__(self, x, *, reduce_output=True):
                        if x.shape[-2] != 1:
                            return super().__call__(x, reduce_output=reduce_output)
                        common = dict(
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            compute_kernel_config=self.decode_compute,
                            program_config=self.split_program,
                        )
                        up = ttnn.linear(x, self.split_weights[0], **common)
                        gate = ttnn.linear(x, self.split_weights[1], **common)
                        hidden = ttnn.mul(
                            gate, up, input_tensor_a_activations=[ttnn.UnaryWithParam(ttnn.UnaryOpType.GELU, 0.0)]
                        )
                        result = ttnn.linear(
                            hidden,
                            self.decode_weights[1],
                            dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG,
                            compute_kernel_config=self.decode_compute,
                            program_config=self.decode_programs[1],
                        )
                        return self.reduce(result) if reduce_output else result

                shared.split_weights = (
                    shared.decode_weights[0][..., : shared.width],
                    shared.decode_weights[0][..., shared.width :],
                )
                shared.split_program = ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                    compute_with_storage_grid_size=(11, 2),
                    in0_block_w=44,
                    out_subblock_h=1,
                    out_subblock_w=1,
                    out_block_h=1,
                    out_block_w=1,
                    per_core_M=1,
                    per_core_N=1,
                    fuse_batch=True,
                    mcast_in0=True,
                )
                shared.__class__ = SplitShared
            if tp == 4 and args.projection_k is not None:
                if args.attention_dram or args.fused_agmm:
                    raise ValueError("Projection K overrides require the interleaved backend")
                attention = decoder.layer.self_attn
                if args.projection_role == "qkv":
                    program, tiles = attention.source.weights.wqkv.program, 88
                elif args.projection_role == "output":
                    program = attention.output_program
                    tiles = 32 if config.layer_types[args.layer] == "sliding_attention" else 64
                else:
                    program, tiles = decoder.layer.moe.router.projection_program, 88
                if args.projection_k <= 0 or tiles % args.projection_k:
                    raise ValueError("Projection K block must divide the input tile width")
                program.in0_block_w = args.projection_k
            if tp == 4 and args.split_qkv:
                if args.fused_agmm or args.attention_dram:
                    raise ValueError("Split QKV control requires interleaved projections")
                projection = decoder.layer.self_attn.source.weights.wqkv
                cfg = decoder.layer.self_attn.config
                q_width = cfg.num_attention_heads * cfg.head_dim
                kv_width = cfg.num_key_value_heads * cfg.head_dim
                cuts = (0, q_width, q_width + kv_width, q_width + 2 * kv_width)
                assert cuts[-1] == projection.weight.shape[-1]
                projection.split_weights = tuple(projection.weight[..., a:b] for a, b in zip(cuts, cuts[1:]))
                projection.split_programs = tuple(
                    ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
                        compute_with_storage_grid_size=(8, weight.shape[-1] // 32 // 2 // 8),
                        in0_block_w=projection.program.in0_block_w,
                        out_subblock_h=1,
                        out_subblock_w=2,
                        per_core_M=1,
                        per_core_N=2,
                        fuse_batch=True,
                        mcast_in0=True,
                    )
                    for weight in projection.split_weights
                )
                original_class = type(projection)

                class SplitQKV(original_class):
                    def __call__(self, value):
                        if value.shape[-2] != 1:
                            return super().__call__(value)
                        value = ttnn.to_memory_config(value, ttnn.L1_MEMORY_CONFIG)
                        parts = [
                            ttnn.linear(
                                value,
                                weight,
                                dtype=ttnn.float32,
                                program_config=program,
                                compute_kernel_config=self.decode_compute,
                                memory_config=ttnn.L1_MEMORY_CONFIG,
                            )
                            for weight, program in zip(self.split_weights, self.split_programs)
                        ]
                        return ttnn.concat(parts, dim=-1, memory_config=ttnn.L1_MEMORY_CONFIG)

                projection.__class__ = SplitQKV
            cfg = decoder.layer.self_attn.config
            cache = [
                upload(torch.zeros(pages, cfg.num_key_value_heads, block, cfg.head_dim), ttnn.bfloat8_b)
                for _ in range(2)
            ]
            page_table = upload(table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            rope = tuple(upload(t.unsqueeze(0)) for t in (cos, sin))
            rope_decode = tuple(upload(t.squeeze(0), layout=ttnn.ROW_MAJOR_LAYOUT) for t in (cos, sin))
            reservations = []
            if args.reserve_full_stack and tp == 4:
                plan = json.loads((root / "doc/multichip_decoder/memory_capacity_plan.json").read_text())
                kind = config.layer_types[args.layer]
                layer_plan = plan["per_device"][kind]
                resident = plan["full_stack_per_device"]["resident_and_reserve_bound_bytes"]
                current_weights = layer_plan["weight_bound"]
                if args.hybrid_experts:
                    dual = plan["hypothetical_dual_expert_layout"]
                    resident = dual["resident_and_reserve_bound_bytes"]
                    current_weights += dual[
                        (
                            "extra_ep_experts_per_sliding_layer_bytes"
                            if kind == "sliding_attention"
                            else "extra_ep_experts_per_full_layer_bytes"
                        )
                    ]
                elif args.expert_parallel:
                    raise ValueError("Reservation accounting currently covers TP or hybrid experts")
                if args.optimized_shared:
                    shared_plan = plan["optimized_shared_decode"]
                    resident += shared_plan["extra_full_stack_bytes"]
                    current_weights += shared_plan[
                        "extra_sliding_layer_bytes" if kind == "sliding_attention" else "extra_full_layer_bytes"
                    ]
                if args.shared_dram:
                    extra_tiles = decoder.layer.shared_mlp.extra_decode_weight_tiles
                    extra_bytes = {"sliding_attention": extra_tiles * 576, "full_attention": extra_tiles * 576}
                    resident += sum(extra_bytes[layer_type] for layer_type in config.layer_types)
                    current_weights += extra_bytes[kind]
                if args.attention_dram:
                    extra_bytes = {}
                    for layer_type in set(config.layer_types):
                        is_sliding = layer_type == "sliding_attention"
                        head_dim = config.head_dim if is_sliding else config.global_head_dim
                        q_heads = config.num_attention_heads // 4
                        if args.attention_dram == "qkv":
                            kv_heads = config.num_key_value_heads if is_sliding else config.num_global_key_value_heads
                            k, n = config.hidden_size, (q_heads + 2 * max(1, kv_heads // 4)) * head_dim
                        else:
                            k, n = q_heads * head_dim, config.hidden_size
                        extra_bytes[layer_type] = (k // 32) * (n // 32) * 1088
                    assert extra_bytes[kind] == decoder.attention_dram_extra_weight_bytes
                    resident += sum(extra_bytes[layer_type] for layer_type in config.layer_types)
                    current_weights += extra_bytes[kind]
                optimized_plan = None
                if decoder.optimized_decode:
                    optimized_plan = json.loads(
                        (root / "doc/optimized_multichip_decoder/final_memory_plan.json").read_text()
                    )
                    resident += optimized_plan["full_stack_extra_dram_weight_bytes_per_device"]
                    current_weights += optimized_plan["per_device_extra_attention_weights"][kind]
                    current_weights += optimized_plan.get("per_device_extra_expert_weights", {}).get(kind, 0)
                # Reserve other layers, tied embeddings, and shared per-kind RoPE.
                # Leave the independent 2 GiB workspace allowance available.
                current_cache = 2 * pages * cfg.num_key_value_heads * (block // 32) * (cfg.head_dim // 32) * 1088
                current_rope = 2 * (cos.numel() + sin.numel()) * 2
                reserve = resident - plan["full_stack_per_device"]["reserved_trace_activation_allocator_bytes"]
                reserve -= current_weights + current_cache + current_rope
                allocation_bytes = 64 * 1024**2
                count = (reserve + allocation_bytes - 1) // allocation_bytes
                reservations = [
                    ttnn.empty(
                        (1, 1, 32768, 1024),
                        device=mesh,
                        dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG,
                    )
                    for _ in range(count)
                ]
                # Each layer keeps private global semaphores even when the
                # writable collective payload pool is shared across layers.
                reservations.extend(
                    type(decoder.ccl)(mesh, 1, decoder.topology) for _ in range(len(config.layer_types) - 1)
                )
                l1_reserve_bytes = 0
                if optimized_plan is not None and decoder.persistent_ccl:
                    # Prime the actual shared pool's complete selected dtype/role
                    # union before prefill, as after a previous decoded request.
                    for role, planes, dtype in (
                        ("attention", 1, ttnn.bfloat8_b),
                        ("attention", 1, ttnn.bfloat16),
                        ("moe_pair", 2, ttnn.bfloat8_b),
                        ("moe_pair", 2, ttnn.bfloat16),
                    ):
                        value = upload(torch.zeros(1, planes, 1, 2816), dtype)
                        decoder.allreduce(value, role=role)
                    ttnn.synchronize_device(mesh)
                    del value
                    l1_reserve_bytes = optimized_plan["pooled_persistent_ccl_bytes_per_device"]
                capacity = dict(
                    resident_layer_ccl_managers=len(config.layer_types),
                    persistent_ccl_l1_payload_reserved_bytes_per_device=l1_reserve_bytes,
                    reserved_other_resident_bytes_per_device=count * allocation_bytes,
                    current_weight_bound=current_weights,
                    current_cache_bytes=current_cache,
                    current_rope_bytes=current_rope,
                    accounting="memory_capacity_plan.json",
                    limitation="Anonymous DRAM reservations exercise capacity, not a full-model stack",
                )
                print("CAPACITY_RESERVED", capacity, flush=True)
            dx = input_upload(x.unsqueeze(0))
            with device_only():
                y = decoder.prefill_forward(dx, rope_mats=rope, page_table=page_table, kv_cache=cache)
            prefill = read(y, "prefill")
            del y
            prefill_times = []
            for sample in range(args.prefill_timing_samples):
                ttnn.synchronize_device(mesh)
                if args.profile and sample == args.prefill_timing_samples - 1:
                    from tracy import signpost

                    signpost("PERF_PREFILL")
                before = time.perf_counter()
                with device_only():
                    y = decoder.prefill_forward(dx, rope_mats=rope, page_table=page_table, kv_cache=cache)
                ttnn.synchronize_device(mesh)
                prefill_times.append((time.perf_counter() - before) * 1e6)
                if args.profile and sample == args.prefill_timing_samples - 1:
                    signpost("PERF_PREFILL_END")
                del y
            timings[tp] = dict(prefill_host_us=prefill_times)
            decoded = []
            decode_times = []
            if args.steps:
                token = input_upload(decode[:, :1].unsqueeze(0))
                pos = upload(torch.tensor([[args.length]], dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
                cache_pos = upload(torch.tensor([args.length], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

                def forward():
                    with device_only():
                        return decoder.decode_forward(
                            token,
                            rope_mats=rope_decode,
                            current_pos=pos,
                            cache_pos=cache_pos,
                            page_table=page_table,
                            kv_cache=cache,
                        )

                trace_id = None
                if args.trace:
                    for _ in range(2):
                        y = forward()
                    trace_id = ttnn.begin_trace_capture(mesh, cq_id=0)
                    y = forward()
                    ttnn.end_trace_capture(mesh, trace_id, cq_id=0)
                decode_times = []
                if args.profile:
                    signpost("PERF_DECODE")
                for step in range(args.steps):
                    for value, dst, dtype, layout in (
                        (decode[:, step : step + 1].unsqueeze(0), token, ttnn.bfloat16, ttnn.TILE_LAYOUT),
                        (
                            torch.tensor([[args.length + step]], dtype=torch.int32),
                            pos,
                            ttnn.uint32,
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                        (
                            torch.tensor([args.length + step], dtype=torch.int32),
                            cache_pos,
                            ttnn.int32,
                            ttnn.ROW_MAJOR_LAYOUT,
                        ),
                    ):
                        host = ttnn.from_torch(
                            value,
                            dtype=dtype,
                            layout=layout,
                            mesh_mapper=(
                                ttnn.ShardTensorToMesh(mesh, dim=-1)
                                if dst is token and tp == 4 and args.sharded_residual
                                else mapper
                            ),
                        )
                        ttnn.copy_host_to_device_tensor(host, dst)
                    ttnn.synchronize_device(mesh)
                    before = time.perf_counter()
                    if args.trace:
                        ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                    else:
                        y = forward()
                        ttnn.synchronize_device(mesh)
                    decode_times.append((time.perf_counter() - before) * 1e6)
                    decoded.append(read(y, "decode", step=step))
                    if args.trace and not args.profile:
                        for duplicate in range(args.duplicate_replays):
                            ttnn.execute_trace(mesh, trace_id, cq_id=0, blocking=True)
                            repeated = read(y, "decode", step=step, replay=f"repeat{duplicate + 1}")
                            if not torch.equal(decoded[-1], repeated):
                                fail(
                                    dict(
                                        kind="replay_difference",
                                        tp=tp,
                                        phase="decode",
                                        step=step,
                                        replay=f"repeat{duplicate + 1}",
                                        absolute_position=args.length + step,
                                        changed_elements=int((decoded[-1] != repeated).sum()),
                                        max_abs_diff=float((decoded[-1] - repeated).abs().max()),
                                    )
                                )
                if args.profile:
                    signpost("PERF_DECODE_END")
                if trace_id is not None:
                    ttnn.release_trace(mesh, trace_id)
            timings[tp]["decode_host_us"] = decode_times
            outputs[tp] = [prefill, *decoded]
            if args.check_cache:
                caches[tp] = [[ttnn.to_torch(t).float() for t in ttnn.get_device_tensors(c)] for c in cache]
            print("TP_DONE", tp, flush=True)
        finally:
            ttnn.close_mesh_device(mesh)
    values = []
    for a, b in zip(outputs.get(1, []), outputs.get(4, [])):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        values.append(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1].item())
    cache_pcc = []
    if len(caches) == 2:
        logical_end = args.length + args.steps
        for ref_ranks, actual_ranks in zip(caches[1], caches[4]):
            source = ref_ranks[0]
            for rank, actual in enumerate(actual_ranks):
                local_heads = actual.shape[1]
                head_start = rank * local_heads if config.layer_types[args.layer] == "sliding_attention" else rank // 2
                wanted = source[:, head_start : head_start + local_heads]

                def ordered(c):
                    return c[table[0].long()].permute(1, 0, 2, 3).reshape(local_heads, extent, -1)[:, :logical_end]

                a, b = ordered(wanted), ordered(actual)
                cache_pcc.append(torch.corrcoef(torch.stack((a.flatten().double(), b.flatten().double())))[0, 1].item())
        if not _pcc_values_pass(cache_pcc):
            args.output.with_suffix(".failure.json").write_text(
                json.dumps(
                    dict(
                        kind="cache_pcc",
                        cache_pcc=cache_pcc,
                        pcc=values,
                        runtime_sha256=runtime_hash,
                        runner_sha256=runner_hash,
                        command=sys.argv,
                        passed=False,
                    ),
                    indent=2,
                )
                + "\n"
            )
            raise AssertionError(cache_pcc)
    result = dict(
        runtime_sha256=runtime_hash,
        runner_sha256=runner_hash,
        optimized_decode=decoder.optimized_decode if 4 in outputs else None,
        attention_precision=decoder.attention_precision if 4 in outputs else None,
        persistent_ccl=decoder.persistent_ccl if 4 in outputs else None,
        ccl_tuning=decoder.ccl_tuning if 4 in outputs else None,
        collective_memory=str(decoder.collective_memory) if 4 in outputs else None,
        dense_geometry=args.dense_geometry,
        moe_ccl_bfp8=args.moe_ccl_bfp8,
        expert_gate_bfp4=args.expert_gate_bfp4,
        expert_gate_dtype_requested=args.expert_gate_dtype,
        expert_fidelity_override=args.expert_fidelity,
        shared_gate_bfp4=args.shared_gate_bfp4,
        shared_down_bfp4=args.shared_down_bfp4,
        expert_activation_bfp8=args.expert_activation_bfp8,
        split_expert_gate=args.split_expert_gate,
        split_shared_gate=args.split_shared_gate,
        projection_k=args.projection_k,
        projection_role=args.projection_role,
        sharded_decode_rope=args.sharded_decode_rope,
        qkv_fidelity=args.qkv_fidelity,
        output_fidelity=args.output_fidelity,
        attention_ccl_dtype=args.attention_ccl_dtype,
        full_attention_ccl_dtype_override=args.full_attention_ccl_dtype,
        command=sys.argv,
        expert_parallel=args.expert_parallel,
        fused_tail=args.fused_tail,
        optimized_shared=args.optimized_shared,
        shared_dram=args.shared_dram,
        attention_dram=args.attention_dram,
        shared_geometry_requested=args.shared_geometry,
        shared_geometry=decoder.shared_geometry if 4 in outputs else None,
        sparse_gate_geometry=args.sparse_gate_geometry,
        sparse_down_geometry=args.sparse_down_geometry,
        grouped_moe_reduce=args.grouped_moe_reduce,
        hybrid_experts=args.hybrid_experts,
        cache_pcc=cache_pcc,
        capacity=capacity,
        layer_type=config.layer_types[args.layer],
        length=args.length,
        steps=args.steps,
        pcc=values,
        passed=_pcc_values_pass(values) if values else None,
        trace=args.trace,
        duplicate_replays_per_position=args.duplicate_replays if args.trace and not args.profile else 0,
        timings=timings,
        all_replicas_equal=not args.sharded_residual,
        sharded_residual=args.sharded_residual,
        topology="ring" if args.ring else "linear",
        fused_agmm=args.fused_agmm,
    )
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        {k: v for k, v in result.items() if k not in ("timings", "pcc")},
        "min_pcc",
        min(values) if values else None,
        flush=True,
    )
    if values:
        assert result["passed"]


if __name__ == "__main__":
    main()
