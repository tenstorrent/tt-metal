"""Serialized real-weight, identical-input fusion candidate measurements."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoConfig

import ttnn

# Isolate historical candidate monkeypatches from the delivered runtime.
import sys
from models.autoports.qwen_qwen3_8_flash_next.tests import optimized_experiment_base
sys.modules["models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder"] = optimized_experiment_base
from models.autoports.qwen_qwen3_8_flash_next import tt as _tt_package
_tt_package.optimized_decoder = optimized_experiment_base
from models.autoports.qwen_qwen3_8_flash_next.tests.test_functional_decoder import (
    CHECKPOINT,
    no_runtime_host_conversion,
)
from models.autoports.qwen_qwen3_8_flash_next.tt.checkpoint import CheckpointState
from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedDecoder
from models.autoports.qwen_qwen3_8_flash_next.tt.fused_decoder import FusedDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--impl", choices=["fused", "optimized"], default="optimized")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--state-workers", type=int, default=32)
    parser.add_argument("--attention-dtype", choices=["bf16", "bfp8", "bfp4"], default="bf16")
    parser.add_argument("--attention-fidelity", choices=["HiFi4", "HiFi2", "LoFi"], default="HiFi4")
    parser.add_argument("--expert-dtype", choices=["bfp8", "bfp4"], default="bfp8")
    parser.add_argument("--down-dtype", choices=["bfp8", "bfp4"], default="bfp8")
    parser.add_argument("--compact-moe", action="store_true")
    parser.add_argument("--logical-router", action="store_true")
    parser.add_argument("--dram", action="store_true")
    parser.add_argument("--dram-cores", default="5,8")
    parser.add_argument("--dram-blocks", default="16,24")
    parser.add_argument("--dram-readers", default="1")
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--sparse-cores", type=int, default=20)
    parser.add_argument("--sparse-blocks", default="4,4")
    parser.add_argument("--sparse-dram", action="store_true")
    parser.add_argument("--sparse-split", action="store_true")
    parser.add_argument("--sparse-fidelity", default="LoFi")
    parser.add_argument("--sparse-exact", action="store_true")
    parser.add_argument("--prefill-grid", default="")
    parser.add_argument("--prefill-fidelity", default="LoFi")
    parser.add_argument("--hc-dtype", default="")
    parser.add_argument("--hc-fidelity", default="HiFi4")
    parser.add_argument("--hc-norm-cores", type=int, default=0)
    parser.add_argument("--prefill-2d", action="store_true")
    parser.add_argument("--prefill-block", type=int, default=8)
    parser.add_argument("--hc-dram-readers", type=int, default=0)
    parser.add_argument("--hc-dram-cores", default="20,2")
    parser.add_argument("--hc-dram-blocks", default="16,5")
    parser.add_argument("--hybrid-moe", action="store_true")
    parser.add_argument("--prefill-full-reduce", action="store_true")
    parser.add_argument("--qsa-sparse", action="store_true")
    parser.add_argument("--qsa-cache", choices=["bf16","fp8"],default="bf16")
    parser.add_argument("--qsa-chunk",type=int,default=128)
    parser.add_argument("--prefill-length", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--decode-position", type=int, default=0)
    parser.add_argument("--hc-unfused-up", action="store_true")
    parser.add_argument("--split-attention", action="store_true")
    parser.add_argument("--split-dram", action="store_true")
    parser.add_argument("--ple-dtype", default="")
    parser.add_argument("--ple-fidelity", default="HiFi4")
    parser.add_argument("--ple-readers", type=int, default=0)
    parser.add_argument("--ple-cores", type=int, default=5)
    parser.add_argument("--ple-block", type=int, default=4)
    parser.add_argument("--ple-split", action="store_true")
    parser.add_argument("--shared-dtype", default="")
    parser.add_argument("--shared-fidelity", default="HiFi4")
    parser.add_argument("--shared-readers", type=int, default=0)
    parser.add_argument("--shared-cores", default="5,2")
    parser.add_argument("--shared-blocks", default="16,10")
    parser.add_argument("--shared-split", action="store_true")
    parser.add_argument("--split-readers", type=int, default=1)
    parser.add_argument("--split-cores", type=int, default=5)
    parser.add_argument("--split-block", type=int, default=4)
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--minimal-cast", action="store_true")
    parser.add_argument("--minimal-block", default="4,4,8")
    parser.add_argument("--hc-keep-sharded",action="store_true")
    parser.add_argument("--ple-norm-cores",type=int,default=0)
    parser.add_argument("--ple-keep-sharded",action="store_true")
    parser.add_argument("--output-dtype",default="")
    parser.add_argument("--output-fidelity",default="")
    parser.add_argument("--fused-routing",action="store_true")
    parser.add_argument("--reduce-bf8",action="store_true")
    parser.add_argument("--hc-adapt-flatten",action="store_true")
    parser.add_argument("--ple-adapt-l1",action="store_true")
    parser.add_argument("--qsa-direct-topk",action="store_true")
    parser.add_argument("--qsa-dynamic-topk",action="store_true")
    parser.add_argument("--qsa-fidelity",default="HiFi4")
    parser.add_argument("--qsa-indexer",action="store_true")
    parser.add_argument("--qsa-indexer-chunk",type=int,default=128)
    parser.add_argument("--moe-padded-topk",action="store_true")
    parser.add_argument("--hc-adapt-binary",action="store_true")
    parser.add_argument("--sparse-down-cores",type=int,default=0)
    parser.add_argument("--gdn-roll",action="store_true")
    parser.add_argument("--gdn-inplace",action="store_true")
    parser.add_argument("--gdn-composite-norm",action="store_true")
    parser.add_argument("--dram-output",choices=["dram","l1","sharded"],default="dram")
    parser.add_argument("--qsa-mask-bounds",action="store_true")
    parser.add_argument("--best-candidate",action="store_true")
    parser.add_argument("--sparse-fused-gate",action="store_true")
    parser.add_argument("--hc-split-read",action="store_true")
    args = parser.parse_args()
    if not 0 <= args.decode_position < args.prefill_length:
        raise ValueError("The benchmark must prefill the decode context")
    from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedGatedDeltaNet
    from models.autoports.qwen_qwen3_8_flash_next.tt.optimized_decoder import OptimizedQSA, OptimizedMoE
    dtypes = dict(bf16=ttnn.bfloat16, bfp8=ttnn.bfloat8_b, bfp4=ttnn.bfloat4_b)
    for cls in (OptimizedGatedDeltaNet, OptimizedQSA):
        cls.output_dtype=dtypes[args.output_dtype] if args.output_dtype else None
        cls.output_fidelity=getattr(ttnn.MathFidelity,args.output_fidelity) if args.output_fidelity else None
        cls.projection_dtype = dtypes[args.attention_dtype]
        cls.projection_fidelity = getattr(ttnn.MathFidelity, args.attention_fidelity)
    OptimizedMoE.dedicated_weight_dtype = dtypes[args.expert_dtype]
    OptimizedMoE.down_dtype = dtypes[args.down_dtype]
    OptimizedGatedDeltaNet.state_workers = args.state_workers
    if args.compact_moe:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_compact_moe_candidate import CompactMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        CompactMoE.logical_router = args.logical_router
        optimized_decoder.OptimizedMoE = CompactMoE
    if args.sparse:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_sparse_candidate import SparseMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        SparseMoE.exact_nnz = args.sparse_exact
        SparseMoE.sparse_down_cores = args.sparse_down_cores
        SparseMoE.sparse_cores = args.sparse_cores
        SparseMoE.gate_block, SparseMoE.down_block = map(int,args.sparse_blocks.split(','))
        SparseMoE.sparse_l1 = not args.sparse_dram
        SparseMoE.sparse_fuse_silu=args.sparse_fused_gate
        SparseMoE.sparse_packed = not args.sparse_split
        SparseMoE.sparse_fidelity = getattr(ttnn.MathFidelity,args.sparse_fidelity)
        optimized_decoder.OptimizedMoE = SparseMoE
    if args.prefill_grid:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_prefill_candidate import NativePrefillMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        NativePrefillMoE.prefill_grid=tuple(map(int,args.prefill_grid.split(',')))
        NativePrefillMoE.prefill_fidelity=getattr(ttnn.MathFidelity,args.prefill_fidelity)
        optimized_decoder.OptimizedMoE=NativePrefillMoE
    if args.dram:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_dram_candidate import install
        install(tuple(map(int,args.dram_cores.split(','))),tuple(map(int,args.dram_blocks.split(','))),tuple(map(int,args.dram_readers.split(','))) if ',' in args.dram_readers else int(args.dram_readers),result_memory=args.dram_output)
    if args.hybrid_moe:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_hybrid_moe_candidate import HybridMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        HybridMoE.sparse_down_cores=args.sparse_down_cores
        HybridMoE.sparse_cores=args.sparse_cores
        HybridMoE.gate_block,HybridMoE.down_block=map(int,args.sparse_blocks.split(','))
        HybridMoE.sparse_l1=not args.sparse_dram
        HybridMoE.prefill_compact=not args.prefill_full_reduce
        optimized_decoder.OptimizedMoE=HybridMoE
    if args.hc_dtype:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_hyper_candidate import ShardedHyperConnection
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        ShardedHyperConnection.fused_up=not args.hc_unfused_up
        ShardedHyperConnection.weight_dtype=dtypes[args.hc_dtype]
        ShardedHyperConnection.fidelity=getattr(ttnn.MathFidelity,args.hc_fidelity)
        ShardedHyperConnection.split_read=args.hc_split_read
        ShardedHyperConnection.adapt_binary=args.hc_adapt_binary
        ShardedHyperConnection.adapt_flatten=args.hc_adapt_flatten
        ShardedHyperConnection.keep_sharded=args.hc_keep_sharded
        ShardedHyperConnection.norm_cores=args.hc_norm_cores
        ShardedHyperConnection.dram_readers=args.hc_dram_readers
        ShardedHyperConnection.dram_cores=tuple(map(int,args.hc_dram_cores.split(',')))
        ShardedHyperConnection.dram_blocks=tuple(map(int,args.hc_dram_blocks.split(',')))
        optimized_decoder.OptimizedHyperConnection=ShardedHyperConnection
    if args.prefill_2d:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_projection_prefill_candidate import install
        install(k_block=args.prefill_block)
    if args.qsa_sparse:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_qsa_sparse_candidate import SparseQSA
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        SparseQSA.direct_topk=args.qsa_direct_topk
        SparseQSA.dynamic_topk=args.qsa_dynamic_topk
        SparseQSA.sparse_fidelity=getattr(ttnn.MathFidelity,args.qsa_fidelity)
        SparseQSA.mask_bounds=args.qsa_mask_bounds
        SparseQSA.indexer=args.qsa_indexer
        SparseQSA.indexer_chunk=args.qsa_indexer_chunk
        SparseQSA.cache_dtype=ttnn.bfloat16 if args.qsa_cache=='bf16' else ttnn.fp8_e4m3
        SparseQSA.sparse_chunk=args.qsa_chunk
        SparseQSA.index_width=((2051+args.qsa_chunk-1)//args.qsa_chunk)*args.qsa_chunk
        optimized_decoder.OptimizedQSA=SparseQSA
    if args.split_attention:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_split_projection_candidate import install
        install(dram=args.split_dram,readers=args.split_readers,core_count=args.split_cores,block=args.split_block)
    if args.ple_dtype:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_ple_candidate import TunedPLE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        TunedPLE.weight_dtype=dtypes[args.ple_dtype]
        TunedPLE.fidelity=getattr(ttnn.MathFidelity,args.ple_fidelity)
        TunedPLE.readers=args.ple_readers
        TunedPLE.cores=args.ple_cores
        TunedPLE.block=args.ple_block
        TunedPLE.split=args.ple_split
        TunedPLE.adapt_l1=args.ple_adapt_l1
        TunedPLE.norm_cores=args.ple_norm_cores
        TunedPLE.keep_sharded=args.ple_keep_sharded
        optimized_decoder.OptimizedPLE=TunedPLE
    if args.shared_dtype:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_shared_candidate import SharedMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        SharedMoE.shared_dtype=dtypes[args.shared_dtype]
        SharedMoE.shared_fidelity=getattr(ttnn.MathFidelity,args.shared_fidelity)
        SharedMoE.shared_readers=args.shared_readers
        SharedMoE.shared_cores=tuple(map(int,args.shared_cores.split(',')))
        SharedMoE.shared_blocks=tuple(map(int,args.shared_blocks.split(',')))
        SharedMoE.shared_split=args.shared_split
        optimized_decoder.OptimizedMoE=SharedMoE
    if args.minimal:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_minimal_candidate import install
        install(cast=args.minimal_cast,blocks=tuple(map(int,args.minimal_block.split(','))))
    if args.fused_routing:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_routing_candidate import RoutedMoE
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        RoutedMoE.padded_topk=args.moe_padded_topk
        RoutedMoE.reduce_bf8=args.reduce_bf8
        optimized_decoder.OptimizedMoE=RoutedMoE
    if args.gdn_roll or args.gdn_inplace or args.gdn_composite_norm:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_gdn_movement_candidate import MovementGDN
        from models.autoports.qwen_qwen3_8_flash_next.tt import optimized_decoder
        MovementGDN.roll=args.gdn_roll
        MovementGDN.inplace=args.gdn_inplace
        MovementGDN.composite_norm=args.gdn_composite_norm
        optimized_decoder.OptimizedGatedDeltaNet=MovementGDN
    if args.best_candidate:
        from models.autoports.qwen_qwen3_8_flash_next.tests.optimized_candidate_policy import install
        install()
    torch.set_num_threads(8)
    root = Path(__file__).resolve().parents[1]
    dest = root / "doc/optimized_decoder" / args.tag
    dest.mkdir(parents=True, exist_ok=True)
    source = root / "tt/fused_decoder.py" if args.impl == "fused" else root / "tests/optimized_experiment_base.py"
    source_snapshot = source.read_bytes()
    (dest / "source.py.txt").write_bytes(source_snapshot)
    (dest / "runner.py.txt").write_bytes(Path(__file__).read_bytes())
    for candidate_source in (root / "tests").glob("optimized_*candidate.py"):
        (dest / (candidate_source.name + ".txt")).write_bytes(candidate_source.read_bytes())
    cfg = AutoConfig.from_pretrained(CHECKPOINT, local_files_only=True).text_config
    state = CheckpointState(CHECKPOINT, f"model.language_model.layers.{args.layer}.")
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    trace = None
    try:
        model = (FusedDecoder if args.impl == "fused" else OptimizedDecoder).from_state_dict(
            state,
            hf_config=cfg,
            layer_idx=args.layer,
            mesh_device=mesh,
            batch_size=1,
            chunk_size=args.chunk_size,
            max_requests=3,
            cache_dir=Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/de4b8e4_ple"),
        )
        model.prepare([1, args.prefill_length])
        identities = {
            name: type(getattr(model, name)).__module__ + "." + type(getattr(model, name)).__qualname__
            for name in ("attention", "mlp", "ple", "attn_hyper", "mlp_hyper")
            if hasattr(model, name)
        }

        def upload(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(x, dtype=dtype, layout=layout, device=mesh)

        rng = torch.Generator().manual_seed(2026)
        x = (
            torch.randn(1, args.prefill_length, 2560, generator=rng).repeat(1, 1, 4) * 0.05
            + torch.randn(1, args.prefill_length, 10240, generator=rng) * 0.005
        ).bfloat16()
        tokens = torch.randint(cfg.vocab_size, (1, args.prefill_length), generator=rng, dtype=torch.int32)
        xd, dd = upload(x.unsqueeze(0)), upload(x[:, :1].unsqueeze(0))
        pd = upload(torch.zeros(1, 32, dtype=torch.int32), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        decode_position = torch.zeros(1,32,dtype=torch.int32)
        decode_position[0,0] = args.decode_position
        dp = upload(decode_position,ttnn.uint32,ttnn.ROW_MAJOR_LAYOUT)
        slots = torch.zeros(1, 32, dtype=torch.int32)
        slots[0, 0] = 1
        sd = upload(slots, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT)
        page = (
            upload(
                torch.randperm(cfg.max_position_embeddings // 32, generator=rng).reshape(1, -1).int(),
                ttnn.int32,
                ttnn.ROW_MAJOR_LAYOUT,
            )
            if args.layer == 3
            else None
        )
        pt = upload(tokens, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        dt = upload(tokens[:, :1], ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT) if args.layer == 1 else None
        common = dict(current_positions=pd, slots=sd, page_table=page)

        def prefill():
            with no_runtime_host_conversion():
                return model.prefill_forward(xd, tokens=pt, **common)

        def decode():
            with no_runtime_host_conversion():
                return model.decode_forward(dd, tokens=dt, **dict(common,current_positions=dp))

        def drain():
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)

        for _ in range(2):
            prefill()
            decode()
            ttnn.synchronize_device(mesh)
            drain()
        if args.profile:
            from tracy import signpost
        else:
            signpost = lambda _: None
        timings = []
        prefill_host = None
        for _ in range(1 if args.profile else 5):
            signpost("PERF_PREFILL")
            before = time.perf_counter()
            out = prefill()
            ttnn.synchronize_device(mesh)
            timings.append((time.perf_counter() - before) * 1000)
            signpost("PERF_PREFILL_END")
            drain()
            prefill_host = ttnn.to_torch(out)
            del out
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = decode()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        drain()
        for _ in range(2):
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            drain()
        samples = []
        for _ in range(1 if args.profile else 5):
            signpost("PERF_DECODE")
            before = time.perf_counter()
            for _ in range(10):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - before) * 1000 / 10)
            signpost("PERF_DECODE_END")
            drain()
        saved = Path("/home/vkovacevic/qwen38-flash/tt-metal-cache/qwen_qwen3_8_flash_next/optimization_outputs")
        saved.mkdir(parents=True, exist_ok=True)
        output_path = saved / (args.tag + ".pt")
        torch.save({"prefill": prefill_host, "decode": ttnn.to_torch(out)}, output_path)
        result = dict(
            impl=args.impl,
            layer=args.layer,
            tag=args.tag,
            seed=2026,
            policy=vars(args),
            runtime_classes=identities,
            batch=1,
            prefill_length=args.prefill_length,
            chunk_size=args.chunk_size,
            position=args.decode_position,
            decode_replays=50 if not args.profile else 10,
            prefill_host_ms=timings,
            traced_decode_ms=samples,
            median_prefill_host_ms=statistics.median(timings),
            median_traced_decode_ms=statistics.median(samples),
            profiler=args.profile,
            tracking=os.environ.get("TT_METAL_TRACE_ALLOC_TRACKING"),
            outputs=str(output_path),
            source_sha256=hashlib.sha256(source_snapshot).hexdigest(),
        )
        (dest / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
