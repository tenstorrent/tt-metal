# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-kind layer stack, arbitrary-start continuation and diverse ragged batch contracts."""

import argparse
import hashlib
import inspect
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipCollectives, MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--grouped", action="store_true")
    ap.add_argument("--stage4", action="store_true")
    ap.add_argument("--movement-control", action="store_true")
    ap.add_argument("--recurrent", action="store_true")
    ap.add_argument("--attention-batch", action="store_true")
    ap.add_argument("--pooled", action="store_true")
    ap.add_argument("--private-state", action="store_true")
    ap.add_argument("--full-pool", action="store_true")
    ap.add_argument("--chunk-gdn", action="store_true")
    ap.add_argument("--folded-conv", type=int)
    ap.add_argument("--precise-ccl", action="store_true")
    ap.add_argument("--table-offsets", action="store_true")
    ap.add_argument("--wave-heads", type=int, default=32)
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--layout", default="sharded")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--prefix-length", type=int, default=31)
    ap.add_argument("--context", type=int, default=1152)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--sample-prefill", action="store_true")
    ap.add_argument("--state-checks", action="store_true")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()
    args.full_pool = args.full_pool or args.chunk_gdn
    args.pooled = args.pooled or args.full_pool
    if args.profile:
        args.benchmark = False
    torch.set_num_threads(8)
    torch.manual_seed(119)
    c = config()
    sd = [weights(i) for i in [0, 3]]
    expected = {} if args.baseline else torch.load(args.reference, weights_only=True)
    saved, records = {}, []
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=64 * 1024 * 1024,
    )
    sharded = not args.baseline and args.layout == "sharded"

    def host(v, integer=False, residual=False, device=False):
        return ttnn.from_torch(
            v.contiguous(),
            device=mesh if device else None,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=(
                ttnn.ShardTensorToMesh(mesh, dim=3) if sharded and residual else ttnn.ReplicateTensorToMesh(mesh)
            ),
        )

    diagnostic_label = "prefill"

    def download(v):
        if args.baseline:
            return ttnn.to_torch(v)
        if sharded:
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        ranks = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(v)]
        agreements = [pcc(ranks[0], other) for other in ranks[1:]]
        if not all(score >= 0.99999 for score in agreements):
            failure = {
                "label": diagnostic_label,
                "rank_pcc": agreements,
                "rank_max_abs_difference": [
                    (ranks[0].float() - other.float()).abs().max().item() for other in ranks[1:]
                ],
                "reference_pcc": (
                    [pcc(expected[diagnostic_label], rank) for rank in ranks] if diagnostic_label in expected else None
                ),
                "records_completed": len(records),
            }
            Path(args.output + ".failure.json").write_text(json.dumps(failure, indent=2) + "\n")
            torch.save(ranks, args.output + ".failure.pt")
            print("RANK_FAILURE", json.dumps(failure), flush=True)
            raise AssertionError("replicated ranks diverged")
        return ranks[0]

    def check(label, v):
        saved[label] = v
        if not args.baseline:
            score = pcc(expected[label], v)
            assert score >= 0.995, (label, score)
            records.append({"label": label, "pcc": score})

    try:
        cls = OptimizedDecoder if args.baseline else MultichipDecoder
        if args.stage4:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_stage4_reference import (
                MultichipDecoder as Stage4Decoder,
            )

            cls = Stage4Decoder
        if args.grouped:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batched_candidate import BatchedCandidate

            cls = BatchedCandidate
            if args.recurrent:
                from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_recurrent_candidate import (
                    RecurrentCandidate,
                )

                cls = RecurrentCandidate
            if args.attention_batch:
                from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_attention_batch_candidate import (
                    AttentionBatchCandidate,
                )

                cls = AttentionBatchCandidate
        if args.pooled:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_pool_candidate import PoolCandidate

            cls = PoolCandidate
            if args.full_pool:
                from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_full_pool_candidate import (
                    FullPoolCandidate,
                )

                cls = FullPoolCandidate
                if args.chunk_gdn:
                    from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_chunk_pool_candidate import (
                        ChunkPoolCandidate,
                    )

                    cls = type("ChunkPool", (ChunkPoolCandidate,), {"wave_heads": args.wave_heads})
        if args.folded_conv:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_folded_conv_candidate import (
                FoldedConvCandidate,
            )

            cls = type("FoldedConv", (FoldedConvCandidate,), {"channel_chunk": args.folded_conv})
        if args.precise_ccl:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_precise_ccl_candidate import (
                PreciseCCLCandidate,
            )

            cls = PreciseCCLCandidate
        if args.table_offsets:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_table_candidate import BatchedTableCandidate

            cls = BatchedTableCandidate
        collectives = None if args.baseline else MultichipCollectives(mesh)
        layers = [
            cls.from_state_dict(
                s,
                hf_config=c,
                layer_idx=i,
                mesh_device=mesh,
                **(
                    {}
                    if args.baseline
                    else {
                        "residual_layout": args.layout,
                        "collectives": collectives,
                        **({"movement_optimized": False} if args.movement_control else {}),
                    }
                ),
            )
            for s, i in zip(sd, [0, 3])
        ]
        if not args.baseline:
            assert layers[0].ar_buffers is layers[1].ar_buffers
            assert layers[0].ar_semaphores is layers[1].ar_semaphores
        capacity = max(args.context, ((args.prefix_length + 2 + 18 + args.steps + 31) // 32) * 32)
        use_pool = not args.private_state and hasattr(layers[0], "allocate_batch_state")
        if use_pool:
            columns = [layer.allocate_batch_state(batch=args.batch, context=capacity) for layer in layers]
            states = [list(pair) for pair in zip(*columns)]
        else:
            states = [[l.allocate_state(context=capacity) for l in layers] for _ in range(args.batch)]
        table_hosts = [torch.randperm(capacity // 32, dtype=torch.int32)[None] for _ in range(args.batch)]
        tables = [host(t, True, device=True) for t in table_hosts]
        lengths = [args.prefix_length + (row % 3) for row in range(args.batch)]
        # Different initial lengths, shuffled tables and three non-aligned starts.
        for part in range(2):
            for row in range(args.batch):
                count = lengths[row] if part == 0 else 17 + row % 2
                start = 0 if part == 0 else lengths[row]
                x = (torch.randn(1, 1, count, 4096) * 0.1).bfloat16()
                tx = host(x, residual=True, device=True)
                with device_only():
                    for layer, st in zip(layers, states[row]):
                        tx = layer.prefill_forward(tx, state=st, start_position=start, page_table=tables[row])
                if args.sample_prefill and count > 96:
                    for start in (0, count // 2, count - 32):
                        check(f"prefill_{part}_{row}_{start}", download(tx[:, :, start : start + 32, :]))
                else:
                    check(f"prefill_{part}_{row}", download(tx))
                if args.profile:
                    ttnn.ReadDeviceProfiler(mesh)
        starts = [lengths[row] + 17 + row % 2 for row in range(args.batch)]
        dec = [host(torch.zeros(1, 1, 1, 4096, dtype=torch.bfloat16), residual=True, device=True) for _ in starts]
        positions = [host(torch.tensor([start], dtype=torch.int32), True, device=True) for start in starts]
        unique_buffers = {id(b): b for ss in states for s in ss for b in buffers(s)}
        snaps = [(b, ttnn.clone(b)) for b in unique_buffers.values()]

        def restore():
            for dst, src in snaps:
                ttnn.copy(src, dst)

        def forward():
            with device_only():
                if hasattr(layers[0], "decode_batch_packed") and layers[0].native_local:
                    packed = ttnn.concat(dec, dim=2) if len(dec) > 1 else dec[0]
                    for idx, layer in enumerate(layers):
                        packed = layer.decode_batch_packed(
                            packed, states=[s[idx] for s in states], current_positions=positions, page_tables=tables
                        )
                    packed = ttnn.to_memory_config(packed, ttnn.DRAM_MEMORY_CONFIG)
                    return [packed[:, :, i : i + 1, :] for i in range(args.batch)]
                rows = dec
                for idx, layer in enumerate(layers):
                    rows = layer.decode_batch(
                        rows, states=[s[idx] for s in states], current_positions=positions, page_tables=tables
                    )
                return rows

        if args.state_checks:
            # Warm inspection slice programs before capture. Their immutable
            # program-cache metadata must not be allocated in trace scratch
            # when state is inspected before a later benchmark replay.
            for row, pair in enumerate(states):
                state_row = getattr(pair[0], "state_row", None)
                if state_row is not None:
                    for name in ("conv", "recurrent"):
                        tensor = getattr(pair[0], name)
                        inspected = (
                            layers[0].read_linear_state(pair[0], name)
                            if hasattr(layers[0], "read_linear_state")
                            else tensor[state_row : state_row + 1]
                        )
                        del inspected
                for logical_page in sorted({0, (starts[row] + args.steps - 1) // 32}):
                    physical_page = int(table_hosts[row][0, logical_page]) + getattr(pair[1], "page_offset", 0)
                    for name in ("key", "value"):
                        inspected = getattr(pair[1], name)[physical_page : physical_page + 1]
                        del inspected

        for _ in range(2):
            warm = forward()
            del warm
            restore()
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        outputs = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        if args.profile:
            ttnn.ReadDeviceProfiler(mesh)
        from ttnn.tools import trace_allocation_tracker

        for out in outputs:
            trace_allocation_tracker.acknowledge_corruptible(out)
        restore()
        for step in range(args.steps):
            for row in range(args.batch):
                x = (torch.randn(1, 1, 1, 4096) * 0.1).bfloat16()
                ttnn.copy_host_to_device_tensor(host(x, residual=True), dec[row])
                ttnn.copy_host_to_device_tensor(
                    host(torch.tensor([starts[row] + step], dtype=torch.int32), True), positions[row]
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            for row, out in enumerate(outputs):
                diagnostic_label = f"decode_{step}_{row}"
                check(diagnostic_label, download(out))
            if args.profile:
                ttnn.ReadDeviceProfiler(mesh)
        if args.state_checks:

            def state_download(t, dim):
                return (
                    ttnn.to_torch(t)
                    if args.baseline
                    else ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=dim))
                )

            for row, pair in enumerate(states):
                conv_tensor, recurrent_tensor = pair[0].conv, pair[0].recurrent
                if getattr(pair[0], "state_row", None) is not None:
                    index = pair[0].state_row
                    conv_tensor = (
                        layers[0].read_linear_state(pair[0], "conv")
                        if hasattr(layers[0], "read_linear_state")
                        else conv_tensor[index : index + 1]
                    )
                    recurrent_tensor = recurrent_tensor[index : index + 1]
                if args.baseline:
                    conv = ttnn.to_torch(conv_tensor)
                else:
                    ranks = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(conv_tensor)]
                    # TP ranks own local Q/K/V channels; reconstruct global Q,K,V order.
                    conv = torch.cat(
                        [
                            torch.cat([v[..., start:end] for v in ranks], dim=-1)
                            for start, end in ((0, 512), (512, 1024), (1024, 2048))
                        ],
                        dim=-1,
                    )
                check(f"conv_{row}", conv)
                check(f"recurrent_{row}", state_download(recurrent_tensor, 1))
                del conv_tensor, recurrent_tensor
                for logical_page in sorted({0, (starts[row] + args.steps - 1) // 32}):
                    physical_page = int(table_hosts[row][0, logical_page]) + getattr(pair[1], "page_offset", 0)
                    for name in ("key", "value"):
                        value = getattr(pair[1], name)[physical_page : physical_page + 1]
                        check(f"{name}_{row}_{logical_page}", state_download(value, 1))
                        del value
        profile_host_ms = None
        if args.profile:
            from tracy import signpost

            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            restore()
            ttnn.synchronize_device(mesh)
            ttnn.ReadDeviceProfiler(mesh)
            signpost("PERF_DECODE")
            begin = time.perf_counter()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            profile_host_ms = (time.perf_counter() - begin) * 1000
            signpost("PERF_DECODE_END")
            ttnn.ReadDeviceProfiler(mesh)
        times = []
        if args.benchmark:
            for _ in range(7):
                restore()
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(20):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                times.append((time.perf_counter() - begin) * 1000 / 20)
        ttnn.release_trace(mesh, trace)
        if args.baseline:
            torch.save(saved, args.reference)
        Path(args.output).write_text(
            json.dumps(
                {
                    "baseline": args.baseline,
                    "implementation_class": cls.__module__ + "." + cls.__qualname__,
                    "command": sys.argv,
                    "implementation_files": {
                        str(Path(inspect.getfile(base)).relative_to(ROOT.parents[2])): hashlib.sha256(
                            Path(inspect.getfile(base)).read_bytes()
                        ).hexdigest()
                        for base in cls.__mro__
                        if base is not object
                    },
                    "grouped": hasattr(layers[0], "decode_batch_packed") and layers[0].native_local,
                    "stage4": args.stage4,
                    "movement_control": args.movement_control,
                    "recurrent": args.recurrent,
                    "attention_batch": args.attention_batch,
                    "pooled": use_pool,
                    "full_pool": args.full_pool,
                    "chunk_gdn": args.chunk_gdn,
                    "wave_heads": args.wave_heads if args.chunk_gdn else None,
                    "candidate_sha256": (
                        hashlib.sha256((ROOT / "tests/multichip_batched_candidate.py").read_bytes()).hexdigest()
                        if args.grouped
                        else None
                    ),
                    "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
                    "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
                    "batch": args.batch,
                    "prefix_length": args.prefix_length,
                    "capacity": capacity,
                    "steps": args.steps,
                    "state_checks": args.state_checks,
                    "sample_prefill": args.sample_prefill,
                    "layout": args.layout,
                    "stack": [0, 3],
                    "shared_collective_workspace": not args.baseline,
                    "starts": starts,
                    "records": records,
                    "trace_replays": args.steps,
                    "decode_stack_ms": statistics.median(times) if times else None,
                    "decode_stack_samples_ms": times,
                    "benchmark": args.benchmark,
                    "profile": args.profile,
                    "profile_decode_host_ms": profile_host_ms,
                    "source_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print("STACK_BATCH_PASS", args.batch, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
