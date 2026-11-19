# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Complete advertised-context execution, sampled TTNN baseline windows and terminal trace."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, clear, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--layout", default="sharded")
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(91)
    c = config()
    n = c.max_position_embeddings
    sd = weights(args.layer)
    x = torch.randn(1, 1, n, 4096, dtype=torch.bfloat16) * 0.1
    table_host = torch.randperm(n // 32, dtype=torch.int32)[None]
    expected = {} if args.baseline else torch.load(args.reference, weights_only=True)
    saved, records = {}, []
    if not args.baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if args.baseline else 4),
        physical_device_ids=[0] if args.baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )

    def upload(v, integer=False, residual=False):
        return ttnn.from_torch(
            v.contiguous(),
            device=mesh,
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            mesh_mapper=(
                ttnn.ShardTensorToMesh(mesh, dim=3)
                if residual and not args.baseline and args.layout == "sharded"
                else ttnn.ReplicateTensorToMesh(mesh)
            ),
        )

    def download(v):
        if args.baseline:
            return ttnn.to_torch(v)
        if args.layout == "sharded":
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=3))
        ranks = [ttnn.to_torch(t) for t in ttnn.get_device_tensors(v)]
        assert all(pcc(ranks[0], other) >= 0.99999 for other in ranks[1:]), "replicated ranks diverged"
        return ranks[0]

    def check(label, v, residual):
        saved[label] = v
        if not args.baseline:
            row = {
                "label": label,
                "pcc": pcc(expected[label], v),
                "contribution_pcc": pcc(expected[label] - residual, v - residual),
            }
            assert min(row["pcc"], row["contribution_pcc"]) >= 0.995, row
            records.append(row)
            print(json.dumps(row), flush=True)

    try:
        layer = (OptimizedDecoder if args.baseline else MultichipDecoder).from_state_dict(
            sd,
            hf_config=c,
            layer_idx=args.layer,
            mesh_device=mesh,
            **({} if args.baseline else {"residual_layout": args.layout}),
        )
        state = layer.allocate_state(context=n)
        table = upload(table_host, True)
        pos = upload(torch.tensor([n - 1], dtype=torch.int32), True)
        dec = upload(x[:, :, n - 1 :, :], residual=True)
        spans = [(0, 32), (n // 2, 32), (n - 64, 32), (n - 32, 27), (n - 5, 4), (n - 1, 1)]
        for length in [n - 5, n - 1, n]:
            clear(state)
            tx = upload(x[:, :, :length, :], residual=True)
            with device_only():
                out = layer.prefill_forward(tx, state=state, page_table=table)
            ttnn.synchronize_device(mesh)
            for start, count in spans:
                if start + count <= length:
                    check(
                        f"prefill_{length}_{start}",
                        download(out[:, :, start : start + count, :]),
                        x[:, :, start : start + count, :],
                    )
            del out, tx
            if length == n - 1:
                snaps = [(b, ttnn.clone(b)) for b in buffers(state)]

                def restore():
                    for dst, src in snaps:
                        ttnn.copy(src, dst)

                for _ in range(2):
                    warm = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
                    restore()
                    del warm
                trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                out = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
                ttnn.end_trace_capture(mesh, trace, cq_id=0)
                from ttnn.tools import trace_allocation_tracker

                trace_allocation_tracker.acknowledge_corruptible(out)
                previous = None
                for repeat in range(3):
                    restore()
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    v = download(out)
                    if previous is not None:
                        assert torch.equal(v, previous)
                    previous = v
                check("decode_terminal", v, x[:, :, n - 1 :, :])
                ttnn.release_trace(mesh, trace)
                del out, snaps
            print("PREFILL_DONE", length, flush=True)
        if args.baseline:
            torch.save(saved, args.reference)
        Path(args.output).write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "baseline": args.baseline,
                    "tracking": os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
                    "skip_program_cache_tracking": os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0"),
                    "layout": args.layout,
                    "prefill_lengths": [n - 5, n - 1, n],
                    "decode_position": n - 1,
                    "deterministic_replays": 3,
                    "records": records,
                    "source_sha256": hashlib.sha256((ROOT / "tt/multichip_decoder.py").read_bytes()).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        )
        print("LONG_CONTEXT_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
