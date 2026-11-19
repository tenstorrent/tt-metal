# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Warmed serial B32 prefill cost of private, pooled RM, and pooled TILE history."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, weights, pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tt.multichip_decoder import MultichipDecoder, MultichipCollectives
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_tiled_history_candidate import TiledHistoryCandidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lengths", default="32,33,1024,1025")
    ap.add_argument("--second-layout", choices=["rm", "tile"], default="tile")
    ap.add_argument("--shared-collectives", action="store_true")
    ap.add_argument("--sync-each", action="store_true")
    ap.add_argument("--clone-outputs", action="store_true")
    ap.add_argument("--diagnostic", action="store_true")
    ap.add_argument("--direct-history", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(927)
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=0)
    rows = []
    try:
        sd = weights(0)
        history_cls = TiledHistoryCandidate
        if args.direct_history:
            from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_history_row_write import DirectHistoryCandidate
            history_cls = DirectHistoryCandidate
        collectives = MultichipCollectives(mesh) if args.shared_collectives else None
        layers = [cls.from_state_dict(sd, hf_config=config(), layer_idx=0, mesh_device=mesh, collectives=collectives)
                  for cls in (MultichipDecoder, history_cls if args.second_layout == "tile" else MultichipDecoder)]
        for length in map(int, args.lengths.split(",")):
            hosts = [(torch.randn(1, 1, length, 4096) * 0.1).bfloat16() for _ in range(args.batch)]
            inputs = [ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh,
                      mesh_mapper=ttnn.ReplicateTensorToMesh(mesh)) for x in hosts]
            expected = None
            for name, layer in (("private", layers[0]), ("pooled_rm", layers[0]), ("pooled_tile", layers[1])):
                capacity = ((length + 31) // 32) * 32
                states = ([layer.allocate_state(context=capacity) for _ in inputs] if name == "private"
                          else layer.allocate_batch_state(batch=args.batch, context=capacity))
                unique = {id(value): value for state in states for value in buffers(state)}
                snapshots = [(value, ttnn.clone(value)) for value in unique.values()]
                samples, diagnostics = [], []
                for repeat in range(18):
                    for dst, src in snapshots:
                        ttnn.copy(src, dst)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    with device_only():
                        outputs = []
                        for x, state in zip(inputs, states):
                            value = layer.prefill_forward(x, state=state)
                            outputs.append(ttnn.clone(value) if args.clone_outputs else value)
                            if args.sync_each:
                                ttnn.synchronize_device(mesh)
                    ttnn.synchronize_device(mesh)
                    elapsed = (time.perf_counter() - begin) * 1000
                    if repeat >= 3:
                        samples.append(elapsed)
                    if args.diagnostic:
                        downloaded = [[ttnn.to_torch(rank) for rank in ttnn.get_device_tensors(out)] for out in outputs]
                        if expected is None:
                            expected = [ranks[0] for ranks in downloaded]
                        for request, (reference, ranks) in enumerate(zip(expected, downloaded)):
                            finite = all(bool(torch.isfinite(rank).all()) for rank in ranks)
                            diagnostics.append(dict(repeat=repeat, request=request, finite=finite,
                                pcc=min(pcc(reference, rank) for rank in ranks) if finite else None,
                                agreement=min(pcc(ranks[0], rank) for rank in ranks[1:]) if finite else None,
                                output_address=outputs[request].buffer_address()))
                        Path(args.output).with_suffix(".diagnostic.json").write_text(json.dumps(
                            dict(layout=name, options=vars(args), records=diagnostics), indent=2)+"\n")
                    if repeat != 17:
                        del outputs
                downloaded = [[ttnn.to_torch(rank) for rank in ttnn.get_device_tensors(out)] for out in outputs]
                if expected is None:
                    expected = [ranks[0] for ranks in downloaded]
                scores = [pcc(ref, rank) for ref, ranks in zip(expected, downloaded) for rank in ranks]
                contributions = [pcc(ref - x, rank - x) for ref, x, ranks in zip(expected, hosts, downloaded)
                                 for rank in ranks]
                agreement = [pcc(ranks[0], rank) for ranks in downloaded for rank in ranks[1:]]
                row = dict(layout=name if name != "pooled_tile" else "second_" + args.second_layout, options=vars(args), batch=args.batch, length=length, prefill_batch_ms=statistics.median(samples),
                           samples_ms=samples, pcc=min(scores), contribution_pcc=min(contributions),
                           rank_agreement=min(agreement), warmups=3, repeats=15)
                rows.append(row)
                Path(args.output).write_text(json.dumps(dict(records=rows, devices=[0, 1, 2, 3],
                    mesh=[1, 4], regime="serial", complete=False), indent=2) + "\n")
                assert min(scores + contributions) >= 0.995 and min(agreement) >= 0.99999, row
                del outputs, downloaded, states, snapshots, unique
                print("HISTORY_PREFILL", name, length, row["prefill_batch_ms"], flush=True)
            del inputs, hosts, expected
        Path(args.output).write_text(json.dumps(dict(records=rows, devices=[0, 1, 2, 3], mesh=[1, 4], regime="serial", complete=True,
            description="32 independent prefills executed serially; host inputs prepared before timing; state reset excluded",
            source_sha256=hashlib.sha256((Path(__file__).parents[1] / "tt/multichip_decoder.py").read_bytes()).hexdigest()), indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
