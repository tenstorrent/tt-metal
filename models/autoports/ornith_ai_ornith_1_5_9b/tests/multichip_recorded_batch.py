# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pooled TP4 decode precision on recorded HF activations and independent outputs."""

import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import config, weights, pcc
from models.autoports.ornith_ai_ornith_1_5_9b.tests.validate_optimized import buffers, device_only
from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_gdn_fidelity_candidate import GdnFidelityCandidate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gdn-fidelity", choices=["LoFi", "HiFi2", "HiFi4"], default="HiFi4")
    ap.add_argument("--approximate", action="store_true")
    ap.add_argument("--family", choices=["separate", "persistent", "bf8", "bf8_persistent"])
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(737)
    data = torch.load(args.inputs, weights_only=True)
    choices = [c for c in data["cases"] if c["layer"] == args.layer]
    cases = [choices[i % len(choices)] for i in range(args.batch)]
    GdnFidelityCandidate.selected_fidelity = args.gdn_fidelity
    GdnFidelityCandidate.approximate = args.approximate
    decoder_cls = GdnFidelityCandidate
    if args.family:
        from models.autoports.ornith_ai_ornith_1_5_9b.tests.multichip_batched_family_candidate import BatchedFamilyCandidate
        decoder_cls = BatchedFamilyCandidate
        decoder_cls.batch_size = args.batch
        decoder_cls.persistent = "persistent" in args.family
        if "bf8" in args.family:
            decoder_cls.communication_dtype = ttnn.bfloat8_b
            decoder_cls.norm_communication_dtype = ttnn.bfloat8_b
    ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 4), trace_region_size=64 * 1024 * 1024)
    records, failed = [], []

    def upload(value, integer=False, device=True):
        return ttnn.from_torch(
            value.contiguous(),
            dtype=ttnn.int32 if integer else ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
            device=mesh if device else None,
            mesh_mapper=(ttnn.ShardTensorToMesh(mesh, dim=3) if args.family and not integer
                         else ttnn.ReplicateTensorToMesh(mesh)),
        )

    def download(value):
        ranks = [ttnn.to_torch(x) for x in ttnn.get_device_tensors(value)]
        return [torch.cat(ranks, dim=-1)] if args.family else ranks

    def check(label, actual, expected, residual):
        scores = [pcc(expected, x) for x in actual]
        contributions = [pcc(expected - residual, x - residual) for x in actual]
        agreement = [pcc(actual[0], x) for x in actual[1:]]
        row = dict(label=label, pcc=min(scores), contribution_pcc=min(contributions),
                   rank_agreement=min(agreement) if agreement else None)
        records.append(row)
        if min(scores + contributions) < 0.995 or (agreement and min(agreement) < 0.99999):
            failed.append(row)

    try:
        layer = decoder_cls.from_state_dict(
            weights(args.layer),
            hf_config=config(),
            layer_idx=args.layer,
            mesh_device=mesh,
            residual_layout="replicated",
        )
        capacity = ((max(c["length"] + c["steps"] for c in cases) + 31) // 32) * 32
        states = layer.allocate_batch_state(batch=args.batch, context=capacity)
        tables = [upload(torch.randperm(capacity // 32, dtype=torch.int32)[None], True) for _ in cases]
        positions = [upload(torch.tensor([c["length"]], dtype=torch.int32), True) for c in cases]
        for i, (case, state, table) in enumerate(zip(cases, states, tables)):
            tx = upload(case["prefill_input"][None])
            with device_only():
                out = layer.prefill_forward(tx, state=state, page_table=table)
            ranks = [x.reshape_as(case["prefill_output"]) for x in download(out)]
            check(f"prefill_{i}", ranks, case["prefill_output"], case["prefill_input"])
            del tx, out
        unique = {id(b): b for state in states for b in buffers(state)}
        snapshots = [(b, ttnn.clone(b)) for b in unique.values()]

        def restore():
            for dst, src in snapshots:
                ttnn.copy(src, dst)

        def packed(step):
            return torch.cat([c["decode_input"][step] for c in cases], dim=1)[None]

        td = upload(packed(0))

        def forward():
            with device_only():
                return layer.decode_batch_packed(td, states=states, current_positions=positions, page_tables=tables)

        for _ in range(2):
            out = forward()
            del out
            restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        out = forward()
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        restore()
        steps = min(c["steps"] for c in cases)
        for step in range(steps):
            ttnn.copy_host_to_device_tensor(upload(packed(step), device=False), td)
            for case, position in zip(cases, positions):
                ttnn.copy_host_to_device_tensor(
                    upload(torch.tensor([case["length"] + step], dtype=torch.int32), True, False), position
                )
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ranks = download(out)
            for i, case in enumerate(cases):
                check(
                    f"decode_{step}_{i}",
                    [rank[0, :, i : i + 1] for rank in ranks],
                    case["decode_output"][step],
                    case["decode_input"][step],
                )
        ttnn.copy_host_to_device_tensor(upload(packed(0), device=False), td)
        for case, position in zip(cases, positions):
            ttnn.copy_host_to_device_tensor(
                upload(torch.tensor([case["length"]], dtype=torch.int32), True, False), position
            )
        samples = []
        for _ in range(5):
            restore()
            ttnn.synchronize_device(mesh)
            begin = time.perf_counter()
            for _ in range(50):
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            samples.append((time.perf_counter() - begin) * 1000 / 50)
        ttnn.release_trace(mesh, trace)
        result = dict(
            vars(args),
            records=records,
            failed=failed,
            passed=not failed,
            input_sha256=hashlib.sha256(Path(args.inputs).read_bytes()).hexdigest(),
            actual_weights={k: str(v.dtype) for k, v in layer.weights.items()},
            implementation_files={inspect.getfile(base): hashlib.sha256(Path(inspect.getfile(base)).read_bytes()).hexdigest()
                                  for base in type(layer).__mro__ if base is not object},
            semantics="Independent HF layer outputs on recorded teacher-forced inputs; two corpora and varied prefix lengths",
            gdn_fp32_dest_acc_en=True,
            decode_ms=statistics.median(samples),
            samples_ms=samples,
            devices=[0, 1, 2, 3],
            mesh=[1, 4],
            regime="serial",
            threshold=0.995,
            traced=True,
            residual_layout="hidden-sharded" if args.family else "replicated",
            steps=steps,
            cases=[dict(corpus=c["corpus"], length=c["length"]) for c in cases],
        )
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
        assert not failed, failed[:3]
        print("RECORDED_BATCH_PASS", args.gdn_fidelity, args.approximate, args.batch)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
