# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Real/synthetic, same-instance boundary, continuation and batch evidence."""

import argparse
import json
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.run_decoder import hf_forward
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def run(args):
    from models.autoports.qwen_qwen3_6_35b_a3b.tests.provenance import record

    if args.output:
        record(args.output)
    torch.set_num_threads(8)
    torch.manual_seed(81)
    c = config()
    if args.synthetic:
        stats = json.loads((Path(__file__).parents[1] / "doc/functional_decoder/weight_stats.json").read_text())[
            str(args.layer)
        ]
        sd = {k: (torch.randn(tuple(v["shape"])) * v["std"] + v["mean"]).bfloat16() for k, v in stats.items()}
    else:
        sd = weights(args.layer)
    hf = layer(sd, args.layer)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh)

        def new():
            return (
                dec.new_recurrent_state()
                if dec.kind == "linear_attention"
                else dec.new_kv_state(context_length=1024, page_order=torch.randperm(32))
            )

        def up(x):
            return ttnn.from_torch(x, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        def read(x, shape):
            return ttnn.to_torch(x).reshape(shape)

        def reset(state):
            for k in ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v"):
                ttnn.copy(ttnn.zeros_like(state[k]), state[k])

        def check(a, b, **meta):
            score = pcc(a, b)
            rows.append(dict(meta, pcc=score))
            assert score >= 0.995, rows[-1]

        if args.mode in ("boundaries", "all"):
            state = new()
            for n in (1, 31, 32, 33, 255, 256, 257, 513, 7):
                reset(state)
                x = torch.randn(1, n, 2048, dtype=torch.bfloat16) * 0.05
                xd = torch.randn(1, 1, 2048, dtype=torch.bfloat16) * 0.05
                hc = DynamicCache(config=c)
                ref = hf_forward(hf, x, hc, 0)
                refd = hf_forward(hf, xd, hc, n)
                plan, tx, td = dec.prepare_prefill(seq_len=n), up(x), up(xd)
                pos = dec.prepare_positions(start_pos=n, seq_len=1) if dec.kind == "full_attention" else None
                with device_only():
                    out = dec.prefill_forward(tx, state=state, plan=plan)
                    outd = dec.decode_forward(td, state=state, positions=pos)
                check(ref, read(out, x.shape), mode="prefill", length=n)
                check(refd, read(outd, xd.shape), mode="decode", position=n)
                if n == 256:
                    reset(state)
                    alternate = dec.prepare_prefill(seq_len=n, chunk_size=32)
                    with device_only():
                        control = dec.prefill_forward(tx, state=state, plan=alternate)
                    check(ref, read(control, x.shape), mode="chunked_vs_unchunked", chunk_size=32, length=n)
                    check(read(out, x.shape), read(control, x.shape), mode="chunk_partition_parity", length=n)
                    del control, alternate
                print("BOUNDARY", n, rows[-2:], flush=True)
                del out, outd, tx, td, plan, pos
            del state
        if args.mode in ("resume", "all"):
            x = torch.randn(1, 529, 2048, dtype=torch.bfloat16) * 0.05
            baseline = hf_forward(hf, x, DynamicCache(config=c), 0)
            for start in (32, 33, 256, 512, 528):
                count = min(17, 529 - start)
                state = new()
                prefix, tail = up(x[:, :start]), up(x[:, start : start + count])
                pp = dec.prepare_prefill(seq_len=start)
                rp = dec.prepare_prefill(seq_len=count, start_pos=start)
                with device_only():
                    prefout = dec.prefill_forward(prefix, state=state, plan=pp)
                    out = dec.prefill_forward(tail, state=state, plan=rp)
                got = read(out, (1, count, 2048))
                expected = baseline[:, start : start + count]
                contribution = x[:, start : start + count]
                perrow = [
                    pcc(expected[:, i] - contribution[:, i], got[:, i] - contribution[:, i]) for i in range(count)
                ]
                assert min(perrow) >= 0.995, (start, perrow)
                reset(state)
                with device_only():
                    bad = dec.prefill_forward(tail, state=state, plan=rp)
                bad = read(bad, got.shape)
                negative = [
                    pcc(expected[:, i] - contribution[:, i], bad[:, i] - contribution[:, i]) for i in range(count)
                ]
                assert min(negative) < 0.995, ("negative control insensitive", start, negative)
                rows.append(
                    dict(
                        mode="resume",
                        start=start,
                        count=count,
                        per_row_pcc=perrow,
                        dropped_history_per_row_pcc=negative,
                    )
                )
                print("RESUME", start, min(perrow), min(negative), flush=True)
                del state, prefix, tail, prefout, out, pp, rp
        if args.mode in ("batch", "all"):
            states, xs, xds, plans, positions, refs, refds = [], [], [], [], [], [], []
            pool = None
            pool_pages = torch.randperm(args.batch * 32 + 3)
            cached_refs = []
            for i in range(args.batch):
                n = (1, 31, 32, 33)[i % 4]
                start = (0, 1, 32, 33)[i % 4]
                x = torch.randn(1, start + n, 2048, dtype=torch.bfloat16) * 0.05
                xd = torch.randn(1, 1, 2048, dtype=torch.bfloat16) * 0.05
                hc = DynamicCache(config=c)
                refs.append(hf_forward(hf, x, hc, 0)[:, start:])
                refds.append(hf_forward(hf, xd, hc, start + n))
                xs.append(up(x[:, start:]))
                xds.append(up(xd))
                if dec.kind == "full_attention":
                    state = dec.new_kv_state(
                        context_length=1024,
                        num_physical_pages=args.batch * 32 + 3,
                        page_order=pool_pages[i * 32 : (i + 1) * 32],
                        kv_cache=pool,
                    )
                    pool = (state["k"], state["v"])
                    states.append(state)
                    cached_refs.append((hc.layers[args.layer].keys.clone(), hc.layers[args.layer].values.clone()))
                else:
                    states.append(new())
                if start:
                    prefix = up(x[:, :start])
                    prefix_plan = dec.prepare_prefill(seq_len=start)
                    with device_only():
                        prefix_out = dec.prefill_forward(prefix, state=states[-1], plan=prefix_plan)
                    del prefix, prefix_plan, prefix_out
                plans.append(dec.prepare_prefill(seq_len=n, start_pos=start))
                positions.append(
                    dec.prepare_positions(start_pos=start + n, seq_len=1) if dec.kind == "full_attention" else None
                )
            with device_only():
                outs = dec.prefill_batch(xs, states=states, plans=plans)
            backups = {}
            for state in states:
                for key in ("conv", "recurrent") if dec.kind == "linear_attention" else ("k", "v"):
                    tensor = state[key]
                    if id(tensor) not in backups:
                        backups[id(tensor)] = (tensor, ttnn.clone(tensor))
            with device_only():
                outds = dec.decode_batch(xds, states=states, positions=positions)
            for i, (ref, refd, out, outd) in enumerate(zip(refs, refds, outs, outds)):
                check(
                    ref,
                    read(out, ref.shape),
                    mode="batch_prefill",
                    batch=args.batch,
                    row=i,
                    start=(0, 1, 32, 33)[i % 4],
                    length=(1, 31, 32, 33)[i % 4],
                )
                check(refd, read(outd, refd.shape), mode="batch_decode", batch=args.batch, row=i)
            if pool is not None:
                host_k, host_v = (ttnn.to_torch(t) for t in pool)
                guards = pool_pages[-3:]
                assert torch.count_nonzero(host_k[guards]) == 0 and torch.count_nonzero(host_v[guards]) == 0
                for i, (key_ref, value_ref) in enumerate(cached_refs):
                    owned = pool_pages[i * 32 : (i + 1) * 32]
                    for label, reference, host in (("key", key_ref, host_k), ("value", value_ref, host_v)):
                        logical = host[owned].permute(1, 0, 2, 3).reshape(1, 2, 1024, 256)[:, :, : reference.shape[2]]
                        check(reference, logical, mode="shared_pool_cache", row=i, tensor=label)
                rows.append(dict(mode="unowned_guard_pages", pages=guards.tolist(), untouched=True))

            def restore_batch():
                for destination, saved in backups.values():
                    ttnn.copy(saved, destination)

            restore_batch()
            ttnn.synchronize_device(mesh)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                traced_outputs = dec.decode_batch(xds, states=states, positions=positions)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            from ttnn.tools import trace_allocation_tracker

            for output in traced_outputs:
                trace_allocation_tracker.acknowledge_corruptible(output)
            first = None
            for repeat in range(3):
                restore_batch()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                got = [read(output, reference.shape).clone() for output, reference in zip(traced_outputs, refds)]
                for i, (reference, actual) in enumerate(zip(refds, got)):
                    check(reference, actual, mode="batch_traced_decode", batch=args.batch, row=i, repeat=repeat)
                if first is None:
                    first = got
                else:
                    assert all(torch.equal(a, b) for a, b in zip(first, got))
            ttnn.release_trace(mesh, trace)
            rows.append(dict(mode="batch_trace_determinism", batch=args.batch, repeats=3, bit_equal=True))
            print("BATCH", args.batch, "PASS", flush=True)
        Path(args.output).write_text(
            json.dumps(
                {
                    "layer": args.layer,
                    "kind": dec.kind,
                    "synthetic": args.synthetic,
                    "runtime_fallback_audit": "pass",
                    "results": rows,
                },
                indent=2,
            )
            + "\n"
        )
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--mode", choices=["all", "boundaries", "resume", "batch"], default="all")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--output", required=True)
    run(p.parse_args())
