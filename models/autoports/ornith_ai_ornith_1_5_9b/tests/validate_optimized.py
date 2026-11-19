# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Real-shape acceptance battery. Run as a module under an external timeout."""

import argparse
import contextlib
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.ornith_ai_ornith_1_5_9b.tests.run_decoder import MODEL, REVISION, ROOT, config, pcc, weights
from models.autoports.ornith_ai_ornith_1_5_9b.tt.optimized_decoder import OptimizedDecoder


class NoTorch(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        raise AssertionError(f"Runtime torch fallback: {func}")


@contextlib.contextmanager
def device_only():
    def forbidden(*args, **kwargs):
        raise AssertionError("Runtime host transfer")

    with (
        NoTorch(),
        patch.object(ttnn, "from_torch", forbidden),
        patch.object(ttnn, "to_torch", forbidden),
        patch.object(ttnn, "copy_host_to_device_tensor", forbidden),
    ):
        yield


def upload(x, mesh, integer=False):
    return ttnn.from_torch(
        x.contiguous(),
        device=mesh,
        dtype=ttnn.int32 if integer else ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT if integer else ttnn.TILE_LAYOUT,
    )


def buffers(state):
    return [
        getattr(state, f.name)
        for f in fields(state)
        if not f.metadata.get("host_metadata") and getattr(state, f.name) is not None
    ]


def clear(state):
    for x in buffers(state):
        if x.layout == ttnn.ROW_MAJOR_LAYOUT:
            # Binary RM kernels cannot write into a preallocated output; copy the
            # temporary zero into the same persistent state allocation instead.
            ttnn.copy(ttnn.multiply(x, 0), x)
        else:
            ttnn.multiply(x, 0, output_tensor=x)


def restore(snapshots):
    for dst, src in snapshots:
        ttnn.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--diagnostic-pcc", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    ap.add_argument("--traced-prefill", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=1024)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--lengths", default="1,31,32,33,1023,1024,1025,2051,17")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--positive-control", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(23)
    lengths = list(map(int, args.lengths.split(",")))
    c = config()
    c._attn_implementation = "sdpa"
    sd = weights(args.layer, synthetic=args.synthetic)
    ref = Qwen3_5DecoderLayer(c, args.layer).bfloat16().eval()
    ref.load_state_dict(sd)
    rope = Qwen3_5TextRotaryEmbedding(c)
    context = ((max(max(lengths), 36) + 32) // 32) * 32
    full = c.layer_types[args.layer] == "full_attention"
    records = []

    def reference(x, start=0, cache=None):
        cache = cache if cache is not None else DynamicCache(config=c)
        n = x.shape[1]
        pos = torch.arange(start, start + n)[None, :]
        mask = None
        if full:
            mask = torch.where(torch.arange(start + n)[None, :] <= pos.T, 0.0, float("-inf"))[None, None]
        with torch.no_grad():
            y = ref(x, position_embeddings=rope(x, pos), attention_mask=mask, past_key_values=cache)
        return y, cache

    def compare(label, expected, actual, residual=None, per_row=False):
        actual = actual.reshape_as(expected)
        score = pcc(expected, actual)
        r = {"label": label, "pcc": score}
        if residual is not None:
            r["contribution_pcc"] = pcc(expected - residual, actual - residual)
            assert args.diagnostic_pcc or r["contribution_pcc"] >= 0.995, r
        if per_row:
            r["min_row_contribution_pcc"] = min(
                pcc(a, b) for a, b in zip((expected - residual)[0], (actual - residual)[0])
            )
            assert args.diagnostic_pcc or r["min_row_contribution_pcc"] >= 0.995, r
        assert args.diagnostic_pcc or score >= 0.995, r
        records.append(r)
        print(json.dumps(r), flush=True)

    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    trace = None
    try:
        decoder_cls = OptimizedDecoder
        if args.baseline:
            from models.autoports.ornith_ai_ornith_1_5_9b.tt.fused_decoder import FusedDecoder

            decoder_cls = FusedDecoder
        layer = decoder_cls.from_state_dict(
            sd, hf_config=c, layer_idx=args.layer, mesh_device=mesh, chunk_size=args.chunk_size
        )
        state = layer.allocate_state(context=context)
        table = upload(torch.arange(context // 32, dtype=torch.int32)[None], mesh, True)
        pos = upload(torch.tensor([0], dtype=torch.int32), mesh, True)
        dec = upload(torch.zeros(1, 1, 1, 4096, dtype=torch.bfloat16), mesh)
        cases = []
        cache_references = []
        prefill_results = []
        for index, n in enumerate(lengths):
            x = (torch.randn(1, n, 4096) * 0.1).bfloat16()
            d = (torch.randn(1, 1, 4096) * 0.1).bfloat16()
            y, hfcache = reference(x)
            yd, _ = reference(d, start=n, cache=hfcache)
            perm = torch.randperm(context // 32, dtype=torch.int32)[None]
            cases.append((x, d, y, yd, perm))
            cache_references.append(hfcache.layers[args.layer])
        # Prepare every tested prefill path before capture. Inputs persist through replay.
        prepared = []
        for x, d, y, yd, perm in cases:
            tx = upload(x[None], mesh)
            hp = ttnn.from_torch(perm, dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT)
            hd = ttnn.from_torch(d[None], dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            hn = ttnn.from_torch(
                torch.tensor([x.shape[1]], dtype=torch.int32), dtype=ttnn.int32, layout=ttnn.ROW_MAJOR_LAYOUT
            )
            clear(state)
            ttnn.copy_host_to_device_tensor(hp, table)
            with device_only():
                o = layer.prefill_forward(tx, state=state, page_table=table)
            prefill_results.append(ttnn.to_torch(o))
            compare(f"prefill_{x.shape[1]}", y, prefill_results[-1], x)
            del o
            ttnn.copy_host_to_device_tensor(hd, dec)
            ttnn.copy_host_to_device_tensor(hn, pos)
            with device_only():
                o = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
            compare(f"eager_decode_{x.shape[1]}", yd, ttnn.to_torch(o), d)
            del o
            prepared.append((tx, hp, hd, hn))
        # Warm state snapshot/copy paths before trace capture.
        snapshots = [(x, ttnn.clone(x)) for x in buffers(state)]
        restore(snapshots)
        ttnn.synchronize_device(mesh)
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = layer.decode_forward(dec, state=state, page_table=table, current_position=pos)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        from ttnn.tools import trace_allocation_tracker

        trace_allocation_tracker.acknowledge_corruptible(out)
        if args.positive_control:
            bad = ttnn.empty((32, 32), device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
            try:
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            except RuntimeError as exc:
                assert "alloc" in str(exc).lower()
                print("POSITIVE_CONTROL_RAISED", str(exc)[:1000], flush=True)
                records.append({"positive_control": "raised"})
            else:
                raise AssertionError("Allocation tracker failed positive control")
            del bad
        else:
            for case_index, ((x, d, y, yd, perm), (tx, hp, hd, hn)) in enumerate(zip(cases, prepared)):
                clear(state)
                ttnn.copy_host_to_device_tensor(hp, table)
                ttnn.copy_host_to_device_tensor(hd, dec)
                ttnn.copy_host_to_device_tensor(hn, pos)
                with device_only():
                    o = layer.prefill_forward(tx, state=state, page_table=table)
                got_prefill = ttnn.to_torch(o)
                assert torch.equal(prefill_results[case_index], got_prefill)
                compare(f"prefill_after_capture_{x.shape[1]}", y, got_prefill, x)
                del o
                for dst, src in snapshots:
                    ttnn.copy(dst, src)
                previous = None
                for _ in range(3):
                    restore(snapshots)
                    with device_only():
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                    actual = ttnn.to_torch(out)
                    if previous is not None:
                        assert torch.equal(previous, actual)
                    previous = actual
                compare(f"traced_decode_{x.shape[1]}", yd, actual, d)
                if full:
                    hf_state = cache_references[case_index]
                    for name, tt_cache, hf_cache in (
                        ("key", state.key, hf_state.keys),
                        ("value", state.value, hf_state.values),
                    ):
                        paged = ttnn.to_torch(tt_cache)
                        logical = (
                            paged[perm[0].long()].permute(1, 0, 2, 3).reshape(1, 4, -1, 256)[:, :, : x.shape[1] + 1]
                        )
                        compare(f"physical_cache_{name}_{x.shape[1]}", hf_cache, logical)
            records.append(
                {
                    "trace_id": int(trace),
                    "captures": 1,
                    "request_releases": 0,
                    "replays": 3 * len(cases),
                    "deterministic": True,
                    "runtime_fallback_audit": "clean",
                }
            )
        if args.benchmark:
            import statistics

            tx = prepared[-1][0]
            samples = []
            for _ in range(7):
                clear(state)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                with device_only():
                    measured = layer.prefill_forward(tx, state=state, page_table=table)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - begin) * 1000)
                del measured
            prefill_samples = samples
            if args.traced_prefill:
                # This fixed-signature optional caller trace is separate from the
                # normal warmed prefill metric, and never changes the decoder API.
                prefill_trace = ttnn.begin_trace_capture(mesh, cq_id=0)
                with device_only():
                    prefill_out = layer.prefill_forward(tx, state=state, page_table=table)
                ttnn.end_trace_capture(mesh, prefill_trace, cq_id=0)
                trace_allocation_tracker.acknowledge_corruptible(prefill_out)
                traced_samples = []
                for _ in range(7):
                    clear(state)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    ttnn.execute_trace(mesh, prefill_trace, cq_id=0, blocking=True)
                    traced_samples.append((time.perf_counter() - begin) * 1000)
                compare("traced_prefill", cases[-1][2], ttnn.to_torch(prefill_out), cases[-1][0])
                records.append(
                    {
                        "traced_prefill_ms": statistics.median(traced_samples),
                        "samples_ms": traced_samples,
                        "length": lengths[-1],
                        "trace_scope": "caller-owned fixed input signature; state reset outside timing",
                    }
                )
                ttnn.release_trace(mesh, prefill_trace)
                del prefill_out
            samples = []
            for _ in range(7):
                restore(snapshots)
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(50):
                    ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                ttnn.synchronize_device(mesh)
                samples.append((time.perf_counter() - begin) * 1000 / 50)
            records.append(
                {
                    "benchmark": True,
                    "prefill_ms": statistics.median(prefill_samples),
                    "decode_ms": statistics.median(samples),
                    "prefill_samples_ms": prefill_samples,
                    "decode_samples_ms": samples,
                    "decode_replays_per_sample": 50,
                    "regime": "serial_chip0",
                    "hardware": {"chip": 0, "dram_banks": mesh.dram_grid_size().x, "regime": "serial"},
                    "implementation": decoder_cls.__name__,
                }
            )
        if args.profile:
            from tracy import signpost

            tx = prepared[-1][0]
            clear(state)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL")
            with device_only():
                prof = layer.prefill_forward(tx, state=state, page_table=table)
            ttnn.synchronize_device(mesh)
            signpost("PERF_PREFILL_END")
            del prof
            restore(snapshots)
            ttnn.synchronize_device(mesh)
            signpost("PERF_DECODE")
            begin = time.perf_counter()
            with device_only():
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
            ttnn.synchronize_device(mesh)
            elapsed = time.perf_counter() - begin
            signpost("PERF_DECODE_END")
            records.append({"traced_host_wall_ms": elapsed * 1000, "regime": "serial_chip0"})
        ttnn.release_trace(mesh, trace)
        trace = None
        # Feasible unchunked control shares target config and real weights.
        if args.resume:
            idx = lengths.index(max(lengths))
            clear(state)
            ttnn.copy_host_to_device_tensor(prepared[idx][1], table)
            layer.chunk_size = ((max(lengths) + 31) // 32) * 32
            with device_only():
                control = layer.prefill_forward(prepared[idx][0], state=state, page_table=table)
            compare("unchunked_control", cases[idx][2], ttnn.to_torch(control), cases[idx][0])
            compare("chunked_vs_unchunked", prefill_results[idx], ttnn.to_torch(control))
            layer.chunk_size = 1024
        # Continuation and batched tests follow the single-row trace lifetime.
        if args.resume:
            n = max(lengths)
            x = cases[lengths.index(n)][0]
            y = cases[lengths.index(n)][2]
            for start in [32, 33, 1024, 1025, n - 1]:
                if start >= n:
                    continue
                count = min(32, n - start)
                clear(state)
                prefix = upload(x[:, :start][None], mesh)
                segment = upload(x[:, start : start + count][None], mesh)
                with device_only():
                    layer.prefill_forward(prefix, state=state, page_table=table)
                    resumed = layer.prefill_forward(segment, state=state, start_position=start, page_table=table)
                observed = ttnn.to_torch(resumed)
                expected = y[:, start : start + count]
                residual = x[:, start : start + count]
                compare(f"resume_{start}", expected, observed, residual, True)
                clear(state)
                with device_only():
                    broken = layer.prefill_forward(segment, state=state, start_position=start, page_table=table)
                bad = ttnn.to_torch(broken).reshape_as(expected)
                scores = [pcc(a, b) for a, b in zip((expected - residual)[0], (bad - residual)[0])]
                assert min(scores) < 0.995, (start, scores)
                records.append(
                    {
                        "negative_control_start": start,
                        "min_row_contribution_pcc": min(scores),
                        "failed_gate_as_expected": True,
                    }
                )
        if args.batch > 1:
            states = [layer.allocate_state(context=context) for _ in range(args.batch)]
            starts = [i % 3 for i in range(args.batch)]
            lens = [31 + i % 4 for i in range(args.batch)]
            rows = []
            tables = []
            refs = []
            decrows = []
            decpos = []
            decrefs = []
            for i, (st, start, n) in enumerate(zip(states, starts, lens)):
                x = (torch.randn(1, start + n + 1, 4096) * 0.1).bfloat16()
                y, cache = reference(x[:, : start + n])
                yd, _ = reference(x[:, start + n :], start=start + n, cache=cache)
                pt = upload(torch.randperm(context // 32, dtype=torch.int32)[None], mesh, True)
                if start:
                    layer.prefill_forward(upload(x[:, :start][None], mesh), state=st, page_table=pt)
                rows.append(upload(x[:, start : start + n][None], mesh))
                tables.append(pt)
                refs.append(y[:, start:])
                decrows.append(upload(x[:, start + n :][None], mesh))
                decpos.append(upload(torch.tensor([start + n], dtype=torch.int32), mesh, True))
                decrefs.append(yd)
            with device_only():
                outs = layer.prefill_batch(rows, states=states, start_positions=starts, page_tables=tables)
            batch_snapshots = [(b, ttnn.clone(b)) for st in states for b in buffers(st)]
            with device_only():
                ds = layer.decode_batch(decrows, states=states, current_positions=decpos, page_tables=tables)
            restore(batch_snapshots)
            del ds
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            with device_only():
                ds = layer.decode_batch(decrows, states=states, current_positions=decpos, page_tables=tables)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for od in ds:
                trace_allocation_tracker.acknowledge_corruptible(od)
            previous = None
            for _ in range(3):
                restore(batch_snapshots)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                actual = [ttnn.to_torch(od) for od in ds]
                if previous is not None:
                    assert all(torch.equal(a, b) for a, b in zip(previous, actual))
                previous = actual
            records.append({"batch_traced": args.batch, "replays": 3, "deterministic": True})
            if args.benchmark:
                samples = []
                for _ in range(7):
                    restore(batch_snapshots)
                    ttnn.synchronize_device(mesh)
                    begin = time.perf_counter()
                    for _ in range(50):
                        ttnn.execute_trace(mesh, trace, cq_id=0, blocking=False)
                    ttnn.synchronize_device(mesh)
                    samples.append((time.perf_counter() - begin) * 1000 / 50)
                records.append(
                    {
                        "batch_benchmark": True,
                        "batch": args.batch,
                        "decode_batch_ms": statistics.median(samples),
                        "decode_samples_ms": samples,
                        "regime": "serial_chip0",
                        "logical_prefix_lengths": lens,
                        "independent_states": True,
                        "decode_replays_per_sample": 50,
                    }
                )
            # Timing groups advance recurrent state repeatedly at a fixed position.
            # Restore the one-step correctness fixture before reading batch outputs.
            if args.benchmark:
                restore(batch_snapshots)
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            ttnn.release_trace(mesh, trace)
            trace = None
            for i, (expected, o, ed, od) in enumerate(zip(refs, outs, decrefs, ds)):
                compare(f"batch_prefill_row_{i}", expected, ttnn.to_torch(o))
                compare(f"batch_decode_row_{i}", ed, ttnn.to_torch(od))
        report = {
            "hardware": {"chip": 0, "dram_banks": mesh.dram_grid_size().x, "regime": "serial"},
            "implementation": decoder_cls.__name__,
            "model": MODEL,
            "revision": REVISION,
            "layer": args.layer,
            "synthetic_weights": args.synthetic,
            "diagnostic_only": args.diagnostic_pcc,
            "kind": c.layer_types[args.layer],
            "context_allocated": context,
            "chunk_size": args.chunk_size,
            "lengths": lengths,
            "batch": args.batch,
            "records": records,
            "implementation_sha256": hashlib.sha256(
                (ROOT / ("tt/fused_decoder.py" if args.baseline else "tt/optimized_decoder.py")).read_bytes()
            ).hexdigest(),
        }
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
        print("DIAGNOSTIC_COMPLETE" if args.diagnostic_pcc else "VALIDATION_PASS", args.output, flush=True)
    finally:
        if trace is not None:
            ttnn.release_trace(mesh, trace)
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
