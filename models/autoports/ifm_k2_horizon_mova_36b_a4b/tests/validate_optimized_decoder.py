# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Same-instance boundary, continuation, trace reuse and determinism validation."""

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import REVISION, ROOT, pcc, reference


@contextmanager
def device_only():
    def forbidden(*a, **kw):
        raise AssertionError("Host conversion inside decoder forward")

    with (
        patch.object(ttnn, "from_torch", forbidden),
        patch.object(ttnn, "to_torch", forbidden),
        patch.object(ttnn, "as_tensor", forbidden),
    ):
        yield


def validate(
    layer=0,
    batch=1,
    lengths=(1, 31, 32, 33, 255, 256, 257, 529, 7),
    synthetic=False,
    page_size=32,
    recorded=True,
    tag=None,
    cache_dtype=None,
):
    torch.set_num_threads(8)
    torch.manual_seed(112)
    c, hf, rope, sd = reference(layer, synthetic=synthetic)
    continuation_length = 34 if page_size == 32 else page_size + 3
    context = max(max(lengths), 67, 31 + continuation_length) + 64
    pages_per = (context + page_size - 1) // page_size
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    results = []
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh):
            return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        def copy(v, to, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            ttnn.copy_host_to_device_tensor(tt(v, dtype, layout, device=None), to)

        model = OptimizedDecoder.from_state_dict(sd, hf_config=c, layer_idx=layer, mesh_device=mesh)
        kv = [
            tt(
                torch.zeros(batch * pages_per, 8, page_size, 128, dtype=torch.bfloat16),
                getattr(ttnn, cache_dtype) if cache_dtype else model.kv_cache_dtype,
            )
            for _ in range(2)
        ]
        page_table = tt(torch.zeros(batch, pages_per, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dx = tt(torch.zeros(1, 1, batch, 2560, dtype=torch.bfloat16))
        dc = tt(torch.zeros(1, batch, 1, 128, dtype=torch.bfloat16))
        ds = tt(torch.zeros(1, batch, 1, 128, dtype=torch.bfloat16))
        pos = tt(torch.zeros(batch, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        decode_args = dict(cos=dc, sin=ds, page_table=page_table, current_pos=pos, kv_cache=kv)
        cases = []
        # The last case explicitly extends a non-page-aligned cached prefix.
        for case_idx, (length, prefix, random_positions) in enumerate(
            [(n, 0, False) for n in lengths] + [(continuation_length, 31, False), (67, 0, True)]
        ):
            total = length + prefix
            if recorded and not synthetic:
                from .optimized_activations import recorded_inputs

                x = recorded_inputs(layer, batch, total + 1, case_idx % 4)
            else:
                x = (torch.randn(batch, total + 1, 2560) * 0.1).bfloat16()
            pt = torch.randperm(batch * pages_per).reshape(batch, pages_per).int()
            cache = DynamicCache(config=c)
            with torch.no_grad():
                cs = rope(x, torch.arange(total).unsqueeze(0).expand(batch, -1))
                mask = torch.full((total, total), float("-inf")).triu(1)[None, None]
                golden = hf(x[:, :total], position_embeddings=cs, attention_mask=mask, past_key_values=cache)
                positions = torch.full((batch,), total, dtype=torch.int32)
                if random_positions:
                    positions = total - 1 - (torch.arange(batch, dtype=torch.int32) * 17 + 5) % min(total, 64)
                dcs = rope(x, positions[:, None])
                if random_positions:
                    expected_rows = []
                    for user in range(batch):
                        user_cache = DynamicCache(config=c)
                        end = int(positions[user])
                        user_cache.update(
                            cache.layers[layer].keys[user : user + 1, :, :end].clone(),
                            cache.layers[layer].values[user : user + 1, :, :end].clone(),
                            layer,
                        )
                        expected_rows.append(
                            hf(
                                x[user : user + 1, total:],
                                position_embeddings=(dcs[0][user : user + 1], dcs[1][user : user + 1]),
                                past_key_values=user_cache,
                            )
                        )
                    expected = torch.cat(expected_rows, dim=0)
                else:
                    expected = hf(x[:, total:], position_embeddings=dcs, past_key_values=cache)

            def part(start, end):
                return (tt(x[:, start:end].unsqueeze(0)), *[tt(v[:, start:end].unsqueeze(1)) for v in cs])

            cases.append(
                dict(
                    length=length,
                    prefix=prefix,
                    total=total,
                    pt=pt,
                    parts=([part(0, prefix)] if prefix else []) + [part(prefix, total)],
                    dx=x[:, total:].reshape(1, 1, batch, 2560),
                    dc=dcs[0].reshape(1, batch, 1, 128),
                    ds=dcs[1].reshape(1, batch, 1, 128),
                    golden=golden[:, prefix:],
                    expected=expected,
                    positions=positions,
                    random_positions=random_positions,
                )
            )

        def prefill(case):
            copy(case["pt"], page_table, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
            for i, (tx, cos, sin) in enumerate(case["parts"]):
                start = case["prefix"] if i else 0
                with device_only():
                    out = model.prefill_forward(
                        tx, cos=cos, sin=sin, page_table=page_table, kv_cache=kv, start_pos=start
                    )
                if i < len(case["parts"]) - 1:
                    out.deallocate(True)
            return out

        def refresh(case):
            copy(case["dx"], dx)
            copy(case["dc"], dc)
            copy(case["ds"], ds)
            copy(case["positions"], pos, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)

        # Every eager path is compiled before the first capture; no new persistent
        # tensor or program-cache entry may be allocated between trace replays.
        for case in cases:
            before = [cpu(cache) for cache in kv]
            out = prefill(case)
            score = pcc(case["golden"], cpu(out).reshape(batch, case["length"], 2560))
            print("WARM_PREFILL", case["length"], case["prefix"], score, flush=True)
            assert score >= 0.995
            out.deallocate(True)
            allowed = torch.zeros(batch * pages_per, page_size, dtype=torch.bool)
            tokens = torch.arange(case["total"])
            for user in range(batch):
                allowed[case["pt"][user, tokens // page_size].long(), tokens % page_size] = True
            for previous, cache in zip(before, kv):
                changed = (cpu(cache) != previous).any(dim=3).any(dim=1)
                assert not torch.any(changed & ~allowed), "Prefill overwrote an unrelated cache row"
            refresh(case)
            before = [cpu(cache) for cache in kv]
            with device_only():
                out = model.decode_forward(dx, **decode_args)
            assert pcc(case["expected"], cpu(out).reshape(batch, 1, 2560)) >= 0.995
            out.deallocate(True)
            allowed.zero_()
            for user, position in enumerate(case["positions"]):
                allowed[case["pt"][user, position // page_size], position % page_size] = True
            for previous, cache in zip(before, kv):
                changed = (cpu(cache) != previous).any(dim=3).any(dim=1)
                assert not torch.any(changed & ~allowed), "Decode overwrote an unrelated cache row"
        out = prefill(cases[0])
        out.deallocate(True)
        refresh(cases[0])
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            traced = model.decode_forward(dx, **decode_args)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        # Repeat an earlier request after all changed shapes/tokens/page ownership.
        for case in cases + [cases[0]]:
            out = prefill(case)
            actual = cpu(out).reshape(batch, case["length"], 2560)
            pre = pcc(case["golden"], actual)
            out.deallocate(True)
            refresh(case)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            actual_d = cpu(traced).reshape(batch, 1, 2560)
            score = pcc(case["expected"], actual_d)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            repeated = cpu(traced).reshape(batch, 1, 2560)
            assert torch.equal(actual_d, repeated), "Repeated decode input not deterministic"
            # Prefill exact repeat, then replay the same trace: allocations must
            # have been freed and the trace must survive the eager work.
            out = prefill(case)
            repeat_prefill = cpu(out).reshape(batch, case["length"], 2560)
            if not torch.equal(actual, repeat_prefill):
                mismatch = actual != repeat_prefill
                print(
                    "REPEAT_MISMATCH",
                    case["length"],
                    case["prefix"],
                    "count",
                    mismatch.sum().item(),
                    "max",
                    (actual.float() - repeat_prefill.float()).abs().max().item(),
                    "indices",
                    mismatch.nonzero()[:40].tolist(),
                    flush=True,
                )
            assert torch.equal(actual, repeat_prefill), "Repeated prefill input not deterministic"
            out.deallocate(True)
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            assert torch.equal(actual_d, cpu(traced).reshape(batch, 1, 2560))
            assert pre >= 0.995 and score >= 0.995
            row = dict(
                layer=layer,
                batch=batch,
                length=case["length"],
                prefix=case["prefix"],
                prefill_pcc=pre,
                traced_decode_pcc=score,
                trace_id=str(tid),
                deterministic=True,
                positions=case["positions"].tolist(),
                page_size=page_size,
            )
            results.append(row)
            print("CASE_PASS", json.dumps(row), flush=True)
        stress_replays = 0
        stress_start = time.monotonic()
        if os.getenv("TT_METAL_WATCHER"):
            # Span a live Watcher=10 poll while repeatedly executing optimized kernels.
            while time.monotonic() - stress_start < 12:
                for _ in range(1000):
                    ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
                    stress_replays += 1
                ttnn.synchronize_device(mesh)
            assert torch.equal(actual_d, cpu(traced).reshape(batch, 1, 2560))
        stress_seconds = time.monotonic() - stress_start
        ttnn.release_trace(mesh, tid)
        report = dict(
            stress_replays=stress_replays,
            stress_seconds=stress_seconds,
            revision=REVISION,
            decoder_class=type(model).__module__ + "." + type(model).__name__,
            decoder_sha256=hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest(),
            activation_source="checkpoint-HF" if recorded and not synthetic else "random",
            cache_dtype=str(kv[0].dtype),
            chunk_size=model.chunk_size,
            watcher=os.getenv("TT_METAL_WATCHER"),
            argv=sys.argv,
            weight_source="synthetic_from_real_stats" if synthetic else "real",
            trace_alloc_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            program_cache_tracking=not os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") == "1",
            captures=1,
            request_time_releases=0,
            teardown_releases=1,
            cache_ownership_and_padding_unchanged=True,
            rows=results,
        )
        outpath = (
            ROOT
            / f"doc/optimized_decoder/validation_layer{layer}_b{batch}_page{page_size}{'_synthetic' if synthetic else ''}{'_'+tag if tag else ''}.json"
        )
        outpath.write_text(json.dumps(report, indent=2))
        print("VALIDATION_PASS", outpath, flush=True)
        return report
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lengths", type=int, nargs="+", default=[1, 31, 32, 33, 255, 256, 257, 529, 7])
    ap.add_argument("--random-inputs", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--page-size", type=int, default=32)
    ap.add_argument("--tag")
    ap.add_argument("--cache-dtype", choices=["bfloat8_b", "bfloat16"])
    args = ap.parse_args()
    validate(
        args.layer,
        args.batch,
        args.lengths,
        args.synthetic,
        args.page_size,
        not args.random_inputs,
        args.tag,
        args.cache_dtype,
    )
