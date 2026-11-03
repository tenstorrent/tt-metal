# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Full-context public prefill and last-position traced decode.

The layer-only input repeats 256 distinct activation rows. This permits exact
HF KV construction from 256 projection/router evaluations and full-position
RoPE, avoiding a quadratic CPU full-layer oracle. The TT decoder still executes
every token through every component. Selected output rows use the unmodified HF
decoder with a full, independently constructed prefix cache (no TT cache reads).
"""

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.multichip_decoder import MultichipDecoder
from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import REVISION, ROOT, pcc, reference
from .multichip_test_utils import device_only


def validate(layer, lengths, baseline=False, options=None, artifact_dir=None):
    options = options or {}
    tt_golden_path = ROOT / f"doc/multichip_decoder/local/full_prefill_baseline_l{layer}.pt"
    tt_golden = {} if baseline else torch.load(tt_golden_path, weights_only=True)
    torch.set_num_threads(16)
    torch.manual_seed(812)
    c, hf, rope, sd = reference(layer)
    maximum = max(lengths)
    from .multichip_activations import recorded_inputs

    pattern = recorded_inputs(layer, 1, 256)
    x = pattern.repeat(1, (maximum + 255) // 256, 1)[:, :maximum].contiguous()
    mod = importlib.import_module(type(hf).__module__)
    with torch.no_grad():
        norm = hf.input_layernorm(pattern)
        kbase = hf.self_attn.k_proj(norm).reshape(1, 256, 8, 128).transpose(1, 2)
        vcache = DynamicCache(config=c)
        hf.self_attn(
            norm,
            position_embeddings=rope(pattern, torch.arange(256)[None]),
            attention_mask=None,
            past_key_values=vcache,
        )
        vbase = vcache.layers[layer].values
        cs = rope(x, torch.arange(maximum)[None])
        keys = kbase.repeat(1, 1, (maximum + 255) // 256, 1)[:, :, :maximum]
        keys = keys * cs[0].unsqueeze(1) + mod.rotate_half(keys) * cs[1].unsqueeze(1)
        values = vbase.repeat(1, 1, (maximum + 255) // 256, 1)[:, :, :maximum].contiguous()

    def oracle(start, end):
        cache = DynamicCache(config=c)
        if start:
            cache.update(keys[:, :, :start].clone(), values[:, :, :start].clone(), layer)
        with torch.no_grad():
            mask = torch.arange(end)[None, :] > torch.arange(start, end)[:, None]
            mask = torch.zeros(end - start, end).masked_fill(mask, float("-inf"))[None, None]
            return hf(
                x[:, start:end],
                position_embeddings=(cs[0][:, start:end], cs[1][:, start:end]),
                attention_mask=mask,
                past_key_values=cache,
            )

    gold = {}
    for n in lengths:
        spans = [(0, 32), (max(0, n // 2 - 16) // 32 * 32, max(0, n // 2 - 16) // 32 * 32 + 32), (n - 32, n)]
        gold[n] = [(a, b, oracle(a, b)) for a, b in spans]
        print("HF_ORACLE_READY", n, flush=True)
    if not baseline:
        ttnn.set_fabric_config(ttnn.FabricConfig.FABRIC_1D_RING)
    mesh = ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1 if baseline else 4),
        physical_device_ids=[0] if baseline else [],
        trace_region_size=32 * 1024 * 1024,
    )
    rows = []
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh, shard=None):
            mapper = ttnn.ReplicateTensorToMesh(mesh) if shard is None else ttnn.ShardTensorToMesh(mesh, dim=shard)
            return ttnn.from_torch(v.contiguous(), dtype=dtype, layout=layout, device=device, mesh_mapper=mapper)

        def cpu(v):
            result = ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            return result if baseline else result[:1]

        model = (OptimizedDecoder if baseline else MultichipDecoder).from_state_dict(
            sd, hf_config=c, layer_idx=layer, mesh_device=mesh, options=options
        )
        ttnn.CONFIG.throw_exception_on_fallback = True
        original_chunk = model._prefill_chunk

        def progress_chunk(*args, **kwargs):
            result = original_chunk(*args, **kwargs)
            end = kwargs["start_pos"] + args[0].shape[2]
            if end % 32768 == 0:
                print("PREFILL_PROGRESS", end, "time", time.time(), flush=True)
            return result

        # Test-only progress logging; no tensor reads or changes to execution.
        model._prefill_chunk = progress_chunk
        tx = tt(x.unsqueeze(0))
        cos, sin = [tt(v.unsqueeze(1)) for v in cs]
        blocks = (maximum + 31) // 32
        pages = torch.randperm(blocks).reshape(1, blocks).int()
        page_table = tt(pages, ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kv = [
            tt(
                torch.zeros(blocks, 8, 32, 128, dtype=torch.bfloat16),
                model.kv_cache_dtype,
                shard=None if baseline else 1,
            )
            for _ in range(2)
        ]
        dx = tt(x[:, -1:].reshape(1, 1, 1, 2560))
        dc, ds = [tt(v[:, -1:].reshape(1, 1, 1, 128)) for v in cs]
        pos = tt(torch.tensor([maximum - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        kwargs = dict(cos=dc, sin=ds, page_table=page_table, current_pos=pos, kv_cache=kv)
        # Full logical requests through the public API; no skipped prefix work.
        for n in lengths:
            print("PREFILL_START", n, "time", time.time(), flush=True)
            start = time.perf_counter()
            with device_only():
                out = model.prefill_forward(
                    tx[:, :, :n, :], cos=cos[:, :, :n, :], sin=sin[:, :, :n, :], page_table=page_table, kv_cache=kv
                )
            ttnn.synchronize_device(mesh)
            seconds = time.perf_counter() - start
            for a, b, expected in gold[n]:
                selected = out[:, :, a:b, :]
                actual = cpu(selected).reshape(1, b - a, 2560).clone()
                key = f"{n}_{a}_{b}"
                if baseline:
                    tt_golden[key] = actual
                tt_score = pcc(actual, tt_golden[key])
                assert tt_score >= 0.995
                score = pcc(expected, actual)
                del selected
                row = dict(length=n, start=a, end=b, pcc=score, ttnn_baseline_pcc=tt_score, seconds=seconds)
                rows.append(row)
                print("LONG_PREFILL", row, flush=True)
                assert score >= 0.995
            out.deallocate(True)
        # Decode at the last advertised position, with all earlier cache entries
        # produced by TT prefill. Replacing the final token avoids position N.
        if baseline:
            torch.save(tt_golden, tt_golden_path)
        out = model.decode_forward(dx, **kwargs)
        eager = pcc(gold[maximum][-1][2][:, -1:], cpu(out).reshape(1, 1, 2560))
        assert eager >= 0.995
        out.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            out = model.decode_forward(dx, **kwargs)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        last = None
        for iteration in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            got = cpu(out).reshape(1, 1, 2560)
            score = pcc(gold[maximum][-1][2][:, -1:], got)
            assert score >= 0.995
            if last is not None:
                assert torch.equal(last, got)
            last = got
            print("LONG_TRACE", maximum - 1, iteration, score, flush=True)
        ttnn.release_trace(mesh, tid)
        report = dict(
            layer=layer,
            baseline=baseline,
            options=options,
            decoder_sha256=hashlib.sha256(
                (ROOT / ("tt/optimized_decoder.py" if baseline else "tt/multichip_decoder.py")).read_bytes()
            ).hexdigest(),
            trace_alloc_tracking=os.getenv("TT_METAL_TRACE_ALLOC_TRACKING", "0"),
            program_cache_tracking=os.getenv("TT_METAL_TRACE_ALLOC_SKIP_PROGRAM_CACHE", "0") != "1",
            revision=REVISION,
            weight_source="real",
            input_distribution="256 checkpoint-HF BF16 activation rows repeated",
            lengths=lengths,
            rows=rows,
            decode_position=maximum - 1,
            traced_decode_pcc=score,
            repeated_decode_equal=True,
            per_device_cache_bytes=blocks * (8 if baseline else 2) * 4 * 1088 * 2,
            cache_dtype=str(model.kv_cache_dtype),
            activation_source="recorded checkpoint HF rows repeated for exact long-context oracle",
        )
        (
            (artifact_dir or ROOT / "doc/multichip_decoder")
            / f'full_prefill_{"baseline" if baseline else "tp4"}_layer{layer}.json'
        ).write_text(json.dumps(report, indent=2))
        print("LONG_CONTEXT_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=[524275, 524288])
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--options", type=json.loads, default={})
    ap.add_argument("--artifact-dir", type=Path)
    args = ap.parse_args()
    validate(args.layer, args.lengths, args.baseline, args.options, args.artifact_dir)
