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
import importlib
import json
import time

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from .run_decoder import REVISION, ROOT, pcc, reference
from .validate_decoder import device_only


def validate(layer, lengths):
    torch.set_num_threads(16)
    torch.manual_seed(812)
    c, hf, rope, sd = reference(layer)
    maximum = max(lengths)
    pattern = (torch.randn(1, 256, 2560) * 0.1).bfloat16()
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
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=mesh):
            return ttnn.from_torch(v, dtype=dtype, layout=layout, device=device)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))

        model = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=layer, mesh_device=mesh)
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
        kv = [tt(torch.zeros(blocks, 8, 32, 128, dtype=torch.bfloat16)) for _ in range(2)]
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
                score = pcc(expected, cpu(selected).reshape(1, b - a, 2560))
                del selected
                row = dict(length=n, start=a, end=b, pcc=score, seconds=seconds)
                rows.append(row)
                print("LONG_PREFILL", row, flush=True)
                assert score >= 0.995
            out.deallocate(True)
        # Decode at the last advertised position, with all earlier cache entries
        # produced by TT prefill. Replacing the final token avoids position N.
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
            revision=REVISION,
            weight_source="real",
            input_distribution="256 distinct BF16 activation rows repeated; std=0.1",
            lengths=lengths,
            rows=rows,
            decode_position=maximum - 1,
            traced_decode_pcc=score,
            repeated_decode_equal=True,
            cache_bytes=blocks * 8 * 32 * 128 * 2 * 2,
        )
        (ROOT / f"doc/functional_decoder/long_context_layer{layer}.json").write_text(json.dumps(report, indent=2))
        print("LONG_CONTEXT_PASS", flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--lengths", type=int, nargs="+", default=[524275, 524288])
    args = ap.parse_args()
    validate(args.layer, args.lengths)
