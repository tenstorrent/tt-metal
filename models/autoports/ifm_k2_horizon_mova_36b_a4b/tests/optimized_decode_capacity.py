# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Near-maximum, non-1024-aligned cache capacity with independent HF KV.

The full-context runner proves TT-produced KV. This regression isolates the
decode page-table extension on a large odd page count without another complete
prefill. All host cache preparation happens before the measured device call.
"""

import argparse
import hashlib
import importlib
import inspect
import json
import time
from importlib.machinery import SourceFileLoader
from pathlib import Path

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.optimized_decoder import OptimizedDecoder
from .fused_reference import REVISION, ROOT, pcc, reference
from .validate_decoder import device_only


def validate(
    layer,
    capacity=524256,
    source=None,
    tag=None,
    cache_dtype=None,
    options=None,
    probe_prefill=False,
    probe_prefill_length=32,
):
    torch.set_num_threads(16)
    torch.manual_seed(812)
    assert capacity % 32 == 0
    c, hf, rope, sd = reference(layer)
    from .optimized_activations import recorded_inputs

    pattern = recorded_inputs(layer, 1, 256)
    mod = importlib.import_module(type(hf).__module__)
    with torch.no_grad():
        norm = hf.input_layernorm(pattern)
        kbase = hf.self_attn.k_proj(norm).reshape(1, 256, 8, 128).transpose(1, 2)
        sample_cache = DynamicCache(config=c)
        hf.self_attn(
            norm,
            position_embeddings=rope(pattern, torch.arange(256)[None]),
            attention_mask=None,
            past_key_values=sample_cache,
        )
        cs = rope(pattern, torch.arange(capacity)[None])
        keys = kbase.repeat(1, 1, (capacity + 255) // 256, 1)[:, :, :capacity]
        keys = keys * cs[0][:, None] + mod.rotate_half(keys) * cs[1][:, None]
        values = sample_cache.layers[layer].values.repeat(1, 1, (capacity + 255) // 256, 1)[:, :, :capacity]
        prefix = DynamicCache(config=c)
        prefix.update(keys[:, :, :-1].clone(), values[:, :, :-1].clone(), layer)
        x = pattern[:, (capacity - 1) % 256 : (capacity - 1) % 256 + 1]
        expected = hf(x, position_embeddings=(cs[0][:, -1:], cs[1][:, -1:]), past_key_values=prefix)
    if probe_prefill:
        query = pattern.index_select(1, torch.arange(capacity - 32, capacity) % 256)
        prefix_p = DynamicCache(config=c)
        prefix_p.update(keys[:, :, :-32].clone(), values[:, :, :-32].clone(), layer)
        mask = torch.arange(capacity)[None, :] > torch.arange(capacity - 32, capacity)[:, None]
        mask = torch.zeros(32, capacity).masked_fill(mask, float("-inf"))[None, None]
        with torch.no_grad():
            expected_p = hf(
                query,
                position_embeddings=(cs[0][:, -32:], cs[1][:, -32:]),
                attention_mask=mask,
                past_key_values=prefix_p,
            )
        del prefix_p, mask
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v.contiguous(), device=mesh, dtype=dtype, layout=layout)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(1, 1, 2560)

        decoder_class = (
            SourceFileLoader("capacity_candidate", str(source)).load_module().OptimizedDecoder
            if source
            else OptimizedDecoder
        )
        model = decoder_class.from_state_dict(sd, hf_config=c, layer_idx=layer, mesh_device=mesh, options=options)
        blocks = capacity // 32
        pages = torch.randperm(blocks)
        caches = []
        for source in (keys, values):
            logical = source[0].reshape(8, blocks, 32, 128).permute(1, 0, 2, 3)
            physical = torch.empty_like(logical)
            physical[pages] = logical
            caches.append(tt(physical, getattr(ttnn, cache_dtype) if cache_dtype else model.kv_cache_dtype))
        dx = tt(x.reshape(1, 1, 1, 2560))
        kwargs = dict(
            cos=tt(cs[0][:, -1:].reshape(1, 1, 1, 128)),
            sin=tt(cs[1][:, -1:].reshape(1, 1, 1, 128)),
            page_table=tt(pages[None].int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            current_pos=tt(torch.tensor([capacity - 1], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT),
            kv_cache=caches,
        )
        with device_only():
            warm = model.decode_forward(dx, **kwargs)
        assert pcc(expected, cpu(warm)) >= 0.995
        warm.deallocate(True)
        tid = ttnn.begin_trace_capture(mesh, cq_id=0)
        with device_only():
            output = model.decode_forward(dx, **kwargs)
        ttnn.end_trace_capture(mesh, tid, cq_id=0)
        previous = None
        scores = []
        for _ in range(3):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=True)
            actual = cpu(output)
            scores.append(pcc(expected, actual))
            assert scores[-1] >= 0.995
            if previous is not None:
                assert torch.equal(previous, actual)
            previous = actual
        start = time.perf_counter()
        for _ in range(100):
            ttnn.execute_trace(mesh, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(mesh)
        decode_ms = (time.perf_counter() - start) * 1000 / 100
        assert torch.equal(previous, cpu(output))
        ttnn.release_trace(mesh, tid)
        prefill_score = None
        prefill_ms = None
        if probe_prefill:
            assert 32 <= probe_prefill_length <= capacity and probe_prefill_length % 32 == 0
            probe_query = pattern.index_select(1, torch.arange(capacity - probe_prefill_length, capacity) % 256)
            px = tt(probe_query.unsqueeze(0))
            pc = tt(cs[0][:, -probe_prefill_length:].unsqueeze(1))
            ps = tt(cs[1][:, -probe_prefill_length:].unsqueeze(1))
            with device_only():
                result_p = model.prefill_forward(
                    px,
                    cos=pc,
                    sin=ps,
                    page_table=kwargs["page_table"],
                    kv_cache=caches,
                    start_pos=capacity - probe_prefill_length,
                )
            actual_p = ttnn.to_torch(result_p, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(
                1, probe_prefill_length, 2560
            )[:, -32:]
            if probe_prefill_length > 32:
                ttnn.synchronize_device(mesh)
                begin = time.perf_counter()
                for _ in range(3):
                    with device_only():
                        result_p = model.prefill_forward(
                            px,
                            cos=pc,
                            sin=ps,
                            page_table=kwargs["page_table"],
                            kv_cache=caches,
                            start_pos=capacity - probe_prefill_length,
                        )
                ttnn.synchronize_device(mesh)
                prefill_ms = (time.perf_counter() - begin) * 1000 / 3
                repeated_p = ttnn.to_torch(result_p, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(
                    1, probe_prefill_length, 2560
                )[:, -32:]
                assert torch.equal(actual_p, repeated_p)
            prefill_score = pcc(expected_p, actual_p)
            print("CAPACITY_PREFILL_PCC", prefill_score, flush=True)
            assert prefill_score >= 0.995
        report = dict(
            long_prefix_prefill_pcc=prefill_score,
            long_prefix_prefill_length=probe_prefill_length if probe_prefill else None,
            long_prefix_prefill_warmed_ms=prefill_ms,
            layer=layer,
            cache_dtype=str(caches[0].dtype),
            effective_options=model.options,
            implementation=type(model).__module__,
            implementation_source_sha256=hashlib.sha256(Path(inspect.getfile(type(model))).read_bytes()).hexdigest(),
            traced_warmed_decode_wall_ms=decode_ms,
            measured_decode_replays=100,
            revision=REVISION,
            cache_capacity=capacity,
            pages=blocks,
            position=capacity - 1,
            traced_decode_pcc=scores,
            deterministic=True,
            weight_source="real",
            cache_source="independent HF projections and RoPE",
            decoder_sha256=hashlib.sha256((ROOT / "tt/optimized_decoder.py").read_bytes()).hexdigest(),
        )
        (
            ROOT / f"doc/optimized_decoder/decode_capacity_layer{layer}_{capacity}{'_'+tag if tag else ''}.json"
        ).write_text(json.dumps(report, indent=2))
        print("DECODE_CAPACITY_PASS", json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=524256)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--probe-prefill", action="store_true")
    parser.add_argument("--probe-prefill-length", type=int, default=32)
    parser.add_argument("--options", type=json.loads, default={})
    parser.add_argument("--cache-dtype", choices=["bfloat8_b", "bfloat16"])
    args = parser.parse_args()
    validate(
        args.layer,
        args.capacity,
        args.source,
        args.tag,
        args.cache_dtype,
        args.options,
        args.probe_prefill,
        args.probe_prefill_length,
    )
