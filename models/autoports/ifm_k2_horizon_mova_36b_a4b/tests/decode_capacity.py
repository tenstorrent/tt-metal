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
import json

import torch
from transformers.cache_utils import DynamicCache

import ttnn

from ..tt.functional_decoder import FunctionalDecoder
from .run_decoder import REVISION, ROOT, pcc, reference
from .validate_decoder import device_only


def validate(layer, capacity=524256):
    torch.set_num_threads(16)
    torch.manual_seed(812)
    assert capacity % 32 == 0
    c, hf, rope, sd = reference(layer)
    pattern = (torch.randn(1, 256, 2560) * 0.1).bfloat16()
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
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:

        def tt(v, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(v.contiguous(), device=mesh, dtype=dtype, layout=layout)

        def cpu(v):
            return ttnn.to_torch(v, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)).reshape(1, 1, 2560)

        model = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=layer, mesh_device=mesh)
        blocks = capacity // 32
        pages = torch.randperm(blocks)
        caches = []
        for source in (keys, values):
            logical = source[0].reshape(8, blocks, 32, 128).permute(1, 0, 2, 3)
            physical = torch.empty_like(logical)
            physical[pages] = logical
            caches.append(tt(physical))
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
        ttnn.release_trace(mesh, tid)
        report = dict(
            layer=layer,
            revision=REVISION,
            cache_capacity=capacity,
            pages=blocks,
            position=capacity - 1,
            traced_decode_pcc=scores,
            deterministic=True,
            weight_source="real",
            cache_source="independent HF projections and RoPE",
            decoder_sha256=hashlib.sha256((ROOT / "tt/functional_decoder.py").read_bytes()).hexdigest(),
        )
        (ROOT / f"doc/functional_decoder/decode_capacity_layer{layer}.json").write_text(json.dumps(report, indent=2))
        print("DECODE_CAPACITY_PASS", json.dumps(report), flush=True)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=524256)
    args = parser.parse_args()
    validate(args.layer, args.capacity)
