# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""Compare bounded prefill against one-chunk TTNN and the unchanged HF layer."""

import argparse
import json
from pathlib import Path

import torch
from transformers import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_8_27b.tests.reference import load_config, load_layer_weights, make_reference
from models.autoports.qwen_qwen3_8_27b.tests.run_decoder import device_only, pcc
from models.autoports.qwen_qwen3_8_27b.tt.functional_decoder import FunctionalDecoder


def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(241)
    c = load_config(args.snapshot)
    weights = load_layer_weights(args.snapshot, args.layer)
    hf = make_reference(c, args.layer, weights)
    x = (torch.randn(2, 258, c.hidden_size) * 0.1).bfloat16()
    cos, sin = Qwen3_5TextRotaryEmbedding(c)(x, torch.arange(258).expand(2, -1))
    mask = torch.full((257, 257), float("-inf"), dtype=torch.bfloat16).triu(1)[None, None]
    cache = DynamicCache(config=c)
    with torch.no_grad():
        expected = hf(
            x[:, :-1],
            position_embeddings=(cos[:, :-1], sin[:, :-1]),
            attention_mask=mask if hf.layer_type == "full_attention" else None,
            past_key_values=cache,
        )
        expected_decode = hf(x[:, -1:], position_embeddings=(cos[:, -1:], sin[:, -1:]), past_key_values=cache)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
    try:

        def upload(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
            return ttnn.from_torch(
                t.contiguous(), device=mesh, dtype=dtype, layout=layout, memory_config=ttnn.DRAM_MEMORY_CONFIG
            )

        decoder = FunctionalDecoder.from_state_dict(weights, hf_config=c, layer_idx=args.layer, mesh_device=mesh)
        dx, cc, ss = upload(x[:, :-1]), upload(cos[:, :-1]), upload(sin[:, :-1])
        next_x, next_cos, next_sin = upload(x[:, -1:]), upload(cos[:, -1:]), upload(sin[:, -1:])
        current_pos = upload(torch.full((2,), 257, dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        table = upload(torch.randperm(18).reshape(2, 9).int(), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        results, decode_results = [], []
        for chunk_size in (128, 257):
            decoder.CHUNK_SIZE = chunk_size
            state = decoder.allocate_state(batch_size=2, num_pages=18)
            with device_only():
                out = decoder.prefill_forward(dx, state=state, page_table=table, cos=cc, sin=ss)
            results.append(ttnn.to_torch(out))
            snapshots = {name: ttnn.clone(tensor) for name, tensor in vars(state).items() if tensor is not None}

            def restore():
                for name, snapshot in snapshots.items():
                    ttnn.copy(snapshot, getattr(state, name))

            def decode():
                with device_only():
                    return decoder.decode_forward(
                        next_x, state=state, page_table=table, current_pos=current_pos, cos=next_cos, sin=next_sin
                    )

            warm = decode()
            ttnn.synchronize_device(mesh)
            ttnn.deallocate(warm)
            restore()
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            decoded = decode()
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            try:
                restore()
                ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
                decode_results.append(ttnn.to_torch(decoded))
            finally:
                ttnn.release_trace(mesh, trace)
        metrics = {
            "kind": hf.layer_type,
            "weights": "real",
            "batch": 2,
            "length": 257,
            "chunked_hf_pcc": pcc(expected, results[0]),
            "unchunked_hf_pcc": pcc(expected, results[1]),
            "chunked_vs_unchunked_pcc": pcc(results[0], results[1]),
            "chunked_traced_decode_hf_pcc": pcc(expected_decode, decode_results[0]),
            "unchunked_traced_decode_hf_pcc": pcc(expected_decode, decode_results[1]),
            "chunked_vs_unchunked_decode_pcc": pcc(decode_results[0], decode_results[1]),
        }
        print(json.dumps(metrics), flush=True)
        assert all(value >= 0.995 for name, value in metrics.items() if name.endswith("pcc"))
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", type=int, choices=(0, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
