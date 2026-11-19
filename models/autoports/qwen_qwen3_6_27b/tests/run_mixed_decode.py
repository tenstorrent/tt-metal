# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Batched traced decode with independent prefix lengths and page ownership."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer, Qwen3_5TextRotaryEmbedding

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc
from models.autoports.qwen_qwen3_6_27b.tt.functional_decoder import PagedState, RecurrentState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(819)
    torch.set_num_threads(8)
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    config._attn_implementation = "sdpa"
    weights = load_layer(args.layer)
    hf = Qwen3_5DecoderLayer(config, args.layer).eval().to(torch.bfloat16)
    hf.load_state_dict(weights)
    rotary = Qwen3_5TextRotaryEmbedding(config)
    lengths = [33, 1, 63, 32]
    prefixes = [torch.randn(1, n, 5120, dtype=torch.bfloat16) * 0.1 for n in lengths]
    tokens = torch.randn(4, 1, 5120, dtype=torch.bfloat16) * 0.1
    expected = []
    with torch.no_grad():
        for row, (n, prefix) in enumerate(zip(lengths, prefixes)):
            cache = DynamicCache(config=config)
            hf(prefix, position_embeddings=rotary(prefix, torch.arange(n).reshape(1, -1)), past_key_values=cache)
            token = tokens[row : row + 1]
            expected.append(hf(token, position_embeddings=rotary(token, torch.tensor([[n]])), past_key_values=cache))
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = decoder_class().from_state_dict(
            weights, **decoder_kwargs(), hf_config=config, layer_idx=args.layer, mesh_device=mesh
        )
        linear = layer.kind == "linear_attention"
        if linear:
            states = [layer.create_recurrent_state() for _ in lengths]
        else:
            state = layer.create_paged_state(torch.randperm(12).reshape(4, 3))
            states = [PagedState(state.key, state.value, state.page_table[r : r + 1]) for r in range(4)]
        for n, prefix, row_state in zip(lengths, prefixes, states):
            rotations = (
                None
                if linear
                else tuple(layer._upload(t.unsqueeze(1)) for t in rotary(prefix, torch.arange(n).reshape(1, -1)))
            )
            output = layer.prefill_forward(layer._upload(prefix), state=row_state, rope=rotations)
            del output
        if linear:
            state = RecurrentState(
                ttnn.concat([s.conv for s in states], dim=0), ttnn.concat([s.recurrent for s in states], dim=0)
            )
        td = layer._upload(tokens)
        kwargs = (
            {}
            if linear
            else {
                "current_pos": ttnn.from_torch(
                    torch.tensor(lengths, dtype=torch.int32),
                    device=mesh,
                    dtype=ttnn.int32,
                    layout=ttnn.ROW_MAJOR_LAYOUT,
                ),
                "rope": tuple(
                    layer._upload(t.unsqueeze(1)) for t in rotary(tokens, torch.tensor(lengths).reshape(4, 1))
                ),
            }
        )
        buffers = [state.conv, state.recurrent] if linear else [state.key, state.value]
        saved = [
            ttnn.from_torch(
                ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)), dtype=t.dtype, layout=t.layout
            )
            for t in buffers
        ]

        def restore():
            for host, device in zip(saved, buffers):
                ttnn.copy_host_to_device_tensor(host, device)

        warm = layer.decode_forward(td, state=state, **kwargs)
        ttnn.synchronize_device(mesh)
        del warm
        restore()
        trace = ttnn.begin_trace_capture(mesh, cq_id=0)
        output = layer.decode_forward(td, state=state, **kwargs)
        ttnn.end_trace_capture(mesh, trace, cq_id=0)
        results = []
        for _ in range(3):
            restore()
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            results.append(ttnn.to_torch(output, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)))
        correlations = [pcc(e, results[-1][r : r + 1]) for r, e in enumerate(expected)]
        result = {
            "provenance": provenance(),
            "layer": args.layer,
            "positions": lengths,
            "per_request_pcc": correlations,
            "traced": True,
            "repeated_bit_equal": all(torch.equal(results[0], r) for r in results[1:]),
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        assert min(correlations) >= 0.995 and result["repeated_bit_equal"]
        ttnn.release_trace(mesh, trace)
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
