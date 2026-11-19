# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Full logical context, reduced layer-only HF oracle at selected output rows.

All tokens contribute to the reference cache/state. Full-attention's oracle
projects every K/V and evaluates selected queries to avoid a quadratic CPU
full-output reference. Linear attention advances HF recurrence over every token.
TT executes the complete decoder for every row, through public chunked prefill.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

import ttnn
from models.autoports.qwen_qwen3_6_27b.tests.evidence import decoder_class, decoder_kwargs, provenance
from models.autoports.qwen_qwen3_6_27b.tests.run_contract import row_pcc
from models.autoports.qwen_qwen3_6_27b.tests.run_decoder import MODEL, REVISION, load_layer, pcc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--length", type=int, required=True)
    ap.add_argument("--drain-chunks", action="store_true", help="Diagnostic dispatch drain after each physical chunk")
    ap.add_argument("--attention-chunk", type=int, help="Diagnostic native SDPA key-chunk override")
    ap.add_argument(
        "--probe-tail-only", action="store_true", help="Diagnostic HF-cache setup; execute final chunk only"
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(16)
    torch.manual_seed(291)
    c = AutoConfig.from_pretrained(MODEL, revision=REVISION, local_files_only=True).text_config
    c._attn_implementation = "sdpa"
    assert args.length <= c.max_position_embeddings
    weights = load_layer(args.layer)
    ref = Qwen3_5DecoderLayer(c, args.layer).eval().to(torch.bfloat16)
    ref.load_state_dict(weights)
    rotary = Qwen3_5TextRotaryEmbedding(c)
    linear = c.layer_types[args.layer] == "linear_attention"
    length = args.length
    x = torch.randn(1, length, 5120, dtype=torch.bfloat16) * 0.1
    d = torch.randn(1, 1, 5120, dtype=torch.bfloat16) * 0.1
    selected = sorted(set([0, 1, 31, 32, 511, 512] + list(range(length - 32, length))))
    selected = [i for i in selected if 0 <= i < length]
    tail_start = (length - 1) // 512 * 512 if args.probe_tail_only else 0
    if args.probe_tail_only:
        assert not linear
        selected = [i for i in selected if i >= tail_start]
    positions = torch.arange(length).reshape(1, -1)
    expected_parts = {}
    with torch.no_grad():
        if linear:
            cache = DynamicCache(config=c)
            for start in range(0, length, 512):
                chunk = x[:, start : start + 512]
                mixed = ref.linear_attn(ref.input_layernorm(chunk), cache_params=cache)
                for i in selected:
                    if start <= i < start + chunk.shape[1]:
                        residual = chunk[:, i - start : i - start + 1] + mixed[:, i - start : i - start + 1]
                        expected_parts[i] = residual + ref.mlp(ref.post_attention_layernorm(residual))
                if start % 8192 == 0:
                    print("HF_PROGRESS", start, flush=True)
            expected = torch.cat([expected_parts[i] for i in selected], dim=1)
            expected_decode = (
                ref(d, position_embeddings=rotary(d, torch.tensor([[length]])), past_key_values=cache)
                if length < c.max_position_embeddings
                else None
            )
        else:
            keys = []
            values = []
            for start in range(0, length, 512):
                chunk = x[:, start : start + 512]
                h = ref.input_layernorm(chunk)
                k = ref.self_attn.k_norm(ref.self_attn.k_proj(h).reshape(1, -1, 4, 256)).transpose(1, 2)
                cos, sin = rotary(chunk, positions[:, start : start + 512])
                _, k = apply_rotary_pos_emb(k, k, cos, sin)
                keys.append(k)
                values.append(ref.self_attn.v_proj(h).reshape(1, -1, 4, 256).transpose(1, 2))
                if start % 8192 == 0:
                    print("HF_PROGRESS", start, flush=True)
            k = torch.cat(keys, dim=2)
            v = torch.cat(values, dim=2)
            del keys, values

            def selected_forward(inp, indices, k, v):
                h = ref.input_layernorm(inp)
                q, gate = ref.self_attn.q_proj(h).reshape(1, -1, 24, 512).chunk(2, dim=-1)
                q = ref.self_attn.q_norm(q).transpose(1, 2)
                cos, sin = rotary(inp, indices.reshape(1, -1))
                q, _ = apply_rotary_pos_emb(q, q, cos, sin)
                mask = torch.arange(k.shape[2])[None, :] <= indices[:, None]
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask.reshape(1, 1, *mask.shape), enable_gqa=True
                )
                out = out.transpose(1, 2).reshape(1, -1, 6144) * gate.reshape(1, -1, 6144).sigmoid()
                residual = inp + ref.self_attn.o_proj(out)
                return residual + ref.mlp(ref.post_attention_layernorm(residual))

            expected = selected_forward(x[:, selected], torch.tensor(selected), k, v)
            if length < c.max_position_embeddings:
                h = ref.input_layernorm(d)
                kd = ref.self_attn.k_norm(ref.self_attn.k_proj(h).reshape(1, 1, 4, 256)).transpose(1, 2)
                cos, sin = rotary(d, torch.tensor([[length]]))
                _, kd = apply_rotary_pos_emb(kd, kd, cos, sin)
                vd = ref.self_attn.v_proj(h).reshape(1, 1, 4, 256).transpose(1, 2)
                expected_decode = selected_forward(
                    d, torch.tensor([length]), torch.cat([k, kd], dim=2), torch.cat([v, vd], dim=2)
                )
            else:
                expected_decode = None
            if not args.probe_tail_only:
                del k, v
    print("HF_REFERENCE_READY", flush=True)
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    try:
        layer = decoder_class().from_state_dict(
            weights, **decoder_kwargs(), hf_config=c, layer_idx=args.layer, mesh_device=mesh
        )
        if args.attention_chunk:
            native_config = ttnn.SDPAProgramConfig

            def diagnostic_config(**kwargs):
                kwargs["k_chunk_size"] = args.attention_chunk
                return native_config(**kwargs)

            ttnn.SDPAProgramConfig = diagnostic_config
        finish = layer._finish
        completed_tokens = 0

        def finish_with_progress(chunk, attention, **kwargs):
            nonlocal completed_tokens
            result = finish(chunk, attention, **kwargs)
            if args.drain_chunks:
                ttnn.synchronize_device(mesh)
            completed_tokens += chunk.shape[1]
            if completed_tokens % 8192 == 0 or completed_tokens == length:
                print("TT_PROGRESS", completed_tokens, "drained" if args.drain_chunks else "enqueued", flush=True)
            return result

        layer._finish = finish_with_progress
        page_extent = (length + (args.attention_chunk or 32)) // (args.attention_chunk or 32)
        page_extent *= (args.attention_chunk or 32) // 32
        table = torch.randperm(page_extent).reshape(1, -1)
        state = layer.create_recurrent_state() if linear else layer.create_paged_state(table)
        if args.probe_tail_only:
            table = ttnn.to_torch(state.page_table, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            page_extent = table.shape[1]
            for name, tensor in (("key", k), ("value", v)):
                padded = torch.nn.functional.pad(tensor[:, :, :length], (0, 0, 0, page_extent * 32 - length))
                logical = padded.reshape(4, page_extent, 32, 256).permute(1, 0, 2, 3).contiguous()
                physical = torch.empty_like(logical)
                physical[table[0]] = logical
                setattr(state, name, layer._upload(physical))
        tx = layer._upload(x[:, tail_start:])
        rope = (
            None
            if linear
            else tuple(layer._upload(t.unsqueeze(1)) for t in rotary(x[:, tail_start:], positions[:, tail_start:]))
        )
        ttnn.synchronize_device(mesh)
        start = time.perf_counter()
        output = layer.prefill_forward(tx, state=state, rope=rope, start_pos=tail_start)
        ttnn.synchronize_device(mesh)
        duration = time.perf_counter() - start
        actual = torch.cat(
            [
                ttnn.to_torch(
                    output[:, i - tail_start : i - tail_start + 1], mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)
                )
                for i in selected
            ],
            dim=1,
        )
        rows = row_pcc(expected.float() - x[:, selected].float(), actual.float() - x[:, selected].float())
        result = {
            "provenance": provenance(),
            "layer": args.layer,
            "length": length,
            "selected_rows": selected,
            "prefill_pcc": pcc(expected, actual),
            "contribution_pcc": rows,
            "prefill_wall_seconds": duration,
            "hf_reference": "all-token cache/state, selected output rows",
            "full_decoder_all_tokens": not args.probe_tail_only,
            "diagnostic_attention_chunk": args.attention_chunk,
            "diagnostic_chunk_drains": args.drain_chunks,
        }
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(result, flush=True)
        assert result["prefill_pcc"] >= 0.995 and torch.isfinite(actual).all()
        assert min(rows) >= 0.995
        if not linear and not args.probe_tail_only:
            resume = max(0, length - 32)
            indices = [i for i in selected if i >= resume]
            resumed = layer.prefill_forward(
                tx[:, resume:],
                state=state,
                start_pos=resume,
                rope=tuple(t[:, :, resume:] for t in rope),
            )
            resumed_host = ttnn.to_torch(resumed, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            expected_resume = expected[:, [selected.index(i) for i in indices]]
            resume_rows = row_pcc(
                expected_resume.float() - x[:, indices].float(),
                resumed_host[:, [i - resume for i in indices]].float() - x[:, indices].float(),
            )
            result["high_position_continuation"] = {"start": resume, "contribution_pcc": resume_rows}
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            assert min(resume_rows) >= 0.995
            del resumed
        if expected_decode is not None:
            td = layer._upload(d)
            dk = (
                {}
                if linear
                else {
                    "current_pos": ttnn.from_torch(
                        torch.tensor([length], dtype=torch.int32),
                        device=mesh,
                        dtype=ttnn.int32,
                        layout=ttnn.ROW_MAJOR_LAYOUT,
                    ),
                    "rope": tuple(layer._upload(t.unsqueeze(1)) for t in rotary(d, torch.tensor([[length]]))),
                }
            )
            buffers = [state.conv, state.recurrent] if linear else []
            saved = [
                ttnn.from_torch(
                    ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0)), dtype=t.dtype, layout=t.layout
                )
                for t in buffers
            ]
            warm = layer.decode_forward(td, state=state, **dk)
            ttnn.synchronize_device(mesh)
            eager_host = ttnn.to_torch(warm, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            del warm
            for h, t in zip(saved, buffers):
                ttnn.copy_host_to_device_tensor(h, t)
            trace = ttnn.begin_trace_capture(mesh, cq_id=0)
            out = layer.decode_forward(td, state=state, **dk)
            ttnn.end_trace_capture(mesh, trace, cq_id=0)
            for h, t in zip(saved, buffers):
                ttnn.copy_host_to_device_tensor(h, t)
            ttnn.execute_trace(mesh, trace, cq_id=0, blocking=True)
            got = ttnn.to_torch(out, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
            result.update(decode_context=length + 1, decode_pcc=pcc(expected_decode, got), decode_traced=True)
            result.update(
                eager_decode_pcc=pcc(expected_decode, eager_host), eager_trace_bit_equal=torch.equal(eager_host, got)
            )
            ttnn.release_trace(mesh, trace)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print("DECODE", result["decode_pcc"], flush=True)
            assert result["decode_pcc"] >= 0.995
            assert result["eager_trace_bit_equal"] and torch.isfinite(got).all()
    finally:
        ttnn.close_mesh_device(mesh)


if __name__ == "__main__":
    main()
