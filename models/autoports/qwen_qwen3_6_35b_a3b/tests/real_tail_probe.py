# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Discriminate decode geometry with real weights and directly prepared HF KV."""

import json
from pathlib import Path
from unittest.mock import patch

import torch
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding, apply_rotary_pos_emb

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.decoder_factory import Decoder as FunctionalDecoder
from models.autoports.qwen_qwen3_6_35b_a3b.tests.reference import config, layer, pcc, weights
from models.autoports.qwen_qwen3_6_35b_a3b.tests.runtime_audit import device_only


def main():
    doc = Path(__file__).parents[1] / "doc/functional_decoder"
    torch.set_num_threads(8)
    torch.manual_seed(154)
    c, sd = config(), weights(3)
    n = c.max_position_embeddings
    x = torch.randn(1, n, 2048, dtype=torch.bfloat16) * 0.05
    expected_tail = torch.load(doc / "oracle_cache/full_262144.pt", weights_only=True)["expected"][:, -256:].clone()
    expected = expected_tail[:, -1:]
    x_tail = x[:, -256:].clone()
    hf = layer(sd, 3)
    rotary = Qwen3_5MoeTextRotaryEmbedding(c)
    kcache = torch.empty(n // 32, 2, 32, 256, dtype=torch.bfloat16)
    vcache = torch.empty_like(kcache)
    with torch.no_grad():
        for off in range(0, n, 1024):
            h = hf.input_layernorm(x[:, off : off + 1024])
            k = hf.self_attn.k_norm(hf.self_attn.k_proj(h).reshape(1, 1024, 2, 256)).transpose(1, 2)
            v = hf.self_attn.v_proj(h).reshape(1, 1024, 2, 256).transpose(1, 2)
            cos, sin = rotary(h, torch.arange(off, off + 1024).unsqueeze(0))
            _, k = apply_rotary_pos_emb(k, k, cos, sin)
            kcache[off // 32 : (off + 1024) // 32] = k.reshape(2, 32, 32, 256).permute(1, 0, 2, 3)
            vcache[off // 32 : (off + 1024) // 32] = v.reshape(2, 32, 32, 256).permute(1, 0, 2, 3)
    print("HF_KV_PREPARED", n, flush=True)
    del hf, x
    xd = x_tail[:, -1:]
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), physical_device_ids=[0], trace_region_size=0)
    rows = []
    try:
        dec = FunctionalDecoder.from_state_dict(sd, hf_config=c, layer_idx=3, mesh_device=mesh)

        def upload(t):
            return ttnn.from_torch(t, device=mesh, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)

        state = dec.new_kv_state(context_length=n, kv_cache=(upload(kcache), upload(vcache)))
        td = upload(xd)
        positions = dec.prepare_positions(start_pos=n - 1, seq_len=1)
        original = ttnn.transformer.paged_scaled_dot_product_attention_decode
        for grid, chunk in (((1, 1), 32), ((8, 8), 32), ((8, 8), 256)):

            def variant(*args, **kwargs):
                kwargs["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=grid, q_chunk_size=32, k_chunk_size=chunk
                )
                return original(*args, **kwargs)

            with patch.object(ttnn.transformer, "paged_scaled_dot_product_attention_decode", variant), device_only():
                out = dec.decode_forward(td, state=state, positions=positions)
            got = ttnn.to_torch(out).reshape(xd.shape)
            row = dict(grid=grid, k_chunk=chunk, pcc=pcc(expected, got), contribution_pcc=pcc(expected - xd, got - xd))
            print("REAL_TAIL", row, flush=True)
            rows.append(row)
            del out
        tx = upload(x_tail)
        plan = dec.prepare_prefill(seq_len=256, start_pos=n - 256)
        original_prefill = ttnn.transformer.chunked_scaled_dot_product_attention
        for chunk in (32, 256):

            def prefill_variant(*args, **kwargs):
                kwargs["program_config"] = ttnn.SDPAProgramConfig(
                    compute_with_storage_grid_size=(8, 8), q_chunk_size=32, k_chunk_size=chunk
                )
                return original_prefill(*args, **kwargs)

            with patch.object(ttnn.transformer, "chunked_scaled_dot_product_attention", prefill_variant), device_only():
                out = dec.prefill_forward(tx, state=state, plan=plan)
            got = ttnn.to_torch(out).reshape(x_tail.shape)
            perrow = {str(n - 256 + i): pcc(expected_tail[:, i], got[:, i]) for i in range(256)}
            row = dict(
                mode="prefill",
                k_chunk=chunk,
                pcc=pcc(expected_tail, got),
                per_row_pcc=perrow,
                minimum_row_pcc=min(perrow.values()),
            )
            rows.append(row)
            print("REAL_PREFILL_TAIL", chunk, row["pcc"], row["minimum_row_pcc"], flush=True)
            del out
    finally:
        ttnn.close_mesh_device(mesh)
    (doc / "real_prefill_tail_probe.json").write_text(json.dumps(dict(context=n, rows=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
