"""Isolate long-context SDPA from TT prefill cache error using exact HF K/V."""

import json
import sys
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextDecoderLayer,
    Gemma4TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

import ttnn
from models.autoports.google_gemma_4_31b_it.tests.run_decoder import pcc, weights
from models.autoports.google_gemma_4_31b_it.tt.functional_decoder import HF_MODEL, HF_REVISION, FunctionalDecoder


@torch.no_grad()
def main():
    torch.set_num_threads(16)
    torch.manual_seed(123)
    c = AutoConfig.from_pretrained(HF_MODEL, revision=HF_REVISION).text_config
    state = weights(5)
    with torch.device("meta"):
        ref = Gemma4TextDecoderLayer(c, 5)
    ref.load_state_dict(state, assign=True)
    x = torch.randn(1, 262144, 5376, dtype=torch.bfloat16)
    cos, sin = Gemma4TextRotaryEmbedding(c)(x, torch.arange(262144)[None], layer_type="full_attention")
    att = ref.self_attn
    ks, vs = [], []
    for start in range(0, 262144, 2048):
        end = start + 2048
        raw = att.k_proj(ref.input_layernorm(x[:, start:end])).view(1, 2048, 4, 512)
        ks.append(
            apply_rotary_pos_emb(att.k_norm(raw), cos[:, start:end], sin[:, start:end], unsqueeze_dim=2).transpose(1, 2)
        )
        vs.append(att.v_norm(raw).transpose(1, 2))
    k, v = torch.cat(ks, dim=2), torch.cat(vs, dim=2)
    del ks, vs
    q = att.q_norm(att.q_proj(ref.input_layernorm(x[:, -1:])).view(1, 1, 32, 512))
    q = apply_rotary_pos_emb(q, cos[:, -1:], sin[:, -1:], unsqueeze_dim=2).transpose(1, 2)
    expected_att = torch.cat(
        [
            torch.nn.functional.scaled_dot_product_attention(
                q[:, i * 8 : (i + 1) * 8],
                k[:, i : i + 1].expand(-1, 8, -1, -1),
                v[:, i : i + 1].expand(-1, 8, -1, -1),
                scale=1.0,
            )
            for i in range(4)
        ],
        dim=1,
    )
    eager_parts, fp32_parts = [], []
    for i in range(4):
        qi = q[:, i * 8 : (i + 1) * 8]
        ki = k[:, i : i + 1].expand(-1, 8, -1, -1)
        vi = v[:, i : i + 1].expand(-1, 8, -1, -1)
        logits = torch.matmul(qi, ki.transpose(-1, -2))
        eager_parts.append(torch.matmul(torch.softmax(logits.float(), dim=-1).to(vi.dtype), vi))
        logits32 = torch.matmul(qi.float(), ki.float().transpose(-1, -2))
        fp32_parts.append(torch.matmul(torch.softmax(logits32, dim=-1), vi.float()).to(vi.dtype))
    eager_attention = torch.cat(eager_parts, dim=1)
    fp32_attention = torch.cat(fp32_parts, dim=1)

    def finish(a):
        r = x[:, -1:] + ref.post_attention_layernorm(att.o_proj(a.transpose(1, 2).reshape(1, 1, -1)))
        return (r + ref.post_feedforward_layernorm(ref.mlp(ref.pre_feedforward_layernorm(r)))) * ref.layer_scalar

    control = dict(
        context=262144,
        layer_type="full_attention",
        attention_sdpa_vs_eager_pcc=pcc(expected_att, eager_attention),
        attention_sdpa_vs_fp32_pcc=pcc(expected_att, fp32_attention),
        layer_sdpa_vs_eager_pcc=pcc(finish(expected_att), finish(eager_attention)),
        layer_sdpa_vs_fp32_pcc=pcc(finish(expected_att), finish(fp32_attention)),
    )
    print("LONG_REFERENCE_CONTROL", control, flush=True)
    Path(__file__).parents[1].joinpath("doc/functional_decoder/long_reference_control.json").write_text(
        json.dumps(control, indent=2) + "\n"
    )
    assert control["layer_sdpa_vs_eager_pcc"] >= 0.9999 and control["layer_sdpa_vs_fp32_pcc"] >= 0.9999
    if "--reference-only" in sys.argv:
        return
    y = x[:, -1:] + ref.post_attention_layernorm(att.o_proj(expected_att.transpose(1, 2).reshape(1, 1, -1)))
    expected = (y + ref.post_feedforward_layernorm(ref.mlp(ref.pre_feedforward_layernorm(y)))) * ref.layer_scalar
    mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)

    def tt(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
        return ttnn.from_torch(t, dtype=dtype, layout=layout, device=mesh)

    rows = []
    try:
        model = FunctionalDecoder.from_state_dict(state, hf_config=c, layer_idx=5, mesh_device=mesh)

        def paged(t):
            return tt(t.reshape(1, 4, 8192, 32, 512).permute(0, 2, 1, 3, 4).reshape(8192, 4, 32, 512).contiguous())

        kc, vc = paged(k), paged(v)
        page = tt(torch.arange(8192, dtype=torch.int32)[None], ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        pos = tt(torch.tensor([262143], dtype=torch.int32), ttnn.int32, ttnn.ROW_MAJOR_LAYOUT)
        dx = tt(x[:, -1:].reshape(1, 1, 1, 5376))
        dc = tt(cos[:, -1:].reshape(1, 1, 1, 512))
        ds = tt(sin[:, -1:].reshape(1, 1, 1, 512))
        tq, tk, tv = model._qkv(dx, dc, ds, decode=True)
        hq = tt(q.transpose(1, 2))
        print("QUERY_PCC", pcc(ttnn.to_torch(tq), q.transpose(1, 2)), flush=True)
        for query_name, query in [("tt", tq), ("hf", hq)]:
            cases = [("default", None)]
            for chunk, cores, exp in [
                (32, 1, True),
                (32, 1, False),
                (32, 16, True),
                (32, 16, False),
                (64, 1, False),
                (64, 16, False),
                (128, 16, False),
            ]:
                cases.append(
                    (
                        f"k{chunk}_cores{cores}_approx{exp}",
                        ttnn.SDPAProgramConfig(
                            compute_with_storage_grid_size=(8, 8),
                            q_chunk_size=32,
                            k_chunk_size=chunk,
                            exp_approx_mode=exp,
                            max_cores_per_head_batch=cores,
                        ),
                    )
                )
            for mode, program in cases:
                a = ttnn.transformer.paged_scaled_dot_product_attention_decode(
                    query,
                    kc,
                    vc,
                    page,
                    cur_pos_tensor=pos,
                    scale=1.0,
                    program_config=program,
                    compute_kernel_config=model.kernel,
                )
                out = model._finish(dx, a, decode=True)
                row = dict(
                    query=query_name,
                    mode=mode,
                    attention_pcc=pcc(ttnn.to_torch(a), expected_att.transpose(1, 2)),
                    layer_pcc=pcc(ttnn.to_torch(out).reshape(1, 1, -1), expected),
                )
                rows.append(row)
                print(row, flush=True)
    finally:
        ttnn.close_mesh_device(mesh)
    Path(__file__).parents[1].joinpath("doc/functional_decoder/long_attention_probe.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
