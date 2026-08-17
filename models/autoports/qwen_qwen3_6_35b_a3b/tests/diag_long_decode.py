# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0

"""
Diagnostic for the long-context full-attention decode PCC.

``test_longest_decode_context[full]`` at position 262143 reports PCC ~0.9974, visibly worse
than every other case in the stage (~0.99999). This script decides *why* rather than assuming.

It sweeps the decode position over an identical random-seeded paged cache and at each position
reports:

  * ``layer_pcc``  - PCC of the whole decoder-layer output vs HF,
  * ``tt_vs_fp32``      - PCC of the **attention branch alone** (the only position-dependent
                          part of the layer) against HF's fp32 ``self_attn``,
  * ``tt_vs_bf16ctl``   - the same TTNN output against a *bf16-operand* HF control: HF's own
                          attention math with q/k/v rounded to bf16 and exact accumulation, i.e.
                          what any correct bf16 implementation must produce,
  * ``bf16ctl_vs_fp32`` - that control against fp32 HF. This is the error floor the precision
                          policy alone imposes, independent of TTNN.
  * ``attn_rms``        - the RMS magnitude of the (fp32) attention branch.

A page-table / index bug would show up as an ``attn_pcc`` cliff at one position. A flat-softmax
conditioning effect shows up as ``attn_rms`` shrinking like 1/sqrt(context) while ``attn_pcc``
stays high, so the *relative* error of a fixed absolute error grows smoothly.

The last row repeats position 262143 with the query scaled up, which peaks the softmax and
therefore removes the cancellation without changing any code path.

    python models/autoports/qwen_qwen3_6_35b_a3b/tests/diag_long_decode.py
"""

import torch

import ttnn
from models.autoports.qwen_qwen3_6_35b_a3b.tests.harness import (
    build_layer_pair,
    from_tt,
    seed_kv_cache,
    to_tt_decode,
    to_tt_positions,
)
from models.autoports.qwen_qwen3_6_35b_a3b.tt import reference as ref

CONTEXT = 262144
POSITIONS = [1023, 8191, 32767, 131071, 262143]


def hf_attn_branch(pair, token, pos, k_all, v_all):
    cache = ref.make_cache(pair.hf_config)
    cache.update(k_all[:, :, :pos], v_all[:, :, :pos], pair.layer_idx)
    cos, sin = ref.rope_cos_sin(pair.hf_config, torch.tensor([[pos]]))
    with torch.no_grad():
        normed = pair.hf.input_layernorm(token)
        out, _ = pair.hf.self_attn(
            hidden_states=normed,
            position_embeddings=(cos, sin),
            attention_mask=None,
            past_key_values=cache,
        )
    return out


def hf_attn_branch_bf16_inputs(pair, token, pos, k_all, v_all):
    """Control: HF's own attention math with the operands rounded to bf16.

    This is what a *correct* bf16 implementation must produce. It uses the layer's own
    ``q_proj``/``q_norm``/``k_proj``/``k_norm``/``o_proj`` modules and HF's
    ``apply_rotary_pos_emb``, and only rounds q/k/v to bf16 before the attention itself —
    exactly the precision the device path carries. If the device output tracks *this* closely
    while diverging from the fp32 reference, the divergence is operand quantisation under an
    ill-conditioned (near-uniform) softmax, not an implementation error.
    """
    attn = pair.hf.self_attn
    cfg = pair.hf_config
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    cos, sin = ref.rope_cos_sin(cfg, torch.tensor([[pos]]))
    with torch.no_grad():
        h = pair.hf.input_layernorm(token)
        q_all, gate = torch.chunk(attn.q_proj(h).view(1, 1, -1, hd * 2), 2, dim=-1)
        gate = gate.reshape(1, 1, -1)
        q = attn.q_norm(q_all.view(1, 1, nh, hd)).transpose(1, 2)
        k_new = attn.k_norm(attn.k_proj(h).view(1, 1, nkv, hd)).transpose(1, 2)
        v_new = attn.v_proj(h).view(1, 1, nkv, hd).transpose(1, 2)
        q, k_new = ref.apply_rotary_pos_emb(q, k_new, cos, sin)

        k = torch.cat([k_all[:, :, :pos], k_new], dim=2).to(torch.bfloat16).float()
        v = torch.cat([v_all[:, :, :pos], v_new], dim=2).to(torch.bfloat16).float()
        q = q.to(torch.bfloat16).float()
        k = k.repeat_interleave(nh // nkv, dim=1)
        v = v.repeat_interleave(nh // nkv, dim=1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), scale=hd**-0.5
        ).float()
        out = out.transpose(1, 2).reshape(1, 1, -1)
        return attn.o_proj(out * torch.sigmoid(gate))


def tt_attn_branch(pair, token, pos):
    device = pair.device
    tt_tok = to_tt_decode(device, token.reshape(1, 1, -1))
    normed = ttnn.rms_norm(
        tt_tok,
        weight=pair.tt.w["input_norm_w"],
        epsilon=pair.cfg.rms_norm_eps,
        compute_kernel_config=pair.tt.compute_config,
    )
    tt_pos = to_tt_positions(device, torch.tensor([pos]))
    out = pair.tt._full_attention_decode(normed, current_pos=tt_pos, page_table=pair.page_table)
    got = from_tt(out).reshape(1, 1, pair.cfg.hidden_size)
    for t in (tt_tok, normed, tt_pos, out):
        ttnn.deallocate(t)
    return got


def main():
    torch.set_num_threads(16)
    device = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
    try:
        pair = build_layer_pair(device, kind="full", max_batch_size=1, supported_context=CONTEXT)
        cfg = pair.cfg
        gen = torch.Generator().manual_seed(4242)
        k_all = torch.randn(1, cfg.num_key_value_heads, CONTEXT, cfg.head_dim, generator=gen)
        v_all = torch.randn(1, cfg.num_key_value_heads, CONTEXT, cfg.head_dim, generator=gen)
        k_all = k_all.to(torch.bfloat16).float()
        v_all = v_all.to(torch.bfloat16).float()
        seed_kv_cache(pair, k_all, v_all, user_id=0)
        token = ref.synthetic_hidden_states(pair.hf_config, 1, 1, seed=3)

        header = (
            f"{'pos':>8} {'layer_pcc':>11} {'tt_vs_fp32':>11} {'tt_vs_bf16ctl':>14} "
            f"{'bf16ctl_vs_fp32':>16} {'attn_rms':>10}"
        )
        print("DIAG " + header)
        for pos, tok, tag in [(p, token, "") for p in POSITIONS] + [(262143, token * 8.0, " q*8")]:
            got_layer = None
            tt_pos = to_tt_positions(device, torch.tensor([pos]))
            tt_tok = to_tt_decode(device, tok.reshape(1, 1, -1))
            out = pair.tt.decode_forward(tt_tok, current_pos=tt_pos, page_table=pair.page_table)
            got_layer = from_tt(out).reshape(1, 1, cfg.hidden_size)
            for t in (tt_tok, tt_pos, out):
                ttnn.deallocate(t)
            # decode wrote one row into the cache; restore so every row sees the same context
            seed_kv_cache(pair, k_all, v_all, user_id=0)

            cache = ref.make_cache(pair.hf_config)
            cache.update(k_all[:, :, :pos], v_all[:, :, :pos], pair.layer_idx)
            want_layer = ref.hf_decode(pair.hf, pair.hf_config, tok, positions=torch.tensor([pos]), cache=cache)

            got_attn = tt_attn_branch(pair, tok, pos)
            seed_kv_cache(pair, k_all, v_all, user_id=0)
            want_attn = hf_attn_branch(pair, tok, pos, k_all, v_all)
            ctl_attn = hf_attn_branch_bf16_inputs(pair, tok, pos, k_all, v_all)

            print(
                f"DIAG {pos:>8} {ref.pcc(got_layer, want_layer):>11.7f} "
                f"{ref.pcc(got_attn, want_attn):>11.7f} {ref.pcc(got_attn, ctl_attn):>14.7f} "
                f"{ref.pcc(ctl_attn, want_attn):>16.7f} {float(want_attn.std()):>10.5f}{tag}"
            )
    finally:
        ttnn.close_mesh_device(device)


if __name__ == "__main__":
    main()
