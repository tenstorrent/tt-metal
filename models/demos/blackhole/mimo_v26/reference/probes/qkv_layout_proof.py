"""qkv layout proof: HF MiMoV2DecoderLayer L0 (global, dense) and L1 (SWA, MoE) on REAL GSM8K chat-template tokens (embeddings), real weights, fp32.
rank = rank-blocked de-interleave (decision); naive = contiguous Q|K|V reading. Prints layer-output rms vs input rms and attention statistics."""
import os

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

torch.set_num_threads(32)
from models.demos.blackhole.mimo_v26.reference import hf_layer as H
from models.demos.blackhole.mimo_v26.reference import loader as Ld

ck = Ld.Ckpt()
cfg = H.hf_config()
from models.demos.blackhole.mimo_v26.reference import prompts as P

t, _ = P.gsm8k_tokens(2, 128)
h = ck.get("model.embed_tokens.weight")[t].float()
tm = H.masks_and_rope(cfg, h)
L0rank = None
mod, _ = H.import_hf()


def attn_stats(layer, x):
    a = layer.self_attn
    xn = layer.input_layernorm(x)
    qkv = a.qkv_proj(xn)
    q, k, v = qkv.split([a.q_size, a.k_size, a.v_size], -1)
    B, S, _ = x.shape
    q = q.view(B, S, a.num_attention_heads, a.head_dim).transpose(1, 2)
    k = k.view(B, S, a.num_key_value_heads, a.head_dim).transpose(1, 2)
    masks, rope, pos = tm
    cos, sin = rope[layer.attention_type]
    r = a.rope_dim
    qr, kr = mod.apply_rotary_pos_emb(q[..., :r], k[..., :r], cos, sin)
    q = torch.cat([qr, q[..., r:]], -1)
    k = torch.cat([kr, k[..., r:]], -1)
    kk = mod.repeat_kv(k, a.num_key_value_groups)
    w = q @ kk.transpose(2, 3) * a.scaling + masks[layer.attention_type]
    p = w.softmax(-1)
    ent = -(p * (p + 1e-20).log()).sum(-1)  # [B,H,S]
    S0 = torch.arange(S).float()
    return dict(
        ent_over_logn=(ent[..., 8:] / torch.log(S0[8:] + 1)).mean().item(),
        p_key0=p[..., 8:, 0].mean().item(),
        logit_rms=w[w > -1e8].pow(2).mean().sqrt().item(),
    )


for i in (0, 1):
    for lay in ("rank", "naive"):
        l = H.build_layer(i, torch.float32, lay, ckpt=ck, cfg=cfg)
        x = h if i == 0 else L0rank
        o = H.run_layer(l, x, cfg, tm)
        if i == 0 and lay == "rank":
            L0rank = o
        print(
            "L",
            i,
            lay,
            "layer out rms %.4f in rms %.4f" % (o.pow(2).mean().sqrt(), x.pow(2).mean().sqrt()),
            attn_stats(l, x),
            flush=True,
        )
