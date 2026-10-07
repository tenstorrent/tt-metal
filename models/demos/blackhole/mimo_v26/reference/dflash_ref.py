"""CPU oracle for the MiMo-V2.6-Flash DFlash drafter (checkpoint dir /mnt/tt-data/MiMo-V2.6-Flash-RL/dflash), written from dflash/dflash.py + dflash/config.json.

Block draft at anchor position p (one round): ctx = target features of positions 0..p-1 (concat of hidden states of layers [0,11,23,35,47] -> fc (20480->4096) ->
hidden_norm; shared by all draft layers), block tokens = [x_p (the verified anchor token), mask x7] embedded with the TARGET embedding (mask row = mask_embedding.pt),
5 layers of (input_layernorm -> attention over [ctx keys (window 1024 before the anchor), 8 block keys (bidirectional)] -> residual -> post_attention_layernorm ->
SwiGLU 16384 -> residual), final norm, TARGET lm_head on the last 7 rows -> draft tokens for positions p+1..p+7.
Attention details (variants flagged because dflash.py as shipped does not implement them and config/checkpoint do): q_norm/k_norm (per-head RMSNorm, weight [128]),
partial rotary 0.5 (first 64 of 128 dims, split-half rotate_half, theta 1e4), V scaled by 0.612, per-head attention sink bias (layers.*.self_attn.attention_sink_bias [64]),
GQA 64/8, scaling 128**-0.5, non-causal inside the block. Variants: sink on/off, vscale 0.612/1.0, rope 'partial'|'full'.
"""
import torch
import torch.nn.functional as F

from . import ref_layer as R

DF = "/mnt/tt-data/MiMo-V2.6-Flash-RL/dflash"
TARGET_LAYERS = (0, 11, 23, 35, 47)
BLOCK = 8
WINDOW = 1024
EPS = 1e-6


def load_dflash(dtype=torch.bfloat16):
    from safetensors import safe_open

    f = safe_open(f"{DF}/dflash_draft_model.safetensors", "pt")
    sd = {k: f.get_tensor(k).to(dtype) for k in f.keys()}
    me = torch.load(f"{DF}/mask_embedding.pt")
    sd["mask_embedding"] = me["embedding"].to(dtype)
    sd["mask_token_id"] = me["mask_token_id"]
    return sd


def head_norm(x, w):  # RMSNorm over the last dim (per head)
    return R.rms_norm(x, w, EPS)


def rope(x, pos, theta=1e4, mode="partial"):
    """x [..., S, 128]; pos [S] float. partial: rotate first 64 dims only."""
    rd = 64 if mode == "partial" else 128
    inv = 1.0 / (theta ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd))
    f = pos[:, None].float() * inv[None]
    e = torch.cat([f, f], -1)
    cos, sin = e.cos().to(x.dtype), e.sin().to(x.dtype)
    r = x[..., :rd]
    r = r * cos + R.rotate_half(r) * sin
    return torch.cat([r, x[..., rd:]], -1) if rd < x.shape[-1] else r


def ctx_features(feats, sd):
    """feats [B,S,5*4096] (layers 0,11,23,35,47 concatenated) -> hidden_norm(fc(feats)) [B,S,4096]."""
    return head_norm(feats @ sd["fc.weight"].T, sd["hidden_norm.weight"])


@torch.no_grad()
def draft_blocks(hn, anchors_tok, anchors_pos, sd, E, variant=None):
    """hn [S,4096] ctx features of ONE sequence; anchors_pos [A] anchor positions p (ctx = positions < p); anchors_tok [A] the token at p.
    Returns draft hidden rows [A,7,4096] (after final norm) for positions p+1..p+7."""
    v = dict(sink=True, vscale=0.612, rope="partial")
    v.update(variant or {})
    dt = hn.dtype
    A = anchors_pos.shape[0]
    S = hn.shape[0]
    mask_e = sd["mask_embedding"].to(dt)
    x = torch.cat([E[anchors_tok].to(dt)[:, None], mask_e.view(1, 1, -1).expand(A, BLOCK - 1, -1)], 1)  # [A,8,4096]
    ctx_pos = torch.arange(S)
    blk_pos = anchors_pos[:, None] + torch.arange(BLOCK)[None]  # [A,8]
    # visibility of ctx keys: position < p and >= p - WINDOW
    vis = (ctx_pos[None] < anchors_pos[:, None]) & (ctx_pos[None] >= anchors_pos[:, None] - WINDOW)  # [A,S]
    for li in range(5):
        p = f"layers.{li}."
        w = lambda n: sd[p + n]
        xn = R.rms_norm(x, w("input_layernorm.weight"), EPS)
        q = (xn @ w("self_attn.q_proj.weight").T).view(A, BLOCK, 64, 128)
        q = head_norm(q, w("self_attn.q_norm.weight")).transpose(1, 2)  # [A,64,8,128]
        kn = head_norm(
            (xn @ w("self_attn.k_proj.weight").T).view(A, BLOCK, 8, 128), w("self_attn.k_norm.weight")
        ).transpose(
            1, 2
        )  # [A,8,8,128]
        vn = (xn @ w("self_attn.v_proj.weight").T).view(A, BLOCK, 8, 128).transpose(1, 2)
        kc = head_norm((hn @ w("self_attn.k_proj.weight").T).view(S, 8, 128), w("self_attn.k_norm.weight")).transpose(
            0, 1
        )  # [8,S,128]
        vc = (hn @ w("self_attn.v_proj.weight").T).view(S, 8, 128).transpose(0, 1)
        kc = rope(kc, ctx_pos.float(), mode=v["rope"])
        # per-anchor rope for q and noise k (positions differ per anchor)
        q = torch.stack([rope(q[a], blk_pos[a].float(), mode=v["rope"]) for a in range(A)])
        kn = torch.stack([rope(kn[a], blk_pos[a].float(), mode=v["rope"]) for a in range(A)])
        if v["vscale"] != 1.0:
            vn, vc = vn * v["vscale"], vc * v["vscale"]
        K = torch.cat([kc[None].expand(A, -1, -1, -1), kn], 2)  # [A,8,S+8,128]
        V = torch.cat([vc[None].expand(A, -1, -1, -1), vn], 2)
        Ke, Ve = K.repeat_interleave(8, 1), V.repeat_interleave(8, 1)  # q head h -> kv head h // 8
        s = (q @ Ke.transpose(2, 3)).float() * 128**-0.5  # [A,64,8,S+8]
        m = torch.cat([vis, torch.ones(A, BLOCK, dtype=torch.bool)], 1)  # [A,S+8]
        s = s.masked_fill(~m[:, None, None, :], float("-inf"))
        if v["sink"]:
            s = torch.cat([s, w("self_attn.attention_sink_bias").float().view(1, 64, 1, 1).expand(A, 64, BLOCK, 1)], -1)
        pr = F.softmax(s - s.max(-1, keepdim=True).values, -1)
        if v["sink"]:
            pr = pr[..., :-1]
        o = (pr.to(dt) @ Ve).transpose(1, 2).reshape(A, BLOCK, 64 * 128)
        x = x + o @ w("self_attn.o_proj.weight").T
        xm = R.rms_norm(x, w("post_attention_layernorm.weight"), EPS)
        x = (
            x
            + (F.silu(xm @ w("mlp.gate_proj.weight").T) * (xm @ w("mlp.up_proj.weight").T))
            @ w("mlp.down_proj.weight").T
        )
    return R.rms_norm(x, sd["norm.weight"], EPS)[:, 1:]
