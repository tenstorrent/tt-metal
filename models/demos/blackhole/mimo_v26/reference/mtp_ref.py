"""CPU oracle for the in-checkpoint MiMo MTP layers (model.mtp.layers.{0,1,2}) under the HYPOTHESES in MTP_NOTES.md (no reference code exists).
Layer k at position t: x = eh_proj(concat of enorm(embed(tok_{t+1+k})) and hnorm(h_prev_t)) -> input_layernorm -> SWA attention (window 128, sinks, theta 1e4,
dims as the main SWA layers) -> residual -> pre_mlp_layernorm -> dense SwiGLU 16384 -> residual -> final_layernorm -> (shared) model.norm? / lm_head.
Variants (all computed, compared on real device-captured last-layer hidden states by `evaluate`):
  order: 'eh' (embedding first, DeepSeek convention) | 'he'
  h_src: 'pre' (main model's last-layer output before model.norm) | 'post' (after model.norm)
  head_norm: 'final' (mtp final_layernorm only) | 'final+norm' (also model.norm) -- applied before the shared lm_head
  res: 'x' (residual stream starts at the eh_proj output) | 'h' ... (only 'x' implemented; the input_layernorm is applied inside the attention branch)
Usage on captures: evaluate(h_last [B,S,4096] (pre model.norm), tokens [B,S+k], k) -> {variant: top-1 accuracy of predicting token t+2+k}.
Not run on a full model: needs the device-captured L47 hidden state (or any hidden state handed in).
"""
import itertools

import torch

from . import loader as Ld
from . import ref_layer as R


def load_mtp(k, dtype=torch.bfloat16, ck=None):
    ck = ck or Ld.Ckpt()
    p = f"model.mtp.layers.{k}."
    g = lambda n: ck.get(p + n)
    sd = {
        n: g(n).to(dtype)
        for n in (
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "pre_mlp_layernorm.weight",
            "final_layernorm.weight",
            "eh_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.attention_sink_bias",
        )
    }
    sd["self_attn.qkv_proj.weight"] = Ld.dequant_qkv(
        1, g("self_attn.qkv_proj.weight"), g("self_attn.qkv_proj.weight_scale_inv")
    ).to(
        dtype
    )  # SWA dims (layer index 1 = SWA)
    for n in ("gate_proj", "up_proj", "down_proj"):
        sd[f"mlp.{n}.weight"] = Ld.dequant_fp8_block(g(f"mlp.{n}.weight"), g(f"mlp.{n}.weight_scale_inv")).to(dtype)
    return sd


@torch.no_grad()
def mtp_layer(h_prev, tok_ids, sd, E, order="eh", q0=0):
    """h_prev [B,S,D]; tok_ids [B,S] = the tokens whose embeddings are fused (t+1+k). Returns the layer output stream [B,S,D] (before final_layernorm) and normed output."""
    dt = h_prev.dtype
    e = R.rms_norm(E[tok_ids].to(dt), sd["enorm.weight"])
    h = R.rms_norm(h_prev, sd["hnorm.weight"])
    x = torch.cat([e, h] if order == "eh" else [h, e], -1) @ sd["eh_proj.weight"].T
    a, _, _ = R.attention(R.rms_norm(x, sd["input_layernorm.weight"]), sd, 1, q0=q0)
    x = x + a
    m = R.rms_norm(x, sd["pre_mlp_layernorm.weight"])
    x = x + R.dense(m, sd)
    return x, R.rms_norm(x, sd["final_layernorm.weight"])


@torch.no_grad()
def evaluate(h_last, tokens, n_layers=1, dtype=torch.bfloat16, ck=None, head_rows=None):
    """h_last [B,S,D]: main-model last-layer output BEFORE model.norm for positions 0..S-1 (the hidden that predicted token t+1);
    tokens [B,S+n_layers+1]: the real token stream (position p holds token p). Layer k predicts token t+2+k at position t (so compares with tokens[:, t+2+k]).
    Chain: layer k>0 takes layer k-1's output stream (variant 'chain_pre'=before final_layernorm or 'chain_post'=after). Returns {variant: [top-1 acc per layer]}.
    """
    ck = ck or Ld.Ckpt()
    E = ck.get("model.embed_tokens.weight")
    W = ck.get("lm_head.weight").to(dtype)
    Wn = ck.get("model.norm.weight")
    B, S, _ = h_last.shape
    res = {}
    for order, h_src, head_norm, chain in itertools.product(
        ("eh", "he"), ("pre", "post"), ("final", "final+norm"), ("chain_pre", "chain_post")
    ):
        h0 = h_last.to(dtype)
        hp = h0 if h_src == "pre" else R.rms_norm(h0, Wn)
        accs = []
        for k in range(n_layers):
            sd = load_mtp(k, dtype, ck) if (k, dtype) not in _CACHE else _CACHE[(k, dtype)]
            _CACHE[(k, dtype)] = sd
            ids = tokens[:, 1 + k : 1 + k + S]
            x, xn = mtp_layer(hp, ids, sd, E, order)
            hn = xn if head_norm == "final" else R.rms_norm(xn, Wn)
            rows = slice(0, S - 1 - k)  # positions whose target token exists
            logits = hn[:, rows].float() @ W.float().T
            tgt = tokens[:, 2 + k : 2 + k + S][:, rows]
            accs.append((logits.argmax(-1) == tgt).float().mean().item())
            hp = x if chain == "chain_pre" else xn
        res[(order, h_src, head_norm, chain)] = accs
    return res


_CACHE = {}
