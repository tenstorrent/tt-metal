"""Per-layer CPU oracle dump (chain over a layer range) for the device tests. Real GSM8K chat-template tokens only.

Per layer `layer_{L}.pt` = {"prefill": {h_in,h_out,attn_in,attn_out,ffn_in,ffn_out [B,S,4096], routing_idx [B*S,8] int64, routing_w [B*S,8] fp32,
route_gap [B*S] fp32 (score+bias margin 8th vs 9th expert)}, "state": KV after prefill (global layers: k [B,4,S,192] post-rope, v [B,4,S,128] already*0.707;
SWA layers: k_ring [B,8,128,192], v_ring [B,8,128,128] with position p in slot p%128), "decode": [per step {dec_in, dec_out [B,1,4096], routing_idx, routing_w}],
"meta": {...}}. dir/tokens.pt = {prefill_tokens, decode_tokens, sha256}. dtype bf16 (default; what the device approximates) or fp32 (noise floor).
Chain: layer L+1 input is the oracle's own layer L output (so the dumps are a consistent prefix chain). With --tf_dir the fp32 pass additionally runs each layer
teacher-forced on the bf16 dump's h_in and stores the bf16-vs-fp32 per-layer difference (noise floor) in noise.pt.

    nice python -m models.demos.blackhole.mimo_v26.reference.ref_prefill_dump --layers 0-5 --S 128 --B 16 --dtype bf16 --out /mnt/tt-data/ssinghal/mimo-dumps/mimo-prefill-s128b16
"""
import argparse
import os
import time

import torch

from . import loader as Ld
from . import prompts as P
from . import ref_layer as R


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def pcc(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return torch.corrcoef(torch.stack([a, b]))[0, 1].item()


def ring(k, S, W=128):
    """last W positions into slots p % W."""
    out = torch.zeros(*k.shape[:2], W, k.shape[-1], dtype=k.dtype)
    for p in range(max(0, S - W), S):
        out[:, :, p % W] = k[:, :, p]
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-5")
    ap.add_argument("--S", type=int, default=128)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--start", type=int, default=0, help="first GSM8K test row")
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--tf_dir",
        default=None,
        help="bf16 dump dir: also run each layer teacher-forced on its h_in (fp32 pass only) -> noise.pt",
    )
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "32")))
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    os.makedirs(a.out, exist_ok=True)
    ck = Ld.Ckpt()
    toks, texts, dec = P.gsm8k_tokens(a.B, a.S, start=a.start, steps=a.steps)
    sha = P.sha256_tokens(toks)
    torch.save(
        {"prefill_tokens": toks, "decode_tokens": dec, "sha256": sha, "texts": texts, "start_row": a.start},
        os.path.join(a.out, "tokens.pt"),
    )
    print("prompts sha256", sha, flush=True)
    E = ck.get("model.embed_tokens.weight")
    h = E[toks].to(dt)
    hd = [E[dec[:, i : i + 1]].to(dt) for i in range(a.steps)]
    layers = parse_layers(a.layers)
    if layers[0] != 0:
        raise SystemExit(
            "chain must start at layer 0 (use --layers 0-N); per-layer inputs for later layers come from the chain"
        )
    noise = {}
    for L in layers:
        t0 = time.time()
        sd = Ld.load_layer(L, dt, ck)
        t1 = time.time()
        r = R.layer_forward(h, sd, L)
        S = a.S
        pre = {k: r[k].to(dt).clone() for k in ("attn_in", "attn_out", "ffn_in", "ffn_out", "h_out")}
        pre["h_in"] = h.clone()
        if L > 0:
            pre.update(
                routing_idx=r["routing_idx"].clone(),
                routing_w=r["routing_w"].float().clone(),
                route_gap=r["route_gap"].float().clone(),
            )
        k, v = r["k"].to(torch.bfloat16), r["v"].to(torch.bfloat16)
        state = {"k_ring": ring(k, S), "v_ring": ring(v, S), "window": 128} if R.is_swa(L) else {"k": k, "v": v}
        decs = []
        kc, vc = r["k"], r["v"]
        for s in range(a.steps):
            rd = R.layer_forward(hd[s], sd, L, kv_past=(kc, vc), q0=S + s)
            kc, vc = torch.cat([kc, rd["k"]], 2), torch.cat([vc, rd["v"]], 2)
            e = {"dec_in": hd[s].clone(), "dec_out": rd["h_out"].to(dt).clone()}
            if L > 0:
                e.update(routing_idx=rd["routing_idx"].clone(), routing_w=rd["routing_w"].float().clone())
            decs.append(e)
            hd[s] = rd["h_out"]
        if a.tf_dir:
            src = torch.load(os.path.join(a.tf_dir, f"layer_{L}.pt"))["prefill"]
            rt = R.layer_forward(src["h_in"].to(dt), sd, L)
            n = {
                "pcc_h_out": pcc(rt["h_out"], src["h_out"]),
                "rel_h_out": ((rt["h_out"].float() - src["h_out"].float()).norm() / rt["h_out"].float().norm()).item(),
            }
            if L > 0:
                sa = rt["routing_idx"].sort(-1).values == src["routing_idx"].sort(-1).values
                n["router_set_equal_frac"] = sa.all(-1).float().mean().item()
                n["router_slot_overlap"] = (
                    (sum((rt["routing_idx"][:, i : i + 1] == src["routing_idx"]).any(-1).float() for i in range(8)) / 8)
                    .mean()
                    .item()
                )
            noise[L] = n
            print("  noise floor (fp32 vs bf16 dump, teacher-forced):", n, flush=True)
        torch.save(
            {
                "prefill": pre,
                "state": state,
                "decode": decs,
                "meta": {"layer": L, "dtype": a.dtype, "S": S, "B": a.B, "sha256": sha},
            },
            os.path.join(a.out, f"layer_{L}.pt"),
        )
        h = r["h_out"]
        print(
            f"layer {L}: load {t1-t0:.0f}s fwd {time.time()-t1:.0f}s  h_out rms {h.float().pow(2).mean().sqrt():.4f}",
            flush=True,
        )
        del sd, r
    if a.tf_dir:
        torch.save(noise, os.path.join(a.out, "noise.pt"))
    if layers[-1] == 47:
        W = ck.get("model.norm.weight")
        hn = R.rms_norm(h[:, -1:], W)
        logits = (hn.float() @ ck.get("lm_head.weight").float().T)[:, 0]
        torch.save({"prefill_logits": logits, "prefill_argmax": logits.argmax(-1)}, os.path.join(a.out, "final.pt"))


if __name__ == "__main__":
    main()
