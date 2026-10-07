"""Teacher-forced single-layer oracle: run ONE decoder layer on a hidden state handed in (e.g. captured from the device) and dump the same fields as
ref_prefill_dump (prefill dict + KV state, optional decode steps). No chain, any layer 0-47.

    python -m models.demos.blackhole.mimo_v26.reference.single_layer --layer 20 --h-in h.pt --out out.pt [--dtype bf16|fp32] [--dec-in dec.pt]

--h-in: a tensor [B,S,4096] or a dict with key h_in (e.g. a layer_{L}.pt "prefill" dict, or a whole layer_{L}.pt). Positions are 0..S-1.
--dec-in (optional): tensor [B,T,4096] of teacher-forced decode-step hidden states for this layer (T steps, fed one by one after the prefill; they are
the device's inputs, not chained from this layer's outputs). Output file = {"prefill","state","decode","meta"} (decode empty without --dec-in).
"""
import argparse

import torch

from . import loader as Ld
from . import ref_layer as R
from .ref_prefill_dump import ring


def load_h(path):
    d = torch.load(path)
    if isinstance(d, dict):
        d = d["prefill"]["h_in"] if "prefill" in d else d["h_in"]
    return d


@torch.no_grad()
def run(L, h, dt=torch.bfloat16, dec=None, ckpt=None, sd=None):
    h = h.to(dt)
    sd = sd or Ld.load_layer(L, dt, ckpt)
    r = R.layer_forward(h, sd, L)
    S = h.shape[1]
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
    decs, kc, vc = [], r["k"], r["v"]
    if dec is not None:
        for s in range(dec.shape[1]):
            rd = R.layer_forward(dec[:, s : s + 1].to(dt), sd, L, kv_past=(kc, vc), q0=S + s)
            kc, vc = torch.cat([kc, rd["k"]], 2), torch.cat([vc, rd["v"]], 2)
            e = {"dec_in": dec[:, s : s + 1].to(dt).clone(), "dec_out": rd["h_out"].to(dt).clone()}
            if L > 0:
                e.update(routing_idx=rd["routing_idx"].clone(), routing_w=rd["routing_w"].float().clone())
            decs.append(e)
    return {
        "prefill": pre,
        "state": state,
        "decode": decs,
        "meta": {"layer": L, "dtype": str(dt), "S": S, "B": h.shape[0], "swa": R.is_swa(L)},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--h-in", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--dec-in", default=None)
    a = ap.parse_args()
    torch.set_num_threads(32)
    dec = load_h(a.dec_in) if a.dec_in else None
    out = run(a.layer, load_h(a.h_in), torch.bfloat16 if a.dtype == "bf16" else torch.float32, dec)
    torch.save(out, a.out)
    print(
        "wrote",
        a.out,
        "layer",
        a.layer,
        "swa" if out["meta"]["swa"] else "global",
        {k: tuple(v.shape) for k, v in out["prefill"].items()},
    )


if __name__ == "__main__":
    main()
