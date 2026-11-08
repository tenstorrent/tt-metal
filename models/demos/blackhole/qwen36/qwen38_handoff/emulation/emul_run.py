import os
import re
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bfp import bfp_quantize
from transformers import AutoConfig, AutoModelForCausalLM

torch.set_num_threads(int(os.environ.get("NT", "16")))
W = "/home/ttuser/atupe/qwen38_work"
OUT = W + "/runs/T3/emul/"
K = W + "/runs/T3/kl/"
PATHS = {"38": "/home/ttuser/atupe/models/Qwen3.8-27B", "36": "/home/runara/models/Qwen3.6-27B"}
# bits: 7 -> BFP8, 3 -> BFP4
CFG = {
    "Q8": dict(gu=3, down=7, gdn_in=7, gdn_out=7, attn=7, lm=7),
    "Q4": dict(gu=3, down=3, gdn_in=7, gdn_out=7, attn=3, lm=7),
    "Q4all": dict(gu=3, down=3, gdn_in=3, gdn_out=7, attn=3, lm=7),
    "Q8noGU": dict(gu=7, down=7, gdn_in=7, gdn_out=7, attn=7, lm=7),
    "E1": dict(gu=7, down=3, gdn_in=7, gdn_out=7, attn=7, lm=7),
    "E2": dict(gu=7, down=7, gdn_in=7, gdn_out=7, attn=3, lm=7),
    "E3": dict(gu=7, down=3, gdn_in=7, gdn_out=7, attn=3, lm=7),
    "E4": dict(gu=7, down=7, gdn_in=3, gdn_out=7, attn=7, lm=7),
    "E5": dict(gu=7, down=7, gdn_in=7, gdn_out=3, attn=7, lm=7),
    "E6": dict(gu=7, down=7, gdn_in=7, gdn_out=7, attn=7, lm=3),
    "E8": dict(gu=7, down=7, gdn_in=7, gdn_out=7, attn=7, lm=7, gate=3),
    "Q0": None,
}


def q_hf(w, bits, chunk=4096):
    """w: HF [out,in] bf16 -> quantize as [K=in,N=out] -> back."""
    out = torch.empty_like(w)
    for c0 in range(0, w.shape[1], chunk):
        x = w[:, c0 : c0 + chunk].float().t().contiguous()
        out[:, c0 : c0 + chunk] = bfp_quantize(x, bits).t().to(w.dtype)
    return out


def q_ab(a, b, bits, tp=4):
    nv = a.shape[0] // tp
    for d in range(tp):
        sl = slice(d * nv, (d + 1) * nv)
        M = torch.cat([a[sl].float().t(), b[sl].float().t(), torch.zeros(a.shape[1], 32 - 2 * nv)], 1).contiguous()
        Q = bfp_quantize(M, bits)
        a[sl] = Q[:, :nv].t().to(a.dtype)
        b[sl] = Q[:, nv : 2 * nv].t().to(b.dtype)


def apply(model, cfg):
    sd = dict(model.named_parameters())
    n = 0
    with torch.no_grad():
        for name, p in sd.items():
            if "visual" in name or name.startswith("mtp"):
                continue
            bits = None
            if name.endswith("mlp.gate_proj.weight") and "gate" in cfg:
                bits = cfg["gate"]
            elif name.endswith("mlp.gate_proj.weight") or name.endswith("mlp.up_proj.weight"):
                bits = cfg["gu"]
            elif name.endswith("mlp.down_proj.weight"):
                bits = cfg["down"]
            elif re.search(r"linear_attn\.in_proj_(qkv|z)\.weight$", name):
                bits = cfg["gdn_in"]
            elif name.endswith("linear_attn.out_proj.weight"):
                bits = cfg["gdn_out"]
            elif re.search(r"self_attn\.(q|k|v|o)_proj\.weight$", name):
                bits = cfg["attn"]
            elif name.endswith("lm_head.weight"):
                bits = cfg["lm"]
            if name.endswith("linear_attn.in_proj_a.weight"):
                q_ab(p.data, sd[name.replace("in_proj_a", "in_proj_b")].data, cfg["gdn_in"])
                n += 2
                continue
            if name.endswith("linear_attn.in_proj_b.weight"):
                continue
            if bits is None:
                continue
            p.data.copy_(q_hf(p.data, bits))
            n += 1
    return n


def run(tag, cname):
    t0 = time.time()
    toks = torch.load(K + "hf38_ref2048.refpt")["reference_tokens"][:, :1024]
    path = PATHS[tag]
    cfg = AutoConfig.from_pretrained(path)
    cfg.rope_scaling = {"factor": 4.0, "original_max_position_embeddings": 32768, "type": "yarn"}
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, dtype=torch.bfloat16)
    model.eval()
    print(tag, cname, "loaded", time.time() - t0, flush=True)
    if CFG[cname] is not None:
        n = apply(model, CFG[cname])
        print("quantized tensors", n, time.time() - t0, flush=True)
    with torch.no_grad():
        lg = model(toks).logits[0, 511:1023].float()
    P = torch.log_softmax(lg, -1)
    torch.save({"logp": P, "argmax": P.argmax(-1)}, OUT + f"{cname}-{tag}_logp.pt")
    print("done", tag, cname, time.time() - t0, flush=True)


if __name__ == "__main__":
    tag = sys.argv[1]
    for c in sys.argv[2:]:
        run(tag, c)
