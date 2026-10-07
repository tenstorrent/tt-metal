"""Per layer: best scalar alpha with device ~ alpha * oracle (fp32, forced device routing for the MoE; bfp8-weight bf16 attention) for ffn_out and attn_out, and the relative error
before / after the rescale. alpha != 1 consistently = a systematic scale bias in the device block (not random rounding noise)."""
import json
import os
import sys

os.environ.setdefault("MIMO_CKPT", "/mnt/tt-data/MiMo-V2.6-Flash-RL")
import torch

from models.demos.blackhole.mimo_v26.reference import loader as Ld
from models.demos.blackhole.mimo_v26.reference import ref_layer as R
from models.demos.blackhole.mimo_v26.reference.bfp import quant_weight_bfp8
from models.demos.blackhole.mimo_v26.reference.probes.device_attribution import moe_forced, rel
from models.demos.blackhole.mimo_v26.reference.ref_prefill_dump import parse_layers

torch.set_num_threads(16)


def alpha(dev, ref):
    dev, ref = dev.float(), ref.float()
    a = ((dev * ref).sum() / (ref * ref).sum()).item()
    return a, rel(dev, ref), rel(dev, a * ref)


with torch.no_grad():
    dev_dir, out = sys.argv[1], sys.argv[2]
    ck = Ld.Ckpt()
    done = {json.loads(l)["L"] for l in open(out)} if os.path.exists(out) else set()
    for L in parse_layers(sys.argv[3]):
        if L in done:
            continue
        d = torch.load(f"{dev_dir}/layer_{L}.pt")["prefill"]
        sd = Ld.load_layer(L, torch.float32, ck)
        sdq = dict(sd)
        for k in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
            sdq[k] = quant_weight_bfp8(sd[k])
        a_ref = R.attention(d["attn_in"].float(), sdq, L)[0]
        fin = d["ffn_in"].reshape(-1, 4096)
        f_ref = moe_forced(fin.float(), sd, d["routing_idx"], d["routing_w"].float())
        aa, ar, aar = alpha(d["attn_out"], a_ref)
        fa, fr, far = alpha(d["ffn_out"].reshape(-1, 4096), f_ref)
        row = {
            "L": L,
            "type": "swa" if R.is_swa(L) else "global",
            "attn": {"alpha": aa, "rel": ar, "rel_after_alpha": aar},
            "ffn_forced": {"alpha": fa, "rel": fr, "rel_after_alpha": far},
        }
        open(out, "a").write(json.dumps(row) + "\n")
        print(
            L,
            row["type"],
            "attn alpha %.4f rel %.4f->%.4f | ffn alpha %.4f rel %.4f->%.4f" % (aa, ar, aar, fa, fr, far),
            flush=True,
        )
