"""python opsum_buckets.py <profiler dir> -> buckets per layer type and a 48-layer extrapolation (kernel ms, mean over devices / sum of per-op max over devices).
Layer layout (tt/layer.py decode): [0] input rms_norm (1 core), attention ops 1..k, post-attn rms_norm (second 1-core op of ~65 us), then FFN (dense L0: 7 ops, MoE: router..all_reduce)."""
import json
import sys

d = sys.argv[1]
S = json.load(open(d + "/opsum.json"))


def split(ops):
    idx = [i for i, o in enumerate(ops) if o["cores"] == 1 and 55e3 < o["mean"] < 80e3]
    assert len(idx) >= 2, idx
    return idx[0], idx[1]


def bucket(name):
    ops = S[name]["ops"]
    a, b = split(ops)
    rng = lambda lo, hi, key: sum(o[key] for o in ops[lo:hi]) / 1e6
    r = {
        "norm_in": ops[a]["mean"] / 1e6,
        "attention": rng(a + 1, b, "mean"),
        "norm_post": ops[b]["mean"] / 1e6,
        "ffn": rng(b + 1, len(ops), "mean"),
        "attention_max": rng(a + 1, b, "max"),
        "ffn_max": rng(b + 1, len(ops), "max"),
        "op2op": S[name]["op2op_ms"],
        "n_attn_ops": b - a - 1,
        "n_ffn_ops": len(ops) - b - 1,
    }
    return r


def avg(names):
    bs = [bucket(n) for n in names if n in S]
    return {k: sum(b[k] for b in bs) / len(bs) for k in bs[0]}


swa, glo, dense = avg(["layer1", "layer2", "layer3"]), avg(["layer5", "layer11"]), bucket("layer0")
print("per layer (kernel ms, mean over devices):")
for n, b in (("dense+global L0", dense), ("moe+swa (L1-3)", swa), ("moe+global (L5,L11)", glo)):
    print(
        f"  {n:22s} norm_in {b['norm_in']:.3f} attention {b['attention']:.3f} ({b['n_attn_ops']:.0f} ops) norm_post {b['norm_post']:.3f} ffn {b['ffn']:.3f} ({b['n_ffn_ops']:.0f} ops)  | attention_max {b['attention_max']:.3f} ffn_max {b['ffn_max']:.3f} op2op {b['op2op']:.3f}"
    )
tot = {}
for k in ("norm_in", "attention", "norm_post", "ffn", "attention_max", "ffn_max", "op2op"):
    tot[k] = dense[k] + 39 * swa[k] + 8 * glo[k]
embed, head = S["embed"]["kernel_ms_mean"], S["head"]["kernel_ms_mean"]
print(
    "48 layers (1 dense-global + 39 swa-moe + 8 global-moe), kernel ms: "
    + ", ".join(f"{k} {v:.2f}" for k, v in tot.items())
)
print(
    f"embedding {embed:.3f} ms, head (norm+lm_head+max+argmax) {head:.3f} ms; sum of mean kernel times {sum(tot[k] for k in ('norm_in','attention','norm_post','ffn')) + embed + head:.2f} ms; "
    f"with per-op max over devices {tot['norm_in']+tot['norm_post']+tot['attention_max']+tot['ffn_max']+embed+head:.2f} ms"
)
