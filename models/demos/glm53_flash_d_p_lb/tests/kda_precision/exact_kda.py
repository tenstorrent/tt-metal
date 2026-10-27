"""Offline: the exact fp32 KDA (the CPU reference's _kda) on the device's own captured KDA inputs.

For each KDA layer: run the reference KDA chunk by chunk over x_00..x_10 (the device's attn_norm inputs), compare the
device's carried state after each chunk (state_<end>.pt) with the exact state on the same inputs, and with the golden
at 51200 / 56320. Last chunk: device output vs exact (from the exact chain, and from the device's own 51200 state).
Saves exact_51200.pt ({layer: {kda_recurrent, kda_conv}}) and metrics.pt.

    python exact_kda.py <capture dir> [layers]
"""

import sys
import time
from pathlib import Path

import torch

from models.demos.common.bringup.reference.golden import Golden, hf_path, load_spec
from models.demos.glm53_flash_d_p.reference.glm_ref import GlmConfig, GlmReference, State, load_layer
from models.demos.glm53_flash_d_p.reference.weights import WeightLoader

torch.set_num_threads(int(__import__("os").environ.get("EK_THREADS", "32")))
cap = Path(sys.argv[1])
spec = load_spec()
path = hf_path(spec)
cfg = GlmConfig.from_json(str(Path(path) / "config.json"))
loader = WeightLoader(path)
g = Golden.for_rung(spec, "s56320")
kda_layers = [i for i in range(45) if cfg.is_kda(i)]
if len(sys.argv) > 2:
    kda_layers = [int(v) for v in sys.argv[2].split(",")]
seq, chunk = 56320, 5120
last = seq - chunk

ref = GlmReference.__new__(GlmReference)
ref.cfg, ref.dtype = cfg, torch.float32
t0 = time.time()
ref.w = {}
for i in kda_layers:
    w = load_layer(loader, cfg, i, torch.float32, eager_experts=False)
    w.experts = None
    w["conv"] = torch.cat([w.pop(f"self_attn.{n}_conv1d.weight") for n in "qkv"])[:, 0].float()
    ref.w[i] = {k: v for k, v in w.items() if k.startswith("self_attn.") or k == "conv"}
print(f"weights {time.time() - t0:.0f}s", flush=True)


class Ctx:
    def __init__(self, start, state):
        self.start, self.state = start, state


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def head_stats(d, e):
    """per head rel / norm ratio of d vs e ([H, K, V])"""
    r = [rel(d[h], e[h]) for h in range(d.shape[0])]
    q = [float(d[h].norm() / e[h].norm().clamp_min(1e-30)) for h in range(d.shape[0])]
    return r, q


st = State(cfg, kda_layers, seq)
dev_st = State(cfg, kda_layers, seq)
metrics = {}
exact51 = {}
for c in range(seq // chunk):
    t = time.time()
    xs = torch.load(cap / f"x_{c:02d}.pt")
    dstate = torch.load(cap / f"state_{(c + 1) * chunk}.pt")
    ys = torch.load(cap / f"y_{c:02d}.pt") if c == seq // chunk - 1 else None
    start = c * chunk
    if start == last:
        prev = torch.load(cap / f"state_{last}.pt")
        for i in kda_layers:
            exact51[i] = {"kda_recurrent": st.rec[i].clone(), "kda_conv": st.conv[i].clone()}
            dev_st.rec[i].copy_(prev[i]["kda_recurrent"].reshape(dev_st.rec[i].shape))
            dev_st.conv[i].copy_(prev[i]["kda_conv"].reshape(dev_st.conv[i].shape))
    for i in kda_layers:
        x = xs[i].float()
        y = ref._kda(i, Ctx(start, st), x)
        e = st.rec[i]
        d = dstate[i]["kda_recurrent"].reshape(e.shape).float()
        r, q = head_stats(d, e)
        m = {
            "rel": rel(d, e),
            "ratio": float(d.norm() / e.norm()),
            "head_rel": r,
            "head_ratio": q,
            "conv_rel": rel(dstate[i]["kda_conv"].float(), st.conv[i].float()),
        }
        if (c + 1) * chunk in (last, seq):
            gs = g.state(i, at=(c + 1) * chunk if (c + 1) * chunk != seq else None)["kda_recurrent"].float()
            m["exact_vs_gold"] = rel(e, gs)
            m["dev_vs_gold"] = rel(d, gs)
            m["exact_vs_gold_head"], _ = head_stats(e, gs)
            m["dev_vs_gold_head"], m["dev_vs_gold_head_ratio"] = head_stats(d, gs)
        if ys is not None:
            yd = ys[i].float()
            m["y_vs_exact_chain"] = rel(yd, y.float())
            yo = ref._kda(i, Ctx(start, dev_st), x)  # exact from the device's own 51200 state
            m["y_vs_exact_devstate"] = rel(yd, yo.float())
            gy = g.layer(10, i)["attn_out"].float()
            m["y_vs_gold"] = rel(yd, gy)
            m["yexact_vs_gold"] = rel(y.float(), gy)
        metrics[(i, (c + 1) * chunk)] = m
        hr = sorted(range(len(r)), key=lambda h: -r[h])[:3]
        extra = ""
        if "dev_vs_gold" in m:
            extra = f" | exact vs gold {m['exact_vs_gold']:.4f}, dev vs gold {m['dev_vs_gold']:.4f}"
        if "y_vs_exact_chain" in m:
            extra += (
                f" | y: vs exact chain {m['y_vs_exact_chain']:.4f}, vs exact(dev state) {m['y_vs_exact_devstate']:.4f},"
                f" vs gold {m['y_vs_gold']:.4f}, exact vs gold {m['yexact_vs_gold']:.4f}"
            )
        print(
            f"[ek] end {(c + 1) * chunk:5d} L{i:02d}: state rel {m['rel']:.4f} ratio {m['ratio']:.4f}; heads med "
            f"{sorted(r)[len(r) // 2]:.4f} max {max(r):.4f} (h{hr[0]} ratio {q[hr[0]]:.3f}, h{hr[1]} {q[hr[1]]:.3f}, "
            f"h{hr[2]} {q[hr[2]]:.3f}); conv {m['conv_rel']:.4f}{extra}",
            flush=True,
        )
    print(f"[ek] chunk {c} {time.time() - t:.0f}s", flush=True)
    del xs
torch.save(exact51, cap / "exact_51200.pt")
torch.save(metrics, cap / "metrics.pt")
print("done", flush=True)
