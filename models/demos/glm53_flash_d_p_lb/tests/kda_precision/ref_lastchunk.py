"""CPU reference, last chunk only (51200..56320) from the golden's own context, with the routed experts' weights
quantized as the device holds them. Measures what bfp4 experts cost within one chunk, everything else exact (fp32).

    python ref_lastchunk.py <mode> [layers]      mode: none | bfp4 | bfp4b (bfp4 of the bf16 weights) | bfp8
                                                 | <mode>+x8 (also the expert input x and h rounded to bfp8)
    layers: comma list of MoE layers to quantize (default all)
"""

import os
import sys
import time

import torch

import models.demos.glm53_flash_d_p.reference.glm_ref as gr
from models.demos.common.bringup.reference.golden import Golden, load_spec

torch.set_num_threads(int(os.environ.get("RL_THREADS", "48")))
mode = sys.argv[1]
only = {int(v) for v in sys.argv[2].split(",")} if len(sys.argv) > 2 else None
wmode, _, amode = mode.partition("+")
gain_fix = wmode.endswith("g")  # bfp4g: routed experts' output / 1.0122 (bfp4's measured gain, layers 4 / 20 / 36)
wmode = wmode.rstrip("g") if gain_fix else wmode
spec = load_spec()
g = Golden.for_rung(spec, "s56320")
tokens = g.tokens()
seq, chunk = 56320, 5120
last = seq - chunk


def bfp(t: torch.Tensor, mbits: int) -> torch.Tensor:
    """Block float as the device packs it (pure torch, no ttnn import: that opens the cluster): blocks of 16
    consecutive elements of the last dim share the exponent E of their largest magnitude; elements round to
    nearest on the grid 2^(E - mbits + 1), magnitude <= (2^mbits - 1) steps (bfp4: mbits 3, bfp8: 7)."""
    x = t.float()
    sh = x.shape
    b = x.reshape(-1, 16)
    m = b.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(m.clamp_min(1e-38)))
    step = torch.exp2(e - (mbits - 1))
    lim = 2**mbits - 1
    q = torch.round(b / step).clamp(-lim, lim) * step
    q = torch.where(m > 0, q, torch.zeros_like(q))
    return q.reshape(sh).to(t.dtype)


def quant_w(w: torch.Tensor) -> torch.Tensor:
    """w [out, in] (torch linear) -> the values the device holds: [in, out] tiles, a block = 16 outputs."""
    if wmode == "none":
        return w
    src = w.t()
    if wmode == "bfp4b":
        src = src.to(torch.bfloat16).float()
    return bfp(src, 7 if wmode == "bfp8" else 3).t().contiguous()


t0 = time.time()
ref = spec.hooks().reference(spec, layers=None, dtype=torch.float32)
print(f"reference loaded {time.time() - t0:.0f}s", flush=True)
orig_ew = ref.expert_weights


def expert_weights(layer):
    f = orig_ew(layer)
    if only is not None and layer not in only:
        return f

    def q(e):  # (no cache: each expert runs once per layer, and the quantized weights are 29 GB a layer)
        if e == 0:
            print(f"  layer {layer} experts quantized ({time.time() - t0:.0f}s)", flush=True)
        return tuple(quant_w(w) for w in f(e))

    return q


ref.expert_weights = expert_weights

if amode == "x8":
    orig_swiglu = gr.clamped_swiglu
    orig_experts = gr.experts_forward

    def act8(x):
        return bfp(x, 7)

    def swiglu8(x, w_gate, w_up, w_down, limit):
        import torch.nn.functional as F

        x = act8(x)
        gg = F.linear(x, w_gate).clamp(max=limit)
        u = F.linear(x, w_up).clamp(min=-limit, max=limit)
        return F.linear(act8(F.silu(gg) * u), w_down)

    def experts8(x, routing, ew, limit):
        gr.clamped_swiglu = swiglu8
        try:
            return orig_experts(x, routing, ew, limit)
        finally:
            gr.clamped_swiglu = orig_swiglu

    gr.experts_forward = experts8

if gain_fix:
    _orig_ef = gr.experts_forward

    def _ef_scaled(*a, **k):
        return (_orig_ef(*a, **k).float() / 1.0122).to(a[0].dtype)

    gr.experts_forward = _ef_scaled

state = ref.new_state(seq)
for i in ref.layer_ids:
    ref.load_state(state, i, g.state(i, at=last), last)
t0 = time.time()
per_layer = {}


def rec(name, t):
    if name.endswith(".out") or name.endswith(".experts_out"):
        per_layer[name] = t.detach().to(torch.bfloat16).clone()


out, logits = ref.forward_chunk(tokens[last:seq], last, state, rec, logits_last_n=chunk)
print(f"chunk {time.time() - t0:.0f}s", flush=True)
ids = logits.float().topk(5, dim=-1).indices
want = tokens[last + 1 : seq]
gm = g.model(10)
t1 = float((ids[: chunk - 1, 0] == want).float().mean())
t5 = float((ids[: chunk - 1] == want[:, None]).any(-1).float().mean())
agree = float((ids[: chunk - 1, 0] == gm["top32_ids"][: chunk - 1, 0].long()).float().mean())


def rel(a, b):
    a, b = a.double(), b.double()
    return float((a - b).norm() / b.norm())


print(
    f"[ref] mode {mode} layers {sorted(only) if only else 'all'}: top1 vs text {t1:.4f} top5 {t5:.4f} "
    f"top1 vs golden argmax {agree:.4f} final norm rel {rel(out, gm['final_norm'].float()):.4f}",
    flush=True,
)
for i in (3, 4, 10, 20, 30, 40, 44):
    k = f"L{i}.out"
    if k in per_layer:
        print(
            f"[ref]   L{i} out rel vs golden {rel(per_layer[k].float(), g.layer(10, i)['out'].float()):.4f}", flush=True
        )
torch.save({"ids": ids, "out": out.to(torch.bfloat16)}, os.environ.get("RL_OUT", f"/tmp/ref_last_{mode}.pt"))
