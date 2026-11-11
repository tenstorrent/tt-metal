"""GPTQ-style 5-significant-bit rounding of the GDN in-projection weights (in_proj_qkv | z | a | b, one shared input)
for LoFi (LoFi truncates the weight to 5 significant bits; weights already on that grid are exact). Calibration: HF fp32 on
CPU over goldens NOT in the 100-item eval set; H = X^T X of each layer's in-proj input. Saves {layer: {qkv, z, a, b}} bf16
(5-bit values are exact in bf16) + a per-layer proxy error table (tr(dW H dW^T) / tr(W H W^T)) for RTN vs GPTQ.
argv: <out.pt> <threads> <n_calib per stratum>"""
import json, sys, time
from pathlib import Path
import torch

SNAP = "/mnt/models/huggingface/hub/models--black-forest-labs--flux-3-image-klein-pup/snapshots/3f3edb52b2c55de0d49feab28598b68f7f8e4830"
G = Path("/data/sraizada/pup-eval/tf/gold")
OUT, NT, NC = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
torch.set_num_threads(NT); torch.set_grad_enabled(False)
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: E402


def r5(t):
    m, e = torch.frexp(t)
    return torch.ldexp(torch.round(m * 32.0) / 32.0, e)


m = Qwen3_5ForConditionalGeneration.from_pretrained(SNAP, dtype=torch.float32, attn_implementation="eager").eval()
mods = {}
for n, mod in m.named_modules():
    if n.endswith("linear_attn.in_proj_qkv"):
        li = int(n.split("layers.")[1].split(".")[0])
        mods[li] = (n.rsplit(".", 1)[0], mod)
print("GDN layers", sorted(mods), flush=True)
H = {li: torch.zeros(4096, 4096, dtype=torch.float64) for li in mods}
cnt = {li: 0 for li in mods}


def hook(li):
    def f(mod, inp):
        x = inp[0].reshape(-1, inp[0].shape[-1]).double()
        H[li].addmm_(x.T, x); cnt[li] += x.shape[0]
    return f


hs = [mod.register_forward_pre_hook(hook(li)) for li, (_n, mod) in mods.items()]
evalset = set(sum(json.load(open(G / "tf_list_100.json")).values(), []))
man = [json.loads(l) for l in (G / "manifest.jsonl").read_text().splitlines()]
pick = []
for strat in ("t2i", "edit", "multiref"):
    c = [r["name"] for r in man if r["stratum"] == strat and r["name"] not in evalset]
    pick += c[:NC]
print("calibration items", pick, flush=True)
for name in pick:
    r = torch.load(G / f"{name}.pt", weights_only=False)
    kw = {}
    if r["pix"] is not None:
        kw = dict(pixel_values=r["pix"].float(), image_grid_thw=r["grid"])
        if r["mm"] is not None:
            mm = torch.zeros(r["ids"].shape, dtype=torch.int32); mm[:, : r["mm"].shape[1]] = r["mm"]
            kw["mm_token_type_ids"] = mm
    t0 = time.time()
    m(input_ids=r["ids"], logits_to_keep=1, **kw)
    print(f"calib {name} {r['ids'].shape[1]} tok {time.time() - t0:.0f}s", flush=True)
for h in hs:
    h.remove()

res, rep = {}, []
for li in sorted(mods):
    pre, _ = mods[li]
    la = m.get_submodule(pre)
    parts = [la.in_proj_qkv.weight, la.in_proj_z.weight, la.in_proj_a.weight, la.in_proj_b.weight]
    sizes = [p.shape[0] for p in parts]
    W0 = torch.cat(parts, 0).to(torch.bfloat16).double()  # checkpoint values [out, in]
    Hl = H[li] / max(cnt[li], 1)
    damp = 0.01 * Hl.diagonal().mean()
    Hd = Hl.clone(); Hd.diagonal().add_(damp)
    perm = torch.argsort(Hd.diagonal(), descending=True)  # act order
    W = W0[:, perm].clone(); Hp = Hd[perm][:, perm]
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hp)), upper=True)
    Q = torch.zeros_like(W); B = 128; n = W.shape[1]
    for i1 in range(0, n, B):
        i2 = min(i1 + B, n)
        W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1); Hi = Hinv[i1:i2, i1:i2]
        for j in range(i2 - i1):
            w = W1[:, j]; q = r5(w); Q1[:, j] = q
            e = (w - q) / Hi[j, j]
            W1[:, j:] -= e.unsqueeze(1) * Hi[j, j:].unsqueeze(0)
            E1[:, j] = e
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    Qg = torch.empty_like(Q); Qg[:, perm] = Q
    Qr = r5(W0)

    def perr(Wq):
        D = Wq - W0
        return float(((D @ Hl) * D).sum() / ((W0 @ Hl) * W0).sum())
    rep.append((li, perr(Qr), perr(Qg)))
    print(f"layer {li}: proxy rel err RTN {rep[-1][1]:.3e} GPTQ {rep[-1][2]:.3e} (x{rep[-1][1] / rep[-1][2]:.2f})", flush=True)
    assert torch.equal(r5(Qg), Qg)
    res[li] = {k: t.to(torch.bfloat16).contiguous() for k, t in zip(("qkv", "z", "a", "b"), torch.split(Qg, sizes, 0))}
torch.save({"layers": res, "report": rep, "calib": pick}, OUT)
print("GPTQ_DONE", OUT, flush=True)
