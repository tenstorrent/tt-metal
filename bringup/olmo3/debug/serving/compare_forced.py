import json, sys, torch

dev = torch.load(sys.argv[1]).float()
ref = torch.load(sys.argv[2]).float()
cfg = json.load(open("scratch/force_3600.json"))
ids = cfg["ids"]
N = cfg["prefill_len"]
n = min(dev.shape[0], ref.shape[0])
print("rows", dev.shape[0], ref.shape[0])
dl = torch.log_softmax(dev[:n], -1)
rl = torch.log_softmax(ref[:n], -1)
tgt = torch.tensor(ids[N : N + n])


def pcc(a, b):
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


P = pcc(dev[:n], ref[:n])
top = (dev[:n].argmax(-1) == ref[:n].argmax(-1)).float()
dlp = dl.gather(1, tgt[:, None])[:, 0]
rlp = rl.gather(1, tgt[:, None])[:, 0]
kl = (rl.exp() * (rl - dl)).sum(-1)
for s in range(0, n, 100):
    sl = slice(s, s + 100)
    print(
        f"abs pos {N+s:5d}-{N+min(s+100,n):5d}: pcc {P[sl].mean():.4f} (min {P[sl].min():.3f})  top1-agree {top[sl].mean():.2f}  mean logp(true tok) dev {dlp[sl].mean():.2f} ref {rlp[sl].mean():.2f}  KL(ref||dev) {kl[sl].mean():.3f}"
    )
