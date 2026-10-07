# t212: PCC / PSNR of the ported decode (B) against the unported one (A), uint8 pixels, frame by frame.
import json, sys, torch

a = torch.load(sys.argv[1])
b = torch.load(sys.argv[2])
assert a.shape == b.shape, (a.shape, b.shape)
t_dim = [i for i, s in enumerate(a.shape) if s in (145, 25)][0]
sa = sb = saa = sbb = sab = sse = 0.0
n = 0
worst = (1e9, -1)
for t in range(a.shape[t_dim]):
    x = a.select(t_dim, t).double()
    y = b.select(t_dim, t).double()
    d = ((x - y) ** 2).sum().item()
    m = x.numel()
    sa += x.sum().item()
    sb += y.sum().item()
    saa += (x * x).sum().item()
    sbb += (y * y).sum().item()
    sab += (x * y).sum().item()
    sse += d
    n += m
    p = 99.0 if d == 0 else 10 * torch.log10(torch.tensor(255.0**2 * m / d)).item()
    worst = min(worst, (p, t))
cov = sab / n - (sa / n) * (sb / n)
va = saa / n - (sa / n) ** 2
vb = sbb / n - (sb / n) ** 2
r = {
    "shape": list(a.shape),
    "pcc": cov / (va * vb) ** 0.5,
    "psnr_db": 99.0 if sse == 0 else 10 * torch.log10(torch.tensor(255.0**2 * n / sse)).item(),
    "worst_frame_psnr_db": worst[0],
    "worst_frame": worst[1],
    "identical": sse == 0,
}
print(json.dumps(r))
json.dump(r, open(sys.argv[3], "w"))
