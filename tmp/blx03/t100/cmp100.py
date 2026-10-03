"""md5 / PSNR / PCC / max diff of two decode outputs. Usage: cmp100.py <ref.pt> <new.pt>"""
import hashlib
import sys

import torch

a = torch.load(sys.argv[1])
b = torch.load(sys.argv[2])
md5 = [hashlib.md5(t.contiguous().numpy().tobytes()).hexdigest()[:12] for t in (a, b)]
a, b = a.float(), b.float()
peak = 255.0 if a.max() > 2 else 1.0
mse = ((a - b) ** 2).mean().item()
psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(peak**2 / mse)).item()
pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
print(
    f"CMP100 md5={md5[0]}/{md5[1]} identical={md5[0] == md5[1]} psnr_db={psnr:.2f} pcc={pcc:.6f} "
    f"max_abs={(a - b).abs().max().item()}"
)
