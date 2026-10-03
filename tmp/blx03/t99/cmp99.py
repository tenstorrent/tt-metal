"""PSNR / PCC / max diff of the fused (on) vs unfused (off) decode outputs. Usage: cmp99.py <off.pt> <on.pt>"""
import sys

import torch

a = torch.load(sys.argv[1]).float()
b = torch.load(sys.argv[2]).float()
peak = 255.0 if a.max() > 2 else 1.0
mse = ((a - b) ** 2).mean().item()
psnr = float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(peak**2 / mse)).item()
pcc = torch.corrcoef(torch.stack([a.flatten(), b.flatten()]))[0, 1].item()
print(f"CMP99 shape={tuple(a.shape)} peak={peak} psnr_db={psnr:.2f} pcc={pcc:.6f} max_abs={(a - b).abs().max().item()}")
