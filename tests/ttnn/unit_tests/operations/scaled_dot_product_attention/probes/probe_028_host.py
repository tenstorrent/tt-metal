"""Host-only model of the normalize path to localize the 28% flip source.
No device. Compares: exact mean (RNE) vs. (sum(V)) * recip_approx(S), where
recip_approx applies the measured SFPU recip relative error."""
import torch

torch.manual_seed(42)
S, D = 512, 64
shape = (1, 1, S, D)

# fine grid V on 2^-9 in [0.5,1)
Vf = (0.5 + torch.randint(0, 256, shape) * (2**-9)).to(torch.bfloat16)

sumV = Vf.double().sum(-2, keepdim=True)  # exact fp32-representable sum
exact_mean = sumV / S  # exact 1/512
rne = exact_mean.float().to(torch.bfloat16).float()

# Measured device recip: INV=0.001953054 for l=512 vs exact 0.001953125
recip_exact = 1.0 / S
recip_dev = 0.001953054
rel_err = recip_dev / recip_exact - 1.0
print(f"recip rel_err = {rel_err:.3e}  (abs {recip_dev-recip_exact:.3e})")

# Model 1: only recip error (multiplicative), then RNE pack
o = sumV.float()  # O accumulate assumed exact (matmul exact for P=1)
out_recip = (o.double() * recip_dev).float().to(torch.bfloat16).float()
flips_recip = (out_recip != rne).float().mean().item() * 100
print(f"Model[recip-only] flips = {flips_recip:.2f}%")


# Model 2: exact recip, but O carries a tf32-truncation per-accumulate (matmul)
# tf32: 10-bit mantissa. Truncate each V to tf32 before summing.
def to_tf32(x):
    b = x.float().view(torch.int32)
    b = (b + 0x1000) & ~0x1FFF  # RNE-ish to 10-bit mantissa (19-bit total)
    return b.view(torch.float32)


o_tf32 = to_tf32(Vf.float()).double().sum(-2, keepdim=True).float()
out_tf32 = (o_tf32.double() * recip_exact).float().to(torch.bfloat16).float()
flips_tf32 = (out_tf32 != rne).float().mean().item() * 100
print(f"Model[tf32 matmul, exact recip] flips = {flips_tf32:.2f}%")

# Model 3: both
out_both = (o_tf32.double() * recip_dev).float().to(torch.bfloat16).float()
print(f"Model[both] flips = {(out_both != rne).float().mean().item()*100:.2f}%")

# Distance to midpoint distribution: how many exact means are within rel_err of a tie?
em = exact_mean.float().flatten().double()
# bf16 ulp at each value
import math


def bf16_ulp(v):
    e = torch.floor(torch.log2(torch.abs(v).clamp_min(1e-30)))
    return torch.pow(2.0, e - 7)


ulp = bf16_ulp(em)
# fractional position within ulp grid
q = em / ulp
frac = q - torch.floor(q)  # 0..1, .5 is midpoint
dist_to_mid = (frac - 0.5).abs()
within = (dist_to_mid < abs(rel_err) * em / ulp).float().mean().item() * 100
print(f"frac within recip-rel-err of midpoint = {within:.2f}%")
