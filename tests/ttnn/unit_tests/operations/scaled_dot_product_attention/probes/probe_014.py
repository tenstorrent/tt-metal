import torch, ttnn, math
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention

dev = ttnn.open_device(device_id=0)
S, D = 512, 64
shape = (1, 1, S, D)
torch.manual_seed(42)


def run(Q, K, V):
    tq = ttnn.from_torch(Q, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tk = ttnn.from_torch(K, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    tv = ttnn.from_torch(V, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    return ttnn.to_torch(scaled_dot_product_attention(tq, tk, tv)).float()


def ref(Q, K, V):
    q, k, v = Q.float(), K.float(), V.float()
    s = q @ k.transpose(-2, -1) / math.sqrt(D)
    return torch.softmax(s, -1) @ v


def stats(name, out, exp):
    expb = exp.to(torch.bfloat16).float()
    flips = (out != expb).float().mean().item()
    print(
        f"{name}: flip_rate={flips*100:.3f}% maxdiff={(out-expb).abs().max():.6f} vs_fp32_max={(out-exp).abs().max():.6f}"
    )


# A) V = ones: O must be exactly 1 regardless of P
Q = torch.rand(shape, dtype=torch.bfloat16)
K = torch.rand(shape, dtype=torch.bfloat16)
V1 = torch.ones(shape, dtype=torch.bfloat16)
out = run(Q, K, V1)
print(f"A V=ones: max|O-1| = {(out-1).abs().max():.3e}, frac!=1 = {(out!=1).float().mean()*100:.3f}%")

# B) Q=K=0: P exactly uniform; O = mean(V)
Z = torch.zeros(shape, dtype=torch.bfloat16)
Vr = torch.rand(shape, dtype=torch.bfloat16)
stats("B Q=K=0", run(Z, Z, Vr), ref(Z, Z, Vr))

# C) full uniform case
stats("C uniform", run(Q, K, Vr), ref(Q, K, Vr))
ttnn.close_device(dev)
