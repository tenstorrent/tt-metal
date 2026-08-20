import torch, ttnn
from eval.golden_tests.rms_norm.helpers import pytorch_rms_norm, create_ttnn_input_tensor
from ttnn.operations.rms_norm import rms_norm


def pcc(a, b):
    a = a.flatten().float()
    b = b.flatten().float()
    a = a - a.mean()
    b = b - b.mean()
    return (a @ b / (a.norm() * b.norm() + 1e-30)).item()


CASES = [
    # (shape, in_dtype, gamma_dtype, layout)
    ((1, 1, 32, 7168), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # focus
    ((1, 1, 8192, 1024), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # prefill_1024
    ((1, 1, 512, 1024), ttnn.bfloat8_b, ttnn.bfloat8_b, ttnn.TILE_LAYOUT),
    ((1, 1, 512, 1024), ttnn.float32, ttnn.float32, ttnn.TILE_LAYOUT),
    ((1, 1, 64, 4095), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # w_nonalign
    ((1, 1, 32, 17), ttnn.bfloat16, ttnn.bfloat16, ttnn.TILE_LAYOUT),  # smallest
]

torch.manual_seed(7)
for shape, dt, gdt, lay in CASES:
    W = shape[-1]
    x = torch.randn(shape)
    # gamma with a LOUD per-column signature: if a column's gamma were stale /
    # unfetched, the ratio out/(x/rms) would not match it.
    g = (torch.arange(W, dtype=torch.float32) % 31) * 0.1 + 0.5
    exp = pytorch_rms_norm(x, gamma=g)
    tx = create_ttnn_input_tensor(x, device, dtype=dt, layout=lay)
    tg = ttnn.from_torch(g, dtype=gdt, layout=ttnn.TILE_LAYOUT, device=device)
    out = ttnn.to_torch(rms_norm(tx, gamma=tg))
    p = pcc(out, exp)
    print(f"{str(shape):<22} {str(dt):<20} gamma={str(gdt):<20} pcc={p:.6f}")
