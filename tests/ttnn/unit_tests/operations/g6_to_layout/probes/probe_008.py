import torch, ttnn


def render(v):
    p = ttnn.from_torch(v, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)
    return ttnn.to_torch(p).to(torch.bfloat16).reshape(v.shape)


SPECIAL_BITS = (0x8000, 0x0001, 0x8001, 0x007F, 0x7F80, 0xFF80, 0x7FC1)
torch.manual_seed(0)
x = torch.randn(32, 32, dtype=torch.bfloat16)
f = x.view(torch.int16)
for i, b in enumerate(SPECIAL_BITS):
    f[i, 0] = b - 0x10000 if b >= 0x8000 else b
r = render(x)
rb = r.view(torch.int16)
for i in range(7):
    print(
        hex(SPECIAL_BITS[i]),
        "row",
        i,
        "in:",
        [hex(int(v) & 0xFFFF) for v in f[i, :4]],
        "out:",
        [hex(int(v) & 0xFFFF) for v in rb[i, :4]],
    )
torch.manual_seed(1)
y = torch.randn(1, 32, 64, 256, dtype=torch.bfloat16)
ry = render(y).view(torch.int16)
print(
    "neg zeros:",
    int((ry == -0x8000).sum()),
    "pos zeros:",
    int((ry == 0).sum()),
    "exp0:",
    int(((ry & 0x7F80) == 0).sum()),
)
