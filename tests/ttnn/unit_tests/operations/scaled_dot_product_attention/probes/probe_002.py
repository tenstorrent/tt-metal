import torch, ttnn
from ttnn.operations.scaled_dot_product_attention import scaled_dot_product_attention
from tests.ttnn.utils_for_testing import comp_pcc

dev = ttnn.open_device(device_id=0)


def ref(Q, K, V):
    return torch.nn.functional.scaled_dot_product_attention(Q.float(), K.float(), V.float())


def run(name, Q, K, V):
    exp = ref(Q, K, V)
    to = lambda t: ttnn.from_torch(
        t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    out = ttnn.to_torch(scaled_dot_product_attention(to(Q), to(K), to(V))).float()
    pcc = comp_pcc(exp, out, 0.0)
    diff = out - exp
    rms = torch.sqrt((diff**2).mean()) / exp.std()
    # got/true ratio
    m = exp.abs() > 1e-6
    r = out[m] / exp[m]
    print(
        f"{name}: pcc={pcc[1]}  relRMS={rms:.4f}  ratio med={r.median():.4f} p5={r.quantile(0.05):.4f} p95={r.quantile(0.95):.4f}"
    )


torch.manual_seed(42)
for name, scale, shape in [
    ("uniform_32", None, (1, 1, 32, 32)),
    ("uniform_128", None, (1, 4, 128, 64)),
    ("large_128", "L", (1, 4, 128, 64)),
    ("neg_128", "N", (1, 4, 128, 64)),
]:
    if scale == "L":
        Q = (torch.randn(shape) * 10).to(torch.bfloat16)
        K = (torch.randn(shape) * 10).to(torch.bfloat16)
        V = (torch.randn(shape) * 10).to(torch.bfloat16)
    elif scale == "N":
        Q = -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)
        K = -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)
        V = -(torch.rand(shape, dtype=torch.bfloat16) + 0.5)
    else:
        Q = torch.rand(shape, dtype=torch.bfloat16)
        K = torch.rand(shape, dtype=torch.bfloat16)
        V = torch.rand(shape, dtype=torch.bfloat16)
    run(name, Q, K, V)
ttnn.close_device(dev)
