import math, struct, torch, ttnn
from ttnn.operations.rms_norm.perf_experiments.root_finalize_scope.finalize_bench import (
    create_sharded_memory_config,
    run_op,
)

TILE = 32
W = 1024
EPS = 1e-12


def bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


rows = 2
n = rows * TILE
g = torch.Generator().manual_seed(7)
stat = torch.empty(n, TILE, dtype=torch.float32)
stat[:, 0] = ((2.0 * torch.randn(n, W, generator=g, dtype=torch.float32)) ** 2).sum(-1)
stat[:, 1:] = (1.0e4 * (1.0 + torch.arange(1, TILE, dtype=torch.float32))).unsqueeze(0).expand(n, TILE - 1)
x = torch.randn(n, TILE, generator=torch.Generator().manual_seed(11), dtype=torch.float32)
xb = x.to(torch.bfloat16).to(torch.float64)
device = ttnn.open_device(device_id=0)


def go(hoff):
    st = ttnn.from_torch(
        stat,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((n, TILE)),
    )
    xt = ttnn.from_torch(
        x.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((n, TILE)),
    )
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape([n, TILE]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, create_sharded_memory_config((n, TILE))
    )
    res = run_op(
        st,
        out,
        mode="consumer",
        variant="base",
        handoff=hoff,
        inv_w_bits=bits(1 / W),
        eps_bits=bits(EPS),
        eps_w_bits=bits(EPS * W),
        half_log2_w=5,
        rows=rows,
        x_tensor=xt,
    )
    return ttnn.to_torch(res).to(torch.float64)


try:
    o_raw = go("xfer_raw")  # HOFF=1 -> no finalize, raw stat
    exp_raw = xb * stat[:, 0].to(torch.float64).unsqueeze(1)
    print("raw   out[0,:3]", o_raw[0, :3].tolist())
    print("raw   exp[0,:3]", exp_raw[0, :3].tolist())
    print("raw   ratio", (o_raw[0, :3] / exp_raw[0, :3]).tolist())
    o_fin = go("inplace_copy")
    s0 = torch.rsqrt(stat[:, 0].to(torch.float64) / W + EPS)
    print("fin   out[0,:3]", o_fin[0, :3].tolist())
    print("fin   exp[0,:3]", (xb * s0.unsqueeze(1))[0, :3].tolist())
finally:
    ttnn.close_device(device)
