import torch, ttnn
from ttnn.operations.examples.reduce_accumulate import run_op, create_sharded_memory_config, input_shape

TILE = 32


def mk(device, dim, N, seed=13):
    torch.manual_seed(seed)
    H, W = input_shape(dim, N)
    data = torch.rand(H, W)
    if dim == "row":
        golden = data.to(torch.float64).mean(dim=1)
    elif dim == "col":
        golden = data.to(torch.float64).mean(dim=0)
    else:
        golden = data.to(torch.float64).mean().reshape(1)
    xd = ttnn.from_torch(
        data.to(torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=create_sharded_memory_config((H, W)),
    )
    return xd, golden


def readout(out, dim):
    t = ttnn.to_torch(out).to(torch.float64)
    if dim == "row":
        return t[:, 0]
    if dim == "col":
        return t[0, :]
    return t[0, 0].reshape(1)


device = ttnn.open_device(device_id=0)
try:
    for dim in ("row", "col", "scalar"):
        for N in (1, 3, 8):
            xd, g = mk(device, dim, N)
            print(f"\n-- dim={dim} N={N}  golden[:3]={[round(v,4) for v in g[:3].tolist()]}")
            for v in ("helper", "fast"):
                out = run_op(xd, variant=v, dim=dim, num_tiles=N, accum="fp32", kernel_iters=1)
                r = readout(out, dim)
                mx = (r - g).abs().max().item()
                print(f"   {v:7s} out[:3]={[round(x,4) for x in r[:3].tolist()]}  max_abs={mx:.5f}")
finally:
    ttnn.close_device(device)
