import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
from eval.sharding import auto_shard_config

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi2
c.fp32_dest_acc_en = False
c.math_approx_mode = False


def ref(x, g, eps=1e-6):
    xf = x.to(torch.float32)
    return (xf / torch.sqrt(torch.mean(xf**2, dim=-1, keepdim=True) + eps)) * g.to(torch.float32).reshape(-1)


torch.manual_seed(0)
CASES = [
    ((1, 224, 11008), ML.BLOCK_SHARDED),
    ((1, 1, 96, 6144), ML.WIDTH_SHARDED),
    ((1, 1, 160, 11008), ML.WIDTH_SHARDED),
]
for shape, ml in CASES:
    x = torch.randn(*shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    print(
        "## ",
        shape,
        str(ml).split(".")[-1],
        "shard",
        list(mc.shard_spec.shape),
        "nc",
        mc.shard_spec.grid.num_cores(),
        mc.shard_spec.grid.bounding_box(),
    )
    try:
        xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        gd = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = rms_norm(xd, gamma=gd, compute_kernel_config=c, memory_config=xd.memory_config())
        a = ttnn.to_torch(out).float()
        e = ref(x, g)
        pcc = torch.corrcoef(torch.stack([a.flatten(), e.flatten()]))[0, 1].item()
        rms = ((a - e).pow(2).mean().sqrt() / e.pow(2).mean().sqrt()).item()
        print(f"OK  pcc={pcc:.6f} rms={rms:.5f}")
    except Exception as ex:
        print("ERR", type(ex).__name__, str(ex)[:300])
ttnn.close_device(dev)
