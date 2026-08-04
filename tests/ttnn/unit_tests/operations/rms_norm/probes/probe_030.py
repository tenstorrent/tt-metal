import torch, ttnn
from ttnn.operations.rms_norm import rms_norm
import ttnn.operations.rms_norm.rms_norm_program_descriptor as pd
from eval.sharding import auto_shard_config

ML = ttnn.TensorMemoryLayout
dev = ttnn.open_device(device_id=0)
c = ttnn.ComputeConfigDescriptor()
c.math_fidelity = ttnn.MathFidelity.HiFi2
c.fp32_dest_acc_en = False
c.math_approx_mode = False
CASES = [
    ((1, 1, 160, 11008), ML.HEIGHT_SHARDED),
    ((13, 777, 1023), ML.WIDTH_SHARDED),
    ((99991, 64), ML.BLOCK_SHARDED),
    ((1, 224, 11008), ML.HEIGHT_SHARDED),
]
torch.manual_seed(0)
print("BUDGET unreserved", ttnn.get_max_worker_l1_unreserved_size())
for shape, ml in CASES:
    x = torch.randn(*shape, dtype=torch.bfloat16)
    g = torch.randn(shape[-1], dtype=torch.bfloat16)
    mc = auto_shard_config(list(shape), ml, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=dev)
    xd = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    gd = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    print(
        "## ",
        shape,
        str(ml).split(".")[-1],
        "shard",
        list(mc.shard_spec.shape),
        "nc",
        mc.shard_spec.grid.num_cores(),
        "shard_bytes",
        pd._shard_l1_bytes(xd),
        flush=True,
    )
    try:
        out = rms_norm(xd, gamma=gd, compute_kernel_config=c, memory_config=xd.memory_config())
        print("OK", flush=True)
    except Exception as ex:
        print("ERR", str(ex)[:700].replace("\n", " | "), flush=True)
    del xd, gd
ttnn.close_device(dev)
