import torch, ttnn
from ttnn.operations.g6_reshard import g6_reshard, last_plan, dump_roles
from ttnn.operations.g6_reshard import g6_reshard as mod_fn
import ttnn.operations.g6_reshard.g6_reshard as opmod
import ttnn.operations.g6_reshard.g6_reshard_program_descriptor as pdm

device = ttnn.open_device(device_id=0)
device.enable_program_cache()


def cfg(grid, shard, rm=True):
    (x0, y0), (x1, y1) = grid
    crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, list(shard), ttnn.ShardOrientation.ROW_MAJOR if rm else ttnn.ShardOrientation.COL_MAJOR),
    )


in_cfg = cfg(((0, 0), (10, 9)), (224, 96))
out_cfg = cfg(((0, 0), (7, 7)), (256, 128))
t = torch.randn(1, 1, 2048, 1024).to(torch.bfloat16)
x = ttnn.from_torch(t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=in_cfg)
xh = ttnn.to_torch(x)
outs = []
for i in range(3):
    o = g6_reshard(x, out_cfg)
    outs.append(o)
    print(
        "call",
        i,
        "cache entries",
        device.num_program_cache_entries(),
        "plan cache",
        len(opmod._PLAN_CACHE),
        "eq",
        torch.equal(ttnn.to_torch(o).view(torch.int16), xh.view(torch.int16)),
        "out addr",
        o.buffer_address(),
    )
print(last_plan())
roles = dump_roles(x, out_cfg)
print(len(roles), roles[0])
print(roles[-1])
# patch semantics: does pd.kernels[i].common_runtime_args reference in place?
pd = next(iter(opmod._PLAN_CACHE.values()))
k = pd.kernels[0]
print("type crt", type(k.common_runtime_args), list(k.common_runtime_args))
pd.kernels[0].common_runtime_args[0] = 12345
print("after in-place via pd.kernels[0]:", list(pd.kernels[0].common_runtime_args))
ttnn.close_device(device)
