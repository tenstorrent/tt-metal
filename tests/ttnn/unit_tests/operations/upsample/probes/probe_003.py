import torch, ttnn
from eval.memory import height_shards
from ttnn.operations.upsample import upsample
device = ttnn.open_device(device_id=0)
try:
    def run(shape, f, cores_override=None):
        grid = device.compute_with_storage_grid_size()
        count, shard = height_shards(shape, False, grid.x * grid.y)
        if cores_override: count, shard = cores_override, (shard[0]*shard[0]//shard[0] if False else (shape[0]*shape[1]*shape[2])//cores_override, shape[3])
        mc = ttnn.create_sharded_memory_config(shape=shard, core_grid=ttnn.num_cores_to_corerangeset(count, grid, row_wise=True),
            strategy=ttnn.ShardStrategy.HEIGHT, orientation=ttnn.ShardOrientation.ROW_MAJOR, use_height_and_width_as_shard_shape=True)
        t = torch.randn(*shape).to(torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=mc)
        y = upsample(x, f, memory_config=x.memory_config())
        sh, sw = (f, f) if isinstance(f, int) else f
        ref = t.repeat_interleave(sh, 1).repeat_interleave(sw, 2)
        out = ttnn.to_torch(y)
        print(shape, f, "cores", count, "shard", shard, "equal", torch.equal(out, ref), "out mc", y.memory_config())
    run((1,4,4,64), (2,2), cores_override=2)
    run((1,4,4,64), (2,2))
    run((1,20,20,512), (2,2))
    run((2,8,8,128), 3)
    run((1,16,16,32), 1)
finally:
    ttnn.close_device(device)
