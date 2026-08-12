import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:

    def check(name, shape, in_mem, out_mem, levers, pad=None):
        t = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=in_mem)
        out = _dispatch(tt, out_mem, use_multicore=True, levers=levers, **(pad or {}))
        got = ttnn.to_torch(out)
        ok = torch.equal(got.float(), t.float())
        print(f"{name:44s} levers={levers} exact={ok}")
        if not ok:
            print("  max diff", (got.float() - t.float()).abs().max().item())

    D = ttnn.DRAM_MEMORY_CONFIG
    L1 = ttnn.L1_MEMORY_CONFIG
    for lv in (dict(), dict(stateful_reads=0), dict(stateful_reads=1)):
        check("smallest [1,1,32,64]", (1, 1, 32, 64), D, D, lv)
        check("smallest_aligned [1,1,32,32]", (1, 1, 32, 32), D, D, lv)
        check("tall_narrow-ish [1,1,256,64]", (1, 1, 256, 64), D, D, lv)
        check("tail geom [1,1,96,288]", (1, 1, 96, 288), D, D, lv)
        check("l1_to_l1 [1,1,128,256]", (1, 1, 128, 256), L1, L1, lv)
        check("dram->l1 [1,1,64,128]", (1, 1, 64, 128), D, L1, lv)

    # sharded same-spec -> fold
    grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    sh = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(grid, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    for lv in (dict(), dict(fold_resident=0), dict(fold_resident=1)):
        check("sharded_small [1,1,512,64]", (1, 1, 512, 64), sh, sh, lv)
finally:
    ttnn.close_device(dev)
