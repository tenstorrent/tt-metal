import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:

    def check(name, shape, in_mem, out_mem, levers):
        t = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=in_mem)
        got = ttnn.to_torch(_dispatch(tt, out_mem, use_multicore=True, levers=levers))
        ok = torch.equal(got.float(), t.float())
        print(f"{name:30s} {levers} exact={ok}")

    D, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    g4 = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(3, 0))})
    sh = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(g4, (128, 64), ttnn.ShardOrientation.ROW_MAJOR),
    )
    for lv in (dict(), dict(wait_upfront=0), dict(tilize_uninit=0), dict(fold_resident=1)):
        check("sharded_small", (1, 1, 512, 64), sh, sh, lv)
        check("crossover_in", (1, 1, 512, 64), sh, D, lv)
    for lv in (dict(), dict(wait_upfront=0), dict(stateful_reads=1), dict(fast_addrgen=1)):
        check("smallest", (1, 1, 32, 64), D, D, lv)
        check("tail geom [1,1,96,288]", (1, 1, 96, 288), D, D, lv)
        check("big [1,1,512,512]", (1, 1, 512, 512), D, D, lv)
finally:
    ttnn.close_device(dev)
