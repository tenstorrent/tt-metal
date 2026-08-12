import torch, ttnn
from ttnn.operations.tilize.tilize import _dispatch

dev = ttnn.open_device(device_id=0)
try:

    def check(name, shape, in_mem, out_mem, levers):
        t = torch.arange(torch.tensor(shape).prod().item()).reshape(shape).to(torch.bfloat16)
        tt = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=in_mem)
        got = ttnn.to_torch(_dispatch(tt, out_mem, use_multicore=True, levers=levers))
        ok = torch.equal(got.float(), t.float())
        print(f"{name:34s} {levers} exact={ok}")
        if not ok:
            print("  max diff", (got.float() - t.float()).abs().max().item())

    D, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
    for lv in (
        dict(),
        dict(fast_addrgen=0),
        dict(fast_addrgen=1, stateful_reads=1),
        dict(stateful_reads=1, fast_addrgen=0),
    ):
        check("smallest", (1, 1, 32, 64), D, D, lv)
        check("smallest_aligned", (1, 1, 32, 32), D, D, lv)
        check("tall [1,1,256,64]", (1, 1, 256, 64), D, D, lv)
        check("tail geom [1,1,96,288]", (1, 1, 96, 288), D, D, lv)
        check("rank3 [3,64,128]", (3, 64, 128), D, D, lv)
        check("rank2 [64,192]", (64, 192), D, D, lv)
        check("l1_to_l1 [1,1,128,256]", (1, 1, 128, 256), L1, L1, lv)
        check("dram->l1 [1,1,64,128]", (1, 1, 64, 128), D, L1, lv)
        check("big [1,1,512,512]", (1, 1, 512, 512), D, D, lv)
finally:
    ttnn.close_device(dev)
