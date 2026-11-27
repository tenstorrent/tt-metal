import torch, ttnn
from ttnn.operations.g6_copy import g6_copy, describe_plan
from ttnn.operations.g6_copy import g6_copy_program_descriptor as d

device = ttnn.open_device(device_id=0)
try:
    shape = (1, 12, 128, 128)
    s = torch.randn(shape)
    t = torch.randn(shape)
    src = ttnn.from_torch(
        s, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    dst = ttnn.from_torch(
        t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    print("device id", src.device().id())
    p = describe_plan(src)
    print({k: v for k, v in p.items() if k != "roles"})
    desc = d.get_program_descriptor(src, dst)
    desc.kernels[0].common_runtime_args[0] = 12345
    print("in-place patch:", desc.kernels[0].common_runtime_args[0] == 12345)
    out = g6_copy(src, dst)
    print("same obj", out is dst)
    r = ttnn.to_torch(dst)
    print("bit equal", torch.equal(r.view(torch.int32), s.view(torch.int32)))
finally:
    ttnn.close_device(device)
