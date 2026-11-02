import math, torch, ttnn
import ttnn.operations.linear_gelu.linear_gelu_program_descriptor as pd
from ttnn.operations.linear_gelu import linear_gelu

device = ttnn.open_device(device_id=0)
pd.BLOCK_W_CANDIDATES = (2,)
orig = pd._candidate
g = device.compute_with_storage_grid_size()
pd._candidate = lambda t, bw, ms, ns, tb: orig(t, bw, ms, ns, tb) if (ms, ns) == (g.y, g.x) else None
for batch, seq in [(8, 512), (1, 416), (1, 96), (1, 128)]:
    torch.manual_seed(0)
    x = torch.randn((batch, 1, seq, 768)).to(torch.bfloat16)
    w = (torch.randn((768, 3072)) / math.sqrt(768)).to(torch.bfloat16)
    b = torch.randn((3072,)).to(torch.bfloat16)
    tile = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    out = linear_gelu(
        ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile),
        ttnn.from_torch(w, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile),
        ttnn.from_torch(b, memory_config=ttnn.DRAM_MEMORY_CONFIG, **tile),
        memory_config=ttnn.L1_MEMORY_CONFIG,
    )
    print(
        pd.select_blocking(
            ttnn.from_torch(x, memory_config=ttnn.L1_MEMORY_CONFIG, **tile),
            pd.ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True),
        )
    )
    ref = torch.nn.functional.gelu(x.double().reshape(-1, 768) @ w.double() + b.double(), approximate="tanh")
    act = ttnn.to_torch(out).reshape(-1, 3072).double()
    err = (act - ref).abs().reshape(-1, 32, 96, 32).amax(dim=(1, 3))
    bad = (err > 0.2).nonzero()
    print(
        batch,
        seq,
        "bad tiles",
        err.max().item(),
        len(bad),
        "m tiles",
        sorted(set(bad[:, 0].tolist()))[:40],
        "n tiles",
        sorted(set(bad[:, 1].tolist()))[:40],
    )
ttnn.close_device(device)
