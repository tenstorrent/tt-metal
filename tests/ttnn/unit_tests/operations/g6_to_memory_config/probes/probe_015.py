# Verifier probe: SUPPORTED claims in_buffer=dram x sharded in_scheme; no production case has a
# DRAM-sharded input. Check exactness + placement for each regime such an input selects.
import torch
import ttnn
from ttnn.operations.g6_to_memory_config import g6_to_memory_config, plan_of

device = ttnn.open_device(device_id=0)
try:
    dram = device.dram_grid_size()
    nb = dram.x
    dram_grid = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nb - 1, 0))])

    def dram_sharded(scheme, shard):
        return ttnn.MemoryConfig(
            scheme, ttnn.BufferType.DRAM, ttnn.ShardSpec(dram_grid, shard, ttnn.ShardOrientation.ROW_MAJOR)
        )

    l1_height = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(
            ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 0))]),
            (192, 5120),
            ttnn.ShardOrientation.ROW_MAJOR,
        ),
    )
    shape = [1536, 5120]
    wcfg = dram_sharded(ttnn.TensorMemoryLayout.WIDTH_SHARDED, (1536, 5120 // nb))
    hcfg = dram_sharded(ttnn.TensorMemoryLayout.HEIGHT_SHARDED, (1536 // nb, 5120))
    cases = [
        ("DRAM width -> L1 interleaved", wcfg, ttnn.L1_MEMORY_CONFIG),
        ("DRAM width -> DRAM interleaved", wcfg, ttnn.DRAM_MEMORY_CONFIG),
        ("DRAM width -> L1 height", wcfg, l1_height),
        ("DRAM height -> DRAM height", hcfg, hcfg),
        ("DRAM height -> L1 interleaved", hcfg, ttnn.L1_MEMORY_CONFIG),
    ]
    for name, in_cfg, out_cfg in cases:
        try:
            x = ttnn.from_torch(
                torch.randn(shape).bfloat16(),
                dtype=ttnn.bfloat16,
                layout=ttnn.TILE_LAYOUT,
                device=device,
                memory_config=in_cfg,
            )
            regime = plan_of(x, out_cfg)["regime"]
            y = g6_to_memory_config(x, out_cfg)
            ok = torch.equal(ttnn.to_torch(y), ttnn.to_torch(x)) and y.memory_config() == out_cfg
            print("RES", name, regime, "exact+placement" if ok else "MISMATCH")
            y.deallocate()
            x.deallocate()
        except Exception as e:
            print("RES", name, "EXC", repr(e)[:300])
finally:
    ttnn.close_device(device)
