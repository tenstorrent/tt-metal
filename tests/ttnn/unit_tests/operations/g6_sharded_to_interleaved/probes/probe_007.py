# BB-05 branch check: per-bank L1 bytes of the p1 output, computed through the op's own helper.
import torch, ttnn
from ttnn.operations.g6_sharded_to_interleaved.g6_sharded_to_interleaved import (
    l1_output_bytes_per_bank,
    _check_l1_output_budget,
    _num_l1_banks,
)

device = ttnn.open_device(device_id=0)
try:
    crs = ttnn.CoreRangeSet(
        [
            ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(10, 8)),
            ttnn.CoreRange(ttnn.CoreCoord(0, 9), ttnn.CoreCoord(3, 9)),
        ]
    )
    mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(crs, (320, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    x = ttnn.from_torch(
        torch.randn(1, 128, 256, 32).bfloat16(),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=mc,
    )
    print("bb05 l1 banks", _num_l1_banks(device), "per-bank bytes", l1_output_bytes_per_bank(x))
    _check_l1_output_budget(x)
    print("bb05 within budget: ok")
    ttnn.deallocate(x)
finally:
    ttnn.close_device(device)
