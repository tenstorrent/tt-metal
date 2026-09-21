"""Environment smoke, not an SDPA performance measurement."""
import json
import torch
import ttnn

torch.set_num_threads(8)
device = ttnn.open_device(device_id=0, trace_region_size=32 * 1024**2)
try:
    x = torch.ones((1, 1, 32, 32), dtype=torch.bfloat16)
    a = ttnn.from_torch(x, device=device, layout=ttnn.TILE_LAYOUT)
    actual = ttnn.to_torch(ttnn.matmul(a, a))
    assert torch.equal(actual, torch.full_like(x, 32))
    print(json.dumps({"environment_smoke": "pass", "logical_device": 0,
                      "grid": str(device.compute_with_storage_grid_size()),
                      "torch": torch.__version__, "ttnn": ttnn.__file__}))
finally:
    ttnn.close_device(device)
