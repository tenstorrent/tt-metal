# Device type, stability of id(tensor.device()) used in the program cache key, L1/DRAM bank counts.
import torch, ttnn

device = ttnn.open_device(device_id=0)
try:
    x = ttnn.from_torch(torch.zeros(32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    ids = {id(x.device()) for _ in range(5)}
    print(
        f"PROBE device type={type(device).__name__} tensor.device() type={type(x.device()).__name__} distinct ids={len(ids)} same_as_handle={id(device) in ids}"
    )
    print(
        f"PROBE has id()={hasattr(x.device(), 'id')} l1_banks={ttnn.get_memory_view(device, ttnn.BufferType.L1).num_banks} dram_banks={ttnn.get_memory_view(device, ttnn.BufferType.DRAM).num_banks} grid={device.compute_with_storage_grid_size()}"
    )
finally:
    ttnn.close_device(device)
