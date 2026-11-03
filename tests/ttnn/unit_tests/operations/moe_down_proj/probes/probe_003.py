import sys, torch, ttnn

sys.path.insert(0, "tests/ttnn/unit_tests/operations/moe_down_proj")
import test_moe_down_proj as T

device = ttnn.open_device(device_id=0)
x_dev, w_dev, c_dev, xq, wq = T.make_operands(device, 4096, T.LAYER0)
y = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 8416, 768]), ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, device, ttnn.L1_MEMORY_CONFIG
)
print("RES x addr", x_dev.buffer_address(), "y addr", y.buffer_address())
print("RES l1 size", device.l1_size_per_core() if hasattr(device, "l1_size_per_core") else None)
print(
    "RES base",
    ttnn.device.get_max_worker_l1_unreserved_size()
    if hasattr(ttnn.device, "get_max_worker_l1_unreserved_size")
    else None,
)
print("RES attrs", [a for a in dir(device) if "l1" in a.lower() or "alloc" in a.lower()])
ttnn.close_device(device)
