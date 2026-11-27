import torch, ttnn, importlib
from ttnn.operations.g6_typecast import g6_typecast

m = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")
d = ttnn.open_device(device_id=0)
for shp in [(1, 2048, 2048, 3), (1, 1024, 1024, 3)]:
    x = ttnn.from_torch(torch.zeros(shp, dtype=torch.bfloat16), layout=ttnn.ROW_MAJOR_LAYOUT, device=d)
    g6_typecast(x, ttnn.uint8)
    print("PLAN", shp, m.last_plan, x.buffer_aligned_page_size(), x.buffer_page_size())
ttnn.close_device(d)
