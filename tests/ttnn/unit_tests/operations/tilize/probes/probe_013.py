import torch, ttnn


def show(name, t):
    print(
        f"{name}: logical={list(t.shape)} padded={list(t.padded_shape)} layout={t.layout} pages={t.buffer_num_pages()} page_size={t.buffer_page_size()}"
    )


# 1. rank-0 RM tensor on device
try:
    x = torch.randn(())
    tt = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    show("rank0 RM", tt)
except Exception as e:
    print("rank0 RM FAILED:", e)

# 2. allocate TILE with unaligned logical shape
for shp in ([1, 1, 50, 50], [1, 1], [3, 50, 96], [1, 1, 32, 50]):
    try:
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(shp), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
        )
        show(f"alloc {shp}", t)
    except Exception as e:
        print(f"alloc {shp} FAILED:", e)

# 3. view-reshape to a bigger padded shape
t = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 128, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
)
show("alloc 128", t)
try:
    v = ttnn.reshape(t, ttnn.Shape([1, 1, 50, 50]), ttnn.Shape([1, 1, 128, 128]))
    show("view 50/128", v)
    print("addr same:", v.buffer_address() == t.buffer_address())
    pr = v.cpu().to_torch_with_padded_shape()
    print("padded readback shape", list(pr.shape))
    print("logical readback shape", list(ttnn.to_torch(v).shape))
except Exception as e:
    print("view FAILED:", type(e).__name__, e)

# 4. w-only beyond tile-round
t2 = ttnn.allocate_tensor_on_device(
    ttnn.Shape([1, 1, 32, 128]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, ttnn.DRAM_MEMORY_CONFIG
)
try:
    v2 = ttnn.reshape(t2, ttnn.Shape([1, 1, 32, 50]), ttnn.Shape([1, 1, 32, 128]))
    show("view 32x50/32x128", v2)
    print("addr same:", v2.buffer_address() == t2.buffer_address())
except Exception as e:
    print("view2 FAILED:", type(e).__name__, e)
