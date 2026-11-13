"""kv-design A4: bytes per row of an fp8_e4m3 ROW_MAJOR [1,1,N,656] tensor (SCALED_FP8 KV cache) and of bf16 [1,1,N,576] on one Blackhole chip, from the DRAM allocator view. Prints RES."""
import json

import ttnn

md = ttnn.open_mesh_device(ttnn.MeshShape(1, 1))
N = 131072


def alloc_total():
    v = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    return (
        v.total_bytes_allocated_per_bank * v.num_banks if hasattr(v, "num_banks") else v.total_bytes_allocated_per_bank
    )


def measure(shape, dtype):
    a0 = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    t = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), dtype, ttnn.ROW_MAJOR_LAYOUT, md, ttnn.DRAM_MEMORY_CONFIG)
    a1 = ttnn.get_memory_view(md, ttnn.BufferType.DRAM)
    d = (a1.total_bytes_allocated_per_bank - a0.total_bytes_allocated_per_bank) * a0.num_banks
    r = {
        "shape": shape,
        "dtype": str(dtype),
        "delta_bytes_all_banks": d,
        "bytes_per_row": d / shape[2],
        "num_banks": a0.num_banks,
    }
    try:
        r["aligned_page_size"] = t.buffer_aligned_page_size()
    except Exception as e:  # noqa
        r["aligned_page_size_err"] = str(e)[:80]
    ttnn.deallocate(t)
    return r


for shp, dt in (([1, 1, N, 656], ttnn.fp8_e4m3), ([1, 1, N, 576], ttnn.bfloat16), ([1, 1, N, 704], ttnn.fp8_e4m3)):
    print("RES " + json.dumps(measure(shp, dt)), flush=True)
ttnn.close_mesh_device(md)
