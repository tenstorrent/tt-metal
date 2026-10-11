import torch

import ttnn

m = ttnn.open_mesh_device(ttnn.MeshShape(1, 1), trace_region_size=0)
try:
    q = ttnn.from_torch(torch.randn(1, 32, 32, 128), device=m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    k = ttnn.zeros((4, 2, 32, 128), device=m, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    p = ttnn.from_torch(torch.arange(4, dtype=torch.int32)[None], device=m, dtype=ttnn.int32)
    ck = ttnn.init_device_compute_kernel_config(m.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True)
    pc = ttnn.SDPAProgramConfig(compute_with_storage_grid_size=ttnn.CoreCoord(8, 8), q_chunk_size=32, k_chunk_size=32)
    for kw in (
        {},
        {"scale": 0.08838834764831845},
        {"sliding_window_size": 2048},
        {"program_config": pc},
        {"compute_kernel_config": ck},
    ):
        print("TRY", kw, flush=True)
        try:
            o = ttnn.transformer.chunked_scaled_dot_product_attention(q, k, k, p, 0, **kw)
            ttnn.synchronize_device(m)
            print("PASS", flush=True)
        except (TypeError, RuntimeError) as e:
            print(type(e).__name__, str(e)[:600], flush=True)
finally:
    ttnn.close_mesh_device(m)
