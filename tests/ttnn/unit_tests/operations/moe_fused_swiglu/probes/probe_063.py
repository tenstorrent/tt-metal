"""Probe: can the ring-joint-SDPA device-compare primitives (slice -> ne -> max) run on
this op's output format (bfloat8_b, TILE, DRAM interleaved) at the focus shape?"""
import time
import torch
import ttnn

device = ttnn.open_device(device_id=0)
try:
    cap, emb, rows = 5120, 7168, 256
    a = ttnn.from_torch(
        torch.randn(1, 1, cap, emb, dtype=torch.bfloat16),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )
    b = ttnn.clone(a)

    t0 = time.time()
    sa = ttnn.slice(a, [0, 0, 0, 0], [1, 1, rows, emb])
    sb = ttnn.slice(b, [0, 0, 0, 0], [1, 1, rows, emb])
    print("slice ok", sa.shape, sa.dtype, sa.layout)
    ne = ttnn.ne(sa, sb, dtype=ttnn.bfloat16)
    print("ne ok", ne.shape, ne.dtype)
    m = ttnn.max(ne)
    print("max ok", m.shape, m.dtype)
    ttnn.synchronize_device(device)
    print(
        f"equal-case compare wall time: {(time.time()-t0)*1e3:.2f} ms  marker={float(ttnn.to_torch(ttnn.from_device(m)).item())}"
    )

    # now poison one element and confirm the marker fires
    t = ttnn.to_torch(a)
    t[0, 0, 5, 7] += 4.0
    c = ttnn.from_torch(
        t, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    sc = ttnn.slice(c, [0, 0, 0, 0], [1, 1, rows, emb])
    m2 = ttnn.max(ttnn.ne(sa, sc, dtype=ttnn.bfloat16))
    print("poisoned marker =", float(ttnn.to_torch(ttnn.from_device(m2)).item()))

    # merge semantics
    m3 = ttnn.maximum(m, m2)
    print("merged marker =", float(ttnn.to_torch(ttnn.from_device(m3)).item()))
finally:
    ttnn.close_device(device)
