"""Verify transpose(gate,1,3)+reshape == direct reshape for the size-1-neighbor MoE case; time both."""
import ttnn, torch, time

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
# gate sparse_matmul output shape observed in denoise MoE: (1,1,1,E=128,32,192)
E, T, I = 128, 32, 192
x = torch.randn(1, 1, 1, E, T, I) * 0.1
xt = ttnn.from_torch(
    x.bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG
)


def old():  # transpose(1,3) then reshape (what shared code does)
    g = ttnn.transpose(xt, 1, 3)
    g = ttnn.reshape(g, (1, E, T, I))
    return g


def new():  # direct reshape (hypothesized equivalent, no data move)
    return ttnn.reshape(xt, (1, E, T, I))


try:
    o = ttnn.to_torch(old()).float()
    n = ttnn.to_torch(new()).float()
    print(f"out={tuple(o.shape)}  bitexact={torch.equal(o,n)}  max|diff|={(o-n).abs().max():.2e}")
    for f, name in [(old, "transpose(1,3)+reshape"), (new, "direct reshape")]:
        for _ in range(3):
            y = f()
            ttnn.synchronize_device(dev)
            y.deallocate(True)
        N = 30
        t0 = time.perf_counter()
        for _ in range(N):
            y = f()
            ttnn.synchronize_device(dev)
            y.deallocate(True)
        print(f"{name:26s}: {(time.perf_counter()-t0)/N*1e3:.3f} ms/call")
except Exception as e:
    import traceback

    traceback.print_exc()
    print("FAILED", type(e).__name__, str(e)[:200])
finally:
    ttnn.close_device(dev)
